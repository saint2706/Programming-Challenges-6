import hashlib

import numpy as np
import pytest
from embed import (
    CachedEncoder,
    ChunkedEncoder,
    EmbeddingCache,
    EmbeddingScorer,
    chunk_tokens,
    pick_backend,
)


def test_chunk_tokens_covers_everything_with_overlap_and_bounded_size():
    chunks = chunk_tokens(list(range(600)), window=256, stride=192)
    assert chunks[0][0] == 0
    assert chunks[-1][-1] == 599
    assert all(len(c) <= 256 for c in chunks)
    assert {t for c in chunks for t in c} == set(range(600))


def test_chunk_tokens_short_and_empty_inputs_give_one_chunk():
    assert chunk_tokens([1, 2, 3], window=256, stride=192) == [[1, 2, 3]]
    assert chunk_tokens([], window=256, stride=192) == [[]]


class FakeRunner:
    """Maps each window to a normalised bag-of-ids vector; records its calls."""

    batch_size = 4
    seq_len = 16
    dim = 8

    def __init__(self):
        self.calls = []

    def run(self, input_ids, attention_mask):
        self.calls.append((input_ids.copy(), attention_mask.copy()))
        out = np.zeros((input_ids.shape[0], self.dim))
        for r, (ids, mask) in enumerate(zip(input_ids, attention_mask, strict=True)):
            for tok, m in zip(ids, mask, strict=True):
                if m:
                    out[r, tok % self.dim] += 1
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1, norm)


def _encoder(runner=None, **kw):
    runner = runner or FakeRunner()
    return ChunkedEncoder(
        tokenize=lambda s: [int(t) for t in s.split()],
        runner=runner,
        cls_id=1,
        sep_id=2,
        pad_id=0,
        stride=kw.get("stride", 8),
        max_chunks=kw.get("max_chunks", 8),
    )


def test_windows_are_wrapped_in_special_tokens_and_padded_to_the_static_shape():
    runner = FakeRunner()
    _encoder(runner).encode(["5 6 7"])
    ids, mask = runner.calls[0]
    assert ids.shape == (4, 16) and mask.shape == (4, 16)  # padded to the static batch
    assert ids[0, :5].tolist() == [1, 5, 6, 7, 2]
    assert mask[0].tolist() == [1] * 5 + [0] * 11
    assert (
        mask[1, 0] == 1
    )  # padding rows keep one live token so softmax never sees all -inf


def test_long_text_uses_several_windows_and_max_chunks_caps_them():
    long = " ".join(str(i % 40 + 3) for i in range(100))

    def real_windows(
        runner,
    ):  # real windows start with [CLS]=1, padding rows with pad=0
        return sum(int((ids[:, 0] == 1).sum()) for ids, _ in runner.calls)

    uncapped, capped = FakeRunner(), FakeRunner()
    _encoder(uncapped).encode([long])
    _encoder(capped, max_chunks=2).encode([long])
    assert real_windows(uncapped) > 2
    assert real_windows(capped) == 2


def test_vectors_are_unit_norm_and_empty_text_is_finite():
    vecs = _encoder().encode(["5 6 7", "", "9 9 9 9"])
    assert vecs.shape == (3, 8)
    assert np.isfinite(vecs).all()
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6)


def test_encode_nothing_returns_an_empty_matrix_of_the_right_width():
    assert _encoder().encode([]).shape == (0, 8)


class FakeEncoder:
    """Deterministic per-text unit vectors; ``noise`` corrupts them."""

    def __init__(self, noise=0.0, seed=0, fail=False):
        self.noise, self.seed, self.fail = noise, seed, fail

    def encode(self, texts):
        if self.fail:
            raise RuntimeError("device lost")
        rows = []
        for t in texts:
            base = np.random.default_rng(
                int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2**32
            ).normal(size=16)
            jitter = (
                np.random.default_rng(self.seed + 1).normal(size=16)
                * self.noise
                * np.linalg.norm(base)
            )
            v = base + jitter
            rows.append(v / np.linalg.norm(v))
        return np.array(rows)


PROBE = [
    "data analyst with sql",
    "registered nurse on a ward",
    "executive chef",
    "python developer",
]


def test_pick_backend_rejects_a_corrupted_backend_and_takes_the_next_good_one():
    choice = pick_backend(
        [("bad", lambda: FakeEncoder(noise=2.0)), ("good", lambda: FakeEncoder())],
        PROBE,
        reference=FakeEncoder(),
    )
    assert choice.name == "good"
    assert choice.tried[0][0] == "bad" and "rejected" in choice.tried[0][1]
    assert choice.tried[1][1].startswith("ok")


def test_pick_backend_skips_a_backend_that_raises_or_cannot_be_built():
    def cannot_build():
        raise ImportError("no openvino")

    choice = pick_backend(
        [
            ("nobuild", cannot_build),
            ("crash", lambda: FakeEncoder(fail=True)),
            ("good", lambda: FakeEncoder()),
        ],
        PROBE,
        reference=FakeEncoder(),
    )
    assert choice.name == "good"
    assert [s.split(":")[0] for _, s in choice.tried] == [
        "unavailable",
        "unavailable",
        "ok",
    ]


def test_pick_backend_falls_back_to_the_reference_when_every_candidate_fails():
    ref = FakeEncoder()
    choice = pick_backend(
        [("bad", lambda: FakeEncoder(noise=2.0))],
        PROBE,
        reference=ref,
        reference_name="torch-cpu",
    )
    assert choice.encoder is ref and choice.name == "torch-cpu"


def test_cache_roundtrip_persists_and_keys_depend_on_the_model(tmp_path):
    path = tmp_path / "emb.sqlite"
    inner = FakeEncoder()
    enc = CachedEncoder(inner, EmbeddingCache(path), model_id="m1")
    first = enc.encode(["a", "b", "a"])
    again = CachedEncoder(
        FakeEncoder(fail=True), EmbeddingCache(path), model_id="m1"
    ).encode(["b", "a"])
    np.testing.assert_allclose(
        again, first[[1, 0]], atol=1e-6
    )  # served from disk, inner never called
    with pytest.raises(RuntimeError):
        CachedEncoder(
            FakeEncoder(fail=True), EmbeddingCache(path), model_id="m2"
        ).encode(["a"])


def test_cache_encodes_each_distinct_text_once():
    class Counting(FakeEncoder):
        calls = 0

        def encode(self, texts):
            Counting.calls += len(texts)
            return super().encode(texts)

    enc = CachedEncoder(Counting(), EmbeddingCache(":memory:"), model_id="m")
    enc.encode(["a", "b", "a", "a"])
    assert Counting.calls == 2


def test_embedding_scorer_ranks_by_cosine_and_fit_is_a_noop():
    enc = FakeEncoder()
    scorer = EmbeddingScorer(enc).fit([])
    docs = ["alpha doc", "beta doc", "gamma doc"]
    out = scorer.score("beta doc", docs)
    assert out.argmax() == 1 and out[1] == pytest.approx(1.0)
    assert scorer.score_matrix(["alpha doc", "gamma doc"], docs).shape == (2, 3)


@pytest.fixture(scope="module")
def real_torch_encoder():
    pytest.importorskip("transformers")
    from embed import TorchRunner, load_chunked

    try:
        return load_chunked(TorchRunner)
    except OSError:
        pytest.skip("embedding model not available offline")


def test_real_model_puts_related_texts_closer(real_torch_encoder):
    v = real_torch_encoder.encode(
        [
            "senior data analyst: sql, tableau, dashboards",
            "business intelligence analyst using sql and tableau",
            "line cook preparing sauces in a busy kitchen",
        ]
    )
    assert v[0] @ v[1] > v[0] @ v[2]


def test_openvino_cpu_matches_torch_on_the_real_model(real_torch_encoder):
    pytest.importorskip("openvino")
    from embed import OpenVinoRunner, load_chunked

    ov_enc = load_chunked(lambda model_id: OpenVinoRunner(model_id, "CPU"))
    texts = [
        "senior data analyst: sql, tableau, dashboards",
        "line cook preparing sauces",
    ]
    cos = (ov_enc.encode(texts) * real_torch_encoder.encode(texts)).sum(axis=1)
    assert cos.min() > 0.99
