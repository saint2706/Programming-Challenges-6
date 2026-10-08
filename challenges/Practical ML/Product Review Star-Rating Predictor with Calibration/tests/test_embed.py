import numpy as np
import pytest
from helpers import FakeRunner, fake_tokenize
from review_stars import embed
from review_stars.embed import (
    BackendChoice,
    BucketedEncoder,
    embed_texts,
    pick_backend,
    run_windows,
)

DIM = 8


def runners(batch=4, cls=FakeRunner, **kw):
    return {b: cls(b, batch, DIM, **kw) for b in embed.BUCKETS}


def encoder(rs=None):
    return BucketedEncoder(fake_tokenize, rs or runners(), pad_id=0)


# ---------------------------------------------------------------- windows and the encoder


def test_run_windows_pads_the_last_batch_and_returns_one_row_per_input():
    r = FakeRunner(64, 4, DIM)
    ids = np.arange(1, 10 * 64 + 1).reshape(10, 64) % 50 + 3
    mask = np.ones_like(ids)
    out = run_windows(r, ids, mask)
    assert out.shape == (10, DIM)
    solo = run_windows(FakeRunner(64, 4, DIM), ids[7:8], mask[7:8])
    assert np.allclose(out[7], solo[0])
    assert run_windows(r, ids[:0], mask[:0]).shape == (0, DIM)


def test_padding_rows_keep_one_live_token_so_attention_never_sees_an_all_masked_row():
    r = FakeRunner(64, 4, DIM)
    ids = np.full((6, 64), 7)
    run_windows(r, ids, np.ones_like(ids))
    last = r.masks[-1]  # 2 real rows, 2 padding rows
    assert (last.sum(axis=1) >= 1).all() and last[2:, 0].tolist() == [1, 1]
    assert last[2:, 1:].sum() == 0


def test_encode_gives_unit_vectors_in_input_order_with_token_counts():
    texts = ["great", "this is a somewhat longer review of the thing", "ok"]
    X, n_tok = encoder().encode(texts)
    assert X.shape == (3, DIM) and X.dtype == np.float32 and n_tok.dtype == np.int32
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)
    assert n_tok.tolist() == [3, 11, 3]  # [CLS] + one id per word + [SEP]
    X2, _ = encoder().encode(texts[::-1])
    assert np.allclose(X2, X[::-1])


def test_each_review_goes_to_the_smallest_bucket_that_holds_it():
    rs = runners()
    short = "word " * 10
    mid = "word " * 100
    long = "word " * 200
    huge = "word " * 2000  # truncated at 512 tokens
    X, n_tok = BucketedEncoder(fake_tokenize, rs, pad_id=0).encode(
        [short, mid, long, huge]
    )
    used = {b: r.calls for b, r in rs.items()}
    assert used == {64: 1, 128: 1, 256: 1, 512: 1}
    assert n_tok.tolist() == [12, 102, 202, 512]
    assert np.isfinite(X).all()


def test_the_vector_does_not_depend_on_the_bucket_or_on_what_else_is_in_the_batch():
    base = "a few words about a dishwasher"
    alone, _ = encoder().encode([base])
    crowd, _ = encoder().encode(["x " * 200, base, "y", "z " * 80])
    assert np.allclose(alone[0], crowd[1], atol=1e-6)


def test_empty_one_word_emoji_and_non_english_reviews_encode_to_finite_unit_vectors():
    texts = ["", "Great", "\U0001f44d\U0001f44d", "Très bien, merci", "非常好", "   "]
    X, n_tok = encoder().encode(texts)
    assert np.isfinite(X).all() and np.allclose(
        np.linalg.norm(X, axis=1), 1.0, atol=1e-6
    )
    assert n_tok.min() >= 2  # [CLS] and [SEP] even for an empty review


def test_encoding_nothing_returns_empty_arrays():
    X, n_tok = encoder().encode([])
    assert X.shape == (0, DIM) and n_tok.shape == (0,)


# ---------------------------------------------------------------- the self-check


class Scaled(FakeRunner):
    """A runner that disagrees with the reference at one sequence length."""

    def __init__(self, seq_len, batch_size, dim, bad_at=(), nan_at=()):
        super().__init__(seq_len, batch_size, dim)
        self.bad_at, self.nan_at = bad_at, nan_at

    def run(self, input_ids, attention_mask):
        out = super().run(input_ids, attention_mask)
        if self.seq_len in self.nan_at:
            out = out.copy()
            out[0, 0] = np.nan
        if self.seq_len in self.bad_at:
            rng = np.random.default_rng(self.seq_len)
            out = out + 5.0 * rng.normal(size=out.shape)
        return out


def reference():
    return runners(batch=2)


def build(**kw):
    return lambda: runners(cls=Scaled, **kw)


def crash():
    raise RuntimeError("device lost")


def choose(candidates, min_cos=0.99):
    return pick_backend(
        candidates, fake_tokenize, pad_id=0, reference=reference(), min_cos=min_cos
    )


def test_pick_backend_takes_the_first_candidate_that_agrees_on_every_bucket():
    choice = choose(
        [("bad", build(bad_at=(128,))), ("good", build()), ("also", build())]
    )
    assert choice.name == "good"
    assert (
        choice.tried[0][1].startswith("rejected: mean cosine")
        and "128" in choice.tried[0][1]
    )
    assert choice.tried[1][1].startswith("ok: mean cosine")
    assert set(choice.cosines) == set(embed.BUCKETS)


def test_a_device_that_is_only_wrong_on_the_long_buckets_is_still_rejected():
    # the real OpenVINO CPU device returned NaN at 128+ tokens and was fine at 64
    choice = choose([("cpu", build(nan_at=(128, 256, 512)))])
    assert choice.name == "torch-cpu"
    verdict = choice.tried[0][1]
    assert (
        verdict.startswith("rejected") and "non-finite" in verdict and "128" in verdict
    )


def test_a_crashing_candidate_is_skipped_and_the_reference_is_the_last_resort():
    choice = choose([("gpu", crash), ("bad", build(bad_at=(64,)))])
    assert choice.name == "torch-cpu"
    assert choice.tried[0][1].startswith("unavailable: RuntimeError")
    assert choice.tried[-1] == ("torch-cpu", "ok: reference")
    X, _ = choice.encoder.encode(["still works"])
    assert np.isfinite(X).all()


def test_the_npu_is_never_tried():
    with pytest.raises(ValueError, match="NPU"):
        embed.default_choice(devices=("GPU", "NPU"))


def test_probe_windows_fill_every_bucket_to_capacity_with_real_tokens():
    wins = embed.probe_windows(fake_tokenize, pad_id=0)
    assert set(wins) == set(embed.BUCKETS)
    for b, (ids, mask) in wins.items():
        assert ids.shape == mask.shape and ids.shape[1] == b
        assert (
            mask.sum(axis=1).max() == b
        )  # one probe review is long enough to fill the window
        assert mask.sum(axis=1).min() >= 2


# ---------------------------------------------------------------- shards


class CountingEncoder:
    """Wraps an encoder, counts texts, and can die after some number of calls."""

    def __init__(self, inner, die_on_call=None):
        self.inner, self.calls, self.texts, self.die_on_call = inner, 0, 0, die_on_call

    def encode(self, texts):
        self.calls += 1
        if self.die_on_call == self.calls:
            raise KeyboardInterrupt("simulated interruption")
        self.texts += len(texts)
        return self.inner.encode(texts)


def make_choice(enc=None, name="fake"):
    return BackendChoice(enc or CountingEncoder(encoder()), name, [(name, "ok")], {})


TEXTS = [f"review number {i} " + "word " * (i % 40) for i in range(23)]


def test_embed_texts_computes_shards_once_and_reuses_them(tmp_path):
    asks = []

    def get_choice():
        asks.append(1)
        return make_choice()

    X1, n1, info1 = embed_texts(TEXTS, tmp_path, get_choice, shard_size=10)
    X2, n2, info2 = embed_texts(TEXTS, tmp_path, get_choice, shard_size=10)
    assert len(asks) == 1  # a backend is only built on a miss
    assert X1.shape == (23, DIM) and np.array_equal(X1, X2) and np.array_equal(n1, n2)
    assert info1["shards"] == {"total": 3, "reused": 0, "computed": 3}
    assert info2["shards"] == {"total": 3, "reused": 3, "computed": 0}
    assert info2["backend"] == "fake"


def test_an_interrupted_run_resumes_with_only_the_missing_shards(tmp_path):
    dying = CountingEncoder(encoder(), die_on_call=3)
    with pytest.raises(KeyboardInterrupt):
        embed_texts(TEXTS, tmp_path, lambda: make_choice(dying), shard_size=10)
    assert len(list(tmp_path.glob("shard-*.json"))) == 2  # two shards finished
    fresh = CountingEncoder(encoder())
    X, _, info = embed_texts(TEXTS, tmp_path, lambda: make_choice(fresh), shard_size=10)
    assert info["shards"] == {"total": 3, "reused": 2, "computed": 1}
    assert fresh.texts == 3  # only the last, partial shard
    clean, _, _ = embed_texts(
        TEXTS, tmp_path / "other", lambda: make_choice(), shard_size=10
    )
    assert np.array_equal(X, clean)


def test_shard_keys_depend_on_the_model_the_version_the_buckets_and_every_text():
    k = embed.shard_key("m", ["a", "b"])
    assert k == embed.shard_key("m", ["a", "b"])
    assert k != embed.shard_key("m2", ["a", "b"]) and k != embed.shard_key(
        "m", ["a", "c"]
    )
    assert k != embed.shard_key("m", ["ab"]) and k != embed.shard_key("m", ["b", "a"])


def test_a_stale_shard_from_another_model_or_embedding_version_is_not_reused(
    tmp_path, monkeypatch
):
    embed_texts(TEXTS, tmp_path, make_choice, model_id="old-model", shard_size=10)
    fresh = CountingEncoder(encoder())
    _, _, info = embed_texts(
        TEXTS, tmp_path, lambda: make_choice(fresh), model_id="new-model", shard_size=10
    )
    assert info["shards"]["reused"] == 0 and fresh.texts == 23
    monkeypatch.setattr(embed, "EMBED_VERSION", embed.EMBED_VERSION + 1)
    again = CountingEncoder(encoder())
    _, _, info = embed_texts(
        TEXTS, tmp_path, lambda: make_choice(again), model_id="new-model", shard_size=10
    )
    assert info["shards"]["reused"] == 0 and again.texts == 23


def test_a_corrupt_shard_is_recomputed_not_trusted(tmp_path):
    embed_texts(TEXTS, tmp_path, make_choice, shard_size=10)
    victim = min(tmp_path.glob("shard-*.npz"))
    victim.write_bytes(b"not an npz")
    fresh = CountingEncoder(encoder())
    X, _, info = embed_texts(TEXTS, tmp_path, lambda: make_choice(fresh), shard_size=10)
    assert info["shards"] == {"total": 3, "reused": 2, "computed": 1}
    assert np.isfinite(X).all()


def test_a_shard_with_the_wrong_shape_is_recomputed(tmp_path):
    embed_texts(TEXTS, tmp_path, make_choice, shard_size=10)
    victim = min(tmp_path.glob("shard-*.npz"))
    np.savez(victim, X=np.zeros((3, 2), np.float32), n_tok=np.zeros(3, np.int32))
    _, _, info = embed_texts(TEXTS, tmp_path, make_choice, shard_size=10)
    assert info["shards"]["computed"] == 1


def test_embedding_no_texts_returns_empty_arrays_without_building_a_backend(tmp_path):
    def get_choice():
        raise AssertionError("no backend should be built for nothing")

    X, n_tok, info = embed_texts([], tmp_path, get_choice)
    assert X.shape == (0, 0) and n_tok.shape == (0,) and info["shards"]["total"] == 0


# ---------------------------------------------------------------- the real runners, tiny and offline


def test_the_openvino_runner_matches_torch_on_a_tiny_random_modernbert():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("openvino")
    cfg = transformers.ModernBertConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        bos_token_id=1,
        eos_token_id=2,
        global_attn_every_n_layers=2,
        local_attention=16,
    )
    torch.manual_seed(0)
    model = transformers.ModernBertModel(cfg).eval()
    rng = np.random.default_rng(0)
    ids = rng.integers(3, 100, size=(4, 64)).astype(np.int64)
    mask = np.ones_like(ids)
    mask[1, 40:] = 0
    mask[2, 10:] = 0
    ref = embed.TorchRunner(model, seq_len=64, batch_size=4).run(ids, mask)
    got = embed.OpenVinoRunner(model, seq_len=64, batch_size=4, device="CPU").run(
        ids, mask
    )
    a = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    b = got / np.linalg.norm(got, axis=1, keepdims=True)
    assert np.isfinite(got).all() and (a * b).sum(axis=1).min() > 0.999
