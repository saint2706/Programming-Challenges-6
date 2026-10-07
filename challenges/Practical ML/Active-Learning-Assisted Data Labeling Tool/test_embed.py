import zlib

import embed
import numpy as np
import pytest
from embed import BackendChoice, TextEncoder, embed_cached, pick_backend


class FakeRunner:
    batch_size, seq_len, dim = 4, 8, 6

    def __init__(self):
        self.ids, self.masks = [], []

    def run(self, input_ids, attention_mask):
        self.ids.append(input_ids.copy())
        self.masks.append(attention_mask.copy())
        out = np.zeros((input_ids.shape[0], self.dim))
        for r, (row, live) in enumerate(zip(input_ids, attention_mask, strict=True)):
            for tok, m in zip(row, live, strict=True):
                if m:
                    out[r, tok % self.dim] += 1
        return out


def fake_tokenize(texts):
    ids = np.zeros((len(texts), 8), dtype=np.int64)
    mask = np.zeros_like(ids)
    for i, t in enumerate(texts):
        toks = [ord(ch) for ch in t][:8]
        ids[i, : len(toks)] = toks
        mask[i, : len(toks)] = 1
    return ids, mask


TEXTS = [f"intent {i}" for i in range(10)]  # 10 texts = 2 full batches + a partial one


def test_encode_gives_unit_vectors_in_input_order_whatever_the_batching():
    enc = TextEncoder(fake_tokenize, FakeRunner())
    X = enc.encode(TEXTS)
    assert X.shape == (10, 6) and X.dtype == np.float32
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)
    for i in (0, 4, 9):
        assert np.allclose(
            TextEncoder(fake_tokenize, FakeRunner()).encode([TEXTS[i]])[0], X[i]
        )
    assert np.allclose(enc.encode(TEXTS[::-1]), X[::-1])


def test_empty_input_and_blank_text_do_not_crash():
    enc = TextEncoder(fake_tokenize, FakeRunner())
    assert enc.encode([]).shape == (0, 6)
    assert np.isfinite(enc.encode(["", "x"])).all()


def test_padding_rows_keep_one_live_token_so_softmax_never_sees_an_all_masked_row():
    runner = FakeRunner()
    TextEncoder(fake_tokenize, runner).encode(TEXTS)
    last = runner.masks[-1]  # 2 real rows, 2 padding rows
    assert last[2:, 0].tolist() == [1, 1] and last[2:, 1:].sum() == 0


def vec(text, salt=0):
    v = np.random.default_rng(zlib.crc32(text.encode()) + salt).normal(size=12)
    return v / np.linalg.norm(v)


class Const:
    def __init__(self, fn):
        self.fn = fn

    def encode(self, texts):
        return np.stack([self.fn(t) for t in texts])


def wrong(t):
    v = vec(t) + 2.5 * vec(t, salt=99)
    return v / np.linalg.norm(v)


PROBE = ["a", "b", "c", "d"]


def test_pick_backend_takes_the_first_candidate_that_matches_the_reference():
    built = []

    def make(name, fn):
        def build():
            built.append(name)
            return Const(fn)

        return name, build

    choice = pick_backend(
        [make("bad", wrong), make("good", vec), make("also-good", vec)],
        PROBE,
        Const(vec),
    )
    assert choice.name == "good" and built == ["bad", "good"]
    assert choice.tried[0][1].startswith("rejected: mean cosine")
    assert choice.tried[1][1].startswith("ok")


def crash():
    raise RuntimeError("device lost")


def test_pick_backend_skips_a_crashing_candidate_and_falls_back_to_the_reference():
    choice = pick_backend(
        [("gpu", crash), ("bad", lambda: Const(wrong))], PROBE, Const(vec)
    )
    assert choice.name == "torch-cpu"
    assert choice.tried[0][1].startswith("unavailable: RuntimeError")
    assert choice.tried[1][1].startswith("rejected")
    assert choice.tried[-1] == ("torch-cpu", "ok: reference")


def test_the_npu_is_never_tried():
    with pytest.raises(ValueError, match="NPU"):
        embed.default_choice(devices=("GPU", "NPU"))


def test_cache_key_depends_on_the_model_and_every_text():
    k = embed.cache_key("m", ["a", "b"])
    assert k == embed.cache_key("m", ["a", "b"])
    assert k != embed.cache_key("m2", ["a", "b"]) and k != embed.cache_key(
        "m", ["a", "c"]
    )
    assert k != embed.cache_key("m", ["ab"])


def test_embed_cached_asks_for_a_backend_only_on_a_miss(tmp_path):
    calls = []

    def get_choice():
        calls.append(1)
        return BackendChoice(Const(vec), "good", [("good", "ok")])

    texts = ["a", "b", "c"]
    X1, info1 = embed_cached(texts, tmp_path, get_choice, model_id="m")
    X2, info2 = embed_cached(texts, tmp_path, get_choice, model_id="m")
    assert len(calls) == 1 and np.array_equal(X1, X2) and X1.dtype == np.float32
    assert info1["backend"] == info2["backend"] == "good" and info2["model"] == "m"
    embed_cached(["a", "b", "z"], tmp_path, get_choice, model_id="m")
    embed_cached(texts, tmp_path, get_choice, model_id="other")
    assert len(calls) == 3
