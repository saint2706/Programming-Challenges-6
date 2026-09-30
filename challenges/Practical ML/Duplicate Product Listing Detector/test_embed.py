"""Tests for embed.py -- fake encoders only; the real-model test skips if it can't load."""

from pathlib import Path

import embed
import numpy as np
import polars as pl
import pytest
from PIL import Image


class FakeTextEncoder:
    dim = 4

    def __init__(self):
        self.calls: list[int] = []

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.calls.append(len(texts))
        # Deterministic, distinguishable vectors: hash the text into 4 floats.
        return np.array(
            [[len(t), t.count("a") + 1, t.count("e") + 1, 1.0] for t in texts],
            dtype=np.float32,
        )


class FakeImageEncoder:
    dim = 3

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        return np.array(
            [[im.size[0], im.size[1], 1.0] for im in images], dtype=np.float32
        )


def write_image(path: Path, size=(8, 6)) -> None:
    Image.new("RGB", size, (200, 10, 10)).save(path)


def make_df(tmp_path: Path, n: int = 5) -> pl.DataFrame:
    img_dir = tmp_path / "train_images"
    img_dir.mkdir()
    rows = []
    for i in range(n):
        name = f"{i}.jpg"
        write_image(img_dir / name, size=(8 + i, 6))
        rows.append(
            {"posting_id": f"p{i}", "image": name, "title": f"title number {i}"}
        )
    return pl.DataFrame(rows)


def test_l2_normalize_gives_unit_rows():
    out = embed.l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)


def test_l2_normalize_leaves_zero_rows_zero_instead_of_nan():
    out = embed.l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(out[0], [0.0, 0.0])


def test_embed_catalog_shapes_order_and_normalization(tmp_path: Path):
    df = make_df(tmp_path, 5)
    text_enc = FakeTextEncoder()
    result = embed.embed_catalog(
        df, tmp_path, text_enc, FakeImageEncoder(), batch_size=2
    )
    assert result.posting_ids == df["posting_id"].to_list()
    assert result.text.shape == (5, 4) and result.image.shape == (5, 3)
    np.testing.assert_allclose(np.linalg.norm(result.text, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(result.image, axis=1), 1.0, rtol=1e-5)
    assert text_enc.calls == [2, 2, 1]  # batched, nothing dropped


def test_embed_catalog_unreadable_image_becomes_zero_vector(tmp_path: Path):
    df = make_df(tmp_path, 3)
    (tmp_path / "train_images" / "1.jpg").write_bytes(b"not an image")
    result = embed.embed_catalog(
        df, tmp_path, FakeTextEncoder(), FakeImageEncoder(), batch_size=2
    )
    np.testing.assert_array_equal(result.image[1], np.zeros(3, dtype=np.float32))
    assert result.missing_images == ["p1"]
    assert np.linalg.norm(result.image[0]) == pytest.approx(1.0, rel=1e-5)


def test_embed_catalog_missing_image_file_is_reported_not_raised(tmp_path: Path):
    df = make_df(tmp_path, 2)
    (tmp_path / "train_images" / "0.jpg").unlink()
    result = embed.embed_catalog(
        df, tmp_path, FakeTextEncoder(), FakeImageEncoder(), batch_size=4
    )
    assert result.missing_images == ["p0"]


def test_cache_round_trip(tmp_path: Path):
    df = make_df(tmp_path, 3)
    result = embed.embed_catalog(
        df, tmp_path, FakeTextEncoder(), FakeImageEncoder(), batch_size=2
    )
    cache = tmp_path / "emb.npz"
    embed.save_embeddings(cache, result)
    loaded = embed.load_embeddings(cache, expected_ids=result.posting_ids)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.text, result.text)
    np.testing.assert_array_equal(loaded.image, result.image)
    assert loaded.missing_images == result.missing_images


def test_cache_miss_when_catalog_changed(tmp_path: Path):
    df = make_df(tmp_path, 3)
    result = embed.embed_catalog(
        df, tmp_path, FakeTextEncoder(), FakeImageEncoder(), batch_size=2
    )
    cache = tmp_path / "emb.npz"
    embed.save_embeddings(cache, result)
    assert embed.load_embeddings(cache, expected_ids=["p0", "p1", "other"]) is None
    assert embed.load_embeddings(tmp_path / "nope.npz", expected_ids=["p0"]) is None


def test_real_models_smoke(tmp_path: Path):
    """End-to-end with the real SigLIP 2 + multilingual-e5 models (skips offline)."""
    try:
        text_enc = embed.MultilingualTextEncoder()
        image_enc = embed.SigLIP2ImageEncoder()
    except Exception as exc:  # noqa: BLE001 -- any load failure (offline, no disk cache, HF error types vary) means skip, never fail
        pytest.skip(f"real models unavailable: {exc}")
    t = text_enc.encode_texts(
        [
            "Paper Bag Victoria Secret",
            "Kantong kertas Victoria Secret",
            "Samsung phone case",
        ]
    )
    assert t.shape[0] == 3
    t = embed.l2_normalize(t)
    assert (
        t[0] @ t[1] > t[0] @ t[2]
    )  # cross-lingual near-duplicate beats an unrelated title
    red = Image.new("RGB", (64, 64), (220, 20, 20))
    i = image_enc.encode_images(
        [red, red.copy(), Image.new("RGB", (64, 64), (20, 20, 220))]
    )
    i = embed.l2_normalize(i)
    assert i[0] @ i[1] > i[0] @ i[2]
