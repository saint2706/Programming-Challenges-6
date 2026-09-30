"""Tests for pipeline.py -- fake encoders, a tiny generated catalog, no network."""

import json
from pathlib import Path

import data
import embed
import numpy as np
import pipeline
import polars as pl
import pytest
from PIL import Image


class FakeText:
    dim = 16  # > number of groups, so no two groups share a vector

    def __init__(self):
        self.calls = 0

    def encode_texts(self, texts):
        self.calls += 1
        # Same "product N" prefix -> same vector, so duplicates are text-identical.
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, int(t.split()[1]) % self.dim] = 1.0
        return out


class FakeImage:
    dim = 16  # > number of groups, so no two groups share a vector

    def encode_images(self, images):
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, im in enumerate(images):
            out[i, im.size[0] % self.dim] = 1.0
        return out


def make_catalog(data_dir: Path, n_groups: int = 12) -> pl.DataFrame:
    img_dir = data_dir / "train_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for g in range(n_groups):
        for j in range(2 + g % 2):
            name = f"{n}.jpg"
            Image.new("RGB", (10 + g, 10), (g * 10, 0, 0)).save(img_dir / name)
            rows.append(
                {
                    "posting_id": f"train_{n}",
                    "image": name,
                    "image_phash": f"{n:016x}",
                    "title": f"product {g} variant {j}",
                    "label_group": g,
                }
            )
            n += 1
    df = pl.DataFrame(rows)
    df.write_csv(data_dir / "train.csv")
    return df


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch):
    make_catalog(tmp_path)
    monkeypatch.setattr(
        data, "fetch_catalog", lambda data_dir=tmp_path: data_dir / "train.csv"
    )
    monkeypatch.setattr(data, "fetch_images", lambda names, data_dir=tmp_path: [])
    return tmp_path


def test_build_slice_keeps_whole_groups_and_writes_slice_csv(workdir: Path):
    df = pipeline.build_slice(n_groups=5, seed=0, data_dir=workdir)
    full = data.load_catalog(workdir / "train.csv")
    full_sizes = dict(full.group_by("label_group").len().iter_rows())
    for g, size in df.group_by("label_group").len().iter_rows():
        assert size == full_sizes[g]
    assert df["label_group"].n_unique() == 5
    assert pipeline.load_slice(workdir).equals(df)


def test_embed_slice_uses_cache_second_time(workdir: Path):
    df = pipeline.build_slice(n_groups=6, seed=0, data_dir=workdir)
    text = FakeText()
    a = pipeline.embed_slice(df, workdir, text, FakeImage())
    calls_after_first = text.calls
    b = pipeline.embed_slice(df, workdir, text, FakeImage())
    assert text.calls == calls_after_first  # served from embeddings.npz
    np.testing.assert_array_equal(a.text, b.text)


def test_embed_slice_recomputes_when_slice_changes(workdir: Path):
    df = pipeline.build_slice(n_groups=6, seed=0, data_dir=workdir)
    text = FakeText()
    pipeline.embed_slice(df, workdir, text, FakeImage())
    before = text.calls
    df2 = pipeline.build_slice(n_groups=6, seed=1, data_dir=workdir)
    pipeline.embed_slice(df2, workdir, text, FakeImage())
    assert text.calls > before


def test_split_slice_is_group_disjoint_and_keeps_embeddings_aligned(workdir: Path):
    df = pipeline.build_slice(n_groups=10, seed=0, data_dir=workdir)
    emb = pipeline.embed_slice(df, workdir, FakeText(), FakeImage())
    (vdf, vemb), (tdf, temb) = pipeline.split_slice(df, emb, val_fraction=0.4, seed=0)
    assert set(vdf["label_group"]).isdisjoint(set(tdf["label_group"]))
    assert vdf.height + tdf.height == df.height
    assert vemb.posting_ids == vdf["posting_id"].to_list()
    assert temb.posting_ids == tdf["posting_id"].to_list()
    assert vemb.text.shape[0] == vdf.height and temb.image.shape[0] == tdf.height


def test_run_evaluation_end_to_end_and_report_round_trip(workdir: Path):
    df = pipeline.build_slice(n_groups=12, seed=0, data_dir=workdir)
    emb = pipeline.embed_slice(df, workdir, FakeText(), FakeImage())
    report, _val, _test = pipeline.run_evaluation(
        df, emb, workdir, k=5, val_fraction=0.4, seed=0
    )
    assert set(report.modes) == {"text", "image", "fused"}
    assert report.modes["fused"].f1 > 0.9  # fake encoders make duplicates identical
    path = workdir / "report.json"
    pipeline.save_report(path, report, settings={"k": 5})
    loaded = pipeline.load_report(path)
    assert loaded["modes"]["fused"]["threshold"] == pytest.approx(
        report.modes["fused"].threshold
    )
    assert loaded["settings"] == {"k": 5}
    assert json.loads(path.read_text())["candidate_recall"] == pytest.approx(
        report.candidate_recall
    )


def test_load_report_missing_file_gives_actionable_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="evaluate"):
        pipeline.load_report(tmp_path / "nope.json")


def test_embeddings_subset_selects_rows():
    emb = embed.Embeddings(
        ["a", "b", "c"], np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32), ["b"]
    )
    sub = emb.subset([2, 1])
    assert sub.posting_ids == ["c", "b"]
    np.testing.assert_array_equal(sub.text, np.eye(3, dtype=np.float32)[[2, 1]])
    assert sub.missing_images == ["b"]
