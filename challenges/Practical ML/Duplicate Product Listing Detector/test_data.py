"""Tests for data.py -- all network-free (the Kaggle calls are monkeypatched)."""

import io
import zipfile
from pathlib import Path

import data
import polars as pl
import pytest


def make_catalog(group_sizes: list[int]) -> pl.DataFrame:
    rows = []
    n = 0
    for g, size in enumerate(group_sizes):
        for _ in range(size):
            rows.append(
                {
                    "posting_id": f"train_{n}",
                    "image": f"{n:032x}.jpg",
                    "image_phash": f"{n:016x}",
                    "title": f"product {g} listing {n}",
                    "label_group": g,
                }
            )
            n += 1
    return pl.DataFrame(rows)


def test_load_catalog_reads_expected_columns(tmp_path: Path):
    p = tmp_path / "train.csv"
    make_catalog([2, 3]).write_csv(p)
    df = data.load_catalog(p)
    assert df.columns == ["posting_id", "image", "image_phash", "title", "label_group"]
    assert df.height == 5


def test_load_catalog_rejects_missing_columns(tmp_path: Path):
    p = tmp_path / "train.csv"
    pl.DataFrame({"posting_id": ["a"], "title": ["x"]}).write_csv(p)
    with pytest.raises(ValueError, match="missing columns"):
        data.load_catalog(p)


def test_sample_groups_keeps_whole_groups():
    df = make_catalog([2, 3, 4, 5, 2, 3, 6, 2])
    sliced = data.sample_groups(df, n_groups=4, seed=0)
    full_sizes = dict(df.group_by("label_group").len().iter_rows())
    sliced_sizes = dict(sliced.group_by("label_group").len().iter_rows())
    assert len(sliced_sizes) == 4
    for group, size in sliced_sizes.items():
        assert size == full_sizes[group]


def test_sample_groups_is_deterministic_and_seed_sensitive():
    df = make_catalog([2] * 40)
    a = data.sample_groups(df, n_groups=10, seed=1)
    b = data.sample_groups(df, n_groups=10, seed=1)
    c = data.sample_groups(df, n_groups=10, seed=2)
    assert a["posting_id"].to_list() == b["posting_id"].to_list()
    assert set(a["label_group"]) != set(c["label_group"])


def test_sample_groups_caps_at_available_groups():
    df = make_catalog([2, 2, 2])
    assert data.sample_groups(df, n_groups=99, seed=0).height == 6


def test_split_groups_has_no_group_leakage():
    df = make_catalog([2, 3, 2, 4, 2, 3, 2, 2, 5, 2])
    val, test = data.split_groups(df, val_fraction=0.4, seed=0)
    assert set(val["label_group"]).isdisjoint(set(test["label_group"]))
    assert val.height + test.height == df.height
    assert val.height > 0 and test.height > 0


def test_true_pairs_counts_n_choose_2_per_group():
    df = make_catalog([2, 3, 4])  # 1 + 3 + 6 pairs
    pairs = data.true_pairs(df)
    assert len(pairs) == 10
    assert all(a < b for a, b in pairs)


def test_true_pairs_never_crosses_groups():
    df = make_catalog([2, 2])
    group_of = dict(zip(df["posting_id"], df["label_group"], strict=True))
    for a, b in data.true_pairs(df):
        assert group_of[a] == group_of[b]


def test_extract_if_zip_unwraps_zip_masquerading_as_csv(tmp_path: Path):
    target = tmp_path / "train.csv"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("train.csv", "a,b\n1,2\n")
    target.write_bytes(buf.getvalue())
    data.extract_if_zip(target)
    assert target.read_text() == "a,b\n1,2\n"


def test_extract_if_zip_leaves_plain_files_alone(tmp_path: Path):
    target = tmp_path / "x.jpg"
    target.write_bytes(b"\xff\xd8\xff\xe0plain")
    data.extract_if_zip(target)
    assert target.read_bytes() == b"\xff\xd8\xff\xe0plain"


def make_archive(path: Path, names: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for n in names:
            z.writestr(f"train_images/{n}", b"\xff\xd8\xff\xe0" + n.encode())
        z.writestr("train.csv", "a,b\n")


def test_extract_images_pulls_only_requested_members(tmp_path: Path):
    archive = tmp_path / "shopee.zip"
    make_archive(archive, ["a.jpg", "b.jpg", "c.jpg"])
    missing = data.extract_images(archive, ["a.jpg", "c.jpg"], tmp_path)
    assert missing == []
    assert sorted(p.name for p in (tmp_path / "train_images").iterdir()) == [
        "a.jpg",
        "c.jpg",
    ]
    assert (tmp_path / "train_images" / "a.jpg").read_bytes().endswith(b"a.jpg")


def test_extract_images_reports_names_absent_from_archive(tmp_path: Path):
    archive = tmp_path / "shopee.zip"
    make_archive(archive, ["a.jpg"])
    assert data.extract_images(archive, ["a.jpg", "nope.jpg"], tmp_path) == ["nope.jpg"]


def test_extract_images_is_resumable_and_replaces_empty_files(tmp_path: Path):
    archive = tmp_path / "shopee.zip"
    make_archive(archive, ["a.jpg", "b.jpg"])
    img_dir = tmp_path / "train_images"
    img_dir.mkdir()
    (img_dir / "a.jpg").write_bytes(b"KEEP")  # already extracted: left alone
    (img_dir / "b.jpg").write_bytes(b"")  # truncated leftover: re-extracted
    data.extract_images(archive, ["a.jpg", "b.jpg"], tmp_path)
    assert (img_dir / "a.jpg").read_bytes() == b"KEEP"
    assert (img_dir / "b.jpg").read_bytes().endswith(b"b.jpg")


def test_extract_images_rejects_path_traversal_members(tmp_path: Path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("train_images/../../escaped.jpg", b"x")
    data.extract_images(archive, ["../../escaped.jpg"], tmp_path)
    assert not (tmp_path.parent / "escaped.jpg").exists()
    assert not (tmp_path / "escaped.jpg").exists()


def test_fetch_images_downloads_archive_only_when_missing(tmp_path: Path, monkeypatch):
    calls: list[Path] = []

    def fake_download(data_dir: Path) -> Path:
        calls.append(data_dir)
        archive = data_dir / data.ARCHIVE_NAME
        make_archive(archive, ["a.jpg"])
        return archive

    monkeypatch.setattr(data, "fetch_archive", fake_download)
    assert data.fetch_images(["a.jpg"], tmp_path) == []
    assert len(calls) == 1
    (tmp_path / "train_images" / "a.jpg").unlink()
    data.fetch_images(["a.jpg"], tmp_path)  # archive already on disk now
    assert len(calls) == 1  # second call reused the archive already on disk


class _Resp:
    def __init__(self, status_code: int, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}


class _HTTPError(Exception):
    def __init__(self, status_code: int, headers: dict | None = None):
        super().__init__(f"{status_code} error")
        self.response = _Resp(status_code, headers)


class _FlakyApi:
    def __init__(self, failures: list[int], data_dir: Path):
        self.failures = failures
        self.data_dir = data_dir
        self.calls = 0

    def competition_download_files(
        self, competition: str, path: str, quiet: bool = True
    ) -> None:
        self.calls += 1
        if self.failures:
            raise _HTTPError(self.failures.pop(0))
        make_archive(Path(path) / data.ARCHIVE_NAME, ["a.jpg"])


def test_fetch_archive_backs_off_and_retries_on_429(tmp_path: Path, monkeypatch):
    api = _FlakyApi([429, 429], tmp_path)
    sleeps: list[float] = []
    monkeypatch.setattr(data, "_kaggle_api", lambda: api)
    monkeypatch.setattr(data.time, "sleep", sleeps.append)
    archive = data.fetch_archive(tmp_path)
    assert archive.exists() and api.calls == 3
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]  # exponential backoff


def test_fetch_archive_does_not_retry_other_http_errors(tmp_path: Path, monkeypatch):
    api = _FlakyApi([403], tmp_path)
    monkeypatch.setattr(data, "_kaggle_api", lambda: api)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    with pytest.raises(_HTTPError):
        data.fetch_archive(tmp_path)
    assert api.calls == 1


def test_fetch_archive_gives_up_after_max_attempts(tmp_path: Path, monkeypatch):
    api = _FlakyApi([429] * 10, tmp_path)
    monkeypatch.setattr(data, "_kaggle_api", lambda: api)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    with pytest.raises(_HTTPError):
        data.fetch_archive(tmp_path, attempts=3)
    assert api.calls == 3


def test_fetch_archive_honors_retry_after_header(tmp_path: Path, monkeypatch):
    class Api(_FlakyApi):
        def competition_download_files(self, competition, path, quiet=True):
            if not self.calls:
                self.calls += 1
                raise _HTTPError(429, {"Retry-After": "900"})
            super().competition_download_files(competition, path, quiet)

    api = Api([], tmp_path)
    sleeps: list[float] = []
    monkeypatch.setattr(data, "_kaggle_api", lambda: api)
    monkeypatch.setattr(data.time, "sleep", sleeps.append)
    data.fetch_archive(tmp_path)
    assert sleeps == [900.0]  # the server's instruction beats our 30s first guess


def test_fetch_archive_ignores_garbage_retry_after(tmp_path: Path, monkeypatch):
    class Api(_FlakyApi):
        def competition_download_files(self, competition, path, quiet=True):
            if not self.calls:
                self.calls += 1
                raise _HTTPError(429, {"Retry-After": "soon"})
            super().competition_download_files(competition, path, quiet)

    api = Api([], tmp_path)
    sleeps: list[float] = []
    monkeypatch.setattr(data, "_kaggle_api", lambda: api)
    monkeypatch.setattr(data.time, "sleep", sleeps.append)
    data.fetch_archive(tmp_path, base_delay=30.0)
    assert sleeps == [30.0]
