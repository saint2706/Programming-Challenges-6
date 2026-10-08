import io
import json

import numpy as np
import pytest
from active_labeling import data

CLASSES = [f"c{i}" for i in range(5)]


class FakeResponse(io.BytesIO):
    def __init__(self, payload, length=None):
        super().__init__(payload)
        self.headers = {
            "Content-Length": str(len(payload) if length is None else length)
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_refuses_non_https_urls(tmp_path):
    with pytest.raises(ValueError, match="https"):
        data.download("http://example.com/x.csv", tmp_path / "x.csv")


def test_download_writes_atomically_and_skips_existing_files(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        data.urllib.request,
        "urlopen",
        lambda url, timeout: calls.append(url) or FakeResponse(b"a,b\n1,2\n"),
    )
    dest = tmp_path / "sub" / "x.csv"
    assert data.download("https://example.com/x.csv", dest) == dest
    assert dest.read_bytes() == b"a,b\n1,2\n" and not list(dest.parent.glob("*.part"))
    data.download("https://example.com/x.csv", dest)
    assert len(calls) == 1


def test_a_truncated_download_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        data.urllib.request,
        "urlopen",
        lambda url, timeout: FakeResponse(b"abc", length=99),
    )
    dest = tmp_path / "x.csv"
    with pytest.raises(OSError, match="truncated"):
        data.download("https://example.com/x.csv", dest)
    assert not dest.exists() and not list(tmp_path.glob("*.part"))


def test_fetch_requests_each_file_once(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        data, "download", lambda url, dest: seen.append((url, dest.name)) or dest
    )
    data.fetch(tmp_path, base_url="https://example.com/b/")
    assert seen == [("https://example.com/b/" + n, n) for n in data.FILES]


def test_load_raw_maps_intents_to_ids_and_keeps_multiline_text(banking_dir):
    raw = data.load_raw(banking_dir)
    assert raw.classes == CLASSES and len(raw.train_texts) == 153
    assert raw.train_y[raw.train_texts.index("multi\nline text")] == 1
    assert raw.test_y[:5].tolist() == [0, 1, 2, 3, 4]


def test_load_raw_reports_missing_files_and_unknown_intents(
    banking_dir, tmp_path_factory
):
    with pytest.raises(FileNotFoundError, match="fetch"):
        data.load_raw(tmp_path_factory.mktemp("empty"))
    (banking_dir / "categories.json").write_text(json.dumps(CLASSES[:4]))
    with pytest.raises(ValueError, match="not in categories.json"):
        data.load_raw(banking_dir)


def test_dedupe_keeps_the_first_occurrence_and_counts_rows():
    keep, counts = data.dedupe(["a", "b", "a", "c", "b", "a"])
    assert keep.tolist() == [0, 1, 3] and counts.tolist() == [3, 2, 1]


def test_make_splits_is_stratified_disjoint_deduplicated_and_leaves_test_alone(
    banking_dir,
):
    raw = data.load_raw(banking_dir)
    s = data.make_splits(raw, seed=0, val_size=30)
    assert len(s.val_texts) == 30 and len(s.pool_texts) == 151 - 30
    assert len(set(s.pool_texts)) == len(s.pool_texts)
    assert not set(s.pool_texts) & set(s.val_texts)
    assert s.n_duplicates == 2 and 1 <= s.pool_count.min() and s.pool_count.max() <= 3
    total = np.bincount(np.concatenate([s.pool_y, s.val_y]), minlength=5)
    got = np.bincount(s.val_y, minlength=5)
    assert (
        np.abs(got - 30 * total / total.sum()) < 1.0
    ).all()  # stratified: within 1 of proportional
    assert s.test_texts == raw.test_texts and np.array_equal(s.test_y, raw.test_y)
    assert s.test_overlap == 2 and s.classes == CLASSES


def test_make_splits_is_reproducible_and_seed_dependent(banking_dir):
    raw = data.load_raw(banking_dir)
    a, b = data.make_splits(raw, 0, 30), data.make_splits(raw, 0, 30)
    assert a.pool_texts == b.pool_texts and a.val_texts == b.val_texts
    assert data.make_splits(raw, 1, 30).val_texts != a.val_texts


def test_load_pool_csv_drops_blank_and_duplicate_rows_and_keeps_gold(tmp_path):
    path = tmp_path / "pool.csv"
    path.write_text(
        "text,intent\nhello there,greet\n  ,greet\nhello there,greet\nbye now,\nbye for good,leave\n"
    )
    texts, gold = data.load_pool_csv(path, label_col="intent")
    assert texts == ["hello there", "bye now", "bye for good"]
    assert gold == ["greet", None, "leave"]
    assert data.load_pool_csv(path)[1] is None


def test_load_pool_csv_rejects_bad_input_with_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        data.load_pool_csv(tmp_path / "nope.csv")
    path = tmp_path / "pool.csv"
    path.write_text("body,intent\nhello,greet\n")
    with pytest.raises(ValueError, match="no 'text' column"):
        data.load_pool_csv(path)
    with pytest.raises(ValueError, match="no 'label' column"):
        data.load_pool_csv(path, text_col="body", label_col="label")
    path.write_text("text\n \n\n")
    with pytest.raises(ValueError, match="no non-blank"):
        data.load_pool_csv(path)


def test_read_classes_strips_blank_lines_and_rejects_duplicates_and_stubs(tmp_path):
    path = tmp_path / "classes.txt"
    path.write_text("alpha\n\n beta \ngamma\n")
    assert data.read_classes(path) == ["alpha", "beta", "gamma"]
    path.write_text("alpha\nalpha\n")
    with pytest.raises(ValueError, match="duplicate"):
        data.read_classes(path)
    path.write_text("alpha\n")
    with pytest.raises(ValueError, match="at least two"):
        data.read_classes(path)
