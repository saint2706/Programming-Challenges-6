from __future__ import annotations

import hashlib

import httpx
import polars as pl
import pytest
from fraud_flagging import data


def _client(body: bytes, status: int = 200) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, content=body)
        )
    )


def _hashes(body: bytes) -> dict:
    return {
        "md5": hashlib.md5(body, usedforsecurity=False).hexdigest(),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def test_fetch_checks_both_hashes_before_naming_the_file(tmp_path):
    body = b"x" * 5000
    dest = tmp_path / "a.parquet"
    data.fetch(
        "https://example.org/a", dest, **_hashes(body), client=_client(body), chunk=1024
    )
    assert dest.read_bytes() == body and not dest.with_suffix(".parquet.part").exists()


@pytest.mark.parametrize("wrong", ["md5", "sha256"])
def test_fetch_rejects_a_wrong_hash_and_leaves_nothing(tmp_path, wrong):
    body = b"abc"
    hashes = _hashes(body) | {wrong: "0" * (32 if wrong == "md5" else 64)}
    dest = tmp_path / "a.parquet"
    with pytest.raises(data.DownloadError, match="checksum"):
        data.fetch("https://example.org/a", dest, **hashes, client=_client(body))
    assert list(tmp_path.iterdir()) == []


def test_fetch_reports_an_http_error_cleanly(tmp_path):
    with pytest.raises(data.DownloadError, match="failed"):
        data.fetch(
            "https://example.org/a",
            tmp_path / "a.parquet",
            md5="0" * 32,
            sha256="0" * 64,
            client=_client(b"", status=404),
        )
    assert list(tmp_path.iterdir()) == []


def test_download_keeps_an_existing_file_unless_forced(installed, monkeypatch):
    calls = []
    monkeypatch.setattr(data, "fetch", lambda *a, **k: calls.append(a) or a[1])
    assert data.download().name == "creditcard.parquet" and calls == []
    data.download(force=True)
    assert len(calls) == 1


def test_load_raw_gives_a_numeric_label_and_the_row_index(installed, published):
    raw = data.load_raw()
    assert raw.height == published.height and raw["y"].dtype == pl.Int8
    assert raw["tx_id"].to_list() == list(range(raw.height))
    assert int(raw["y"].sum()) == int((published["Class"].cast(pl.String) == "1").sum())


def test_load_raw_rejects_a_table_without_the_expected_columns(home):
    (home / "data").mkdir()
    pl.DataFrame({"Time": [0.0], "Amount": [1.0]}).write_parquet(
        home / "data" / "creditcard.parquet"
    )
    with pytest.raises(ValueError, match="missing columns"):
        data.load_raw()


def test_clean_drops_exact_duplicates_and_keeps_the_first_with_its_row_index(
    published, raw
):
    clean = data.clean(raw)
    assert raw.height - clean.height == 25
    assert clean["Time"].is_sorted() and clean["tx_id"].n_unique() == clean.height
    dup = raw.filter(raw.select(data.RAW_COLUMNS).is_duplicated())
    first_ids = dup.group_by(data.RAW_COLUMNS).agg(pl.col("tx_id").min())["tx_id"]
    assert set(first_ids) <= set(clean["tx_id"])


def test_summary_counts_the_duplicates(raw):
    s = data.summary(raw)
    assert s["rows"] == raw.height and s["duplicate_rows"] == 25
    assert s["rows_after_dedup"] == raw.height - 25 and s["time_sorted"]
    assert s["fraud_amount_median"] < s["legit_amount_median"]
