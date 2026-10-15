from __future__ import annotations

import hashlib
from datetime import date

import httpx
import polars as pl
import pytest
from helpers import one_row
from stockout_risk import data


def test_daily_reduces_the_hourly_lists_to_window_features():
    status = [0] * 24
    for h in (
        9,
        10,
        18,
        19,
        20,
        21,
    ):  # six window hours out of stock, the last four are 18-21
        status[h] = 1
    status[2] = 1  # a night hour: outside the window, so not counted
    sales = [0.0] * 24
    sales[7], sales[8], sales[12], sales[15] = 1.0, 2.0, 3.0, 4.0
    row = data.daily(one_row(status, sales)).row(0, named=True)
    assert row["oos_window"] == 6
    assert row["oos_open"] == 0 and row["oos_close"] == 1
    assert row["oos_last4"] == 4
    assert row["first_oos"] == 3  # hour 9 is offset 3 into the window
    assert row["sales_in_stock"] == pytest.approx(10.0)
    assert row["early_share"] == pytest.approx(3.0 / 10.0)  # hours 6-11 sold 1 + 2
    assert row["dt"] == date(2024, 6, 1)


def test_daily_with_no_stockout_and_no_sales():
    row = data.daily(one_row()).row(0, named=True)
    assert row["oos_window"] == 0 and row["first_oos"] == data.WINDOW_HOURS
    assert (
        row["early_share"] is None or row["early_share"] != row["early_share"]
    )  # NaN: no sales


def test_daily_renames_and_sorts(raw):
    d = data.daily(raw)
    assert {"sale", "promo", "holiday", "cat1", "mgmt_id"} <= set(d.columns)
    assert d.select(pl.col("dt").is_sorted()).item() or d.equals(
        d.sort(["store_id", "product_id", "dt"])
    )
    assert d.height == raw.height


def test_daily_flags_equal_the_published_count_column(raw):
    d = data.daily(raw)
    published = raw.sort(["store_id", "product_id", "dt"])["stock_hour6_22_cnt"]
    assert (d["oos_window"].cast(pl.Int32) == published).all()


def test_daily_rejects_missing_columns_and_wrong_list_length(raw):
    with pytest.raises(ValueError, match="missing columns"):
        data.daily(raw.drop("discount"))
    short = raw.head(3).with_columns(pl.col("hours_sale").list.head(23))
    with pytest.raises(ValueError, match="24 hourly values"):
        data.daily(short)


def test_summary_cross_checks_the_columns(raw):
    s = data.summary(raw)
    assert s["window_count_matches_flags"] == 1.0 and s["sales_sum_matches"] == 1.0
    assert s["series"] == 400 and s["days"] == 60


def _client(body: bytes, status: int = 200) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, content=body)
        )
    )


def test_fetch_checks_the_hash_before_naming_the_file(tmp_path):
    body = b"x" * 5000
    dest = tmp_path / "a.parquet"
    data.fetch(
        "https://example.org/a",
        dest,
        hashlib.sha256(body).hexdigest(),
        _client(body),
        chunk=1024,
    )
    assert dest.read_bytes() == body and not dest.with_suffix(".parquet.part").exists()


def test_fetch_leaves_nothing_behind_on_a_wrong_hash_or_http_error(tmp_path):
    dest = tmp_path / "a.parquet"
    with pytest.raises(data.DownloadError, match="sha256"):
        data.fetch("https://example.org/a", dest, "0" * 64, _client(b"abc"))
    with pytest.raises(data.DownloadError, match="failed"):
        data.fetch("https://example.org/a", dest, "0" * 64, _client(b"", status=404))
    assert list(tmp_path.iterdir()) == []


def test_download_keeps_existing_files(home, monkeypatch):
    calls = []
    monkeypatch.setattr(data, "fetch", lambda *a, **k: calls.append(a) or a[1])
    (home / "data").mkdir()
    for name in data.SHA256:
        (home / "data" / f"{name}.parquet").write_bytes(b"x")
    data.download()
    assert calls == []
    data.download(force=True)
    assert len(calls) == 2
