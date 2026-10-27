"""The simulated feed and the warehouse helpers: line keys, late arrivals, interval parsing, transactions
and the content hash."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from helpers import SAMPLE
from star_warehouse.feed import LAG_MIN_DAYS, LAG_SPAN_DAYS, build_feed
from star_warehouse.warehouse import connect, content_hash, parse_interval, transaction


def _lines(rows: list[tuple]) -> pl.DataFrame:
    cols = [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "price",
        "customer_id",
        "country",
    ]
    return pl.DataFrame(rows, schema=cols, orient="row").with_columns(
        pl.col("invoice_date").cast(pl.Date), pl.col("price").cast(pl.Decimal(18, 4))
    )


def test_same_product_twice_on_one_invoice_gets_distinct_keys():
    """Regression: two lines of one product on one invoice at different quantities must not share a key."""
    rows = [
        (
            "536365",
            "85123A",
            "CREAM HANGER",
            2,
            date(2010, 12, 1),
            2.5,
            17850,
            "United Kingdom",
        ),
        (
            "536365",
            "85123A",
            "CREAM HANGER",
            6,
            date(2010, 12, 1),
            2.5,
            17850,
            "United Kingdom",
        ),
        (
            "536365",
            "85123A",
            "CREAM HANGER",
            2,
            date(2010, 12, 1),
            2.5,
            17850,
            "United Kingdom",
        ),
    ]
    feed = build_feed(_lines(rows))
    assert feed["line_key"].n_unique() == 3
    assert sorted(feed["line_key"].to_list()) == [
        "536365|85123A|0",
        "536365|85123A|1",
        "536365|85123A|2",
    ]


def test_feed_is_deterministic_regardless_of_input_order():
    rows = [
        ("1", "A1", "X", 1, date(2010, 1, 5), 1.0, 1, "France"),
        ("1", "A1", "X", 3, date(2010, 1, 5), 1.0, 1, "France"),
        ("2", "B2", "Y", 2, date(2010, 2, 1), 4.0, None, "Spain"),
    ]
    forward = build_feed(_lines(rows))
    backward = build_feed(_lines(list(reversed(rows))))
    assert forward.equals(backward)


def test_sample_feed_has_unique_keys_and_a_plausible_late_share():
    feed = pl.read_parquet(SAMPLE)
    assert feed["line_key"].n_unique() == feed.height
    assert (feed["arrival_date"] >= feed["invoice_date"]).all()
    lag = (feed["arrival_date"] - feed["invoice_date"]).dt.total_days()
    late = feed.filter(lag > 0).with_columns(lag=lag.filter(lag > 0))
    assert late["lag"].min() >= LAG_MIN_DAYS
    assert late["lag"].max() < LAG_MIN_DAYS + LAG_SPAN_DAYS
    invoices = feed["invoice"].n_unique()
    late_invoices = late["invoice"].n_unique()
    assert 0.01 < late_invoices / invoices < 0.03  # about 2% by construction
    # Every line of one invoice arrives together: the lateness is a property of the invoice.
    per_invoice = feed.group_by("invoice").agg(n=pl.col("arrival_date").n_unique())
    assert per_invoice["n"].max() == 1


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2010-02", (date(2010, 2, 1), date(2010, 3, 1))),
        ("2010-12", (date(2010, 12, 1), date(2011, 1, 1))),
        ("2010-03-05/2010-03-09", (date(2010, 3, 5), date(2010, 3, 9))),
    ],
)
def test_parse_interval(text, expected):
    assert parse_interval(text) == expected


@pytest.mark.parametrize(
    "text", ["2010", "2010-13", "2010-03-09/2010-03-05", "yesterday"]
)
def test_parse_interval_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_interval(text)


def test_transaction_rolls_back_all_of_a_task_on_error(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    con.execute("INSERT INTO ops.schedule_state VALUES ('a', TIMESTAMP '2010-01-01')")
    with pytest.raises(RuntimeError), transaction(con):
        con.execute("DELETE FROM ops.schedule_state")
        raise RuntimeError("boom")
    assert con.execute("SELECT count(*) FROM ops.schedule_state").fetchone()[0] == 1
    con.close()


def test_content_hash_sees_data_but_not_the_run_log(tmp_path):
    con = connect(tmp_path / "h.duckdb")
    before = content_hash(con)
    con.execute(
        "INSERT INTO ops.run VALUES ('r', DATE '2010-01-01', DATE '2010-02-01', 'x', 'ok', NULL, NULL, NULL)"
    )
    assert content_hash(con) == before
    con.execute("INSERT INTO dw.dim_country VALUES (1, 'France', 'Western Europe')")
    assert content_hash(con) != before
    con.close()
