"""The star answers equal an independent polars computation from the raw feed, for every grouping.

Amounts are compared as floats with a 1e-6 tolerance: every value has four decimals, so the float sums
are exact to far better than a cent. The line counts and the keys are compared exactly.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from helpers import SAMPLE, frame, load, oracle_lines, raw_clean, star_lines
from star_warehouse.warehouse import connect

GROUPINGS = [
    ["month_start", "country"],
    ["month_start"],
    ["year"],
    ["year", "quarter"],
    ["region"],
    ["weekday"],
    ["stock_code"],
    ["customer_id"],
]


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    db = tmp_path_factory.mktemp("oracle") / "w.duckdb"
    outcome = load(db)
    assert outcome.status == "succeeded", outcome.failed_tasks
    con = connect(db)
    yield con
    con.close()


def _summary(frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return (
        frame.with_columns(
            pl.col("customer_id").fill_null(-1),
            amount=pl.col("amount").cast(pl.Float64),
        )
        .group_by([*keys, "kind"])
        .agg(
            amount=pl.col("amount").sum(),
            lines=pl.len(),
            quantity=pl.col("quantity").sum(),
        )
        .sort([*keys, "kind"])
    )


@pytest.mark.parametrize("keys", GROUPINGS, ids=lambda k: "+".join(k))
def test_star_equals_oracle_for_grouping(warehouse, keys):
    star = _summary(star_lines(warehouse), keys)
    oracle = _summary(oracle_lines(), keys)
    assert star.height == oracle.height, f"{keys}: group count differs"
    joined = star.join(
        oracle, on=[*keys, "kind"], how="full", suffix="_o", coalesce=True
    )
    assert joined.height == star.height
    assert joined["lines"].to_list() == joined["lines_o"].to_list()
    assert joined["quantity"].to_list() == joined["quantity_o"].to_list()
    diff = (joined["amount"] - joined["amount_o"]).abs().max()
    assert diff is None or diff < 1e-6, f"{keys}: amounts differ by {diff}"


def test_totals_match_raw_file(warehouse):
    """The whole-file answer, which every grouping above must add up to."""
    clean = raw_clean()
    totals = warehouse.execute(
        """SELECT (SELECT sum(sales_amount) FROM dw.fact_sales)
                + (SELECT sum(return_amount) FROM dw.fact_returns) AS net,
                (SELECT count(*) FROM dw.fact_sales) AS sales_n,
                (SELECT count(*) FROM dw.fact_returns) AS returns_n"""
    ).fetchone()
    assert float(totals[0]) == pytest.approx(float(clean["amount"].sum()), abs=1e-6)
    assert totals[1] == clean.filter(~pl.col("is_return")).height
    assert totals[2] == clean.filter(pl.col("is_return")).height


def test_customer_country_is_point_in_time(warehouse):
    """Each line's customer country is the one the customer's SCD2 version gives on that day.

    The oracle takes the most frequent country of the customer's kept lines on the invoice date, with
    ties broken alphabetically. That is the definition the dimension's rebuild uses, so the check
    catches a version that is stale, shifted by a day, or resolved against the wrong date.
    """
    clean = raw_clean().filter(pl.col("customer_id").is_not_null())
    per_day = (
        clean.group_by(["customer_id", "invoice_date", "country"])
        .agg(n=pl.len())
        .sort(
            ["customer_id", "invoice_date", "n", "country"],
            descending=[False, False, True, False],
        )
        .group_by(["customer_id", "invoice_date"], maintain_order=True)
        .first()
        .select("customer_id", "invoice_date", expected=pl.col("country"))
    )
    expected = clean.join(per_day, on=["customer_id", "invoice_date"]).select(
        "line_key", "expected"
    )
    star = frame(
        warehouse,
        """SELECT f.line_key, cu.country AS actual
           FROM (SELECT line_key, customer_key FROM dw.fact_sales
                 UNION ALL SELECT line_key, customer_key FROM dw.fact_returns) f
           JOIN dw.dim_customer cu ON cu.customer_key = f.customer_key
           WHERE cu.customer_id IS NOT NULL""",
    )
    checked = expected.join(star, on="line_key")
    assert checked.height == expected.height
    mismatches = checked.filter(pl.col("expected") != pl.col("actual"))
    assert mismatches.height == 0, mismatches.head()


def test_sample_is_the_committed_sample():
    feed = pl.read_parquet(SAMPLE)
    assert feed.height == 87862
    assert feed["line_key"].n_unique() == feed.height
    assert date(2009, 12, 1) <= feed["invoice_date"].min()
