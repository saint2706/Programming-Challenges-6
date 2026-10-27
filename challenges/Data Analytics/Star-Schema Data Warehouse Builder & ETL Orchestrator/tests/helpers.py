"""Shared test helpers: the committed sample, a fresh warehouse, and an independent polars oracle.

The oracle re-derives every star answer from the raw feed file with polars only. It shares no code with
the SQL tasks, so agreement is evidence rather than the same code agreeing with itself.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import polars as pl
from star_warehouse.pipeline import RunOutcome, run_interval

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data" / "feed_sample.parquet"
LOOKUP = ROOT / "sample_data" / "ref_country_region.csv"
WHOLE = (date(2009, 1, 1), date(2013, 1, 1))  # covers every arrival date in the feed


def load(
    db: Path, feed: Path = SAMPLE, start: date = WHOLE[0], end: date = WHOLE[1]
) -> RunOutcome:
    """One interval through the whole DAG, no sleeping between retries."""
    return run_interval(
        db, feed, LOOKUP, start, end, backoff=0.0, sleep=lambda _s: None
    )


def raw_clean(feed: Path = SAMPLE) -> pl.DataFrame:
    """The cleaning rules, re-stated in polars: a kept line is a product line with a positive price
    and a quantity whose sign matches its kind (sale: positive, return: cancelled invoice, negative)."""
    cancelled = pl.col("invoice").str.starts_with("C")
    return (
        pl.read_parquet(feed)
        .filter(
            pl.col("stock_code").str.contains(r"^\d{5}[A-Za-z]{0,2}$"),
            pl.col("price") > 0,
            pl.when(cancelled)
            .then(pl.col("quantity") < 0)
            .otherwise(pl.col("quantity") > 0),
        )
        .with_columns(
            amount=(pl.col("quantity") * pl.col("price")).cast(pl.Decimal(18, 4)),
            is_return=cancelled,
            month_start=pl.col("invoice_date").dt.truncate("1mo"),
            year=pl.col("invoice_date").dt.year(),
            quarter=pl.col("invoice_date").dt.quarter(),
            weekday=pl.col("invoice_date").dt.weekday(),
        )
    )


def region_map() -> dict[str, str]:
    return dict(pl.read_csv(LOOKUP).select("country", "region").iter_rows())


def frame(con: duckdb.DuckDBPyConnection, sql: str) -> pl.DataFrame:
    """A query result as polars. Goes through fetchall because DuckDB's ``.pl()`` needs pyarrow."""
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    return pl.DataFrame(
        cur.fetchall(), schema=names, orient="row", infer_schema_length=None
    )


def star_lines(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Every fact row (sales and returns) joined to its dimensions, one row per invoice line."""
    return frame(
        con,
        """
        SELECT 'sale' AS kind, d.month_start, d.year, d.quarter, d.day_of_week AS weekday,
               k.country, k.region, p.stock_code, cu.customer_id,
               f.sales_amount AS amount, f.quantity, f.line_key, f.customer_key
        FROM dw.fact_sales f
        JOIN dw.dim_date d ON d.date_key = f.invoice_date_key
        JOIN dw.dim_country k ON k.country_key = f.country_key
        JOIN dw.dim_product p ON p.product_key = f.product_key
        JOIN dw.dim_customer cu ON cu.customer_key = f.customer_key
        UNION ALL
        SELECT 'return', d.month_start, d.year, d.quarter, d.day_of_week,
               k.country, k.region, p.stock_code, cu.customer_id,
               f.return_amount, f.quantity, f.line_key, f.customer_key
        FROM dw.fact_returns f
        JOIN dw.dim_date d ON d.date_key = f.invoice_date_key
        JOIN dw.dim_country k ON k.country_key = f.country_key
        JOIN dw.dim_product p ON p.product_key = f.product_key
        JOIN dw.dim_customer cu ON cu.customer_key = f.customer_key
        """,
    )


def oracle_lines(feed: Path = SAMPLE) -> pl.DataFrame:
    """The same rows as ``star_lines``, derived from the raw feed by polars alone."""
    regions = region_map()
    clean = raw_clean(feed)
    return clean.select(
        kind=pl.when(pl.col("is_return"))
        .then(pl.lit("return"))
        .otherwise(pl.lit("sale")),
        month_start=pl.col("month_start").cast(pl.Date),
        year="year",
        quarter="quarter",
        weekday="weekday",
        country="country",
        region=pl.col("country").replace_strict(regions, default="Unknown"),
        stock_code="stock_code",
        customer_id="customer_id",
        amount="amount",
        quantity="quantity",
        line_key="line_key",
    )
