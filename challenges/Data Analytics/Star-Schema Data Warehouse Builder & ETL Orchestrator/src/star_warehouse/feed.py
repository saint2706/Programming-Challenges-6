"""The simulated source feed: Online Retail II invoice lines, each with an arrival date.

Online Retail II is a single finished extract, so arrival times are simulated, deterministically:

* every line has a stable ``line_key`` (invoice, stock code, and the occurrence index among the
  lines of that invoice and product, in a fixed sort order, because a customer really can buy the
  same product twice on one invoice, at different quantities or prices);
* about 2% of invoices are "late": they arrive 30 to 119 days after their invoice date. The late
  rule is a hash of the invoice number, so all lines of one invoice arrive together.

A pipeline run for an interval loads the rows whose ``arrival_date`` falls in that interval. A late
invoice therefore lands in a later run, for an earlier invoice month, and must restate the facts.
"""

from __future__ import annotations

import polars as pl

ROW_COLUMNS = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]
FEED_COLUMNS = ["line_key", *ROW_COLUMNS, "arrival_date"]
LATE_PERCENT = 2  # share of invoices that arrive late
LAG_MIN_DAYS = 30
LAG_SPAN_DAYS = 90


def build_feed(lines: pl.DataFrame) -> pl.DataFrame:
    """``lines`` holds ``ROW_COLUMNS`` as read from the source; returns ``FEED_COLUMNS``."""
    # Sorting first makes the occurrence index deterministic; the window then numbers each invoice-product group.
    frame = (
        lines.select(ROW_COLUMNS)
        .with_columns(
            pl.col("invoice_date").cast(pl.Date),
            pl.col("quantity").cast(pl.Int64),
            pl.col("price").cast(pl.Decimal(18, 4)),
            pl.col("customer_id").cast(pl.Int64),
        )
        .sort(ROW_COLUMNS, nulls_last=True)
        .with_columns(occurrence=pl.int_range(pl.len()).over(["invoice", "stock_code"]))
    )
    invoice_hash = pl.col("invoice").hash()
    late = (invoice_hash % 100) < LATE_PERCENT
    lag_days = (
        pl.when(late)
        .then(LAG_MIN_DAYS + (invoice_hash // 100) % LAG_SPAN_DAYS)
        .otherwise(0)
        .cast(pl.Int32)
    )
    return (
        frame.with_columns(
            line_key=pl.concat_str(
                pl.col("invoice"),
                pl.col("stock_code"),
                pl.col("occurrence").cast(pl.String),
                separator="|",
            ),
            arrival_date=(pl.col("invoice_date").cast(pl.Int32) + lag_days).cast(
                pl.Date
            ),
        )
        .select(FEED_COLUMNS)
        .sort("line_key")
    )
