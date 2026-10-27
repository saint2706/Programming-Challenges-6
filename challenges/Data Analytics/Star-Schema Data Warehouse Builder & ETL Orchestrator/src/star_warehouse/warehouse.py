"""The DuckDB file: schemas, transactions, run logs and a content hash used to prove idempotence.

Schemas: ``raw`` (what the feed delivered), ``staging`` (cleaned lines), ``dw`` (star schema) and
``ops`` (run and task logs, schedule state). ``ops`` is excluded from the content hash because it
records wall-clock times and is supposed to differ between runs.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import duckdb

DDL = """
CREATE SCHEMA IF NOT EXISTS raw;
CREATE TABLE IF NOT EXISTS raw.feed_lines (
    line_key VARCHAR, invoice VARCHAR, stock_code VARCHAR, description VARCHAR,
    quantity BIGINT, invoice_date DATE, price DECIMAL(18, 4), customer_id BIGINT,
    country VARCHAR, arrival_date DATE
);
CREATE SCHEMA IF NOT EXISTS staging;
CREATE TABLE IF NOT EXISTS staging.lines (
    line_key VARCHAR, invoice VARCHAR, stock_code VARCHAR, description VARCHAR,
    quantity BIGINT, invoice_date DATE, price DECIMAL(18, 4), customer_id BIGINT,
    country VARCHAR, arrival_date DATE, is_cancellation BOOLEAN, kind VARCHAR,
    exclusion VARCHAR, revenue DECIMAL(18, 4)
);
CREATE SCHEMA IF NOT EXISTS dw;
CREATE TABLE IF NOT EXISTS dw.dim_date (
    date_key INTEGER, full_date DATE, year INTEGER, quarter INTEGER, month INTEGER,
    month_start DATE, day_of_week INTEGER, is_weekend BOOLEAN
);
CREATE TABLE IF NOT EXISTS dw.dim_country (
    country_key UBIGINT, country VARCHAR, region VARCHAR
);
CREATE TABLE IF NOT EXISTS dw.dim_customer (
    customer_key UBIGINT, customer_id BIGINT, country VARCHAR, valid_from DATE,
    valid_to DATE, is_current BOOLEAN, version_no INTEGER
);
CREATE TABLE IF NOT EXISTS dw.dim_product (
    product_key UBIGINT, stock_code VARCHAR, description VARCHAR, list_price DECIMAL(18, 4),
    valid_from DATE, valid_to DATE, is_current BOOLEAN, version_no INTEGER
);
CREATE TABLE IF NOT EXISTS dw.dim_line_flags (
    flags_key INTEGER, is_cancellation BOOLEAN, is_guest BOOLEAN, is_negative_quantity BOOLEAN
);
CREATE TABLE IF NOT EXISTS dw.fact_sales (
    line_key VARCHAR, invoice VARCHAR, invoice_date_key INTEGER, arrival_date_key INTEGER,
    product_key UBIGINT, customer_key UBIGINT, country_key UBIGINT, flags_key INTEGER,
    quantity BIGINT, unit_price DECIMAL(18, 4), sales_amount DECIMAL(18, 4),
    list_amount DECIMAL(18, 4)
);
CREATE TABLE IF NOT EXISTS dw.fact_returns (
    line_key VARCHAR, invoice VARCHAR, invoice_date_key INTEGER, arrival_date_key INTEGER,
    product_key UBIGINT, customer_key UBIGINT, country_key UBIGINT, flags_key INTEGER,
    quantity BIGINT, unit_price DECIMAL(18, 4), return_amount DECIMAL(18, 4)
);
CREATE TABLE IF NOT EXISTS dw.agg_country_month (
    month_start DATE, country_key UBIGINT, country VARCHAR, sales_amount DECIMAL(38, 4),
    returns_amount DECIMAL(38, 4), net_amount DECIMAL(38, 4), sales_lines BIGINT,
    return_lines BIGINT
);
CREATE SCHEMA IF NOT EXISTS ops;
CREATE TABLE IF NOT EXISTS ops.run (
    run_id VARCHAR, interval_start DATE, interval_end DATE, trigger VARCHAR, status VARCHAR,
    started_at TIMESTAMP, finished_at TIMESTAMP, detail VARCHAR
);
CREATE TABLE IF NOT EXISTS ops.task_run (
    run_id VARCHAR, task VARCHAR, attempt INTEGER, status VARCHAR, started_at TIMESTAMP,
    finished_at TIMESTAMP, error VARCHAR
);
CREATE TABLE IF NOT EXISTS ops.schedule_state (
    schedule VARCHAR, last_fire TIMESTAMP
);
"""

CONTENT_TABLES = [
    "raw.feed_lines",
    "staging.lines",
    "dw.dim_date",
    "dw.dim_country",
    "dw.dim_customer",
    "dw.dim_product",
    "dw.dim_line_flags",
    "dw.fact_sales",
    "dw.fact_returns",
    "dw.agg_country_month",
]

_INTERVAL = re.compile(r"^(\d{4}-\d{2})$")
_RANGE = re.compile(r"^(\d{4}-\d{2}-\d{2})/(\d{4}-\d{2}-\d{2})$")


def connect(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(DDL)
    return con


@contextmanager
def transaction(con: duckdb.DuckDBPyConnection) -> Iterator[None]:
    """All-or-nothing: a task's deletes and inserts commit together or not at all."""
    con.execute("BEGIN TRANSACTION")
    try:
        yield
    except BaseException:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")


def content_hash(con: duckdb.DuckDBPyConnection) -> str:
    """A digest of every content table, rows in a canonical order. Equal digests mean equal data."""
    import hashlib

    digest = hashlib.sha256()
    for table in CONTENT_TABLES:
        rows = con.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()
        digest.update(table.encode())
        digest.update(repr(rows).encode())
    return digest.hexdigest()


def parse_interval(text: str) -> tuple[date, date]:
    """``2010-03`` is the calendar month; ``2010-03-05/2010-03-09`` is an explicit ``[start, end)``."""
    if match := _INTERVAL.match(text):
        start = datetime.strptime(match.group(1), "%Y-%m").date()
        end = start.replace(
            year=start.year + (start.month // 12), month=start.month % 12 + 1
        )
        return start, end
    if match := _RANGE.match(text):
        start, end = (date.fromisoformat(g) for g in match.groups())
        if end <= start:
            raise ValueError(f"interval end must be after its start: {text!r}")
        return start, end
    raise ValueError(f"interval must be YYYY-MM or YYYY-MM-DD/YYYY-MM-DD, got {text!r}")
