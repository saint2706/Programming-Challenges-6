"""The retail database every query runs against, and the guard in front of it."""

from __future__ import annotations

from pathlib import Path

import duckdb

from sql_profiler.fetch_data import TABLES
from sql_profiler.paths import project_root


def data_dir(root: Path | None = None) -> Path:
    """The full tables when fetched, otherwise the committed sample."""
    root = root or project_root()
    for folder in ("data", "sample_data"):
        if all((root / folder / f"{t}.parquet").exists() for t in TABLES):
            return root / folder
    raise FileNotFoundError(
        f"no retail tables under {root}/data or {root}/sample_data; "
        "run `uv run --group fetch python -m sql_profiler.fetch_data`"
    )


def connect(
    root: Path | None = None, threads: int | None = None
) -> duckdb.DuckDBPyConnection:
    """An in-memory database holding the four tables, with external file access switched off.

    Switching it off after loading means a query typed into the profiler cannot read or write any
    file on the machine (``read_csv('C:/...')``, ``COPY ... TO``), which cannot be undone on the
    connection.
    """
    folder = data_dir(root)
    con = duckdb.connect()
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    for table in TABLES:
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_parquet(?)",
            [str(folder / f"{table}.parquet")],
        )
    con.execute("SET enable_external_access = false")
    return con


def check_read_only(con: duckdb.DuckDBPyConnection, sql: str) -> str:
    """The query text if it is exactly one SELECT (or WITH ... SELECT); otherwise ``ValueError``."""
    try:
        statements = con.extract_statements(sql)
    except duckdb.Error as exc:
        raise ValueError(f"cannot parse the query: {exc}") from exc
    if len(statements) != 1:
        raise ValueError(f"expected exactly one statement, got {len(statements)}")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise ValueError(
            f"only SELECT queries can be profiled, not {statements[0].type.name}"
        )
    return sql
