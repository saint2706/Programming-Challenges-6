"""DuckDB adapter: the shared SQL in ``workloads.py`` over views on the Parquet or CSV files.

The same SQL runs as the oracle with an exact ``count(DISTINCT ...)``; the timed engine uses
DuckDB's ``approx_count_distinct`` (a HyperLogLog), which the verifier checks against the bound.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa

from dist_pipeline.engines.base import Dataset
from dist_pipeline.workloads import Query, render_sql

LABEL = "DuckDB (SQL)"
APPROX_DISTINCT = "approx_count_distinct(pu_location_id)"
EXACT_DISTINCT = "count(DISTINCT pu_location_id)"


def _source_sql(ds: Dataset) -> str:
    if ds.fmt == "csv":
        return f"read_csv('{ds.source}', header = true, auto_detect = true)"
    return f"read_parquet('{ds.parquet_glob}')"


def connect(ds: Dataset) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW trips AS SELECT * FROM {_source_sql(ds)}")
    con.execute(
        'CREATE VIEW zones AS SELECT CAST("LocationID" AS INTEGER) AS location_id, '
        f"\"Borough\" AS borough FROM read_csv('{ds.zones_csv.as_posix()}', header = true)"
    )
    return con


def run_sql(con: duckdb.DuckDBPyConnection, query: Query, distinct: str) -> pa.Table:
    sql = render_sql(query, trips="trips", zones="zones", distinct=distinct)
    return con.sql(sql).to_arrow_table()


def start(ds: Dataset, opts: dict[str, object]) -> duckdb.DuckDBPyConnection:
    return connect(ds)


def run(
    session: duckdb.DuckDBPyConnection,
    query: Query,
    ds: Dataset,
    opts: dict[str, object],
) -> pa.Table:
    return run_sql(session, query, APPROX_DISTINCT)


def stop(session: duckdb.DuckDBPyConnection) -> None:
    session.close()
