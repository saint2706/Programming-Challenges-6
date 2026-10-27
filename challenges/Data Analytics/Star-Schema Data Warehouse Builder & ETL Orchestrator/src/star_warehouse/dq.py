"""Data-quality gates: each raises ``DataQualityError`` (permanent, so the DAG does not retry it)."""

from __future__ import annotations

import duckdb

from star_warehouse.dag import PermanentError


class DataQualityError(PermanentError):
    """A gate failed; the message names the table, the check and the counts."""


def expect(check: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise DataQualityError(f"{check}: expected {expected!r}, got {actual!r}")


def scalar(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None):
    return con.execute(sql, params or []).fetchone()[0]
