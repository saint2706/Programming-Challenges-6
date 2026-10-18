"""Run the pipeline in DuckDB, optionally with one source column deliberately changed.

The perturbed copy of a source keeps its row order and all its other columns. Four ways of changing
a column exist because each one wakes up a different kind of dependency:

* ``shift``: a type-aware change (numbers ``x*3+7``, text ``x || '~'``, dates ``+11 days``, booleans
  negated). Changes values and keys but keeps the null pattern.
* ``shuffle``: the same values in another order (seeded). Keeps every distribution, so it changes
  *which row* a value sits on: moves filter and join outcomes without inventing new values.
* ``nullify``: every 7th row becomes NULL. Wakes the fallback branch of COALESCE and the
  null-handling of filters.
* ``constant``: every row gets the column's smallest value. Collapses join keys and group keys.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from data_lineage.project import Project

STRATEGIES = ("shift", "shuffle", "nullify", "constant")
SHUFFLE_SEED = 20091201


def qi(name: str) -> str:
    """A quoted SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def _is_numeric(sql_type: str) -> bool:
    t = sql_type.upper()
    return any(
        k in t
        for k in ("INT", "DECIMAL", "DOUBLE", "FLOAT", "REAL", "NUMERIC", "HUGEINT")
    )


def shifted(column: str, sql_type: str) -> str:
    c, t = qi(column), sql_type.upper()
    if t == "BOOLEAN":
        return f"NOT {c}"
    if _is_numeric(t):
        return f"{c} * 3 + 7"
    if t == "DATE":
        return f"{c} + 11"  # DATE + INTERVAL would silently turn the column into a TIMESTAMP
    if t.startswith("TIMESTAMP"):
        return f"{c} + INTERVAL 11 DAY"
    return f"{c} || '~'"


def perturbed_select(
    base: str, columns: dict[str, str], target: str, strategy: str
) -> str:
    """A SELECT that reproduces ``base`` with ``target`` changed by ``strategy``."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    if target not in columns:
        raise KeyError(target)
    names = ", ".join(qi(c) for c in columns)
    t = qi(target)
    if strategy == "shuffle":
        others = ", ".join(
            f"b.{qi(c)}" if c != target else f"v._v AS {t}" for c in columns
        )
        return (
            f"WITH b AS (SELECT {names}, row_number() OVER () AS _rn FROM {qi(base)}), "
            f"p AS (SELECT _rn, row_number() OVER (ORDER BY hash(_rn + {SHUFFLE_SEED}), _rn) AS _pr FROM b), "
            f"v AS (SELECT {t} AS _v, row_number() OVER (ORDER BY _rn) AS _vr FROM b) "
            f"SELECT {others} FROM b JOIN p USING (_rn) JOIN v ON v._vr = p._pr ORDER BY b._rn"
        )
    if strategy == "constant":
        new = f"(SELECT min({t}) FROM {qi(base)})"
        body = ", ".join(qi(c) if c != target else f"{new} AS {t}" for c in columns)
        return f"SELECT {body} FROM {qi(base)}"
    numbered = f"(SELECT *, row_number() OVER () AS _rn FROM {qi(base)})"
    if strategy == "nullify":
        new = f"CASE WHEN _rn % 7 = 0 THEN NULL ELSE {t} END"
    else:
        new = shifted(target, columns[target])
    body = ", ".join(qi(c) if c != target else f"{new} AS {t}" for c in columns)
    return f"SELECT {body} FROM {numbered} ORDER BY _rn"


def connect() -> duckdb.DuckDBPyConnection:
    """A fresh in-memory database set up so repeated runs give identical results."""
    con = duckdb.connect(":memory:")
    con.execute("SET threads = 1")
    con.execute("SET preserve_insertion_order = true")
    return con


def load_sources(
    con: duckdb.DuckDBPyConnection,
    project: Project,
    schemas: dict[str, dict[str, str]],
    perturb: tuple[str, str, str] | None = None,
) -> None:
    """Create every source as a table; ``perturb`` is ``(source, column, strategy)``."""
    for src in project.sources:
        con.execute(f"CREATE TABLE {qi(src.name)} AS SELECT * FROM {src.relation()}")
        if perturb and perturb[0] == src.name:
            con.execute(
                f"ALTER TABLE {qi(src.name)} RENAME TO {qi('_base_' + src.name)}"
            )
            select = perturbed_select(
                "_base_" + src.name, schemas[src.name], perturb[1], perturb[2]
            )
            con.execute(f"CREATE TABLE {qi(src.name)} AS {select}")
            con.execute(f"DROP TABLE {qi('_base_' + src.name)}")


def run_models(
    con: duckdb.DuckDBPyConnection, project: Project, order: list[str]
) -> None:
    by_name = {m.name: m for m in project.models}
    for name in order:
        con.execute(f"CREATE TABLE {qi(name)} AS {by_name[name].sql}")


def actual_columns(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    return {r[0]: r[1] for r in con.execute(f"DESCRIBE {qi(table)}").fetchall()}


_FLOATY = ("DOUBLE", "FLOAT", "REAL", "DECIMAL")


def unique_key(con: duckdb.DuckDBPyConnection, table: str) -> str | None:
    """The first non-float column whose values are all present and all different, if any."""
    n = con.execute(f"SELECT count(*) FROM {qi(table)}").fetchone()[0]
    if n == 0:
        return None
    for column, sql_type in actual_columns(con, table).items():
        if any(k in sql_type.upper() for k in _FLOATY):
            continue  # a float that happens to be unique is not an identifier
        c = qi(column)
        distinct, nulls = con.execute(
            f"SELECT count(DISTINCT {c}), count(*) FILTER (WHERE {c} IS NULL) FROM {qi(table)}"
        ).fetchone()
        if distinct == n and nulls == 0:
            return column
    return None


@dataclass(frozen=True, slots=True)
class Prints:
    """Fingerprints of one table: row count, then per column a plain and a key-aligned hash."""

    rows: int
    key: str | None
    plain: dict[str, str]
    aligned: dict[str, str]  # empty when the table has no key


def fingerprints(
    con: duckdb.DuckDBPyConnection, table: str, key: str | None = None
) -> Prints:
    """Hash every column two ways.

    *Plain*: the sorted list of the column's values, so row order is ignored. *Aligned*: the
    sorted list of ``(key, value)`` pairs, so a value moving to another row counts as a change.
    Ranks and scores need that: they are a permutation of 1..N whatever the data, and only the
    pairing with the row shows that they moved.
    """
    n = con.execute(f"SELECT count(*) FROM {qi(table)}").fetchone()[0]
    columns = actual_columns(con, table)

    def text(column: str) -> str:
        c = qi(column)
        if any(k in columns[column].upper() for k in _FLOATY):
            inner = f"CAST(round(CAST({c} AS DOUBLE), 4) AS VARCHAR)"
        else:
            inner = f"CAST({c} AS VARCHAR)"
        return f"coalesce({inner}, '<null>')"

    def digest(value: str) -> str:
        return con.execute(
            f"SELECT md5(coalesce(string_agg({value}, '|' ORDER BY {value}), '')) FROM {qi(table)}"
        ).fetchone()[0]

    plain = {column: digest(text(column)) for column in columns}
    aligned = (
        {column: digest(f"{text(key)} || '#' || {text(column)}") for column in columns}
        if key is not None
        else {}
    )
    return Prints(n, key, plain, aligned)
