"""Discover datasets in SQLite files, DuckDB files and folders of Parquet/CSV/JSON, and profile them.

Every source is opened read-only. Profiling runs one aggregate query per table (row count, and per
column the non-null count, distinct count, min and max) plus a small distinct-value sample per
column, used for PII detection and shown in the catalog.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from data_catalog.pii import tag_column

FILE_KINDS = {
    ".parquet": "parquet",
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
}
SQLITE_EXT = {".sqlite", ".sqlite3", ".db"}
DUCKDB_EXT = {".duckdb", ".ddb"}
SAMPLE = 50


@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool
    nulls: int = 0
    distinct: int = 0
    min: str | None = None
    max: str | None = None
    samples: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class Dataset:
    kind: str  # sqlite | duckdb | parquet | csv | json
    source: str  # database file, or the data file itself
    name: str  # table name, or file stem
    rows: int = 0
    columns: list[ColumnInfo] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    # declared foreign keys: (column, referenced table name in the same source, referenced column)
    foreign_keys: list[tuple[str, str, str]] = field(default_factory=list)
    fingerprint: str = ""

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.source}#{self.name}"

    def schema_signature(self) -> list[tuple[str, str]]:
        return [(c.name, c.type) for c in self.columns]


def qi(name: str) -> str:
    """Quote an SQL identifier (both SQLite and DuckDB use double quotes)."""
    return '"' + name.replace('"', '""') + '"'


def ql(text: str) -> str:
    """Quote an SQL string literal."""
    return "'" + text.replace("'", "''") + "'"


def _fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def is_sqlite(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False


def discover(paths: list[Path]) -> Iterator[tuple[str, Path]]:
    """(kind, path) for every source under the given files and folders; symlinks are not followed."""
    for p in paths:
        if p.is_dir():
            for root, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                for f in sorted(files):
                    entry = Path(root) / f
                    # A symlink inside a crawled folder could point anywhere on disk.
                    if not f.startswith(".") and not entry.is_symlink():
                        yield from discover([entry])
        elif p.is_file():
            ext = p.suffix.lower()
            if ext in FILE_KINDS:
                yield FILE_KINDS[ext], p
            elif ext in DUCKDB_EXT:
                yield "duckdb", p
            elif (ext in SQLITE_EXT or is_sqlite(p)) and is_sqlite(p):
                yield "sqlite", p
        elif not p.exists():
            raise FileNotFoundError(p)


# ---- per-engine access --------------------------------------------------------------------------


@contextmanager
def connect(kind: str, source: str):
    """A read-only connection plus the FROM clause for a dataset name."""
    if kind == "sqlite":
        uri = Path(source).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as con:
            yield con, qi
    elif kind == "duckdb":
        # A crawled .duckdb is untrusted: a view inside it could read any local file
        # (read_csv('~/.ssh/...')), so external access is off for these connections.
        cfg = {"enable_external_access": False}
        with closing(duckdb.connect(source, read_only=True, config=cfg)) as con:
            yield con, qi
    else:
        reader = {"parquet": "read_parquet", "csv": "read_csv", "json": "read_json"}[
            kind
        ]
        with closing(duckdb.connect()) as con:
            yield con, lambda _name: f"{reader}({ql(source)})"


def _profile(con, frm: str, ds: Dataset) -> None:
    exprs = ["count(*)"]
    for c in ds.columns:
        col = qi(c.name)
        exprs += [
            f"count({col})",
            f"count(DISTINCT {col})",
            f"CAST(min({col}) AS VARCHAR)",
            f"CAST(max({col}) AS VARCHAR)",
        ]
    row = con.execute(f"SELECT {', '.join(exprs)} FROM {frm}").fetchone()
    ds.rows = row[0]
    for i, c in enumerate(ds.columns):
        nonnull, c.distinct, c.min, c.max = row[1 + 4 * i : 5 + 4 * i]
        c.nulls = ds.rows - nonnull
        vals = con.execute(
            f"SELECT DISTINCT CAST({qi(c.name)} AS VARCHAR) FROM {frm} "
            f"WHERE {qi(c.name)} IS NOT NULL LIMIT {SAMPLE}"
        ).fetchall()
        c.samples = [v[0] for v in vals]
        c.tags = tag_column(c.name, c.samples)
        # The catalog records that a column holds PII, never the PII itself.
        if "pii" in c.tags:
            c.samples, c.min, c.max = [], None, None


def _sqlite(path: Path) -> list[Dataset]:
    out = []
    with connect("sqlite", str(path)) as (con, frm):
        names = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for name in names:
            ds = Dataset("sqlite", str(path), name, fingerprint=_fingerprint(path))
            info = con.execute(f"PRAGMA table_info({qi(name)})").fetchall()
            ds.columns = [
                ColumnInfo(r[1], r[2] or "ANY", not r[3] and not r[5]) for r in info
            ]
            ds.primary_key = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5]]
            for r in con.execute(f"PRAGMA foreign_key_list({qi(name)})"):
                ds.foreign_keys.append((r[3], r[2], r[4] or ""))
            _profile(con, frm(name), ds)
            out.append(ds)
    return out


def _duckdb(path: Path) -> list[Dataset]:
    out = []
    with connect("duckdb", str(path)) as (con, _frm):
        tables = con.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_catalog = current_database() ORDER BY 1, 2"
        ).fetchall()
        cons = con.execute(
            "SELECT schema_name, table_name, constraint_type, constraint_column_names, "
            "referenced_table, referenced_column_names FROM duckdb_constraints()"
        ).fetchall()
        for schema, table in tables:
            name = table if schema == "main" else f"{schema}.{table}"
            ds = Dataset("duckdb", str(path), name, fingerprint=_fingerprint(path))
            cols = con.execute(
                "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
                "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
                [schema, table],
            ).fetchall()
            ds.columns = [ColumnInfo(n, t, nl == "YES") for n, t, nl in cols]
            for s, t, kind, cols_, ref, ref_cols in cons:
                if (s, t) != (schema, table):
                    continue
                if kind == "PRIMARY KEY":
                    ds.primary_key = list(cols_)
                elif kind == "FOREIGN KEY":
                    ds.foreign_keys += [
                        (a, ref, b) for a, b in zip(cols_, ref_cols, strict=False)
                    ]
            _profile(con, f"{qi(schema)}.{qi(table)}", ds)
            out.append(ds)
    return out


def _file(kind: str, path: Path) -> list[Dataset]:
    ds = Dataset(kind, str(path), path.stem, fingerprint=_fingerprint(path))
    with connect(kind, str(path)) as (con, frm):
        desc = con.execute(f"DESCRIBE SELECT * FROM {frm(ds.name)}").fetchall()
        ds.columns = [ColumnInfo(r[0], r[1], True) for r in desc]
        _profile(con, frm(ds.name), ds)
    return [ds]


def crawl_source(kind: str, path: Path) -> list[Dataset]:
    if kind == "sqlite":
        return _sqlite(path)
    if kind == "duckdb":
        return _duckdb(path)
    return _file(kind, path)


def distinct_values(
    ds_kind: str, source: str, name: str, column: str, cap: int
) -> set[str] | None:
    """All distinct non-null values of a column as text, or None if there are more than `cap`."""
    with connect(ds_kind, source) as (con, frm):
        table = frm(name)
        if ds_kind == "duckdb" and "." in name:
            schema, t = name.split(".", 1)
            table = f"{qi(schema)}.{qi(t)}"
        rows = con.execute(
            f"SELECT DISTINCT CAST({qi(column)} AS VARCHAR) FROM {table} "
            f"WHERE {qi(column)} IS NOT NULL LIMIT {cap + 1}"
        ).fetchall()
    return None if len(rows) > cap else {r[0] for r in rows}
