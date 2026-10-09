"""Reading a source into a table of strings with a stable row number.

Everything is read as text on purpose: whether ``"05"`` equals ``5``, or ``"\\N"`` means missing, is a
decision for the comparison step to make visibly, not for the CSV reader to make silently.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from data_reconciler.config import ConfigError, Source
from data_reconciler.paths import project_root

ROW = "_row"
DATA_FILES = ("openflights_airports.dat", "ourairports_airports.csv")


def data_dir(root: Path | None = None) -> Path:
    """The full data folder when both files were fetched, otherwise the committed sample."""
    root = root or project_root()
    for folder in ("data", "sample_data"):
        if all((root / folder / name).exists() for name in DATA_FILES):
            return root / folder
    raise FileNotFoundError(
        f"no airport files under {root}/data or {root}/sample_data; "
        "run `uv run python -m data_reconciler.fetch_data`"
    )


def resolve(path: str, root: Path | None = None) -> Path:
    """``{data}`` becomes the data folder; other relative paths are relative to the project root."""
    root = root or project_root()
    if "{data}" in path:
        return Path(path.replace("{data}", str(data_dir(root))))
    p = Path(path)
    return p if p.is_absolute() else root / p


def load_source(source: Source, root: Path | None = None) -> pl.DataFrame:
    """The in-scope rows of the source: see ``load_split``."""
    return load_split(source, root)[0]


def load_split(
    source: Source, root: Path | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(in_scope, out_of_scope)`` as strings plus ``_row`` (0-based position in the file).

    ``derive`` adds columns from Polars SQL expressions and ``scope`` splits rows by one; both see the
    raw columns. A row for which the scope is NULL is out of scope. ``_row`` is assigned before the
    split, so it always points back at the file.
    """
    path = resolve(source.path, root)
    if not path.exists():
        raise ConfigError(f"{source.name}: file not found: {path}")
    frame = pl.read_csv(
        path,
        has_header=source.header,
        new_columns=list(source.columns) if not source.header else None,
        null_values=list(source.null_values),
        separator=source.delimiter,
        infer_schema_length=0,
        encoding="utf8",
        truncate_ragged_lines=False,
    )
    if not source.header and len(source.columns) != frame.width:
        raise ConfigError(
            f"{source.name}: file has {frame.width} columns but the config names {len(source.columns)}"
        )
    frame = frame.with_columns(pl.int_range(pl.len()).alias(ROW))
    try:
        if source.derive:
            frame = frame.with_columns(
                *(pl.sql_expr(expr).alias(name) for name, expr in source.derive.items())
            )
        inside = (
            pl.sql_expr(source.scope).fill_null(False) if source.scope else pl.lit(True)
        )
        kept, dropped = frame.filter(inside), frame.filter(~inside)
    except (pl.exceptions.PolarsError, pl.exceptions.SQLInterfaceError) as exc:
        raise ConfigError(f"{source.name}: bad derive/scope expression: {exc}") from exc
    return kept, dropped


def require_columns(frame: pl.DataFrame, columns: list[str], source: Source) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ConfigError(
            f"{source.name} has no column(s) {missing}; it has {sorted(frame.columns)}"
        )
