"""Small hand-made sources and jobs."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from data_reconciler import config

FIELDS = [
    {"name": "code", "left": "code", "right": "code", "compare": "code"},
    {"name": "name", "left": "name", "right": "title", "compare": "text"},
    {
        "name": "qty",
        "left": "qty",
        "right": "qty",
        "compare": "number",
        "abs_tolerance": 1,
    },
    {
        "name": "where",
        "left": ["lat", "lon"],
        "right": ["lat", "lon"],
        "compare": "geo",
        "tolerance_km": 1,
    },
    {
        "name": "country",
        "left": "country",
        "right": "iso",
        "compare": "crosswalk",
        "min_support": 3,
    },
]


def write(path: Path, rows: list[dict]) -> str:
    pl.DataFrame(rows, infer_schema_length=None).write_csv(path)
    return str(path)


def job(left: str, right: str, **extra) -> config.Config:
    raw = {
        "left": {"name": "L", "path": left},
        "right": {"name": "R", "path": right},
        "keys": [{"left": "code", "right": "code"}],
        "fields": FIELDS,
    }
    raw.update(extra)
    return config.parse(raw)
