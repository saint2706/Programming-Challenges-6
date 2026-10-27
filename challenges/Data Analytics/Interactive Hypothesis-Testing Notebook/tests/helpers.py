from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from hypothesis_wizard import data

ROOT = Path(__file__).resolve().parents[1]


@cache
def reference() -> dict[str, Any]:
    """Base-R results written by scripts/make_reference.R."""
    return json.loads(
        (ROOT / "tests" / "reference" / "r_tests.json").read_text("utf-8")
    )


def table(key: str) -> pl.DataFrame:
    return data.load(key)


def values(
    df: pl.DataFrame, column: str, *, where: tuple[str, object] | None = None
) -> np.ndarray:
    frame = (
        df
        if where is None
        else df.filter(pl.col(where[0]).cast(pl.Utf8) == str(where[1]))
    )
    return frame[column].cast(pl.Float64).drop_nulls().to_numpy()
