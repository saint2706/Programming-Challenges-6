"""Measurement level of a column: what kind of question its values can answer."""

from __future__ import annotations

from typing import Literal

import polars as pl

Level = Literal["binary", "nominal", "ordinal", "continuous"]


def level_of(values: pl.Series, *, ordinal: bool = False) -> Level:
    """Binary, nominal, ordinal or continuous.

    Text and booleans are categories. Numbers with at most ten distinct whole values are ordinal
    (a 1-5 rating, a quality score), otherwise continuous. Integers with many values are treated as
    continuous (body mass in grams is), so a count that must be treated as ordinal is set by the
    caller with ``ordinal=True``. ``ordinal=True`` overrides everything except binary.
    """
    present = values.drop_nulls()
    distinct = present.n_unique()
    if distinct == 2:
        return "binary"
    if ordinal:
        return "ordinal"
    if not present.dtype.is_numeric():
        return "nominal"
    whole = present.dtype.is_integer() or bool((present == present.round(0)).all())
    if whole and distinct <= 10:
        return "ordinal"
    return "continuous"
