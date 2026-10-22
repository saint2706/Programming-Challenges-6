"""Turning table columns into a Cox design matrix: numbers pass through, categories become dummies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from survival_toolkit.types import SurvivalError


@dataclass(frozen=True)
class Design:
    x: np.ndarray  # rows with complete covariates only
    names: list[str]
    keep: np.ndarray  # boolean mask over the input rows: which rows made it
    levels: dict[str, list[str]]  # categorical column -> [reference, other levels...]


def encode(
    frame: pl.DataFrame,
    columns: list[str],
    *,
    reference: dict[str, str] | None = None,
) -> Design:
    """Numeric and boolean columns as they are; text columns as 0/1 dummies.

    A text column becomes one dummy per level except its reference level, which is the one named
    in ``reference`` or else the most common (the stable choice: a rare reference level gives
    huge, noisy hazard ratios). Rows missing any covariate are left out and reported by ``keep``.
    """
    reference = reference or {}
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SurvivalError(
            f"no column {missing[0]!r}; columns are {', '.join(frame.columns)}"
        )
    keep = np.ones(frame.height, dtype=bool)
    for c in columns:
        keep &= frame[c].is_not_null().to_numpy()
        if frame[c].dtype.is_float():
            keep &= frame[c].is_finite().fill_null(False).to_numpy()
    sub = frame.filter(pl.Series(keep))
    cols: list[np.ndarray] = []
    names: list[str] = []
    levels: dict[str, list[str]] = {}
    for c in columns:
        s = sub[c]
        if s.dtype == pl.Boolean or s.dtype.is_numeric():
            cols.append(s.cast(pl.Float64).to_numpy())
            names.append(c)
        else:
            text = s.cast(pl.String)
            counts = text.value_counts().sort(["count", c], descending=[True, False])
            ordered = counts[c].to_list()
            ref = reference.get(c, ordered[0])
            if ref not in ordered:
                raise SurvivalError(f"reference level {ref!r} is not a value of {c!r}")
            rest = sorted(v for v in ordered if v != ref)
            if not rest:
                raise SurvivalError(
                    f"{c!r} has a single level, so it cannot be a covariate"
                )
            levels[c] = [ref, *rest]
            arr = text.to_numpy()
            for level in rest:
                cols.append((arr == level).astype(float))
                names.append(f"{c}[{level}]")
    return Design(np.column_stack(cols), names, keep, levels)
