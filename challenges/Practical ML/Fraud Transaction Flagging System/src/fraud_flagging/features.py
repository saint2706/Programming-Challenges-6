"""Per-transaction features. Each one comes from that transaction's own fields.

The data has no card, merchant or customer id, so there is no velocity or history feature to build,
and scoring a transaction needs nothing else: not the rest of the stream, not the labels.

Groups (the ablation drops one at a time):

* ``pca``: ``V1`` to ``V28``, the published principal components.
* ``amount``: ``log_amount``; ``amount_frac``, the cents part, so round sums are visible;
  ``amount_zero``, a zero-amount transaction (27 of the 492 published frauds, 1,798 legitimate ones).
* ``time``: the time of day as a circle. ``Time`` counts seconds from the first transaction, whose
  clock time is not published, so this is "time of day relative to the start of the data".
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fraud_flagging.data import RAW_COLUMNS, V_COLUMNS

DAY = 86_400.0
GROUPS: dict[str, list[str]] = {
    "pca": V_COLUMNS,
    "amount": ["log_amount", "amount_frac", "amount_zero"],
    "time": ["tod_sin", "tod_cos"],
}
FEATURES = [f for group in GROUPS.values() for f in group]


def validate(df: pl.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` cannot be scored: missing columns, non-finite values, bad ranges."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for col in RAW_COLUMNS:
        if not df.schema[col].is_numeric():
            raise ValueError(f"column {col} must be numeric, not {df.schema[col]}")
        values = df[col]
        if values.is_null().any() or not values.cast(pl.Float64).is_finite().all():
            raise ValueError(f"column {col} has missing or non-finite values")
    if (df["Amount"] < 0).any():
        raise ValueError("Amount must not be negative")
    if (df["Time"] < 0).any():
        raise ValueError("Time must not be negative")


def build(df: pl.DataFrame) -> pl.DataFrame:
    """``df`` plus the engineered feature columns (the input columns are kept)."""
    validate(df)
    cents = (pl.col("Amount") * 100).round() % 100
    angle = 2 * np.pi * (pl.col("Time") % DAY) / DAY
    return df.with_columns(
        pl.col("Amount").log1p().alias("log_amount"),
        (cents / 100).alias("amount_frac"),
        (pl.col("Amount") == 0).cast(pl.Float64).alias("amount_zero"),
        angle.sin().alias("tod_sin"),
        angle.cos().alias("tod_cos"),
        (pl.col("Time") // 3600).cast(pl.Int32).alias("hour"),
    )


def without(group: str) -> list[str]:
    """All features except one group."""
    drop = set(GROUPS[group])
    return [f for f in FEATURES if f not in drop]
