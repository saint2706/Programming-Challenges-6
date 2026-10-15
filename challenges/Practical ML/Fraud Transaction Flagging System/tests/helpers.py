"""A small synthetic table in the published ULB schema (Time, V1..V28, Amount, Class).

Fraud shifts a few components (V3, V4, V10, V12, V14), tends to be small, and is a little more
common in the first day; a few rows are exact duplicates and some amounts are zero, as in the real
file. ``Class`` is a categorical of "0"/"1" strings, as OpenML ships it.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from fraud_flagging.data import V_COLUMNS


def synthetic_published(
    n: int = 30_000, seed: int = 0, duplicates: int = 25
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0, 172_792, n)).round()
    rate = np.where(time < 86_400, 0.013, 0.008)
    fraud = rng.random(n) < rate
    v = rng.normal(0, 1, (n, 28))
    for col, shift in ((2, -1.6), (3, 1.8), (9, -2.0), (11, -2.2), (13, -2.6)):
        v[fraud, col] += shift
    amount = np.where(
        fraud, rng.lognormal(2.0, 1.4, n), rng.lognormal(3.2, 1.2, n)
    ).round(2)
    amount[rng.random(n) < 0.01] = 0.0
    table = pl.DataFrame(
        {
            "Time": time,
            **{c: v[:, i] for i, c in enumerate(V_COLUMNS)},
            "Amount": amount,
        }
    )
    table = table.with_columns(
        pl.Series("Class", np.where(fraud, "1", "0")).cast(pl.Categorical)
    )
    if duplicates:
        table = pl.concat([table, table.head(duplicates)]).sort(
            "Time", maintain_order=True
        )
    return table


def raw_with_labels(published: pl.DataFrame) -> pl.DataFrame:
    """The published table the way ``data.load_raw`` returns it."""
    return (
        published.with_row_index("tx_id")
        .with_columns(pl.col("Class").cast(pl.String).cast(pl.Int8).alias("y"))
        .drop("Class")
    )
