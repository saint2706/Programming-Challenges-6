"""Real-data fixtures: NYC taxi trips (March 2019) cut into daily batches.

`sample_data/taxis.csv` is the 6,433-row taxi sample shipped with seaborn's example
datasets (mwaskom/seaborn-data), itself a slice of the NYC TLC trip records. Real
data matters here: it has real nulls, real heavy tails and a real weekday/weekend
rhythm, so "normal batch-to-batch noise" is the actual thing the monitor must ignore.

One column is added: `trip_id` (`TX-000001`, unique), because the source has no key
and uniqueness / format checks need one.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from data_quality.paths import project_root

HERE = project_root()
TAXI_CSV = HERE / "sample_data" / "taxis.csv"

# Days 1-21 train the baseline; days 22-31 are held out as real, clean incoming batches.
BASELINE_LAST_DAY = 21


def load_taxi(path: Path = TAXI_CSV) -> pl.DataFrame:
    df = pl.read_csv(path, try_parse_dates=True, infer_schema_length=None).sort(
        "pickup"
    )
    ids = [f"TX-{i:06d}" for i in range(1, df.height + 1)]
    return df.with_columns(pl.Series("trip_id", ids)).select(
        "trip_id", pl.exclude("trip_id")
    )


def daily_batches(df: pl.DataFrame) -> dict[date, pl.DataFrame]:
    """One batch per pickup day. The single Feb-28 trip is folded into March 1."""
    day = pl.col("pickup").dt.date().clip(lower_bound=date(2019, 3, 1))
    out: dict[date, pl.DataFrame] = {}
    for (d,), part in df.with_columns(day.alias("_day")).group_by(
        "_day", maintain_order=True
    ):
        out[d] = part.drop("_day")
    return out


def split(
    df: pl.DataFrame,
) -> tuple[list[tuple[str, pl.DataFrame]], list[tuple[str, pl.DataFrame]]]:
    """(baseline batches, held-out clean batches) as (name, frame) pairs."""
    train, held = [], []
    for d, part in daily_batches(df).items():
        (train if d.day <= BASELINE_LAST_DAY else held).append(
            (f"taxi_{d.isoformat()}.csv", part)
        )
    return train, held
