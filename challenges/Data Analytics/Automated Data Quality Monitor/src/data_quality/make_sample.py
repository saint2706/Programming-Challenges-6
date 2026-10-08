"""Write the demo CSV batches (idempotent; everything is derived from sample_data/taxis.csv).

    uv run python -m data_quality.make_sample

sample_data/batches/train/      taxi days 1-21, the known-good baseline batches
sample_data/batches/incoming/   taxi days 22-31, real clean batches to check
sample_data/batches/drifted/    one held-out day with a documented corruption planted in it
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl

from data_quality.corruptions import CORRUPTIONS
from data_quality.taxi import HERE, load_taxi, split

OUT = HERE / "sample_data" / "batches"
# The corruptions shipped as ready-made files (all of them are exercised by evaluate.py).
DEMO = (
    "rename_column",
    "text_in_numeric",
    "nulls_fare_15pct",
    "fare_x1.5",
    "negative_fares",
    "payment_mix",
    "new_payment_type",
    "duplicate_keys",
    "key_format",
    "epoch_zero_dates",
    "volume_tenth",
    "stale_batch",
)


def _write(df: pl.DataFrame, path: Path) -> None:
    df.write_csv(path, datetime_format="%Y-%m-%d %H:%M:%S")


def main() -> None:
    train, held = split(load_taxi())
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("train", "incoming", "drifted"):
        (OUT / sub).mkdir(parents=True)
    for name, df in train:
        _write(df, OUT / "train" / name)
    for name, df in held:
        _write(df, OUT / "incoming" / name)

    base_day = held[0][1]
    rng = np.random.default_rng(7)
    by_name = {c.name: c for c in CORRUPTIONS}
    for name in DEMO:
        _write(
            by_name[name].apply(base_day, rng),
            OUT / "drifted" / f"{held[0][0][:-4]}__{name}.csv",
        )
    print(
        f"wrote {len(train)} train, {len(held)} incoming and {len(DEMO)} drifted batches under {OUT}"
    )


if __name__ == "__main__":
    main()
