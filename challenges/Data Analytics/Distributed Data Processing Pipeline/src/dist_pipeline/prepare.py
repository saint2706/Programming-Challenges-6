"""Turn the raw TLC monthly files into the datasets every engine reads.

Two scales are built from the same rows, so the scale ladder is a prefix-free subset rather than a
different sample:

* ``1m``  : March 2019 only (``yellow_tripdata_2019-03``), the month ``Automated Data Quality Monitor``
  already uses.
* ``12m`` : all of 2019.

Normalization (identical for every engine, recorded in ``_stats.json``):

* keep the 11 columns the workloads use, with lowercase names and fixed types;
* keep only rows whose pickup falls in 2019 (a handful of TLC rows carry 2008-2018 pickups);
* write part files of about 1,000,000 rows, one row group each. TLC's monthly files hold a single
  row group, which would hand file-based engines one partition per month. Part files give every
  engine the same partitioning to start from.

``data/csv/1m/trips.csv`` is the same March 2019 rows exported to CSV by this script (the TLC's own
CSV copies are not served), used for the CSV-versus-Parquet leg.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from dist_pipeline.download import MONTHS
from dist_pipeline.paths import data_dir

PART_ROWS = 1_000_000

SCALES: dict[str, tuple[str, ...]] = {
    "1m": (MONTHS[2],),
    "12m": MONTHS,
}

COLUMNS = {
    "VendorID": ("vendor_id", pl.Int32),
    "tpep_pickup_datetime": ("pickup_ts", pl.Datetime("us")),
    "tpep_dropoff_datetime": ("dropoff_ts", pl.Datetime("us")),
    "passenger_count": ("passenger_count", pl.Int32),
    "trip_distance": ("trip_distance", pl.Float64),
    "PULocationID": ("pu_location_id", pl.Int32),
    "DOLocationID": ("do_location_id", pl.Int32),
    "payment_type": ("payment_type", pl.Int32),
    "fare_amount": ("fare_amount", pl.Float64),
    "tip_amount": ("tip_amount", pl.Float64),
    "total_amount": ("total_amount", pl.Float64),
}


def raw_path(month: str) -> Path:
    return data_dir() / "raw" / f"yellow_tripdata_{month}.parquet"


def normalize_month(month: str) -> tuple[pl.DataFrame, int]:
    """One month as the normalized frame, plus the number of rows dropped for a pickup outside 2019."""
    raw = pl.read_parquet(raw_path(month), columns=list(COLUMNS))
    raw_rows = raw.height
    df = raw.select(
        [pl.col(src).cast(dtype).alias(dst) for src, (dst, dtype) in COLUMNS.items()]
    ).filter(pl.col("pickup_ts").dt.year() == 2019)
    return df, raw_rows - df.height


def build_scale(scale: str, out_root: Path | None = None) -> dict[str, object]:
    """Write ``data/normalized/<scale>/part-NNNN.parquet`` and ``_stats.json``."""
    out = (out_root or data_dir()) / "normalized" / scale
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("part-*.parquet"):
        old.unlink()
    part = 0
    rows = dropped = 0
    per_month: dict[str, int] = {}
    for month in SCALES[scale]:
        df, dropped_here = normalize_month(month)
        dropped += dropped_here
        per_month[month] = df.height
        for start in range(0, df.height, PART_ROWS):
            chunk = df.slice(start, PART_ROWS)
            chunk.write_parquet(
                out / f"part-{part:04d}.parquet", row_group_size=PART_ROWS
            )
            part += 1
            rows += chunk.height
    stats = {
        "scale": scale,
        "months": list(SCALES[scale]),
        "rows": rows,
        "rows_per_month": per_month,
        "rows_dropped_pickup_outside_2019": dropped,
        "part_files": part,
        "bytes": sum(p.stat().st_size for p in out.glob("part-*.parquet")),
    }
    (out / "_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def build_csv(scale: str = "1m", out_root: Path | None = None) -> dict[str, object]:
    """The CSV leg: the normalized rows of ``scale`` written as one CSV file."""
    src = (out_root or data_dir()) / "normalized" / scale
    out = (out_root or data_dir()) / "csv" / scale
    out.mkdir(parents=True, exist_ok=True)
    target = out / "trips.csv"
    pl.scan_parquet(src / "part-*.parquet").sink_csv(target)
    stats = {"scale": scale, "bytes": target.stat().st_size}
    (out / "_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return stats
