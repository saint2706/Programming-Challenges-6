"""What every engine adapter is given: where the normalized data and the zone lookup live."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from dist_pipeline.paths import data_dir


@dataclass(frozen=True)
class Dataset:
    scale: str
    fmt: str  # "parquet" (part files) or "csv" (one file)
    root: Path

    @property
    def location(self) -> Path:
        if self.fmt == "parquet":
            return self.root / "normalized" / self.scale
        return self.root / "csv" / self.scale

    @property
    def source(self) -> str:
        """A folder for Parquet (every part file in it), one file for CSV, as a forward-slash path."""
        if self.fmt == "parquet":
            return self.location.as_posix()
        return (self.location / "trips.csv").as_posix()

    @property
    def parquet_glob(self) -> str:
        return (self.location / "part-*.parquet").as_posix()

    @property
    def parts(self) -> list[Path]:
        return sorted(self.location.glob("part-*.parquet"))

    @property
    def zones_csv(self) -> Path:
        return self.root / "raw" / "taxi_zone_lookup.csv"

    def zones_lazy(self) -> pl.LazyFrame:
        return pl.scan_csv(self.zones_csv).select(
            location_id=pl.col("LocationID").cast(pl.Int32),
            borough=pl.col("Borough").cast(pl.String),
        )

    def zones_frame(self) -> pl.DataFrame:
        return self.zones_lazy().collect()


def dataset(scale: str, fmt: str = "parquet", root: Path | None = None) -> Dataset:
    if fmt not in ("parquet", "csv"):
        raise ValueError(f"unknown format {fmt!r}")
    return Dataset(scale=scale, fmt=fmt, root=root or data_dir())


def has_data(ds: Dataset) -> bool:
    if ds.fmt == "parquet":
        return bool(ds.parts) and ds.zones_csv.exists()
    return (ds.location / "trips.csv").exists() and ds.zones_csv.exists()
