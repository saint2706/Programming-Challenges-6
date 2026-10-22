"""The three bundled tables, and loading any CSV/TSV/Parquet file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from report_generator.paths import project_root


@dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    file: str
    row_label: str
    time: str | None = None
    ignore: tuple[str, ...] = ()
    sum_columns: tuple[str, ...] = ()
    mean_columns: tuple[str, ...] = ()
    note: str = ""
    derived: dict[str, str] = field(default_factory=dict)  # column -> what it is


DATASETS: dict[str, Dataset] = {
    "bike": Dataset(
        name="bike",
        title="Daily bike rentals, Washington DC, 2011-2012",
        file="bike.parquet",
        row_label="days",
        time="date",
        note="one row per day: weather, calendar flags and rentals by casual and registered riders",
    ),
    "retail": Dataset(
        name="retail",
        title="Online gift retailer invoice lines, 2009-2011",
        file="retail.parquet",
        row_label="invoice lines",
        time="invoice_date",
        ignore=("invoice", "description"),
        note="every invoice line of 600 customers; returns appear as negative quantities",
        derived={"revenue": "quantity x price"},
    ),
    "ames": Dataset(
        name="ames",
        title="House sales in Ames, Iowa, 2006-2010",
        file="ames.parquet",
        row_label="sales",
        time="sale_month",
        note="one row per sale: price, size, neighbourhood and condition",
    ),
}


def data_dir() -> Path:
    return project_root() / "sample_data"


def read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pl.read_csv(
            path, separator=sep, infer_schema_length=10_000, null_values=["NA", ""]
        )
    raise ValueError(f"cannot read {path.suffix} files; use CSV, TSV or Parquet")


def load(name_or_path: str) -> tuple[Dataset | None, pl.DataFrame]:
    if name_or_path in DATASETS:
        spec = DATASETS[name_or_path]
        path = data_dir() / spec.file
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run `uv run autoreport fetch`")
        frame = read_table(path)
        if spec.name == "retail":
            frame = frame.with_columns(
                (pl.col("quantity") * pl.col("price")).alias("revenue")
            )
        return spec, frame
    path = Path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{name_or_path!r} is neither a bundled dataset ({', '.join(DATASETS)}) nor a file"
        )
    return None, read_table(path)
