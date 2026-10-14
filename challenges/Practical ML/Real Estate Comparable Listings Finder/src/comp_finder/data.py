"""King County house sales: fetch, cache, clean, and the three time windows.

21,613 sales from 2014-05-02 to 2015-05-27 (OpenML dataset 42092, the Kaggle ``harlfoxem`` copy of
King County's public sales records; licence CC0). It is read as parquet with polars, so pandas is
never imported.

Things that matter for comparable-sales work and are handled here or downstream:

* The table is sorted by sale date. Comparable selection is "every earlier sale", which is then a
  prefix of the table, and the split by window is a prefix too.
* 177 property ids sold twice inside the year. ``n_prior`` counts earlier sales of the same id; the
  headline evaluation keeps only first sales, because a house's own earlier sale is a near-perfect
  comp that an appraiser rarely has.
* One row says 33 bedrooms on 1,620 sq ft (a typo). Rows are never edited; ``features`` winsorises.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from comp_finder.paths import project_root

PARQUET_URL = "https://data.openml.org/datasets/0004/42092/dataset_42092.pq"
CACHE_PATH = project_root() / "data" / "king_county_sales.parquet"

RAW = [
    "id",
    "date",
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]


@dataclass(frozen=True)
class Windows:
    """Fit on sales before ``val_start``, tune on the next window, report on ``test_start`` onward."""

    val_start: date = date(2015, 1, 1)
    test_start: date = date(2015, 3, 1)


def load_raw(*, force_refresh: bool = False) -> pl.DataFrame:
    """The raw dataset, fetched from OpenML on first use and read from the parquet cache after."""
    if not force_refresh and CACHE_PATH.exists():
        return pl.read_parquet(CACHE_PATH)
    df = pl.read_parquet(PARQUET_URL)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(CACHE_PATH)
    return df


def clean(df: pl.DataFrame) -> pl.DataFrame:
    """Typed, date-sorted sales with ``sold``, ``month`` (year*12 + month) and ``n_prior``."""
    missing = [c for c in RAW if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = (
        df.select(RAW)
        .with_columns(
            pl.col("date").str.slice(0, 8).str.to_date("%Y%m%d").alias("sold"),
            pl.col("zipcode").cast(pl.Utf8),
            pl.col("price").cast(pl.Float64),
            *[
                pl.col(c).cast(pl.Float64)
                for c in ("bedrooms", "bathrooms", "floors", "lat", "long")
            ],
        )
        .drop("date")
        .filter(pl.col("price") > 0)
        .sort("sold", "id", maintain_order=True)
    )
    return out.with_columns(
        (pl.col("sold").dt.year() * 12 + pl.col("sold").dt.month()).alias("month"),
        pl.col("id").cum_count().over("id").alias("n_prior") - 1,
    )


def load() -> pl.DataFrame:
    return clean(load_raw())


def window_counts(df: pl.DataFrame, w: Windows) -> tuple[int, int, int]:
    """Rows in (fit, validation, test); the table is date-sorted so these are prefixes."""
    sold = df["sold"].to_numpy()
    n_fit = int(np.searchsorted(sold, np.datetime64(w.val_start), side="left"))
    n_val_end = int(np.searchsorted(sold, np.datetime64(w.test_start), side="left"))
    return n_fit, n_val_end - n_fit, df.height - n_val_end


# What a custom subject must say, and what it may leave out (with the default used).
REQUIRED_SUBJECT = (
    "lat",
    "long",
    "sqft_living",
    "sqft_lot",
    "bedrooms",
    "bathrooms",
    "floors",
    "grade",
    "condition",
)
OPTIONAL_SUBJECT = {
    "waterfront": 0,
    "view": 0,
    "sqft_basement": 0,
    "yr_renovated": 0,
    "yr_built": None,
    "sqft_living15": None,  # defaults to sqft_living
    "sqft_lot15": None,  # defaults to sqft_lot
    "zipcode": None,  # defaults to the zip of the nearest sale
}
BOUNDS = {"lat": (47.1, 47.8), "long": (-122.6, -121.2)}


def subject_frame(record: dict, sold: date, sales: pl.DataFrame) -> pl.DataFrame:
    """One unsold property as a table row (price NaN, id -1) shaped like ``sales``."""
    unknown = set(record) - set(REQUIRED_SUBJECT) - set(OPTIONAL_SUBJECT)
    missing = [f for f in REQUIRED_SUBJECT if record.get(f) is None]
    if unknown or missing:
        raise ValueError(f"unknown fields {sorted(unknown)}, missing fields {missing}")
    rec = {**{k: v for k, v in OPTIONAL_SUBJECT.items()}, **record}
    for f in (
        "sqft_living",
        "sqft_lot",
        "bedrooms",
        "bathrooms",
        "floors",
        "grade",
        "condition",
    ):
        if not float(rec[f]) > 0:
            raise ValueError(f"{f} must be positive")
    for f, (lo, hi) in BOUNDS.items():
        if not lo <= float(rec[f]) <= hi:
            raise ValueError(f"{f} {rec[f]} is outside King County ({lo} to {hi})")
    rec["sqft_living15"] = rec["sqft_living15"] or rec["sqft_living"]
    rec["sqft_lot15"] = rec["sqft_lot15"] or rec["sqft_lot"]
    rec["yr_built"] = rec["yr_built"] or int(sales["yr_built"].median())
    if rec["zipcode"] is None:
        near = (
            (sales["lat"] - float(rec["lat"])) ** 2
            + (sales["long"] - float(rec["long"])) ** 2
        ).arg_min()
        rec["zipcode"] = sales["zipcode"][near]
    rec["sqft_above"] = rec["sqft_living"] - rec["sqft_basement"]
    rec |= {
        "id": -1,
        "price": float("nan"),
        "sold": sold,
        "month": sold.year * 12 + sold.month,
        "n_prior": 0,
    }
    return pl.DataFrame({c: [rec[c]] for c in sales.columns}, schema=sales.schema)
