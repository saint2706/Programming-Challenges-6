"""Rebuild the bundled tables in ``sample_data/`` from their public sources."""

from __future__ import annotations

import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

BIKE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
AMES_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/ames.csv"
RETAIL_SIBLING = "Customer Segmentation Explorer (RFM Analysis)/sample_data/transactions_sample.parquet"

SEASONS = {1: "winter", 2: "spring", 3: "summer", 4: "fall"}
WEATHER = {
    1: "clear",
    2: "mist or cloud",
    3: "light rain or snow",
    4: "heavy rain or snow",
}
DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

AMES_COLUMNS = [
    "Neighborhood", "Bldg_Type", "House_Style", "Central_Air", "Sale_Condition", "Sale_Type",
    "Gr_Liv_Area", "Lot_Area", "Year_Built", "Overall_Cond", "Full_Bath", "Bedroom_AbvGr",
    "Garage_Cars", "Total_Bsmt_SF", "Fireplaces", "Sale_Price",
]  # fmt: skip


def _get(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "report-generator/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def build_bike(raw_zip: bytes) -> pl.DataFrame:
    """UCI Bike Sharing, daily file, with the normalised weather columns turned back into units.

    The data's own readme gives the scales: temperature / 41 C, feeling temperature / 50 C,
    humidity / 100, wind speed / 67 km/h. ``cnt`` is ``casual + registered`` in every row.
    """
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        day = pl.read_csv(zf.read("day.csv"))
    return day.select(
        pl.col("dteday").str.to_date().alias("date"),
        pl.col("season").replace_strict(SEASONS).alias("season"),
        pl.col("weekday").replace_strict(dict(enumerate(DAYS))).alias("weekday"),
        pl.col("holiday").cast(pl.Boolean),
        pl.col("workingday").cast(pl.Boolean),
        pl.col("weathersit").replace_strict(WEATHER).alias("weather"),
        (pl.col("temp") * 41).round(2).alias("temp_c"),
        (pl.col("atemp") * 50).round(2).alias("feels_like_c"),
        (pl.col("hum") * 100).round(1).alias("humidity_pct"),
        (pl.col("windspeed") * 67).round(1).alias("wind_kmh"),
        "casual",
        "registered",
        pl.col("cnt").alias("rentals"),
    )


def build_ames(raw_csv: bytes) -> pl.DataFrame:
    ames = pl.read_csv(io.BytesIO(raw_csv), infer_schema_length=5000)
    return ames.select(
        pl.date(pl.col("Year_Sold"), pl.col("Mo_Sold"), 1).alias("sale_month"),
        *AMES_COLUMNS,
    )


def fetch_all(dest: Path, repo_challenges: Path | None = None) -> dict[str, int]:
    dest.mkdir(parents=True, exist_ok=True)
    rows: dict[str, int] = {}
    bike = build_bike(_get(BIKE_URL))
    bike.write_parquet(dest / "bike.parquet", compression="zstd")
    rows["bike.parquet"] = bike.height
    ames = build_ames(_get(AMES_URL))
    ames.write_parquet(dest / "ames.parquet", compression="zstd")
    rows["ames.parquet"] = ames.height
    if repo_challenges is not None:
        src = repo_challenges / "Data Analytics" / RETAIL_SIBLING
        if src.exists():
            shutil.copyfile(src, dest / "retail.parquet")
            rows["retail.parquet"] = pl.read_parquet(dest / "retail.parquet").height
    return rows


SOURCES = {
    "bike.parquet": {
        "source": "UCI Bike Sharing Dataset, daily file (Fanaee-T and Gama, 2013), CC BY 4.0",
        "url": BIKE_URL,
        "cleaning": "categorical codes decoded; weather columns rescaled to C, % and km/h",
    },
    "retail.parquet": {
        "source": "UCI Online Retail II (Chen, 2019), CC BY 4.0",
        "url": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        "origin": "copied from challenges/Data Analytics/Customer Segmentation Explorer (RFM "
        "Analysis)/sample_data: every line of 600 random customers; revenue is derived at load",
    },
    "ames.parquet": {
        "source": "Ames, Iowa housing (De Cock, 2011) via R modeldata::ames (Rdatasets)",
        "url": AMES_URL,
        "cleaning": "16 of 74 columns kept; sale_month built from Year_Sold and Mo_Sold",
    },
}


def write_sources(dest: Path, rows: dict[str, int]) -> None:
    notes = {name: {**meta, "rows": rows.get(name)} for name, meta in SOURCES.items()}
    (dest / "SOURCES.json").write_text(
        json.dumps(notes, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
