"""Download the three bundled datasets and write them, cleaned, to ``sample_data/``."""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import polars as pl

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)
RDATASETS = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/"

SOURCES = {
    "telco.parquet": {
        "source": "IBM Telco customer churn sample dataset (Cognos Analytics sample data)",
        "url": TELCO_URL,
        "rows": 7043,
        "cleaning": "TotalCharges blanks -> null; churned = Churn == 'Yes'; columns otherwise as-is",
    },
    "lung.csv": {
        "source": "NCCTG lung cancer data (Loprinzi et al., 1994), R survival::lung",
        "url": RDATASETS + "cancer.csv",
        "rows": 228,
        "cleaning": "dots in column names -> underscores; event = (status == 2)",
    },
    "gbsg.csv": {
        "source": "German Breast Cancer Study Group (Schumacher et al., 1994), R survival::gbsg",
        "url": RDATASETS + "gbsg.csv",
        "rows": 686,
        "cleaning": "rownames dropped; event = status (1 = recurrence or death)",
    },
}


def _get(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "survival-toolkit/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _csv(raw: bytes) -> pl.DataFrame:
    return pl.read_csv(
        io.BytesIO(raw), null_values=["NA", " ", ""], infer_schema_length=10_000
    )


def fetch_all(dest: Path) -> dict[str, int]:
    dest.mkdir(parents=True, exist_ok=True)
    rows: dict[str, int] = {}

    telco = _csv(_get(TELCO_URL)).with_columns(
        (pl.col("Churn") == "Yes").alias("churned"),
        pl.col("TotalCharges").cast(pl.Float64, strict=False),
    )
    telco.write_parquet(dest / "telco.parquet", compression="zstd")
    rows["telco.parquet"] = telco.height

    lung = _csv(_get(RDATASETS + "cancer.csv")).drop("rownames")
    lung = lung.rename({c: c.replace(".", "_") for c in lung.columns})
    lung = lung.with_columns((pl.col("status") == 2).cast(pl.Int8).alias("event"))
    lung.write_csv(dest / "lung.csv", line_terminator="\n")
    rows["lung.csv"] = lung.height

    gbsg = _csv(_get(RDATASETS + "gbsg.csv")).drop("rownames")
    gbsg = gbsg.with_columns(pl.col("status").cast(pl.Int8).alias("event"))
    gbsg.write_csv(dest / "gbsg.csv", line_terminator="\n")
    rows["gbsg.csv"] = gbsg.height

    notes = {name: {**meta, "rows": rows[name]} for name, meta in SOURCES.items()}
    (dest / "SOURCES.json").write_text(
        json.dumps(notes, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return rows
