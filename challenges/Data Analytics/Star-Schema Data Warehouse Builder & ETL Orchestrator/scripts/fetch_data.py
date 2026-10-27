"""Rebuild the feed from the original public source.

Run with:  uv run --group fetch python scripts/fetch_data.py

Source: UCI Machine Learning Repository, "Online Retail II" (Chen, 2019, CC BY 4.0). The zip is
cached in ``.cache/`` (gitignored) and its SHA-256 is checked before it is read.

Writes:

* ``data/feed.parquet``: the full feed (not committed, about 1M rows).
* ``sample_data/feed_sample.parquet``: every line of 600 random customers (committed; tests fall back
  to it when the full feed is absent). Same customer sample as the RFM challenge (seed 20091201).
* ``sample_data/SOURCES.json``: download hash, row counts and the late-arrival share.

The workbook has one sheet per year and both sheets contain 2010-12-01 to 2010-12-09. Those invoices
are identical in both, so they are dropped from the second sheet by invoice number, never by row.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile

import polars as pl
from star_warehouse.feed import LATE_PERCENT, build_feed
from star_warehouse.paths import project_root

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ZIP_SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
SHEETS = ("Year 2009-2010", "Year 2010-2011")
SAMPLE_CUSTOMERS = 600
SAMPLE_SEED = 20091201
RENAME = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def download(cache) -> object:
    cache.mkdir(exist_ok=True)
    target = cache / "online_retail_ii.zip"
    if not target.exists():
        print(f"downloading {URL}")
        request = urllib.request.Request(
            URL, headers={"User-Agent": "star-warehouse-fetch/1.0"}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            target.write_bytes(response.read())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != ZIP_SHA256:
        raise SystemExit(f"checksum mismatch for {target}: {digest}")
    return target


def read_sheets(xlsx) -> tuple[pl.DataFrame, int]:
    """Both sheets stacked, second-sheet invoices already present in the first removed."""
    frames = [
        pl.read_excel(
            xlsx,
            sheet_name=sheet,
            engine="calamine",
            schema_overrides={
                "Invoice": pl.Utf8,
                "StockCode": pl.Utf8,
                "Customer ID": pl.Int64,
            },
        ).rename(RENAME)
        for sheet in SHEETS
    ]
    first, second = frames
    overlap = second.filter(
        pl.col("invoice").is_in(first["invoice"].unique().implode())
    )
    stacked = pl.concat(
        [
            first,
            second.join(overlap.select("invoice").unique(), on="invoice", how="anti"),
        ]
    )
    return stacked, overlap.height


def main() -> None:
    root = project_root()
    cache, data, sample_dir = root / ".cache", root / "data", root / "sample_data"
    data.mkdir(exist_ok=True)
    sample_dir.mkdir(exist_ok=True)

    archive = download(cache)
    with zipfile.ZipFile(archive) as zf:
        zf.extract("online_retail_II.xlsx", cache)
    lines, overlap_rows = read_sheets(cache / "online_retail_II.xlsx")
    feed = build_feed(lines)
    feed.write_parquet(data / "feed.parquet", compression="zstd")

    customers = lines["customer_id"].drop_nulls().unique().sort()
    chosen = customers.sample(SAMPLE_CUSTOMERS, seed=SAMPLE_SEED)
    sample = feed.filter(pl.col("customer_id").is_in(chosen.implode()))
    sample.write_parquet(sample_dir / "feed_sample.parquet", compression="zstd")

    late_lines = feed.filter(pl.col("arrival_date") > pl.col("invoice_date")).height
    (sample_dir / "SOURCES.json").write_text(
        json.dumps(
            {
                "url": URL,
                "sha256": ZIP_SHA256,
                "rows": feed.height,
                "overlap_rows_removed": overlap_rows,
                "late_arrival_percent_of_invoices": LATE_PERCENT,
                "late_lines": late_lines,
                "sample_customers": SAMPLE_CUSTOMERS,
                "sample_rows": sample.height,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"{feed.height:,} lines, {overlap_rows:,} sheet-overlap rows removed, "
        f"{late_lines:,} late lines, sample {sample.height:,} lines"
    )


if __name__ == "__main__":
    main()
