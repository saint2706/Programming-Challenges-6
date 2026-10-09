"""Rebuild the transaction files from the original public source.

Run with:  uv run --group fetch python -m rfm_explorer.fetch_data

Source: UCI Machine Learning Repository, "Online Retail II" (Chen, 2019, CC BY 4.0): every
invoice line of a UK gift-ware wholesaler between 2009-12-01 and 2011-12-09.

Writes:

* ``data/online_retail_ii.parquet``: the full file (not committed, about 1M rows).
* ``sample_data/transactions_sample.parquet``: every row of 600 random customers (committed;
  tests and the dashboard fall back to it when the full file is absent).
* ``sample_data/SOURCES.json``: the download's SHA-256 and what the overlap fix removed.

The workbook has one sheet per year and the two sheets both contain 2010-12-01 to 2010-12-09.
Those 1,088 invoices are identical in both, so they are dropped from the second sheet by invoice
number, never by exact row: a customer really can buy the same product twice on one invoice.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile

import polars as pl

from rfm_explorer.paths import project_root

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
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
        req = urllib.request.Request(
            URL, headers={"User-Agent": "rfm-explorer-fetch/1.0"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            target.write_bytes(resp.read())
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
    return pl.concat(
        [
            first,
            second.join(overlap.select("invoice").unique(), on="invoice", how="anti"),
        ]
    ), overlap.height


def main() -> None:
    root = project_root()
    cache, data, sample_dir = root / ".cache", root / "data", root / "sample_data"
    data.mkdir(exist_ok=True)
    sample_dir.mkdir(exist_ok=True)

    archive = download(cache)
    with zipfile.ZipFile(archive) as zf:
        zf.extract("online_retail_II.xlsx", cache)
    frame, overlap_rows = read_sheets(cache / "online_retail_II.xlsx")
    frame.write_parquet(data / "online_retail_ii.parquet", compression="zstd")

    customers = frame["customer_id"].drop_nulls().unique().sort()
    chosen = customers.sample(SAMPLE_CUSTOMERS, seed=SAMPLE_SEED)
    sample = frame.filter(pl.col("customer_id").is_in(chosen.implode())).sort(
        "invoice_date", "invoice"
    )
    sample.write_parquet(sample_dir / "transactions_sample.parquet", compression="zstd")

    (sample_dir / "SOURCES.json").write_text(
        json.dumps(
            {
                "url": URL,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "rows": frame.height,
                "overlap_rows_removed": overlap_rows,
                "sample_customers": SAMPLE_CUSTOMERS,
                "sample_rows": sample.height,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"{frame.height:,} rows, {overlap_rows:,} sheet-overlap rows removed, sample {sample.height:,} rows"
    )


if __name__ == "__main__":
    main()
