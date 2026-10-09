"""Build the four retail tables from the original public source.

Run with:  uv run --group fetch python -m sql_profiler.fetch_data

Source: UCI Machine Learning Repository, "Online Retail II" (Chen, 2019, CC BY 4.0): every invoice
line of a UK gift-ware wholesaler between 2009-12-01 and 2011-12-09. The workbook is one flat table;
this splits it into the normalized schema a real shop database would have:

* ``invoice_lines(invoice, stock_code, quantity, price)``     about 1.04M rows
* ``invoices(invoice, invoice_date, customer_id, country)``   about 53k rows
* ``products(stock_code, description)``                       about 5k rows
* ``customers(customer_id, first_invoice_date)``              about 5.9k rows

Writes ``data/*.parquet`` (not committed) and ``sample_data/*.parquet``: every row belonging to 600
random customers' invoices, 300 random guest invoices (no customer id) and their products (committed, so tests and demos work
offline). Rows are written in invoice order, which is the order real data arrives in and is what
makes DuckDB's min/max "zonemap" pruning work on ``invoice``.

The two workbook sheets overlap on 2010-12-01 to 2010-12-09; those invoices are identical in both,
so the second sheet's copies are dropped by invoice number.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

from sql_profiler.paths import project_root

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
SHEETS = ("Year 2009-2010", "Year 2010-2011")
SAMPLE_CUSTOMERS = 600
SAMPLE_GUEST_INVOICES = (
    300  # invoices with no customer id, so NULL-key behaviour shows up
)
SAMPLE_SEED = 20091201
TABLES = ("invoice_lines", "invoices", "products", "customers")


def download(cache: Path) -> Path:
    cache.mkdir(exist_ok=True)
    target = cache / "online_retail_ii.zip"
    if not target.exists():
        print(f"downloading {URL}")
        req = urllib.request.Request(
            URL, headers={"User-Agent": "sql-profiler-fetch/1.0"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            target.write_bytes(resp.read())
    return target


def read_flat(xlsx: Path) -> tuple[pl.DataFrame, int]:
    """Both sheets stacked, with the second sheet's copies of already-seen invoices removed."""
    first, second = (
        pl.read_excel(
            xlsx,
            sheet_name=sheet,
            engine="calamine",
            schema_overrides={
                "Invoice": pl.Utf8,
                "StockCode": pl.Utf8,
                "Customer ID": pl.Int64,
            },
        )
        for sheet in SHEETS
    )
    seen = first["Invoice"].unique().implode()
    overlap = second.filter(pl.col("Invoice").is_in(seen)).height
    return pl.concat([first, second.filter(~pl.col("Invoice").is_in(seen))]), overlap


def normalize(flat: pl.DataFrame) -> dict[str, pl.DataFrame]:
    flat = flat.rename(
        {
            "Invoice": "invoice",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "price",
            "Customer ID": "customer_id",
            "Country": "country",
        }
    )
    lines = flat.select("invoice", "stock_code", "quantity", "price")
    invoices = (
        flat.group_by("invoice", maintain_order=True)
        .agg(
            pl.col("invoice_date").min(),
            pl.col("customer_id").first(),
            pl.col("country").first(),
        )
        .sort("invoice_date", "invoice")
    )
    # one description per code: the most common non-null one (descriptions drift over time)
    products = (
        flat.filter(pl.col("description").is_not_null())
        .group_by("stock_code", "description")
        .len()
        .sort("len", "description", descending=[True, False])
        .group_by("stock_code", maintain_order=True)
        .first()
        .select("stock_code", "description")
        .sort("stock_code")
    )
    customers = (
        invoices.filter(pl.col("customer_id").is_not_null())
        .group_by("customer_id")
        .agg(first_invoice_date=pl.col("invoice_date").min())
        .sort("customer_id")
    )
    order = invoices.select("invoice", rank=pl.int_range(pl.len()))
    lines = (
        lines.join(order, on="invoice").sort("rank", maintain_order=True).drop("rank")
    )
    return {
        "invoice_lines": lines,
        "invoices": invoices,
        "products": products,
        "customers": customers,
    }


def sample_of(tables: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    chosen = tables["customers"]["customer_id"].sample(
        SAMPLE_CUSTOMERS, seed=SAMPLE_SEED
    )
    invoices = tables["invoices"].filter(pl.col("customer_id").is_in(chosen.implode()))
    guests = (
        tables["invoices"]
        .filter(pl.col("customer_id").is_null())
        .sample(SAMPLE_GUEST_INVOICES, seed=SAMPLE_SEED)
    )
    invoices = pl.concat([invoices, guests]).sort("invoice_date", "invoice")
    lines = tables["invoice_lines"].join(
        invoices.select("invoice"), on="invoice", how="semi"
    )
    return {
        "invoice_lines": lines,
        "invoices": invoices,
        "products": tables["products"].join(
            lines.select("stock_code").unique(), on="stock_code", how="semi"
        ),
        "customers": tables["customers"].filter(
            pl.col("customer_id").is_in(chosen.implode())
        ),
    }


def main() -> None:
    root = project_root()
    archive = download(root / ".cache")
    with zipfile.ZipFile(archive) as zf:
        zf.extract("online_retail_II.xlsx", root / ".cache")
    flat, overlap = read_flat(root / ".cache" / "online_retail_II.xlsx")
    tables = normalize(flat)
    sample = sample_of(tables)
    for folder, subset in (("data", tables), ("sample_data", sample)):
        (root / folder).mkdir(exist_ok=True)
        for name, frame in subset.items():
            frame.write_parquet(root / folder / f"{name}.parquet", compression="zstd")
    (root / "sample_data" / "SOURCES.json").write_text(
        json.dumps(
            {
                "url": URL,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "sheet_overlap_rows_removed": overlap,
                "rows": {n: f.height for n, f in tables.items()},
                "sample_rows": {n: f.height for n, f in sample.items()},
            },
            indent=2,
        )
        + "\n"
    )
    print(
        {n: f.height for n, f in tables.items()},
        "sample:",
        {n: f.height for n, f in sample.items()},
    )


if __name__ == "__main__":
    main()
