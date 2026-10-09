"""Loading the invoice lines and cleaning them into something RFM can be computed from.

Every row that is dropped is counted in a ledger (rows and pounds), so the dashboard can show what
the numbers exclude instead of hiding it. Rules run in this order, and a row is counted under the
first rule that removes it:

1. ``non_product``: stock codes that are not a product (postage ``POST``/``DOT``, manual lines
   ``M``, discounts ``D``, ``BANK CHARGES``, ``AMAZONFEE``, ``ADJUST``, samples, test rows...).
   A product code is five digits plus up to two letters (``85123A``).
2. ``bad_price``: a price of zero or less (stock write-offs and adjustments).
3. ``bad_quantity``: a quantity of zero or less on a normal invoice, or a positive quantity on a
   cancellation. Real returns are negative quantities on ``C`` invoices and are kept.
4. ``no_customer``: no customer id, mostly guest checkout. They cannot be segmented, but their
   share of revenue is reported so nobody mistakes the segmented total for the whole business.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from rfm_explorer.paths import project_root

PRODUCT_CODE = r"^\d{5}[A-Za-z]{0,2}$"
RAW_COLUMNS = [
    "invoice",
    "stock_code",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]
FULL_FILE = "data/online_retail_ii.parquet"
SAMPLE_FILE = "sample_data/transactions_sample.parquet"


def find_data_file(root: Path | None = None) -> Path:
    """The full dataset when it has been fetched, otherwise the committed sample."""
    root = root or project_root()
    for name in (FULL_FILE, SAMPLE_FILE):
        if (root / name).exists():
            return root / name
    raise FileNotFoundError(
        f"neither {FULL_FILE} nor {SAMPLE_FILE} found under {root}; "
        "run `uv run --group fetch python -m rfm_explorer.fetch_data`"
    )


def load_raw(path: Path | None = None) -> pl.DataFrame:
    return pl.read_parquet(path or find_data_file(), columns=RAW_COLUMNS)


def clean(raw: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(lines, ledger)``: lines with ``revenue`` and ``is_return``; ledger of what was dropped."""
    lines = raw.with_columns(
        revenue=pl.col("quantity") * pl.col("price"),
        is_return=pl.col("invoice").str.starts_with("C"),
    )
    rules = [
        (
            "non_product",
            ~pl.col("stock_code").str.contains(PRODUCT_CODE).fill_null(False),
        ),
        ("bad_price", pl.col("price") <= 0),
        (
            "bad_quantity",
            (pl.col("is_return") & (pl.col("quantity") >= 0))
            | (~pl.col("is_return") & (pl.col("quantity") <= 0)),
        ),
        ("no_customer", pl.col("customer_id").is_null()),
    ]
    ledger = []
    for name, drop in rules:
        drop = drop.fill_null(False)  # an unknown flag must not silently remove the row
        dropped = lines.filter(drop)
        ledger.append(
            {
                "rule": name,
                "rows": dropped.height,
                "revenue": float(dropped["revenue"].sum() or 0.0),
            }
        )
        lines = lines.filter(~drop)
    kept = lines.select(
        "invoice",
        "invoice_date",
        "customer_id",
        "country",
        "quantity",
        "price",
        "revenue",
        "is_return",
    )
    return kept, pl.DataFrame(
        ledger, schema={"rule": pl.Utf8, "rows": pl.Int64, "revenue": pl.Float64}
    )
