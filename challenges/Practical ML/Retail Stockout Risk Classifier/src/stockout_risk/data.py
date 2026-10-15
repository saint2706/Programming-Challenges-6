"""FreshRetailNet-50K: download, load and reduce the hourly lists to one row per store-product-day.

The data (CC BY 4.0, arXiv 2505.16319, ``Dingdong-Inc/FreshRetailNet-50K`` on Hugging Face) are
50,000 store-product series of a Chinese fresh-food retailer: 898 stores, 865 products, 90 days in
``train`` (2024-03-28 to 2024-06-25) and the following 7 days in ``eval``. Every row is one
store-product-day with 24 hourly sales and 24 hourly stock flags.

What the columns mean was settled by looking at the rows, not the dataset card:

* ``hours_stock_status`` is 1 when the product is *out of stock* in that hour.
* ``stock_hour6_22_cnt`` is the count of those hours among hours 6..21, a 16-hour selling window.
  It matches the flags on every row, so the flags are used and the column is only cross-checked.
* ``hours_sale`` sums to ``sale_amount`` on every row. There is no on-hand quantity anywhere:
  inventory enters only as the hourly in-stock flag.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from stockout_risk.paths import data_dir

URL = "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/main/data/{}.parquet"
# sha256 of the LFS objects, from the Hugging Face tree API.
SHA256 = {
    "train": "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d",
    "eval": "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e",
}
SERIES = ["store_id", "product_id"]
WINDOW = (6, 22)  # the selling hours: 6..21
WINDOW_HOURS = WINDOW[1] - WINDOW[0]
EARLY = (6, 12)  # the morning part of the window, for the sales-profile feature

RAW_COLUMNS = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
    "dt",
    "sale_amount",
    "hours_sale",
    "stock_hour6_22_cnt",
    "hours_stock_status",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]


class DownloadError(RuntimeError):
    pass


def fetch(
    url: str,
    dest: Path,
    sha256: str,
    client: httpx.Client | None = None,
    chunk: int = 1 << 20,
) -> Path:
    """Stream ``url`` to ``dest``, checking the sha256 before the file takes its final name."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=60)
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with part.open("wb") as out:
                for block in response.iter_bytes(chunk):
                    digest.update(block)
                    out.write(block)
    except httpx.HTTPError as exc:
        part.unlink(missing_ok=True)
        raise DownloadError(f"download of {url} failed: {exc}") from exc
    finally:
        if own:
            client.close()
    if digest.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"{dest.name}: sha256 {digest.hexdigest()} is not the expected {sha256}"
        )
    part.replace(dest)
    return dest


def download(
    *, force: bool = False, client: httpx.Client | None = None
) -> dict[str, Path]:
    """Both parquet files in ``data/``; files already there are kept unless ``force``."""
    paths = {}
    for name, sha in SHA256.items():
        dest = data_dir() / f"{name}.parquet"
        paths[name] = (
            dest
            if dest.exists() and not force
            else fetch(URL.format(name), dest, sha, client)
        )
    return paths


def load_raw() -> pl.DataFrame:
    """``train`` and ``eval`` stacked, with a ``part`` column saying which file a row came from."""
    paths = download()
    return pl.concat(
        [
            pl.read_parquet(p).with_columns(pl.lit(name).alias("part"))
            for name, p in paths.items()
        ]
    )


def daily(raw: pl.DataFrame) -> pl.DataFrame:
    """One typed row per store-product-day, sorted, with the hourly lists reduced to features.

    Reductions of the two 24-hour lists (all about the selling window, hours 6..21):

    * ``oos_window``: hours out of stock, recomputed from the flags.
    * ``oos_open`` / ``oos_close``: out of stock in the first / last selling hour. The close state
      is what tomorrow starts from.
    * ``oos_last4``: out-of-stock hours among the last four.
    * ``first_oos``: first out-of-stock hour as an offset into the window (16 when there is none).
    * ``sales_in_stock``: units sold in hours that were in stock; ``early_share``: the part of the
      window's sales that fell in the morning hours.
    """
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for col in ("hours_sale", "hours_stock_status"):
        if raw[col].list.len().n_unique() != 1 or raw[col].list.len()[0] != 24:
            raise ValueError(f"{col} must hold 24 hourly values on every row")
    status = raw["hours_stock_status"].list.to_array(24).to_numpy().astype(np.int8)
    sales = raw["hours_sale"].list.to_array(24).to_numpy().astype(np.float32)
    lo, hi = WINDOW
    flags = status[:, lo:hi]
    window_sales = sales[:, lo:hi]
    sold = (window_sales * (flags == 0)).sum(axis=1)
    total = window_sales.sum(axis=1)
    early = window_sales[:, EARLY[0] - lo : EARLY[1] - lo].sum(axis=1)
    out = raw.select(
        pl.col(c)
        for c in RAW_COLUMNS
        if c not in ("hours_sale", "hours_stock_status", "dt")
    ).with_columns(
        raw["dt"].str.to_date().alias("dt"),
        pl.Series("oos_window", flags.sum(axis=1).astype(np.int8)),
        pl.Series("oos_open", flags[:, 0]),
        pl.Series("oos_close", flags[:, -1]),
        pl.Series("oos_last4", flags[:, -4:].sum(axis=1).astype(np.int8)),
        pl.Series(
            "first_oos",
            np.where(flags.any(axis=1), flags.argmax(axis=1), WINDOW_HOURS).astype(
                np.int8
            ),
        ),
        pl.Series("sales_in_stock", sold),
        pl.Series(
            "early_share",
            np.divide(early, total, out=np.full_like(total, np.nan), where=total > 0),
        ),
    )
    return out.rename(
        {
            "sale_amount": "sale",
            "management_group_id": "mgmt_id",
            "first_category_id": "cat1",
            "second_category_id": "cat2",
            "third_category_id": "cat3",
            "precpt": "precip",
            "avg_temperature": "temp",
            "avg_humidity": "humidity",
            "avg_wind_level": "wind",
            "holiday_flag": "holiday",
            "activity_flag": "promo",
        }
    ).sort([*SERIES, "dt"])


def load() -> pl.DataFrame:
    return daily(load_raw())


def summary(raw: pl.DataFrame) -> dict:
    """Facts about the raw table that the README quotes, including the column cross-check."""
    flags = raw["hours_stock_status"].list.slice(WINDOW[0], WINDOW_HOURS).list.sum()
    return {
        "rows": raw.height,
        "series": raw.select(SERIES).unique().height,
        "stores": raw["store_id"].n_unique(),
        "products": raw["product_id"].n_unique(),
        "first_day": raw["dt"].min(),
        "last_day": raw["dt"].max(),
        "days": raw["dt"].n_unique(),
        "window_count_matches_flags": float(
            (flags == raw["stock_hour6_22_cnt"]).mean()
        ),
        "sales_sum_matches": float(
            ((raw["hours_sale"].list.sum() - raw["sale_amount"]).abs() < 1e-3).mean()
        ),
        "any_oos_day_share": float((flags > 0).mean()),
    }
