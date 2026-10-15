"""The ULB credit-card fraud data: download, check, clean.

284,807 card transactions by European cardholders over two days in September 2013 (OpenML dataset
1597; Dal Pozzolo et al., the ULB Machine Learning Group; OpenML lists the licence as "Public"). 492 are fraud
(0.173%). ``V1`` to ``V28`` are principal components of the original (undisclosed) features, so they
carry no meaning a person can read; ``Time`` is the seconds since the first transaction and
``Amount`` the transaction amount (the currency is not stated, so amounts are in "currency units").
There is no card, merchant or customer id, so nothing here can use a card's own history.

OpenML publishes no checksum for the file. The server's ETag is its md5, which is checked; the
sha256 below was computed at first download and is pinned from then on (trust on first use).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import polars as pl

from fraud_flagging.paths import data_dir

URL = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
MD5 = "1593844f40edbdabaf5bddec4649e1c4"
SHA256 = "b7efcb35a428bbe22347a05d2437d9177bab07ce61e51214a17bec584ad9496d"
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_COLUMNS = ["Time", *V_COLUMNS, "Amount"]


class DownloadError(RuntimeError):
    pass


def fetch(
    url: str,
    dest: Path,
    *,
    md5: str,
    sha256: str,
    client: httpx.Client | None = None,
    chunk: int = 1 << 20,
) -> Path:
    """Stream ``url`` to ``dest``; both hashes must match before the file gets its final name."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    sha, md = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=60)
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with part.open("wb") as out:
                for block in response.iter_bytes(chunk):
                    sha.update(block)
                    md.update(block)
                    out.write(block)
    except httpx.HTTPError as exc:
        part.unlink(missing_ok=True)
        raise DownloadError(f"download of {url} failed: {exc}") from exc
    finally:
        if own:
            client.close()
    if md.hexdigest() != md5 or sha.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(f"{dest.name}: checksum mismatch (md5 {md.hexdigest()})")
    part.replace(dest)
    return dest


def download(*, force: bool = False, client: httpx.Client | None = None) -> Path:
    """``data/creditcard.parquet``; kept when it exists unless ``force``."""
    dest = data_dir() / "creditcard.parquet"
    if dest.exists() and not force:
        return dest
    return fetch(URL, dest, md5=MD5, sha256=SHA256, client=client)


def load_raw() -> pl.DataFrame:
    """The table as published, with ``Class`` as an ``Int8`` column ``y`` and the file's row index."""
    df = pl.read_parquet(download())
    missing = [c for c in [*RAW_COLUMNS, "Class"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return (
        df.with_row_index("tx_id")
        .with_columns(pl.col("Class").cast(pl.String).cast(pl.Int8).alias("y"))
        .drop("Class")
    )


def clean(raw: pl.DataFrame) -> pl.DataFrame:
    """Time-ordered, with exact duplicate transactions dropped (the first one is kept).

    1,081 rows repeat another row in every column including the time (32 of them frauds). A copy
    is not a second observation, and in a random split it would sit on both sides of the split, so
    it goes. ``tx_id`` keeps pointing at the row in the published file.
    """
    return raw.unique(subset=RAW_COLUMNS, keep="first", maintain_order=True).sort(
        "Time", "tx_id"
    )


def load() -> pl.DataFrame:
    return clean(load_raw())


def summary(raw: pl.DataFrame) -> dict:
    """Facts about the published table that the README quotes."""
    dedup = clean(raw)
    return {
        "rows": raw.height,
        "frauds": int(raw["y"].sum()),
        "fraud_rate": float(raw["y"].mean()),
        "hours": float(raw["Time"].max() / 3600),
        "time_sorted": bool(raw["Time"].is_sorted()),
        "duplicate_rows": raw.height - dedup.height,
        "duplicate_frauds": int(raw["y"].sum() - dedup["y"].sum()),
        "rows_after_dedup": dedup.height,
        "frauds_after_dedup": int(dedup["y"].sum()),
        "fraud_amount_median": float(raw.filter(pl.col("y") == 1)["Amount"].median()),
        "legit_amount_median": float(raw.filter(pl.col("y") == 0)["Amount"].median()),
    }
