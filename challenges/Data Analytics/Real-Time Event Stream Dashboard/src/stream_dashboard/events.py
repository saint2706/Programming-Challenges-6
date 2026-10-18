"""Load the real clickstream and, optionally, make some of it arrive out of order.

The REES46 file is sorted by event time, so replayed as-is it can never exercise a watermark. The
disorder here is *injected* and seeded: a chosen fraction of events is delayed in transit by a random
amount. Their event times stay the real ones; only the order they reach the broker changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from stream_dashboard.engine import Event
from stream_dashboard.paths import project_root

COLUMNS = (
    "event_time",
    "event_type",
    "product_id",
    "brand",
    "price",
    "user_id",
    "user_session",
)


def default_path() -> Path:
    full = project_root() / "data" / "events_raw.csv"
    return full if full.exists() else project_root() / "sample_data" / "events.csv.gz"


def load_events(path: Path | None = None, limit: int | None = None) -> pl.DataFrame:
    """Columns: ts_ms, type, product_id, brand, price_cents, user_id; sorted by event time."""
    df = pl.read_csv(
        path or default_path(),
        schema_overrides={"user_session": pl.String, "brand": pl.String},
        n_rows=limit,
    )
    return df.select(
        pl.col("event_time")
        .str.to_datetime(time_zone="UTC", time_unit="ms")
        .dt.epoch("ms")
        .alias("ts_ms"),
        pl.col("event_type").alias("type"),
        pl.col("product_id"),
        pl.col("brand").fill_null(""),
        (pl.col("price") * 100).round().cast(pl.Int64).alias("price_cents"),
        pl.col("user_id"),
    ).sort("ts_ms", maintain_order=True)


@dataclass(frozen=True, slots=True)
class Disorder:
    """``fraction`` of events are held back by up to ``max_delay_ms``; ``straggler_fraction`` of
    them by ``max_delay_ms`` to ``straggler_ms`` instead (too late for a normal watermark)."""

    fraction: float = 0.0
    max_delay_ms: int = 20_000
    straggler_fraction: float = 0.0
    straggler_ms: int = 300_000
    seed: int = 7


def arrival_order(df: pl.DataFrame, disorder: Disorder | None = None) -> pl.DataFrame:
    """Add ``arrival_ms`` and return the frame ordered by it (ties keep event order)."""
    d = disorder or Disorder()
    n = df.height
    rng = np.random.default_rng(d.seed)
    delay = np.zeros(n, dtype=np.int64)
    if d.fraction > 0:
        hit = rng.random(n) < d.fraction
        delay[hit] = rng.integers(1, d.max_delay_ms + 1, size=int(hit.sum()))
    if d.straggler_fraction > 0:
        hit = rng.random(n) < d.straggler_fraction
        delay[hit] = rng.integers(
            d.max_delay_ms + 1, d.straggler_ms + 1, size=int(hit.sum())
        )
    out = df.with_columns(pl.Series("arrival_ms", df["ts_ms"].to_numpy() + delay))
    return out.sort("arrival_ms", maintain_order=True)


def to_event(row: dict, produced_ms: int = 0) -> Event:
    return Event(
        ts_ms=int(row["ts_ms"]),
        type=row["type"],
        product_id=int(row["product_id"]),
        brand=row["brand"] or "",
        price_cents=int(row["price_cents"]),
        user_id=int(row["user_id"]),
        produced_ms=produced_ms,
    )
