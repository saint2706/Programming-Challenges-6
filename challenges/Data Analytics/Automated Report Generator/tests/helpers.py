"""Synthetic tables with patterns planted at known strength, and loaders for the bundled data."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def planted(seed: int = 0) -> pl.DataFrame:
    """Three years of daily rows with one pattern per column and nothing else going on."""
    rng = np.random.default_rng(seed)
    n = 3 * 365
    t = np.arange(n)
    dates = [date(2019, 1, 1) + timedelta(days=int(i)) for i in t]
    year = np.sin(2 * np.pi * (t - 80) / 365.25)  # peaks in early July
    visits = 1000 * (1 + 0.6 * t / n) * (1 + 0.3 * year) * (1 + rng.normal(0, 0.03, n))
    visits[800:807] *= 0.2  # one collapsed week
    temp = 15 + 10 * year + rng.normal(0, 1.5, n)
    score = 100 + rng.normal(0, 3, n) + np.where(t >= 600, 40, 0)
    plan = rng.choice(["basic", "plus", "pro"], n)
    spend = rng.lognormal(3, 0.2, n) * np.select(
        [plan == "basic", plan == "plus"], [1.0, 1.6], 2.6
    )
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(0, 0.5, n)
    noise = rng.normal(size=n)
    return pl.DataFrame(
        {"date": dates, "visits": visits.round(), "temp": temp, "score": score, "plan": plan, "spend": spend,
         "x": x, "y": y, "noise": noise}
    )  # fmt: skip


def noise_table(seed: int, n: int = 730) -> pl.DataFrame:
    """Daily rows of pure noise: independent measures, random categories, no structure to find."""
    rng = np.random.default_rng(seed)
    dates = [date(2020, 1, 1) + timedelta(days=int(i)) for i in range(n)]
    return pl.DataFrame(
        {
            "date": dates,
            "a": rng.normal(100, 10, n),
            "b": rng.normal(50, 5, n),
            "c": rng.normal(10, 2, n),
            "group": rng.choice(["p", "q", "r", "s"], n),
        }
    )


def events(seed: int = 0, n: int = 6000) -> pl.DataFrame:
    """An event table: many rows per day, one customer and one country dominating."""
    rng = np.random.default_rng(seed)
    days = rng.integers(0, 540, n)
    dates = [date(2020, 1, 1) + timedelta(days=int(d)) for d in days]
    weights = 1 / np.arange(1, 301) ** 1.2
    customer = rng.choice(np.arange(1000, 1300), n, p=weights / weights.sum())
    country = rng.choice(["UK", "FR", "DE", "ES"], n, p=[0.7, 0.15, 0.1, 0.05])
    qty = rng.integers(1, 20, n)
    price = rng.uniform(1, 5, n).round(2)
    return pl.DataFrame(
        {
            "order_date": dates,
            "customer_id": customer,
            "country": country,
            "quantity": qty,
            "price": price,
        }
    )


def bundled(name: str) -> pl.DataFrame:
    frame = pl.read_parquet(ROOT / "sample_data" / f"{name}.parquet")
    if name == "retail":
        frame = frame.with_columns(
            (pl.col("quantity") * pl.col("price")).alias("revenue")
        )
    return frame
