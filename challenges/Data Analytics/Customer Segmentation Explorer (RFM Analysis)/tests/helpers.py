"""Builders for small, hand-checkable transaction tables."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

RAW_SCHEMA = {
    "invoice": pl.Utf8,
    "stock_code": pl.Utf8,
    "quantity": pl.Int64,
    "invoice_date": pl.Datetime("ms"),
    "price": pl.Float64,
    "customer_id": pl.Int64,
    "country": pl.Utf8,
}


def raw(*rows: tuple) -> pl.DataFrame:
    """Rows of ``(invoice, stock_code, quantity, 'YYYY-MM-DD', price, customer_id, country)``."""
    return pl.DataFrame(
        [(i, s, q, datetime.fromisoformat(d), p, c, n) for i, s, q, d, p, c, n in rows],
        schema=RAW_SCHEMA,
        orient="row",
    )


def lines(*rows: tuple) -> pl.DataFrame:
    """Cleaned lines from ``(customer_id, 'YYYY-MM-DD', invoice, revenue)``; negative revenue is a return."""
    return pl.DataFrame(
        [
            (inv, datetime.fromisoformat(d), c, "UK", 1, abs(rev), rev, rev < 0)
            for c, d, inv, rev in rows
        ],
        schema={
            "invoice": pl.Utf8,
            "invoice_date": pl.Datetime("ms"),
            "customer_id": pl.Int64,
            "country": pl.Utf8,
            "quantity": pl.Int64,
            "price": pl.Float64,
            "revenue": pl.Float64,
            "is_return": pl.Boolean,
        },
        orient="row",
    )


def population(customers: int = 400, days: int = 540, seed: int = 7) -> pl.DataFrame:
    """Customers with a hidden purchase rate and basket size, so past behaviour predicts future behaviour."""
    rng = np.random.default_rng(seed)
    rows = []
    start = np.datetime64("2020-01-01")
    for cid in range(1, customers + 1):
        rate = rng.gamma(1.0, 0.02)  # purchases per day
        basket = rng.lognormal(4.0, 0.8)
        n = rng.poisson(rate * days)
        for k, day in enumerate(sorted(rng.integers(0, days, n))):
            when = str(start + np.timedelta64(int(day), "D"))
            rows.append(
                (cid, when, f"{cid}-{k}", float(basket * rng.lognormal(0, 0.3)))
            )
    return lines(*rows)
