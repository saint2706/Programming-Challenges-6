"""Recency, frequency and monetary value per customer as of a snapshot date."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Literal

import polars as pl

Monetary = Literal["net", "gross"]


def as_of(snapshot: date) -> datetime:
    """Midnight at the start of ``snapshot``: only activity strictly before it is visible."""
    return datetime.combine(snapshot, time.min)


def last_day(lines: pl.DataFrame) -> date:
    return lines["invoice_date"].max().date()


def window(
    lines: pl.DataFrame, snapshot: date, lookback_days: int | None
) -> pl.DataFrame:
    """Lines in ``[snapshot - lookback_days, snapshot)``; no lookback means all history."""
    end = as_of(snapshot)
    visible = lines.filter(pl.col("invoice_date") < end)
    if lookback_days is None:
        return visible
    return visible.filter(pl.col("invoice_date") >= end - timedelta(days=lookback_days))


def build_rfm(
    lines: pl.DataFrame,
    snapshot: date,
    lookback_days: int | None = 365,
    monetary: Monetary = "net",
) -> pl.DataFrame:
    """One row per customer who bought in the window.

    ``recency_days`` is whole days from the last *purchase* (a refund is not activity) to the
    snapshot, so a purchase the day before the snapshot has recency 1. ``frequency`` counts
    distinct purchase invoices. ``monetary`` is purchases minus refunds (``net``) or purchases
    only (``gross``). A refund cannot be tied to the purchase it reverses, so a customer who
    refunds more than they bought in the window ends up with a negative net value.
    """
    if monetary not in ("net", "gross"):
        raise ValueError(f"monetary must be 'net' or 'gross', not {monetary!r}")
    visible = window(lines, snapshot, lookback_days)
    sales = visible.filter(~pl.col("is_return"))
    spend = visible if monetary == "net" else sales
    per_customer = sales.group_by("customer_id").agg(
        recency_days=(
            pl.lit(as_of(snapshot)).dt.date() - pl.col("invoice_date").max().dt.date()
        ).dt.total_days(),
        frequency=pl.col("invoice").n_unique(),
        country=pl.col("country").sort_by("invoice_date").last(),
    )
    money = spend.group_by("customer_id").agg(monetary=pl.col("revenue").sum())
    return (
        per_customer.join(money, on="customer_id", how="left")
        .sort("customer_id")
        .select("customer_id", "country", "recency_days", "frequency", "monetary")
    )
