"""Cleaned lines to scored customers in one call, shared by the dashboard and the CLI."""

from __future__ import annotations

from datetime import date

import polars as pl

from rfm_explorer.rfm import Monetary, build_rfm
from rfm_explorer.scoring import Method, score_customers


def segment_customers(
    lines: pl.DataFrame,
    snapshot: date,
    lookback_days: int | None = 365,
    monetary: Monetary = "net",
    method: Method = "quintile",
    k: int = 5,
    edges: dict[str, list[float]] | None = None,
    countries: list[str] | None = None,
) -> pl.DataFrame:
    """Score every customer who bought in the window. ``countries`` empty or None means all."""
    if countries:
        lines = lines.filter(pl.col("country").is_in(countries))
    return score_customers(
        build_rfm(lines, snapshot, lookback_days, monetary), method, k=k, edges=edges
    )
