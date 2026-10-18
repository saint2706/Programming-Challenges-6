"""Shared test helpers."""

from __future__ import annotations

import polars as pl
from stream_dashboard.engine import Event
from stream_dashboard.events import Disorder, arrival_order, to_event

HUGE_LATENESS_MS = 24 * 3_600_000


def ev(
    ts_ms, type="view", product=1, brand="acme", cents=1000, user=1, produced=0
) -> Event:
    return Event(ts_ms, type, product, brand, cents, user, produced)


def events_in_order(frame: pl.DataFrame) -> list[Event]:
    return [to_event(r) for r in frame.iter_rows(named=True)]


def messy(events: pl.DataFrame, fraction=0.05, stragglers=0.0, seed=7) -> pl.DataFrame:
    return arrival_order(events, Disorder(fraction, 20_000, stragglers, 300_000, seed))
