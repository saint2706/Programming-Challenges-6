"""Replay events into the stream at ``speed`` event-seconds per wall-second, with optional backpressure.

With ``backpressure="block"`` the producer stops sending while the consumers' backlog (undelivered
plus read-but-unacknowledged) is above ``high_water`` and resumes below ``low_water``. The replay
clock is paused while it waits, so it does not burst to catch up afterwards. With ``"none"`` it keeps
its schedule and the backlog is allowed to grow, which is what the lag panel is there to show.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from stream_dashboard.broker import Broker
from stream_dashboard.events import to_event


@dataclass(frozen=True, slots=True)
class ProducerConfig:
    speed: float = 60.0  # event-seconds per wall-second; 0 means as fast as possible
    batch: int = 500
    backpressure: str = "block"  # "block" or "none"
    high_water: int = 5_000
    low_water: int = 1_000

    def __post_init__(self) -> None:
        if self.backpressure not in ("block", "none"):
            raise ValueError("backpressure must be 'block' or 'none'")
        if self.speed < 0 or self.batch < 1:
            raise ValueError("speed must be >= 0 and batch >= 1")
        if self.low_water > self.high_water:
            raise ValueError("low_water must not exceed high_water")


@dataclass(frozen=True, slots=True)
class ProducerResult:
    sent: int
    blocked_ms: int
    wall_s: float


def produce(
    broker: Broker,
    frame: pl.DataFrame,
    cfg: ProducerConfig | None = None,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    stop: Callable[[], bool] = lambda: False,
) -> ProducerResult:
    """``frame`` must have ``arrival_ms`` (see ``events.arrival_order``) and be sorted by it."""
    cfg = cfg or ProducerConfig()
    r, L = broker.r, broker.layout
    broker.ensure_group()
    rows = frame.to_dicts()
    arrivals = frame["arrival_ms"].to_numpy()
    n = len(rows)
    r.hset(
        L.meta,
        mapping={
            "speed": cfg.speed,
            "total": n,
            "sent": 0,
            "blocked_ms": 0,
            "done": 0,
            "first_ms": int(frame["ts_ms"].min()) if n else 0,
            "last_ms": int(frame["ts_ms"].max()) if n else 0,
            "backpressure": cfg.backpressure,
        },
    )
    start_wall = clock()
    wall0 = start_wall
    first_arrival = int(arrivals[0]) if n else 0
    blocked_total = 0.0
    pos = 0
    while pos < n and not stop():
        if cfg.speed > 0:
            now_event = first_arrival + (clock() - wall0) * 1000.0 * cfg.speed
            end = min(
                int(np.searchsorted(arrivals, now_event, side="right")), pos + cfg.batch
            )
            if end <= pos:
                wait = (arrivals[pos] - now_event) / 1000.0 / cfg.speed
                sleep(min(max(wait, 0.001), 0.05))
                continue
        else:
            end = min(n, pos + cfg.batch)
        if cfg.backpressure == "block" and broker.lag().backlog >= cfg.high_water:
            t_block = clock()
            while broker.lag().backlog > cfg.low_water and not stop():
                sleep(0.01)
            waited = clock() - t_block
            blocked_total += waited
            wall0 += waited  # pause the replay clock
            continue
        stamp = now_ms()
        broker.add_many([(to_event(row), stamp) for row in rows[pos:end]])
        pos = end
        r.hset(L.meta, mapping={"sent": pos, "blocked_ms": int(blocked_total * 1000)})
    r.hset(
        L.meta,
        mapping={"sent": pos, "blocked_ms": int(blocked_total * 1000), "done": 1},
    )
    return ProducerResult(pos, int(blocked_total * 1000), clock() - start_wall)
