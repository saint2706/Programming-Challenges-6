"""Everything the dashboard reads, in one call, as polars frames and plain numbers.

Reading is separate from drawing so it can be tested without a browser, and so the CLI ``status``
command can print the same numbers the charts show.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import polars as pl
import redis

from stream_dashboard.broker import Broker
from stream_dashboard.engine import EVENT_TYPES
from stream_dashboard.windows import (
    DEFAULT_LATENESS_MS,
    DEFAULT_SPECS,
    Layout,
    WindowSpec,
    watermark,
)

WINDOW_SCHEMA = {
    "start_ms": pl.Int64,
    "start": pl.Datetime("ms", "UTC"),
    "n": pl.Int64,
    **{t: pl.Int64 for t in EVENT_TYPES},
    "revenue": pl.Float64,
    "users": pl.Int64,
    "closed": pl.Boolean,
}
TELEMETRY_SCHEMA = {
    "t_ms": pl.Int64,
    "worker": pl.String,
    "read": pl.Int64,
    "applied": pl.Int64,
    "dup": pl.Int64,
    "late": pl.Int64,
    "reclaimed": pl.Int64,
    "busy_ms": pl.Int64,
    "lat_p50": pl.Float64,
    "lat_p95": pl.Float64,
    "lat_max": pl.Float64,
}


@dataclass(frozen=True, slots=True)
class Health:
    stream_len: int
    undelivered: int
    pending: int
    sent: int
    total: int
    blocked_ms: int
    producer_done: bool
    max_event_ms: int | None
    watermark_ms: int | None
    late: dict[str, int]
    speed: float
    backpressure: str
    reclaimed: int
    duplicates: int

    @property
    def backlog(self) -> int:
        return self.undelivered + self.pending


@dataclass(frozen=True, slots=True)
class Snapshot:
    windows: dict[str, pl.DataFrame]  # spec name -> one row per window, oldest first
    health: Health
    telemetry: pl.DataFrame  # one row per worker batch, oldest first
    throughput: pl.DataFrame  # t_s, per_s: events applied per wall second
    consumers: pl.DataFrame  # name, pending, idle_ms
    taken_ms: int


def window_frame(
    r: redis.Redis, layout: Layout, spec: WindowSpec, wm_ms: int | None, last_n: int
) -> pl.DataFrame:
    starts = [int(s) for s in r.zrange(layout.index(spec), -last_n, -1)]
    if not starts:
        return pl.DataFrame(schema=WINDOW_SCHEMA)
    pipe = r.pipeline(transaction=False)
    for s in starts:
        pipe.hgetall(layout.agg(spec, s))
        pipe.scard(layout.users(spec, s))
    out = pipe.execute()
    rows = []
    for i, s in enumerate(starts):
        agg, users = out[2 * i], out[2 * i + 1]
        rows.append(
            {
                "start_ms": s,
                "start": None,
                "n": int(agg.get("n", 0)),
                **{t: int(agg.get(t, 0)) for t in EVENT_TYPES},
                "revenue": int(agg.get("revenue_cents", 0)) / 100,
                "users": int(users),
                "closed": wm_ms is not None and spec.is_closed(s, wm_ms),
            }
        )
    return pl.DataFrame(rows, schema=WINDOW_SCHEMA).with_columns(
        pl.from_epoch("start_ms", time_unit="ms")
        .dt.replace_time_zone("UTC")
        .alias("start")
    )


def top_table(
    r: redis.Redis,
    layout: Layout,
    spec: WindowSpec,
    start_ms: int,
    kind: str,
    k: int = 10,
) -> pl.DataFrame:
    """Top ``k`` products (by events) or brands (by revenue) of one window."""
    if kind == "products":
        raw = r.zrevrange(layout.products(spec, start_ms), 0, k - 1, withscores=True)
        return pl.DataFrame(
            {"product_id": [m for m, _ in raw], "events": [int(s) for _, s in raw]},
            schema={"product_id": pl.String, "events": pl.Int64},
        )
    raw = r.zrevrange(layout.brands(spec, start_ms), 0, k - 1, withscores=True)
    return pl.DataFrame(
        {"brand": [m for m, _ in raw], "revenue": [s / 100 for _, s in raw]},
        schema={"brand": pl.String, "revenue": pl.Float64},
    )


def telemetry_frame(r: redis.Redis, layout: Layout, last_n: int = 2000) -> pl.DataFrame:
    raw = r.xrevrange(layout.telemetry, count=last_n)
    rows = []
    for _, f in reversed(raw):
        rows.append(
            {
                "t_ms": int(f["t"]),
                "worker": f["worker"],
                "read": int(f["read"]),
                "applied": int(f["applied"]),
                "dup": int(f["dup"]),
                "late": int(f["late"]),
                "reclaimed": int(f["reclaimed"]),
                "busy_ms": int(f["busy_ms"]),
                "lat_p50": float(f["lat_p50"]),
                "lat_p95": float(f["lat_p95"]),
                "lat_max": float(f["lat_max"]),
            }
        )
    return pl.DataFrame(rows, schema=TELEMETRY_SCHEMA)


def throughput(tel: pl.DataFrame, horizon_s: int = 120) -> pl.DataFrame:
    """Events applied per wall-clock second, with empty seconds shown as zero."""
    if tel.is_empty():
        return pl.DataFrame(schema={"t_s": pl.Int64, "per_s": pl.Int64})
    per = tel.group_by((pl.col("t_ms") // 1000).alias("t_s")).agg(
        pl.col("applied").sum().alias("per_s")
    )
    hi = int(per["t_s"].max())
    lo = max(int(per["t_s"].min()), hi - horizon_s)
    grid = pl.DataFrame({"t_s": pl.int_range(lo, hi + 1, eager=True)})
    return (
        grid.join(per, on="t_s", how="left")
        .with_columns(pl.col("per_s").fill_null(0))
        .sort("t_s")
    )


def health(r: redis.Redis, layout: Layout, lateness_ms: int) -> Health:
    broker = Broker(r, layout)
    lag = broker.lag()
    meta = r.hgetall(layout.meta)
    stats = r.hgetall(layout.stats)
    max_ms = broker.max_event_ms()
    return Health(
        stream_len=lag.stream_len,
        undelivered=lag.undelivered,
        pending=lag.pending,
        sent=int(meta.get("sent", 0)),
        total=int(meta.get("total", 0)),
        blocked_ms=int(meta.get("blocked_ms", 0)),
        producer_done=meta.get("done") == "1",
        max_event_ms=max_ms,
        watermark_ms=None if max_ms is None else watermark(max_ms, lateness_ms),
        late={k: int(v) for k, v in r.hgetall(layout.late).items()},
        speed=float(meta.get("speed", 0)),
        backpressure=meta.get("backpressure", ""),
        reclaimed=int(stats.get("reclaimed", 0)),
        duplicates=int(stats.get("duplicates", 0)),
    )


def snapshot(
    r: redis.Redis,
    layout: Layout,
    specs: tuple[WindowSpec, ...] = DEFAULT_SPECS,
    lateness_ms: int = DEFAULT_LATENESS_MS,
    last_n: int = 60,
) -> Snapshot:
    h = health(r, layout, lateness_ms)
    tel = telemetry_frame(r, layout)
    cons = Broker(r, layout).consumers()
    return Snapshot(
        windows={
            s.name: window_frame(r, layout, s, h.watermark_ms, last_n) for s in specs
        },
        health=h,
        telemetry=tel,
        throughput=throughput(tel),
        consumers=pl.DataFrame(
            [
                {
                    "name": c["name"],
                    "pending": int(c["pending"]),
                    "idle_ms": int(c["idle"]),
                }
                for c in cons
            ],
            schema={"name": pl.String, "pending": pl.Int64, "idle_ms": pl.Int64},
        ),
        taken_ms=int(time.time() * 1000),
    )
