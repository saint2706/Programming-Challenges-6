"""A consumer-group worker: read, plan, apply atomically, report telemetry.

Any number of workers share the group. The only state they keep is the batch they are on: window
aggregates and the watermark live in Redis, so killing a worker loses nothing. Entries it had read
but not acknowledged sit in the group's pending list until another worker reclaims them after
``claim_idle_ms`` of silence.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from stream_dashboard.broker import Broker
from stream_dashboard.engine import plan_event
from stream_dashboard.windows import (
    DEFAULT_LATENESS_MS,
    DEFAULT_SPECS,
    WindowSpec,
    watermark,
)

TELEMETRY_MAXLEN = 5_000


class SimulatedCrash(RuntimeError):
    """Raised instead of ``os._exit`` when a test wants a crash it can catch."""


@dataclass(slots=True)
class WorkerConfig:
    name: str = "w1"
    specs: tuple[WindowSpec, ...] = DEFAULT_SPECS
    lateness_ms: int = DEFAULT_LATENESS_MS
    batch: int = 200
    block_ms: int = 200
    claim_idle_ms: int = 2_000
    claim_every_s: float = 0.5
    process_delay_ms: float = 0.0  # artificial per-entry cost to make a slow consumer
    crash_after: int | None = (
        None  # die after reading this many entries, before applying them
    )
    hard_crash: bool = True  # os._exit(137) when True, SimulatedCrash when False
    drain: bool = True  # exit once the producer is done and the group has nothing left


@dataclass(slots=True)
class WorkerStats:
    read: int = 0
    applied: int = 0
    duplicates: int = 0
    late: int = 0
    reclaimed: int = 0
    latencies_ms: list[float] = field(default_factory=list)


class Worker:
    def __init__(
        self,
        broker: Broker,
        cfg: WorkerConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.b, self.cfg = broker, cfg or WorkerConfig()
        self.clock, self.now_ms, self.sleep = clock, now_ms, sleep
        self.stats = WorkerStats()
        self._last_claim = -1e9

    # -- one batch -------------------------------------------------------------------------------
    def process(
        self, entries: list[tuple[str, dict]], *, reclaimed: bool = False
    ) -> None:
        cfg, L = self.cfg, self.b.layout
        self.stats.read += len(entries)
        if reclaimed:
            self.stats.reclaimed += len(entries)
        if cfg.crash_after is not None and self.stats.read >= cfg.crash_after:
            if cfg.hard_crash:
                os._exit(137)
            raise SimulatedCrash(cfg.name)
        if cfg.process_delay_ms:
            self.sleep(cfg.process_delay_ms * len(entries) / 1000.0)
        t0 = self.clock()
        max_ts = self.b.max_event_ms()
        items = []
        late_events = 0
        for entry_id, fields in entries:
            ev = self.b.decode(fields)
            max_ts = ev.ts_ms if max_ts is None else max(max_ts, ev.ts_ms)
            plan = plan_event(
                ev, cfg.specs, watermark(max_ts, cfg.lateness_ms), self.b.layout
            )
            late_events += bool(plan.late_specs)
            items.append((entry_id, ev, plan.ops))
        applied = self.b.apply_entries([(i, ev.ts_ms, ops) for i, ev, ops in items])
        done_ms = self.now_ms()
        lat = [
            done_ms - ev.produced_ms
            for (_, ev, _), ok in zip(items, applied, strict=True)
            if ok
        ]
        n_applied = sum(applied)
        self.stats.applied += n_applied
        self.stats.duplicates += len(applied) - n_applied
        self.stats.late += late_events
        self.stats.latencies_ms.extend(lat)
        p50 = float(np.percentile(lat, 50)) if lat else 0.0
        p95 = float(np.percentile(lat, 95)) if lat else 0.0
        pipe = self.b.r.pipeline(transaction=False)
        pipe.xadd(
            L.telemetry,
            {
                "worker": cfg.name,
                "t": done_ms,
                "read": len(entries),
                "applied": n_applied,
                "dup": len(applied) - n_applied,
                "late": late_events,
                "reclaimed": int(reclaimed) * len(entries),
                "busy_ms": int((self.clock() - t0) * 1000),
                "lat_p50": round(p50, 1),
                "lat_p95": round(p95, 1),
                "lat_max": round(max(lat), 1) if lat else 0,
            },
            maxlen=TELEMETRY_MAXLEN,
            approximate=True,
        )
        if reclaimed:
            pipe.hincrby(L.stats, "reclaimed", len(entries))
        if len(applied) - n_applied:
            pipe.hincrby(L.stats, "duplicates", len(applied) - n_applied)
        pipe.execute()

    # -- the loop --------------------------------------------------------------------------------
    def step(self) -> int:
        """Handle one batch (stale work first, then new). Returns the number of entries handled."""
        cfg = self.cfg
        if self.clock() - self._last_claim >= cfg.claim_every_s:
            self._last_claim = self.clock()
            stale = self.b.claim_stale(cfg.name, cfg.claim_idle_ms, cfg.batch)
            if stale:
                self.process(stale, reclaimed=True)
                return len(stale)
        entries = self.b.read_new(cfg.name, cfg.batch, cfg.block_ms)
        if entries:
            self.process(entries)
        return len(entries)

    def finished(self) -> bool:
        meta = self.b.r.hgetall(self.b.layout.meta)
        if meta.get("done") != "1":
            return False
        lag = self.b.lag()
        return lag.undelivered == 0 and lag.pending == 0

    def run(self, stop: Callable[[], bool] = lambda: False) -> WorkerStats:
        self.b.ensure_group()
        while not stop():
            if self.step() == 0 and self.cfg.drain and self.finished():
                break
        return self.stats
