"""Event-time windows, the watermark rule, and the Redis keys a window's results live under.

Everything here is pure (no Redis, no clock) so the rules can be tested exactly and reused by the
offline oracle. Time is integer epoch milliseconds throughout.

Rules, stated once:

* A window ``[start, start + size)`` has ``start`` a multiple of ``hop`` (epoch-aligned). Tumbling
  windows have ``hop == size``; sliding windows have ``hop < size`` so one event lands in
  ``size / hop`` windows.
* The watermark is ``max_event_time_seen - allowed_lateness``. A window is *closed* once
  ``start + size <= watermark``. The watermark never moves backwards.
* An event is added to every window of a spec that is still open. If none is open the event is
  *late* for that spec: it is counted in ``late`` and never added to a window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_UNITS = {"ms": 1, "s": 1_000, "m": 60_000, "h": 3_600_000}
_DURATION = re.compile(r"^(\d+)(ms|s|m|h)$")


def parse_duration(text: str) -> int:
    """``"30s"`` -> 30000. Only whole numbers with one of ms, s, m, h."""
    m = _DURATION.match(text.strip())
    if not m or int(m.group(1)) == 0:
        raise ValueError(
            f"bad duration {text!r}: use a positive whole number with ms, s, m or h"
        )
    return int(m.group(1)) * _UNITS[m.group(2)]


@dataclass(frozen=True, slots=True)
class WindowSpec:
    name: str
    size_ms: int
    hop_ms: int

    def __post_init__(self) -> None:
        if self.size_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("window size and hop must be positive")
        if self.hop_ms > self.size_ms:
            raise ValueError("hop larger than size would leave gaps between windows")
        if self.size_ms % self.hop_ms:
            raise ValueError("size must be a multiple of hop")

    @property
    def kind(self) -> str:
        return "tumbling" if self.hop_ms == self.size_ms else "sliding"

    def starts_containing(self, ts_ms: int) -> list[int]:
        """Start times of every window that contains ``ts_ms``, oldest first."""
        last = ts_ms - ts_ms % self.hop_ms
        return list(range(last - self.size_ms + self.hop_ms, last + 1, self.hop_ms))

    def open_starts(self, ts_ms: int, watermark_ms: int) -> list[int]:
        return [
            s for s in self.starts_containing(ts_ms) if s + self.size_ms > watermark_ms
        ]

    def is_closed(self, start_ms: int, watermark_ms: int) -> bool:
        return start_ms + self.size_ms <= watermark_ms

    @classmethod
    def tumbling(cls, size: str) -> WindowSpec:
        ms = parse_duration(size)
        return cls(f"t{size}", ms, ms)

    @classmethod
    def sliding(cls, size: str, hop: str) -> WindowSpec:
        return cls(f"s{size}-{hop}", parse_duration(size), parse_duration(hop))

    @classmethod
    def parse(cls, text: str) -> WindowSpec:
        """``"1m"`` is tumbling; ``"5m/1m"`` is a 5-minute window hopping every minute."""
        if "/" in text:
            size, hop = text.split("/", 1)
            return cls.sliding(size.strip(), hop.strip())
        return cls.tumbling(text.strip())


DEFAULT_SPECS = (WindowSpec.tumbling("1m"), WindowSpec.sliding("5m", "1m"))
DEFAULT_LATENESS_MS = 30_000


def watermark(max_event_ms: int | None, lateness_ms: int) -> int:
    """The event time before which nothing more is expected; very small when nothing was seen."""
    return -(2**62) if max_event_ms is None else max_event_ms - lateness_ms


@dataclass(frozen=True, slots=True)
class Layout:
    """Every Redis key the pipeline uses, under one prefix so runs and tests never collide."""

    prefix: str = "sd:"
    group: str = "workers"

    @property
    def stream(self) -> str:
        return f"{self.prefix}events"

    @property
    def done(self) -> str:
        """Set of stream entry ids already applied (makes redelivery harmless)."""
        return f"{self.prefix}done"

    @property
    def max_ts(self) -> str:
        return f"{self.prefix}max_event_ms"

    @property
    def telemetry(self) -> str:
        return f"{self.prefix}telemetry"

    @property
    def meta(self) -> str:
        """Producer facts: sent, blocked_ms, speed, first_ms, last_ms, done."""
        return f"{self.prefix}meta"

    @property
    def stats(self) -> str:
        """Running totals across all workers: reclaimed, duplicates (telemetry, not part of the sink)."""
        return f"{self.prefix}stats"

    @property
    def late(self) -> str:
        return f"{self.prefix}late"

    def index(self, spec: WindowSpec) -> str:
        return f"{self.prefix}wins:{spec.name}"

    def agg(self, spec: WindowSpec, start: int) -> str:
        return f"{self.prefix}w:{spec.name}:{start}:agg"

    def users(self, spec: WindowSpec, start: int) -> str:
        return f"{self.prefix}w:{spec.name}:{start}:users"

    def products(self, spec: WindowSpec, start: int) -> str:
        return f"{self.prefix}w:{spec.name}:{start}:products"

    def brands(self, spec: WindowSpec, start: int) -> str:
        return f"{self.prefix}w:{spec.name}:{start}:brands"
