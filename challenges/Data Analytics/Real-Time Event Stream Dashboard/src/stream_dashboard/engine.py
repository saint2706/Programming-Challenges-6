"""What one event does to the window aggregates, as a list of idempotent-or-deduplicated operations.

``plan_event`` is pure: given an event, the window specs and the current watermark it returns the
operations to apply. The Redis sink runs them inside one Lua script together with the dedupe check
and the ``XACK``; ``MemorySink`` runs the same operations on plain dicts and is the in-process
reference the Redis results are compared with.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from stream_dashboard.windows import Layout, WindowSpec, watermark

# Operations are [command, key, *args], every argument a string (JSON-safe, no float formatting).
Op = list[str]

EVENT_TYPES = ("view", "cart", "remove_from_cart", "purchase")


@dataclass(frozen=True, slots=True)
class Event:
    ts_ms: int
    type: str
    product_id: int
    brand: str
    price_cents: int
    user_id: int
    produced_ms: int = 0


@dataclass(slots=True)
class Plan:
    ops: list[Op] = field(default_factory=list)
    late_specs: list[str] = field(default_factory=list)


def plan_event(
    ev: Event, specs: Iterable[WindowSpec], wm_ms: int, layout: Layout
) -> Plan:
    """Operations for ``ev`` given watermark ``wm_ms`` (which already includes ``ev`` itself)."""
    plan = Plan()
    for spec in specs:
        starts = spec.open_starts(ev.ts_ms, wm_ms)
        if not starts:
            plan.late_specs.append(spec.name)
            plan.ops.append(["HINCRBY", layout.late, spec.name, "1"])
            continue
        for start in starts:
            plan.ops.append(["ZADD", layout.index(spec), str(start), str(start)])
            agg = layout.agg(spec, start)
            plan.ops.append(["HINCRBY", agg, "n", "1"])
            plan.ops.append(["HINCRBY", agg, ev.type, "1"])
            if ev.type == "purchase":
                plan.ops.append(["HINCRBY", agg, "revenue_cents", str(ev.price_cents)])
                if ev.brand:
                    plan.ops.append(
                        [
                            "ZINCRBY",
                            layout.brands(spec, start),
                            str(ev.price_cents),
                            ev.brand,
                        ]
                    )
            plan.ops.append(["SADD", layout.users(spec, start), str(ev.user_id)])
            plan.ops.append(
                ["ZINCRBY", layout.products(spec, start), "1", str(ev.product_id)]
            )
    return plan


class MemorySink:
    """The same operations on plain Python containers. Used as the reference implementation."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.sets: dict[str, set[str]] = defaultdict(set)
        self.zsets: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def apply(self, ops: Iterable[Op]) -> None:
        for cmd, key, *args in ops:
            if cmd == "HINCRBY":
                self.hashes[key][args[0]] += int(args[1])
            elif cmd == "SADD":
                self.sets[key].add(args[0])
            elif cmd == "ZADD":
                self.zsets[key][args[1]] = float(args[0])
            elif cmd == "ZINCRBY":
                self.zsets[key][args[1]] += float(args[0])
            else:  # pragma: no cover - plan_event never emits anything else
                raise ValueError(f"unsupported op {cmd}")


def simulate(
    events: Iterable[Event],
    specs: Iterable[WindowSpec],
    lateness_ms: int,
    layout: Layout | None = None,
) -> MemorySink:
    """Process ``events`` in the order given, one at a time, with a watermark that only moves up.

    This is what a single worker does; with more workers the interleaving, and so which stragglers
    count as late, can differ, but every event still ends in exactly one place.
    """
    layout = layout or Layout()
    specs = tuple(specs)
    sink = MemorySink()
    max_ts: int | None = None
    for ev in events:
        max_ts = ev.ts_ms if max_ts is None else max(max_ts, ev.ts_ms)
        sink.apply(plan_event(ev, specs, watermark(max_ts, lateness_ms), layout).ops)
    return sink
