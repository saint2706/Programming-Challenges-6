"""An independent offline answer to "what should every window contain?", and the comparison.

The oracle never touches the stream or the engine's ``plan_event``: it expands each event into its
windows with vectorised polars arithmetic and groups. If the live pipeline (Redis, Lua sink, several
workers, a crash) agrees with this, nothing was lost, duplicated or put in the wrong window.

What is compared depends on lateness. With no late drops every window must match exactly. With late
drops a window can legitimately hold fewer events than the oracle says, so the checks become
conservation ones: a tumbling spec must account for every event (in a window or in ``late``), and no
window may ever hold *more* than the oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
import redis

from stream_dashboard.engine import EVENT_TYPES
from stream_dashboard.snapshot import window_frame
from stream_dashboard.windows import Layout, WindowSpec


def expected_windows(events: pl.DataFrame, spec: WindowSpec) -> pl.DataFrame:
    """Per window: n, per-type counts, revenue, unique users (every event, none dropped)."""
    copies = spec.size_ms // spec.hop_ms
    base = events.with_columns(
        (pl.col("ts_ms") - pl.col("ts_ms") % spec.hop_ms).alias("last")
    )
    expanded = pl.concat(
        [
            base.with_columns((pl.col("last") - k * spec.hop_ms).alias("start_ms"))
            for k in range(copies)
        ]
    )
    return (
        expanded.group_by("start_ms")
        .agg(
            pl.len().alias("n"),
            *[(pl.col("type") == t).sum().cast(pl.Int64).alias(t) for t in EVENT_TYPES],
            (
                pl.when(pl.col("type") == "purchase")
                .then(pl.col("price_cents"))
                .otherwise(0)
                .sum()
                / 100
            ).alias("revenue"),
            pl.col("user_id").n_unique().cast(pl.Int64).alias("users"),
        )
        .sort("start_ms")
    )


@dataclass(slots=True)
class SpecReport:
    spec: str
    windows_expected: int
    windows_found: int
    exact: int  # windows identical to the oracle
    short: int  # windows holding fewer events than the oracle (late drops)
    over: int  # windows holding MORE than the oracle: always a bug
    wrong_other: int  # same n but another column differs: always a bug
    events_in_windows: int
    late: int
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(slots=True)
class VerifyReport:
    total_events: int
    applied: int
    specs: list[SpecReport]

    @property
    def ok(self) -> bool:
        return self.applied == self.total_events and all(s.ok for s in self.specs)


def verify(
    r: redis.Redis,
    layout: Layout,
    events: pl.DataFrame,
    specs: tuple[WindowSpec, ...],
    *,
    applied: int | None = None,
) -> VerifyReport:
    """Compare Redis with the oracle. ``applied`` defaults to the size of the ``done`` set."""
    total = events.height
    done = r.scard(layout.done) if applied is None else applied
    late = {k: int(v) for k, v in r.hgetall(layout.late).items()}
    reports = []
    for spec in specs:
        exp = expected_windows(events, spec)
        got = window_frame(r, layout, spec, None, last_n=max(exp.height, 1) + 10)
        merged = exp.join(got, on="start_ms", how="full", suffix="_got", coalesce=True)
        n_late = late.get(spec.name, 0)
        rep = SpecReport(
            spec.name, exp.height, got.height, 0, 0, 0, 0, int(got["n"].sum()), n_late
        )
        for row in merged.iter_rows(named=True):
            e_n, g_n = row["n"] or 0, row["n_got"] or 0
            if g_n > e_n:
                rep.over += 1
                rep.problems.append(
                    f"window {row['start_ms']}: {g_n} events, oracle says {e_n}"
                )
            elif g_n < e_n:
                rep.short += 1
            else:
                same = all(
                    (row[c] or 0) == (row[f"{c}_got"] or 0)
                    or (
                        c == "revenue"
                        and abs((row[c] or 0) - (row[f"{c}_got"] or 0)) < 0.005
                    )
                    for c in (*EVENT_TYPES, "revenue", "users")
                )
                if same:
                    rep.exact += 1
                else:
                    rep.wrong_other += 1
                    rep.problems.append(
                        f"window {row['start_ms']}: same count, different columns"
                    )
        if n_late == 0 and rep.short:
            rep.problems.append(
                f"{rep.short} windows short although nothing was dropped as late"
            )
        if spec.kind == "tumbling" and rep.events_in_windows + n_late != total:
            rep.problems.append(
                f"tumbling conservation: {rep.events_in_windows} in windows + {n_late} late != {total}"
            )
        reports.append(rep)
    return VerifyReport(total, done, reports)
