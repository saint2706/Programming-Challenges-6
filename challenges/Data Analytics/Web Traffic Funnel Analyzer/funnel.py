"""Web Traffic Funnel Analyzer: raw clickstream CSV -> stage-by-stage drop-off + HTML report.

Run with:
    uv run python funnel.py sample_data/rees46_sample.csv.gz --steps view,cart,purchase -o report.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from plotly.offline import get_plotlyjs
from scipy import stats

MODES = ("ordered", "strict", "any-order")
UNITS = ("session", "user")

# Tried in order and coalesced (Polars infers one format from the first value it
# sees, which would silently null every row written in a different format).
_TS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%.f%#z",
    "%Y-%m-%d %H:%M:%S%.f%#z",
    "%Y-%m-%d %H:%M:%S UTC",
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%d",
)
# A bare number is an epoch. Anything >= 1e11 is milliseconds (1e11 ms is 1973;
# 1e11 s is the year 5138), so the two can't be confused for real data.
_EPOCH_MS_THRESHOLD = 1e11

_UNIT_MS = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}


class DataError(ValueError):
    """The input file or the step definition can't be analyzed (CLI exit code 2)."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def parse_duration(text: str) -> int:
    """'90s' / '30m' / '2h' / '7d' -> milliseconds."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", text)
    if not m or float(m.group(1)) <= 0:
        raise DataError(
            f"bad duration {text!r}: use a positive number and s/m/h/d, e.g. 30m"
        )
    return round(float(m.group(1)) * _UNIT_MS[m.group(2)])


def parse_steps(text: str) -> tuple[tuple[frozenset[str], ...], tuple[str, ...]]:
    """'view,cart,purchase' -> event-name sets and labels. 'a|b' means either event counts."""
    labels: list[str] = []
    sets: list[frozenset[str]] = []
    for token in text.split(","):
        names = [n.strip() for n in token.split("|") if n.strip()]
        if not names:
            raise DataError(f"empty step in {text!r}")
        sets.append(frozenset(names))
        labels.append(" or ".join(names))
    if len(sets) < 2:
        raise DataError("a funnel needs at least two steps")
    return tuple(sets), tuple(labels)


@dataclass(frozen=True)
class Columns:
    user: str = "user_id"
    time: str = "event_time"
    event: str = "event_type"
    session: str | None = None  # None => sessionize by inactivity gap
    segment: str | None = None
    segment_level: int | None = None  # keep the first N parts of a dotted value


@dataclass(frozen=True)
class Config:
    steps: tuple[frozenset[str], ...]
    labels: tuple[str, ...]
    mode: str = "ordered"
    unit: str = "session"
    window_ms: int | None = None
    session_gap_ms: int = 30 * 60_000
    outcome_step: int | None = None  # 1-based; default = last step
    drop_censored: bool = False
    top_segments: int = 8
    min_segment_size: int = 30
    alpha: float = 0.05

    def validate(self) -> None:
        if self.mode not in MODES:
            raise DataError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.unit not in UNITS:
            raise DataError(f"unit must be one of {UNITS}, got {self.unit!r}")
        if self.mode == "any-order":
            seen: set[str] = set()
            for s in self.steps:
                if seen & s:
                    raise DataError(
                        "any-order mode needs steps with disjoint event names"
                    )
                seen |= s
        if self.outcome_step is not None and not 2 <= self.outcome_step <= len(
            self.steps
        ):
            raise DataError(f"--outcome-step must be between 2 and {len(self.steps)}")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class DataQuality:
    rows_read: int = 0
    null_user: int = 0
    null_time: int = 0
    unparseable_time: int = 0
    null_event: int = 0
    null_session: int = 0
    duplicate_events: int = 0
    out_of_order_rows: int = 0  # informational: kept, re-sorted by time
    rows_used: int = 0
    entities: int = 0
    step_events: int = 0
    first_event: str | None = None
    last_event: str | None = None
    unmatched_steps: list[str] = field(default_factory=list)


@dataclass
class StepStat:
    label: str
    count: int
    of_entry: float
    of_entry_ci: tuple[float, float]
    of_prev: float | None
    of_prev_ci: tuple[float, float] | None
    dropped: int
    median_from_prev_s: float | None
    q1_from_prev_s: float | None
    q3_from_prev_s: float | None
    median_from_entry_s: float | None


@dataclass
class SegmentStat:
    name: str
    entrants: int
    counts: list[int]
    conversion: float
    ci: tuple[float, float]
    lift_vs_rest: float | None
    p_value: float | None
    q_value: float | None
    significant: bool


@dataclass
class FunnelResult:
    labels: list[str]
    mode: str
    unit: str
    window_ms: int | None
    outcome_label: str
    entrants: int = 0
    censored_dropped: int = 0
    steps: list[StepStat] = field(default_factory=list)
    segment_col: str | None = None
    segments: list[SegmentStat] = field(default_factory=list)
    omnibus_p: float | None = None
    biggest_leak: dict | None = None
    quality: DataQuality = field(default_factory=DataQuality)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def wilson_interval(
    successes: int, trials: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (NaN, NaN if there are no trials)."""
    if trials <= 0:
        return (math.nan, math.nan)
    z = float(stats.norm.isf(alpha / 2))
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    # the interval always contains p mathematically; clamp away float rounding at 0 and 1
    return (max(0.0, min(p, centre - half)), min(1.0, max(p, centre + half)))


def benjamini_hochberg(p_values: list[float | None]) -> list[float | None]:
    """BH-adjusted q-values; None entries are skipped and stay None."""
    ranked = sorted(
        ((p, i) for i, p in enumerate(p_values) if p is not None), key=lambda t: t[0]
    )
    m = len(ranked)
    out: list[float | None] = [None] * len(p_values)
    running = 1.0
    for rank in range(m, 0, -1):
        p, i = ranked[rank - 1]
        running = min(running, p * m / rank)
        out[i] = running
    return out


# --------------------------------------------------------------------------- #
# Reading and cleaning
# --------------------------------------------------------------------------- #


def read_raw(path: Path) -> pl.DataFrame:
    """Every column as a string; typing happens in `load_events` where errors are counted."""
    try:
        return pl.read_csv(path, infer_schema=False)
    except pl.exceptions.NoDataError:
        return pl.DataFrame()


def parse_time_ms(column: str) -> pl.Expr:
    """String column -> UTC epoch milliseconds (ISO strings in several layouts, or bare epochs)."""
    text = pl.col(column)
    number = text.cast(pl.Float64, strict=False)
    number = pl.when(number.is_finite()).then(number)
    epoch_ms = (
        pl.when(number >= _EPOCH_MS_THRESHOLD)
        .then(number)
        .otherwise(number * 1000)
        .cast(pl.Int64)
    )
    parsed = pl.coalesce(
        [
            text.str.to_datetime(format=fmt, time_zone="UTC", strict=False)
            for fmt in _TS_FORMATS
        ]
    ).dt.epoch("ms")
    return pl.coalesce([epoch_ms, parsed]).alias("ts")


def load_events(
    raw: pl.DataFrame, cols: Columns | None = None
) -> tuple[pl.DataFrame, DataQuality]:
    """Clean a raw string frame into columns: seq, user, ts (epoch ms), event, session, segment.

    Each row is counted under the first check it fails, so the counts add up.
    """
    cols = cols or Columns()
    q = DataQuality(rows_read=raw.height)
    empty = pl.DataFrame(
        schema={
            "seq": pl.UInt32,
            "user": pl.String,
            "ts": pl.Int64,
            "event": pl.String,
            "session": pl.String,
            "segment": pl.String,
        }
    )
    if raw.width == 0:
        return empty, q

    needed = [cols.user, cols.time, cols.event]
    needed += [c for c in (cols.session, cols.segment) if c]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise DataError(
            f"missing required column(s): {', '.join(missing)}; found {raw.columns}"
        )

    def clean(column: str) -> pl.Expr:
        s = pl.col(column).cast(pl.String).str.strip_chars()
        return pl.when(s == "").then(None).otherwise(s)

    segment = clean(cols.segment) if cols.segment else pl.lit(None, pl.String)
    if cols.segment and cols.segment_level:
        segment = segment.str.split(".").list.head(cols.segment_level).list.join(".")
    df = raw.select(
        user=clean(cols.user),
        time_raw=clean(cols.time),
        event=clean(cols.event),
        session=clean(cols.session) if cols.session else pl.lit(None, pl.String),
        segment=segment,
    ).with_row_index("seq")

    def drop_where(frame: pl.DataFrame, cond: pl.Expr) -> tuple[pl.DataFrame, int]:
        n = frame.select(cond.sum()).item()
        return frame.filter(~cond), n

    df, q.null_user = drop_where(df, pl.col("user").is_null())
    df, q.null_time = drop_where(df, pl.col("time_raw").is_null())
    df = df.with_columns(parse_time_ms("time_raw")).drop("time_raw")
    df, q.unparseable_time = drop_where(df, pl.col("ts").is_null())
    df, q.null_event = drop_where(df, pl.col("event").is_null())
    if cols.session:
        df, q.null_session = drop_where(df, pl.col("session").is_null())

    # Rows that arrive earlier in the file than the same user's previous row. They
    # are kept and re-sorted by time; the count says how much the file needed it.
    q.out_of_order_rows = df.select(
        (pl.col("ts") < pl.col("ts").shift(1).over("user")).sum()
    ).item()

    before = df.height
    df = df.unique(
        subset=["user", "ts", "event", "session", "segment"],
        keep="first",
        maintain_order=True,
    )
    q.duplicate_events = before - df.height
    q.rows_used = df.height
    if df.height:
        q.first_event = _iso(df["ts"].min())
        q.last_event = _iso(df["ts"].max())
    return df, q


def _iso(ms: int) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def assign_entities(df: pl.DataFrame, cfg: Config, cols: Columns) -> pl.DataFrame:
    """Add an `entity` column (a session or a user) and sort each entity's events by time.

    With no session column, sessions are cut wherever one user's consecutive
    events are more than `session_gap_ms` apart (a gap of exactly that long stays
    inside the session). Ties in time keep file order.
    """
    if cfg.unit == "user":
        entity = pl.col("user")
    elif cols.session:
        entity = pl.col("session")
    else:
        df = df.sort(["user", "ts", "seq"])
        gap = pl.col("ts") - pl.col("ts").shift(1).over("user")
        new_session = gap.is_null() | (gap > cfg.session_gap_ms)
        number = new_session.cum_sum().over("user").cast(pl.String)
        entity = pl.col("user") + pl.lit("#") + number
    return df.with_columns(entity.alias("entity")).sort(["entity", "ts", "seq"])


# --------------------------------------------------------------------------- #
# Per-entity funnel matching
#
# An event carries a bitmask of the steps it can serve (bit k = step k), so a
# step may accept several event names. Times are integer milliseconds and each
# list is already sorted by (time, file order).
# --------------------------------------------------------------------------- #


@dataclass
class Reach:
    depth: int  # how many steps were completed (0 = never entered)
    times: list[int] = field(default_factory=list)  # ordered/strict: chosen chain
    spans: list[int] = field(default_factory=list)  # any-order: span of first n steps


def reach_ordered(
    times: list[int], masks: list[int], n_steps: int, window: int | None
) -> Reach:
    """Steps in order, other events allowed in between, each step within `window` of the entry.

    Every step-1 event starts an attempt, because with a window a *later* start
    can succeed where an earlier one timed out. Within one attempt, taking the
    earliest matching event for each step is optimal (it leaves the most room).
    The earliest attempt that reaches the greatest depth is reported.
    """
    best = Reach(0)
    for i, mask in enumerate(masks):
        if not mask & 1:
            continue
        limit = None if window is None else times[i] + window
        chain = [times[i]]
        k = 1
        for j in range(i + 1, len(times)):
            if k == n_steps or (limit is not None and times[j] > limit):
                break
            if masks[j] >> k & 1:
                chain.append(times[j])
                k += 1
        if len(chain) > best.depth:
            best = Reach(len(chain), chain)
            if k == n_steps:
                break
        if window is None:
            break  # without a window a later start can never get deeper
    return best


def reach_strict(
    times: list[int], masks: list[int], n_steps: int, window: int | None
) -> Reach:
    """Steps must be consecutive events: nothing else may happen between them."""
    best = Reach(0)
    n = len(times)
    for i in range(n):
        if not masks[i] & 1:
            continue
        limit = None if window is None else times[i] + window
        d = 1
        while (
            d < n_steps
            and i + d < n
            and masks[i + d] >> d & 1
            and (limit is None or times[i + d] <= limit)
        ):
            d += 1
        if d > best.depth:
            best = Reach(d, times[i : i + d])
            if d == n_steps:
                break
    return best


def _min_span(step_times: list[tuple[int, int]], n: int) -> int | None:
    """Smallest time window containing at least one event of each of steps 0..n-1."""
    events = [(t, s) for t, s in step_times if s < n]
    counts = [0] * n
    have = 0
    left = 0
    best: int | None = None
    for t, s in events:
        counts[s] += 1
        have += counts[s] == 1
        while have == n:
            span = t - events[left][0]
            best = span if best is None or span < best else best
            ls = events[left][1]
            counts[ls] -= 1
            have -= counts[ls] == 0
            left += 1
    return best


def reach_any_order(
    times: list[int], masks: list[int], n_steps: int, window: int | None
) -> Reach:
    """The first n steps all happen, in any order, inside one window of length `window`."""
    step_times = [(t, m.bit_length() - 1) for t, m in zip(times, masks) if m]
    spans: list[int] = []
    for n in range(1, n_steps + 1):
        span = _min_span(step_times, n)
        if span is None or (window is not None and span > window):
            break
        spans.append(span)
    return Reach(len(spans), spans=spans)


_REACH = {
    "ordered": reach_ordered,
    "strict": reach_strict,
    "any-order": reach_any_order,
}


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def _quantile(values: list[float], q: float) -> float | None:
    return pl.Series(values).quantile(q, interpolation="linear") if values else None


def analyze(
    df: pl.DataFrame,
    cfg: Config,
    cols: Columns | None = None,
    quality: DataQuality | None = None,
) -> FunnelResult:
    """Run the funnel over cleaned events (see `load_events`)."""
    cfg.validate()
    cols = cols or Columns()
    n_steps = len(cfg.steps)
    outcome = (cfg.outcome_step or n_steps) - 1
    quality = quality or DataQuality(rows_read=df.height, rows_used=df.height)
    result = FunnelResult(
        labels=list(cfg.labels),
        mode=cfg.mode,
        unit=cfg.unit,
        window_ms=cfg.window_ms,
        outcome_label=cfg.labels[outcome],
        segment_col=cols.segment,
        quality=quality,
    )

    seen_events = set(df["event"].unique().to_list()) if df.height else set()
    result.quality.unmatched_steps = [
        label for label, names in zip(cfg.labels, cfg.steps) if not names & seen_events
    ]

    mask_of = {}
    for k, names in enumerate(cfg.steps):
        for name in names:
            mask_of[name] = mask_of.get(name, 0) | (1 << k)

    if df.height == 0:
        empty: list[list[float]] = [[] for _ in range(n_steps)]
        result.steps = _step_stats(cfg, [0] * n_steps, empty, empty)
        return result

    frame = assign_entities(df, cfg, cols)
    frame = frame.with_columns(
        pl.col("event")
        .replace_strict(mask_of, default=0, return_dtype=pl.Int64)
        .alias("mask")
    )
    result.quality.entities = frame["entity"].n_unique()
    result.quality.step_events = frame.select((pl.col("mask") > 0).sum()).item()
    log_end = frame["ts"].max()
    if cfg.mode != "strict":  # strict needs the non-step events to see "in between"
        frame = frame.filter(pl.col("mask") > 0)

    grouped = frame.group_by("entity", maintain_order=True).agg(
        pl.col("ts"), pl.col("mask"), pl.col("segment")
    )
    reach = _REACH[cfg.mode]
    depth_counts = [0] * n_steps
    deltas: list[list[float]] = [
        [] for _ in range(n_steps)
    ]  # seconds from previous step
    from_entry: list[list[float]] = [[] for _ in range(n_steps)]
    entrants: list[tuple[str, int]] = []  # (segment, depth)

    for _entity, times, masks, segments in grouped.iter_rows():
        r = reach(times, masks, n_steps, cfg.window_ms)
        if r.depth == 0:
            continue
        entry = next(i for i, m in enumerate(masks) if m & 1)
        if (
            cfg.drop_censored
            and cfg.window_ms is not None
            and times[entry] > log_end - cfg.window_ms
        ):
            result.censored_dropped += 1
            continue
        for k in range(r.depth):
            depth_counts[k] += 1
        if r.times:
            for k in range(1, r.depth):
                deltas[k].append((r.times[k] - r.times[k - 1]) / 1000)
                from_entry[k].append((r.times[k] - r.times[0]) / 1000)
        else:
            for k in range(1, r.depth):
                from_entry[k].append(r.spans[k] / 1000)
        entrants.append((segments[entry] or "(unknown)", r.depth))

    result.entrants = depth_counts[0]
    result.steps = _step_stats(cfg, depth_counts, deltas, from_entry)
    result.biggest_leak = _biggest_leak(result.steps)
    if cols.segment:
        _segment_stats(result, cfg, entrants, outcome)
    return result


def _step_stats(
    cfg: Config,
    counts: list[int],
    deltas: list[list[float]],
    from_entry: list[list[float]],
) -> list[StepStat]:
    out = []
    entry = counts[0]
    for k, label in enumerate(cfg.labels):
        prev = counts[k - 1] if k else None
        out.append(
            StepStat(
                label=label,
                count=counts[k],
                of_entry=counts[k] / entry if entry else math.nan,
                of_entry_ci=wilson_interval(counts[k], entry, cfg.alpha),
                of_prev=(counts[k] / prev if prev else math.nan) if k else None,
                of_prev_ci=wilson_interval(counts[k], prev, cfg.alpha) if k else None,
                dropped=(prev - counts[k]) if k else 0,
                median_from_prev_s=_quantile(deltas[k], 0.5) if deltas[k] else None,
                q1_from_prev_s=_quantile(deltas[k], 0.25) if deltas[k] else None,
                q3_from_prev_s=_quantile(deltas[k], 0.75) if deltas[k] else None,
                median_from_entry_s=_quantile(from_entry[k], 0.5)
                if from_entry[k]
                else None,
            )
        )
    return out


def _biggest_leak(steps: list[StepStat]) -> dict | None:
    """The transition that loses the most entities in absolute terms."""
    if len(steps) < 2 or steps[0].count == 0:
        return None
    k = max(range(1, len(steps)), key=lambda i: steps[i].dropped)
    return {
        "from": steps[k - 1].label,
        "to": steps[k].label,
        "dropped": steps[k].dropped,
        "drop_rate": 1 - steps[k].of_prev if not math.isnan(steps[k].of_prev) else None,
        "share_of_all_drops": steps[k].dropped / (steps[0].count - steps[-1].count)
        if steps[0].count != steps[-1].count
        else None,
    }


def _segment_stats(
    result: FunnelResult,
    cfg: Config,
    entrants: list[tuple[str, int]],
    outcome: int,
) -> None:
    n_steps = len(cfg.steps)
    sizes: dict[str, int] = {}
    for seg, _ in entrants:
        sizes[seg] = sizes.get(seg, 0) + 1
    ranked = sorted(sizes, key=lambda s: (-sizes[s], s))
    keep = {
        s
        for s in ranked[: cfg.top_segments]
        if sizes[s] >= cfg.min_segment_size and s != "(other)"
    }

    per_seg: dict[str, list[int]] = {}
    size_of: dict[str, int] = {}
    for seg, depth in entrants:
        name = seg if seg in keep else "(other)"
        counts = per_seg.setdefault(name, [0] * n_steps)
        size_of[name] = size_of.get(name, 0) + 1
        for k in range(depth):
            counts[k] += 1
    if not per_seg:
        return

    total_entrants = len(entrants)
    total_converted = sum(c[outcome] for c in per_seg.values())
    names = sorted(per_seg, key=lambda s: (s == "(other)", -size_of[s], s))
    p_values: list[float | None] = []
    stats_rows: list[SegmentStat] = []
    for name in names:
        counts = per_seg[name]
        n = size_of[name]
        a = counts[outcome]
        rest_n = total_entrants - n
        rest_c = total_converted - a
        if rest_n > 0:
            _, p = stats.fisher_exact([[a, n - a], [rest_c, rest_n - rest_c]])
            p_values.append(float(p))
            rest_rate = rest_c / rest_n
            lift = (a / n) / rest_rate if rest_rate > 0 else None
        else:
            p_values.append(None)
            lift = None
        stats_rows.append(
            SegmentStat(
                name=name,
                entrants=n,
                counts=counts,
                conversion=a / n,
                ci=wilson_interval(a, n, cfg.alpha),
                lift_vs_rest=lift,
                p_value=p_values[-1],
                q_value=None,
                significant=False,
            )
        )
    for row, q in zip(stats_rows, benjamini_hochberg(p_values)):
        row.q_value = q
        row.significant = q is not None and q < cfg.alpha
    result.segments = stats_rows

    table = [[per_seg[n][outcome], size_of[n] - per_seg[n][outcome]] for n in names]
    if len(table) >= 2 and 0 < total_converted < total_entrants:
        result.omnibus_p = float(stats.chi2_contingency(table, correction=False).pvalue)


# --------------------------------------------------------------------------- #
# HTML report. Every data-derived string goes through html.escape, including
# the labels handed to Plotly (which renders a few HTML tags in text).
# --------------------------------------------------------------------------- #

_CONFIG = {"displaylogo": False}


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None or math.isnan(seconds):
        return "-"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f}m"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _fmt_window(ms: int | None) -> str:
    return "no time limit" if ms is None else _fmt_duration(ms / 1000)


def _pct(x: float | None, digits: int = 1) -> str:
    return "-" if x is None or math.isnan(x) else f"{100 * x:.{digits}f}%"


def _ci(ci: tuple[float, float] | None) -> str:
    if ci is None or math.isnan(ci[0]):
        return "-"
    return f"{_pct(ci[0])} to {_pct(ci[1])}"


def _fig_html(fig: go.Figure) -> str:
    fig.update_layout(
        template="plotly_white", margin={"l": 10, "r": 10, "t": 30, "b": 10}
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config=_CONFIG)


def _esc(text: object) -> str:
    return html.escape(str(text))


def _funnel_chart(result: FunnelResult) -> str:
    labels = [_esc(s.label) for s in result.steps]
    fig = go.Figure(
        go.Funnel(
            y=labels,
            x=[s.count for s in result.steps],
            textinfo="value+percent initial+percent previous",
            marker={"color": "#2563eb"},
        )
    )
    fig.update_layout(height=120 + 80 * len(labels))
    return _fig_html(fig)


def _time_chart(result: FunnelResult) -> str:
    rows = [s for s in result.steps[1:] if s.median_from_prev_s is not None]
    if not rows:
        return ""
    fig = go.Figure(
        go.Bar(
            x=[_esc(s.label) for s in rows],
            y=[s.median_from_prev_s for s in rows],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [
                    (s.q3_from_prev_s or 0) - (s.median_from_prev_s or 0) for s in rows
                ],
                "arrayminus": [
                    (s.median_from_prev_s or 0) - (s.q1_from_prev_s or 0) for s in rows
                ],
            },
            text=[_fmt_duration(s.median_from_prev_s) for s in rows],
            marker={"color": "#0d9488"},
        )
    )
    fig.update_layout(
        yaxis={
            "title": "median seconds since previous step (bars: quartiles)",
            "type": "log",
        },
        height=340,
    )
    return _fig_html(fig)


def _segment_chart(result: FunnelResult) -> str:
    segs = sorted(result.segments, key=lambda s: s.conversion)
    fig = go.Figure(
        go.Bar(
            y=[_esc(s.name) for s in segs],
            x=[100 * s.conversion for s in segs],
            orientation="h",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [100 * (s.ci[1] - s.conversion) for s in segs],
                "arrayminus": [100 * (s.conversion - s.ci[0]) for s in segs],
            },
            marker={"color": ["#dc2626" if s.significant else "#94a3b8" for s in segs]},
            customdata=[[s.entrants] for s in segs],
            hovertemplate="%{y}<br>%{x:.2f}% of %{customdata[0]} entrants<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={"title": f"% reaching '{_esc(result.outcome_label)}' (95% Wilson CI)"},
        height=140 + 34 * len(segs),
    )
    return _fig_html(fig)


def _segment_funnel_chart(result: FunnelResult) -> str:
    fig = go.Figure()
    for s in result.segments:
        entry = s.counts[0] or 1
        fig.add_trace(
            go.Scatter(
                x=[_esc(label) for label in result.labels],
                y=[100 * c / entry for c in s.counts],
                mode="lines+markers",
                name=_esc(s.name),
            )
        )
    fig.update_layout(
        yaxis={"title": "% of entrants still in the funnel", "type": "log"}, height=420
    )
    return _fig_html(fig)


def _steps_table(result: FunnelResult) -> str:
    rows = []
    for s in result.steps:
        rows.append(
            f"<tr><td>{_esc(s.label)}</td><td>{s.count:,}</td>"
            f"<td>{_pct(s.of_entry, 2)}<br><small>{_ci(s.of_entry_ci)}</small></td>"
            f"<td>{_pct(s.of_prev)}<br><small>{_ci(s.of_prev_ci)}</small></td>"
            f"<td>{s.dropped:,}</td>"
            f"<td>{_fmt_duration(s.median_from_prev_s)}"
            f"<br><small>{_fmt_duration(s.q1_from_prev_s)} to {_fmt_duration(s.q3_from_prev_s)}</small></td>"
            f"<td>{_fmt_duration(s.median_from_entry_s)}</td></tr>"
        )
    return (
        "<table><tr><th>Step</th><th>Entities</th><th>% of entry (95% CI)</th>"
        "<th>% of previous step (95% CI)</th><th>Dropped</th>"
        "<th>Median since previous step<br><small>quartiles</small></th>"
        "<th>Median since entry</th></tr>" + "".join(rows) + "</table>"
    )


def _segments_table(result: FunnelResult) -> str:
    rows = []
    for s in result.segments:
        flag = "yes" if s.significant else "no"
        lift = "-" if s.lift_vs_rest is None else f"{s.lift_vs_rest:.2f}x"
        p = "-" if s.p_value is None else f"{s.p_value:.3g}"
        q = "-" if s.q_value is None else f"{s.q_value:.3g}"
        rows.append(
            f"<tr><td>{_esc(s.name)}</td><td>{s.entrants:,}</td>"
            f"<td>{_pct(s.conversion, 2)}<br><small>{_ci(s.ci)}</small></td>"
            f"<td>{lift}</td><td>{p}</td><td>{q}</td><td>{flag}</td></tr>"
        )
    return (
        "<table><tr><th>Segment</th><th>Entrants</th><th>Conversion (95% CI)</th>"
        "<th>Lift vs rest</th><th>p (Fisher)</th><th>q (BH)</th><th>Significant</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _quality_table(q: DataQuality) -> str:
    items = [
        ("Rows read", q.rows_read),
        ("Dropped: missing user", q.null_user),
        ("Dropped: missing timestamp", q.null_time),
        ("Dropped: unparseable timestamp", q.unparseable_time),
        ("Dropped: missing event name", q.null_event),
        ("Dropped: missing session id", q.null_session),
        ("Dropped: duplicate events", q.duplicate_events),
        ("Rows used", q.rows_used),
        ("Out-of-order rows (kept, re-sorted)", q.out_of_order_rows),
        ("Events matching a funnel step", q.step_events),
        ("Sessions/users in the log", q.entities),
        ("First event", q.first_event or "-"),
        ("Last event", q.last_event or "-"),
    ]
    rows = "".join(
        f"<tr><td>{_esc(k)}</td><td>{v:,}</td></tr>"
        if isinstance(v, int)
        else f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>"
        for k, v in items
    )
    return f"<table>{rows}</table>"


def build_report(result: FunnelResult, source_name: str = "events") -> str:
    q = result.quality
    unit = "sessions" if result.unit == "session" else "users"
    mode_note = {
        "ordered": "steps must happen in order; other events may happen in between",
        "strict": "steps must be consecutive events, with nothing in between",
        "any-order": "the first N steps must all happen, in any order",
    }[result.mode]

    warnings = []
    for label in q.unmatched_steps:
        warnings.append(
            f"Step '{_esc(label)}' matches no event in the data (check the spelling)."
        )
    if result.censored_dropped:
        warnings.append(
            f"{result.censored_dropped:,} {unit} entered too close to the end of the log "
            "for their window to finish and were excluded."
        )
    warn_html = "".join(f"<p class='warn'>{w}</p>" for w in warnings)
    arrow_labels = " &rarr; ".join(_esc(label) for label in result.labels)

    if result.entrants == 0:
        body = "<p class='warn'>No entity reached the first step: nothing to chart. See the data-quality table.</p>"
    else:
        leak = result.biggest_leak
        leak_html = ""
        if leak:
            share = leak["share_of_all_drops"]
            leak_html = (
                f"<p><strong>Biggest leak:</strong> {_esc(leak['from'])} &rarr; {_esc(leak['to'])} "
                f"loses {leak['dropped']:,} {unit}"
                + (
                    f" ({_pct(share, 0)} of everything lost end to end)"
                    if share
                    else ""
                )
                + ".</p>"
            )
        parts = [
            f"<script>{get_plotlyjs()}</script>",
            leak_html,
            "<h2>Funnel</h2>",
            _funnel_chart(result),
            _steps_table(result),
        ]
        time_chart = _time_chart(result)
        if time_chart:
            parts += ["<h2>Time between steps</h2>", time_chart]
        if result.segments:
            omni = (
                ""
                if result.omnibus_p is None
                else f" Chi-square test that conversion differs across segments: p = {result.omnibus_p:.3g}."
            )
            seg_note = (
                f"<p class='note'>A {unit[:-1]} is attributed to the segment of its first "
                f"'{_esc(result.labels[0])}' event. Conversion means reaching "
                f"'{_esc(result.outcome_label)}'. Red bars differ from all other segments at "
                f"BH-adjusted q &lt; 0.05 (Fisher exact, one test per segment).{omni}</p>"
            )
            parts += [
                f"<h2>By {_esc(result.segment_col)}</h2>",
                seg_note,
                _segment_chart(result),
                _segments_table(result),
                "<h3>Where each segment drops out</h3>",
                _segment_funnel_chart(result),
            ]
        body = "".join(parts)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Funnel: {_esc(source_name)}</title>
<style>
body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #111827; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ text-align: left; padding: 4px 16px 4px 0; vertical-align: top; }}
th {{ border-bottom: 1px solid #d1d5db; }}
small {{ color: #6b7280; }}
.note {{ color: #4b5563; }}
.warn {{ color: #b45309; font-weight: 600; }}
</style></head><body>
<h1>Funnel: {_esc(source_name)}</h1>
<p class="note">{result.entrants:,} {unit} entered the funnel ({arrow_labels}).
Mode: {mode_note}. Time limit from entry: {_fmt_window(result.window_ms)}.</p>
{warn_html}
{body}
<h2>Data quality</h2>
{_quality_table(q)}
</body></html>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Stage-by-stage funnel drop-off from a raw clickstream CSV."
    )
    p.add_argument("csv", type=Path, help="event log CSV or CSV.GZ")
    p.add_argument("-o", "--output", type=Path, default=Path("funnel_report.html"))
    p.add_argument("--json", type=Path, help="also write the results as JSON")
    p.add_argument(
        "--steps",
        default="view,cart,purchase",
        help="comma-separated event names; 'a|b' lets either event count for a step",
    )
    p.add_argument("--mode", choices=MODES, default="ordered")
    p.add_argument("--unit", choices=UNITS, default="session")
    p.add_argument(
        "--window", help="max time from entry to each step, e.g. 30m, 24h, 7d"
    )
    p.add_argument(
        "--session-gap", default="30m", help="inactivity that ends a session"
    )
    p.add_argument("--session-col", help="use this session id instead of sessionizing")
    p.add_argument("--user-col", default="user_id")
    p.add_argument("--time-col", default="event_time")
    p.add_argument("--event-col", default="event_type")
    p.add_argument("--segment-col", help="break the funnel down by this column")
    p.add_argument(
        "--segment-level", type=int, help="keep the first N parts of a dotted value"
    )
    p.add_argument("--top-segments", type=int, default=8)
    p.add_argument("--min-segment-size", type=int, default=30)
    p.add_argument(
        "--outcome-step", type=int, help="1-based step segments are compared on"
    )
    p.add_argument("--drop-censored", action="store_true")
    args = p.parse_args(argv)

    try:
        steps, labels = parse_steps(args.steps)
        cfg = Config(
            steps=steps,
            labels=labels,
            mode=args.mode,
            unit=args.unit,
            window_ms=parse_duration(args.window) if args.window else None,
            session_gap_ms=parse_duration(args.session_gap),
            outcome_step=args.outcome_step,
            drop_censored=args.drop_censored,
            top_segments=args.top_segments,
            min_segment_size=args.min_segment_size,
        )
        cfg.validate()
        cols = Columns(
            user=args.user_col,
            time=args.time_col,
            event=args.event_col,
            session=args.session_col,
            segment=args.segment_col,
            segment_level=args.segment_level,
        )
        events, quality = load_events(read_raw(args.csv), cols)
        result = analyze(events, cfg, cols, quality)
    except (OSError, DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.write_text(build_report(result, args.csv.name), encoding="utf-8")
    if args.json:
        args.json.write_text(result.to_json(), encoding="utf-8")
    print(
        f"{quality.rows_used:,}/{quality.rows_read:,} rows used, "
        f"{result.entrants:,} {'sessions' if cfg.unit == 'session' else 'users'} entered "
        f"-> {args.output}"
    )
    for s in result.steps:
        print(f"  {s.label:<20} {s.count:>9,}  {_pct(s.of_entry, 2):>8} of entry")
    for label in quality.unmatched_steps:
        print(f"warning: step '{label}' matches no event in the data", file=sys.stderr)
    if result.entrants == 0:
        print("warning: nobody reached the first step", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
