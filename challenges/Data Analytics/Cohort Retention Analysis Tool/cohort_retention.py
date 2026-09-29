"""Cohort Retention Analysis Tool: raw event log CSV -> cohort retention matrix + HTML report.

Run with:  uv run python cohort_retention.py events.csv --granularity week -o report.html
"""

from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from plotly.offline import get_plotlyjs

GRANULARITIES = ("day", "week", "month")
MODES = ("active", "return")

# Tried in order and coalesced. Polars infers one format from the first value
# it sees, so a file mixing "...Z", "...+02:00" and naive timestamps would
# otherwise parse only the rows that happen to match the first one.
_TS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%.f%#z",
    "%Y-%m-%d %H:%M:%S%.f%#z",
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%d",
)

# 1970-01-05 was a Monday: epoch-day numbers of Monday-truncated weeks are all
# congruent to 4 mod 7.
_EPOCH_MONDAY_OFFSET = 4


@dataclass
class DataQuality:
    rows_read: int = 0
    null_user_id: int = 0
    null_timestamp: int = 0
    unparseable_timestamp: int = 0
    duplicate_rows: int = 0
    users_without_signup: int = 0
    events_before_signup: int = 0
    rows_used: int = 0
    users: int = 0
    first_event: str | None = None
    last_event: str | None = None


@dataclass
class RetentionResult:
    granularity: str
    mode: str
    include_partial: bool
    cohort_labels: list[str] = field(default_factory=list)
    cohort_sizes: list[int] = field(default_factory=list)
    n_offsets: int = 0
    # counts[i][n] / pct[i][n] are None where the cell is not observable yet
    # (right-censored): that period had not (fully) happened when the log ended.
    counts: list[list[int | None]] = field(default_factory=list)
    pct: list[list[float | None]] = field(default_factory=list)
    average_curve: list[float | None] = field(default_factory=list)
    quality: DataQuality = field(default_factory=DataQuality)


# --------------------------------------------------------------------------- #
# Period arithmetic
# --------------------------------------------------------------------------- #


def period_index_expr(column: str, granularity: str) -> pl.Expr:
    """Integer period number of a naive-UTC datetime column (consecutive periods differ by 1)."""
    ts = pl.col(column)
    if granularity == "day":
        return ts.dt.truncate("1d").dt.epoch("d")
    if granularity == "week":
        days = ts.dt.truncate("1w").dt.epoch("d")
        return (days - _EPOCH_MONDAY_OFFSET) // 7
    if granularity == "month":
        return ts.dt.year() * 12 + ts.dt.month() - 1
    raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")


def period_start(index: int, granularity: str) -> date:
    if granularity == "day":
        return date(1970, 1, 1) + timedelta(days=index)
    if granularity == "week":
        return date(1970, 1, 1) + timedelta(days=_EPOCH_MONDAY_OFFSET + 7 * index)
    year, month0 = divmod(index, 12)
    return date(year, month0 + 1, 1)


def period_label(index: int, granularity: str) -> str:
    start = period_start(index, granularity)
    return start.strftime("%Y-%m") if granularity == "month" else start.isoformat()


# --------------------------------------------------------------------------- #
# Loading + cleaning
# --------------------------------------------------------------------------- #


def read_events(path: Path) -> pl.DataFrame:
    """Read every column as a string; typing happens in `analyze` where errors are counted."""
    try:
        return pl.read_csv(path, infer_schema=False)
    except pl.exceptions.NoDataError:
        return pl.DataFrame()


def parse_timestamps(column: str) -> pl.Expr:
    """String column -> naive datetime holding the UTC wall time (offsets converted, naive = UTC).

    The UTC result is stored *naive* on purpose: it makes bucketing by UTC
    calendar boundaries trivial and avoids needing an OS tz database (Windows
    has none by default) just to hold a "UTC" zone.
    """
    text = pl.col(column).str.strip_chars()
    parsed = [
        text.str.to_datetime(format=fmt, time_zone="UTC", strict=False)
        for fmt in _TS_FORMATS
    ]
    return pl.coalesce(parsed).dt.replace_time_zone(None).alias(column)


def analyze(
    df: pl.DataFrame,
    *,
    granularity: str = "week",
    mode: str = "active",
    signup_event: str | None = None,
    include_partial: bool = False,
    user_col: str = "user_id",
    time_col: str = "timestamp",
    event_col: str = "event_name",
) -> RetentionResult:
    """Build the cohort x offset retention matrix from a raw event log.

    mode="active":  cell (cohort, n) = share of the cohort with >= 1 event in
                    period `cohort + n` ("classic" retention).
    mode="return":  cell (cohort, n) = share of the cohort with >= 1 event in
                    period `cohort + n` *or any later period* (unbounded /
                    "returned on or after" retention; monotonically
                    non-increasing in n).
    """
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {GRANULARITIES}, got {granularity!r}"
        )
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    result = RetentionResult(
        granularity=granularity, mode=mode, include_partial=include_partial
    )
    quality = result.quality
    quality.rows_read = df.height

    required = [user_col, time_col] + ([event_col] if signup_event is not None else [])
    missing = [c for c in required if c not in df.columns]
    if df.width == 0 and df.height == 0:
        return (
            result  # a truly empty file: nothing to complain about, nothing to analyze
        )
    if missing:
        raise ValueError(
            f"missing required column(s): {', '.join(missing)}; found {df.columns}"
        )

    cols = [user_col, time_col] + ([event_col] if event_col in df.columns else [])
    events = df.select([pl.col(c).cast(pl.String) for c in cols])
    events = events.with_columns(pl.col(user_col).str.strip_chars())
    # Each row is counted under the first check it fails, so the counts add up.

    is_null_user = pl.col(user_col).is_null() | (pl.col(user_col) == "")
    quality.null_user_id = events.select(is_null_user.sum()).item()
    events = events.filter(~is_null_user)

    raw_ts_blank = pl.col(time_col).is_null() | (
        pl.col(time_col).str.strip_chars() == ""
    )
    quality.null_timestamp = events.select(raw_ts_blank.sum()).item()
    events = events.filter(~raw_ts_blank)

    events = events.with_columns(parse_timestamps(time_col))
    quality.unparseable_timestamp = events.select(
        pl.col(time_col).is_null().sum()
    ).item()
    events = events.filter(pl.col(time_col).is_not_null())

    before = events.height
    events = events.unique(maintain_order=True)
    quality.duplicate_rows = before - events.height

    if signup_event is not None:
        signups = (
            events.filter(pl.col(event_col) == signup_event)
            .group_by(user_col)
            .agg(pl.col(time_col).min().alias("_start"))
        )
        all_users = events.select(user_col).unique().height
        quality.users_without_signup = all_users - signups.height
        events = events.join(signups, on=user_col, how="inner")
        before = events.height
        events = events.filter(pl.col(time_col) >= pl.col("_start")).drop("_start")
        quality.events_before_signup = before - events.height
        # keep the signup timestamps around for cohort assignment
        start_ts = signups.rename({"_start": time_col})
    else:
        start_ts = events.group_by(user_col).agg(pl.col(time_col).min())

    quality.rows_used = events.height
    if events.height == 0:
        return result

    quality.users = events.select(user_col).n_unique()
    quality.first_event = events.select(pl.col(time_col).min()).item().isoformat()
    last_ts = events.select(pl.col(time_col).max()).item()
    quality.last_event = last_ts.isoformat()

    idx = period_index_expr(time_col, granularity)
    last_idx = pl.DataFrame({time_col: [last_ts]}).select(idx).item()

    cohorts = start_ts.select(pl.col(user_col), idx.alias("cohort"))
    activity = (
        events.select(pl.col(user_col), idx.alias("period"))
        .unique()
        .join(cohorts, on=user_col)
        .with_columns((pl.col("period") - pl.col("cohort")).alias("offset"))
    )

    sizes = {
        row["cohort"]: row["size"]
        for row in cohorts.group_by("cohort")
        .agg(pl.len().alias("size"))
        .iter_rows(named=True)
    }
    cohort_ids = sorted(sizes)

    if mode == "active":
        raw = {
            (r["cohort"], r["offset"]): r["users"]
            for r in activity.group_by("cohort", "offset")
            .agg(pl.col(user_col).n_unique().alias("users"))
            .iter_rows(named=True)
        }
    else:
        last_offset_hist = {
            (r["cohort"], r["last"]): r["users"]
            for r in activity.group_by(user_col, "cohort")
            .agg(pl.col("offset").max().alias("last"))
            .group_by("cohort", "last")
            .agg(pl.len().alias("users"))
            .iter_rows(named=True)
        }
        raw = {}
        for c in cohort_ids:
            running = 0
            for n in range(last_idx - c, -1, -1):
                running += last_offset_hist.get((c, n), 0)
                raw[(c, n)] = running

    last_complete = last_idx if include_partial else last_idx - 1
    result.n_offsets = last_idx - cohort_ids[0] + 1
    observed_users = [0] * result.n_offsets
    observed_retained = [0] * result.n_offsets

    for c in cohort_ids:
        size = sizes[c]
        counts_row: list[int | None] = []
        pct_row: list[float | None] = []
        for n in range(result.n_offsets):
            if c + n > last_complete:  # period not (fully) observed: censored, not zero
                counts_row.append(None)
                pct_row.append(None)
                continue
            retained = raw.get((c, n), 0)
            counts_row.append(retained)
            pct_row.append(100.0 * retained / size)
            observed_users[n] += size
            observed_retained[n] += retained
        result.cohort_labels.append(period_label(c, granularity))
        result.cohort_sizes.append(size)
        result.counts.append(counts_row)
        result.pct.append(pct_row)

    result.average_curve = [
        100.0 * observed_retained[n] / observed_users[n] if observed_users[n] else None
        for n in range(result.n_offsets)
    ]
    return result


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #

_UNIT = {"day": "Day", "week": "Week", "month": "Month"}


def _fig_html(fig: go.Figure) -> str:
    return pio.to_html(
        fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False}
    )


def _heatmap(result: RetentionResult) -> str:
    unit = _UNIT[result.granularity]
    y = [f"{lbl} (n={n})" for lbl, n in zip(result.cohort_labels, result.cohort_sizes)]
    x = [f"{unit} {n}" for n in range(result.n_offsets)]
    text = [[("" if v is None else f"{v:.0f}%") for v in row] for row in result.pct]
    fig = go.Figure(
        go.Heatmap(
            z=result.pct,
            x=x,
            y=y,
            text=text,
            texttemplate="%{text}",
            customdata=result.counts,
            hovertemplate="%{y}<br>%{x}: %{z:.1f}% (%{customdata} users)<extra></extra>",
            hoverongaps=False,
            colorscale="Blues",
            zmin=0,
            zmax=100,
            xgap=2,
            ygap=2,
            colorbar={"title": "% retained"},
        )
    )
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(type="category", side="top")
    fig.update_layout(
        plot_bgcolor="#d1d5db",  # masked (censored) cells show as grey background
        height=max(320, 34 * len(y) + 140),
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    return _fig_html(fig)


def _curves(result: RetentionResult) -> str:
    unit = _UNIT[result.granularity]
    fig = go.Figure()
    for label, row in zip(result.cohort_labels, result.pct):
        fig.add_trace(
            go.Scatter(
                x=list(range(result.n_offsets)),
                y=row,
                mode="lines",
                name=label,
                line={"width": 1},
                opacity=0.55,
                connectgaps=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=list(range(result.n_offsets)),
            y=result.average_curve,
            mode="lines+markers",
            name="Weighted average",
            line={"width": 4, "color": "#111827"},
        )
    )
    fig.update_layout(
        xaxis_title=f"{unit}s since cohort start",
        yaxis_title="% of cohort retained",
        yaxis_range=[0, 105],
        height=460,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    return _fig_html(fig)


def _sizes(result: RetentionResult) -> str:
    fig = go.Figure(
        go.Bar(x=result.cohort_labels, y=result.cohort_sizes, marker_color="#2563eb")
    )
    fig.update_layout(
        xaxis_title="Cohort",
        yaxis_title="Users",
        height=340,
        margin={"l": 10, "r": 10, "t": 30},
    )
    fig.update_xaxes(type="category")
    return _fig_html(fig)


def _quality_table(q: DataQuality) -> str:
    rows = [
        ("Rows read", q.rows_read),
        ("Dropped: missing user_id", q.null_user_id),
        ("Dropped: missing timestamp", q.null_timestamp),
        ("Dropped: unparseable timestamp", q.unparseable_timestamp),
        ("Dropped: duplicate rows", q.duplicate_rows),
        ("Dropped: users with no signup event", q.users_without_signup),
        ("Dropped: events before signup", q.events_before_signup),
        ("Rows analyzed", q.rows_used),
        ("Users", q.users),
        ("First event (UTC)", q.first_event or "-"),
        ("Last event (UTC)", q.last_event or "-"),
    ]
    body = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in rows
    )
    return f"<table>{body}</table>"


def build_report(result: RetentionResult, source_name: str = "events") -> str:
    unit = _UNIT[result.granularity].lower()
    if result.mode == "active":
        mode_note = f"a cell is the share of the cohort with at least one event <em>in</em> {unit} N."
    else:
        mode_note = (
            f"a cell is the share of the cohort with at least one event in {unit} N "
            f"<em>or any later {unit}</em> (unbounded retention, never increases)."
        )
    partial_note = (
        f"The final, still-open {unit} is included."
        if result.include_partial
        else f"The final, still-open {unit} is treated as incomplete and masked."
    )

    if not result.cohort_labels:
        charts = "<p class='warn'>No usable events: nothing to chart. See the data-quality table.</p>"
    else:
        bundle = f"<script>{get_plotlyjs()}</script>"
        charts = f"""{bundle}
<h2>Retention heatmap</h2>
<p class="note">Grey cells are <strong>not yet observable</strong> (that period had not
finished when the log ended). They are unknown, not 0%.</p>
{_heatmap(result)}
<h2>Retention curves</h2>
{_curves(result)}
<h2>Cohort sizes</h2>
{_sizes(result)}"""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cohort retention: {html.escape(source_name)}</title>
<style>
body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #111827; }}
table {{ border-collapse: collapse; }}
th, td {{ text-align: left; padding: 2px 14px 2px 0; }}
.note {{ color: #4b5563; }}
.warn {{ color: #b45309; font-weight: 600; }}
</style></head><body>
<h1>Cohort retention: {html.escape(source_name)}</h1>
<p class="note">Cohorts are grouped by {unit} of first event (UTC); {mode_note} {partial_note}</p>
<h2>Data quality</h2>
{_quality_table(result.quality)}
{charts}
</body></html>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Cohort retention analysis from a raw event log CSV."
    )
    p.add_argument(
        "csv", type=Path, help="event log CSV (user_id, timestamp[, event_name])"
    )
    p.add_argument("-o", "--output", type=Path, default=Path("retention_report.html"))
    p.add_argument("--granularity", choices=GRANULARITIES, default="week")
    p.add_argument(
        "--mode",
        choices=MODES,
        default="active",
        help="'active' = active in period N; 'return' = active in period N or later",
    )
    p.add_argument(
        "--signup-event",
        help="cohort = first occurrence of this event (default: first event of any kind)",
    )
    p.add_argument(
        "--include-partial",
        action="store_true",
        help="count the final, still-open period as observed",
    )
    p.add_argument("--user-col", default="user_id")
    p.add_argument("--time-col", default="timestamp")
    p.add_argument("--event-col", default="event_name")
    args = p.parse_args(argv)

    try:
        df = read_events(args.csv)
        result = analyze(
            df,
            granularity=args.granularity,
            mode=args.mode,
            signup_event=args.signup_event,
            include_partial=args.include_partial,
            user_col=args.user_col,
            time_col=args.time_col,
            event_col=args.event_col,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.write_text(build_report(result, args.csv.name), encoding="utf-8")
    q = result.quality
    print(
        f"{q.rows_used}/{q.rows_read} rows used, {q.users} users, "
        f"{len(result.cohort_labels)} cohorts -> {args.output}"
    )
    if not result.cohort_labels:
        print(
            "warning: no usable events; report contains only the data-quality table",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
