"""Statistical anomaly detector for a time-series CSV -> flags + one self-contained HTML report.

Run with:  uv run python anomaly_detector.py sample_data/nyc_taxi.csv --labels sample_data/labels.json -o report.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from detectors import (
    METHODS,
    STL_METHODS,
    Detection,
    DetectorError,
    Params,
    PeriodEstimate,
    detect_period,
    run_method,
)
from evaluate import (
    NabResult,
    nab_score,
    point_counts,
    runs,
    windows_to_index,
    windows_to_truth,
)
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

# Tried in order and coalesced: Polars infers one format from the first value it
# sees, so a file mixing "...Z", "...+02:00" and naive timestamps would otherwise
# parse only the rows that happen to match the first one.
_TS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%.f%#z",
    "%Y-%m-%d %H:%M:%S%.f%#z",
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%d",
)
# Longest run of missing samples we're willing to fill by interpolation, as a
# share of the regular grid. Beyond this the file is mostly holes and any
# "fix" would be inventing the data being analysed.
MAX_FILLED_SHARE = 0.25
# Plotly's SVG scatter gets sluggish past this many points; switch to WebGL.
_WEBGL_ABOVE = 20_000


class SeriesError(ValueError):
    """The file can't be turned into a usable series."""


@dataclass
class DataQuality:
    rows_read: int = 0
    null_timestamp: int = 0
    unparseable_timestamp: int = 0
    null_value: int = 0
    non_numeric_value: int = 0
    non_finite_value: int = 0
    duplicate_timestamps: int = 0
    out_of_order: bool = False
    rows_used: int = 0
    step_seconds: float | None = None
    gaps_filled: int = 0  # samples that were missing and got interpolated
    irregular: bool = False  # timestamps don't sit on a regular grid
    notes: list[str] = field(default_factory=list)


@dataclass
class TimeSeries:
    name: str
    timestamps: np.ndarray  # datetime64[us], strictly increasing
    values: np.ndarray
    filled: np.ndarray  # True for interpolated (not observed) samples
    quality: DataQuality


# --------------------------------------------------------------------- loading


def _parse_timestamps(column: str) -> pl.Expr:
    text = pl.col(column).str.strip_chars()
    parsed = [
        text.str.to_datetime(format=fmt, time_zone="UTC", strict=False)
        for fmt in _TS_FORMATS
    ]
    return pl.coalesce(parsed).dt.replace_time_zone(None)


def load_series(
    path: Path, time_col: str = "timestamp", value_col: str = "value"
) -> TimeSeries:
    """Read a two-column CSV into a clean, regular, sorted series.

    Bad rows are dropped and counted (each row under the first check it fails),
    duplicate timestamps keep the first occurrence, rows are sorted by time, and
    if the samples sit on a regular grid with holes, the holes are filled by
    linear interpolation and marked so detectors never flag an invented point.
    """
    try:
        raw = pl.read_csv(path, infer_schema_length=0)
    except pl.exceptions.NoDataError as exc:
        raise SeriesError(f"{path}: file is empty") from exc
    for col in (time_col, value_col):
        if col not in raw.columns:
            raise SeriesError(
                f"{path}: no column named {col!r} (found: {', '.join(raw.columns)})"
            )

    q = DataQuality(rows_read=raw.height)
    df = raw.select(
        pl.col(time_col).alias("_raw_ts"),
        _parse_timestamps(time_col).alias("ts"),
        pl.col(value_col).str.strip_chars().alias("_raw_val"),
        pl.col(value_col).str.strip_chars().cast(pl.Float64, strict=False).alias("val"),
    )
    blank_ts = df["_raw_ts"].is_null() | (df["_raw_ts"].str.strip_chars() == "")
    q.null_timestamp = int(blank_ts.sum())
    bad_ts = df["ts"].is_null() & ~blank_ts
    q.unparseable_timestamp = int(bad_ts.sum())
    df = df.filter(~blank_ts & ~bad_ts)

    blank_val = df["_raw_val"].is_null() | (df["_raw_val"] == "")
    q.null_value = int(blank_val.sum())
    non_numeric = df["val"].is_null() & ~blank_val
    q.non_numeric_value = int(non_numeric.sum())
    df = df.filter(~blank_val & ~non_numeric)
    finite = df["val"].is_finite()
    q.non_finite_value = int((~finite).sum())
    df = df.filter(finite)

    before = df.height
    if before > 1:
        step_us = np.diff(
            df["ts"].dt.cast_time_unit("us").to_numpy().astype("datetime64[us]")
        )
        q.out_of_order = bool((step_us.astype(np.int64) < 0).any())
    df = df.unique(subset="ts", keep="first", maintain_order=True).sort("ts")
    q.duplicate_timestamps = before - df.height
    if df.height == 0:
        raise SeriesError(f"{path}: no usable rows (see counts: {_quality_summary(q)})")

    ts = df["ts"].dt.cast_time_unit("us").to_numpy().astype("datetime64[us]")
    vals = df["val"].to_numpy().astype(float)
    filled = np.zeros(len(vals), dtype=bool)

    if len(ts) >= 3:
        diffs = np.diff(ts).astype("timedelta64[us]").astype(np.int64)
        step = int(np.median(diffs))
        q.step_seconds = step / 1e6
        pos = (ts - ts[0]).astype("timedelta64[us]").astype(np.int64) / step
        on_grid = bool(np.all(np.abs(pos - np.round(pos)) < 0.01))
        if not on_grid:
            q.irregular = True
            q.notes.append(
                "timestamps are not on a regular grid; detectors treat samples as equally spaced"
            )
        else:
            grid_n = round(pos[-1]) + 1
            missing = grid_n - len(ts)
            if missing:
                if missing / grid_n > MAX_FILLED_SHARE:
                    q.irregular = True
                    q.notes.append(
                        f"{missing} of {grid_n} grid samples are missing (> {MAX_FILLED_SHARE:.0%}); not filling"
                    )
                else:
                    idx = np.round(pos).astype(int)
                    full = np.interp(np.arange(grid_n), idx, vals)
                    mask = np.ones(grid_n, dtype=bool)
                    mask[idx] = False
                    ts = ts[0] + np.arange(grid_n) * np.timedelta64(step, "us")
                    vals, filled = full, mask
                    q.gaps_filled = missing
                    q.notes.append(
                        f"{missing} missing sample(s) filled by linear interpolation; never flagged"
                    )
    q.rows_used = int((~filled).sum())
    return TimeSeries(path.name, ts, vals, filled, q)


def _quality_summary(q: DataQuality) -> str:
    return (
        f"{q.null_timestamp} missing timestamps, {q.unparseable_timestamp} unparseable, "
        f"{q.null_value} missing values, {q.non_numeric_value} non-numeric, "
        f"{q.non_finite_value} non-finite"
    )


def load_windows(
    labels_path: Path, series_name: str, key: str | None = None
) -> list[tuple[str, str]]:
    """Labelled anomaly windows for a series from a NAB-style labels JSON.

    The JSON maps a series path ("realKnownCause/nyc_taxi.csv") to a list of
    [start, end] timestamp pairs. Without an explicit `key`, the entry whose
    file name matches the series' file name is used.
    """
    try:
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SeriesError(f"cannot read labels {labels_path}: {exc}") from exc
    if not isinstance(labels, dict):
        raise SeriesError(
            f"{labels_path}: expected a JSON object mapping series -> windows"
        )
    if key is None:
        matches = [k for k in labels if k.rsplit("/", 1)[-1] == series_name]
        if len(matches) != 1:
            raise SeriesError(
                f"{labels_path}: {len(matches)} entries match {series_name!r}; pass --label-key"
            )
        key = matches[0]
    if key not in labels:
        raise SeriesError(f"{labels_path}: no entry {key!r}")
    try:
        return [(str(a), str(b)) for a, b in labels[key]]
    except (TypeError, ValueError) as exc:
        raise SeriesError(
            f"{labels_path}: entry {key!r} is not a list of [start, end] pairs"
        ) from exc


# ------------------------------------------------------------------- analysis


@dataclass
class Analysis:
    series: TimeSeries
    params: Params
    period_source: str  # "auto", "given", or "none"
    periods: list[int]
    period_estimate: PeriodEstimate | None
    detections: dict[str, Detection] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)  # method -> reason


def analyze(
    ts: TimeSeries,
    methods: list[str] | None = None,
    periods: str | list[int] = "auto",
    params: Params | None = None,
) -> Analysis:
    """Run the requested detectors. `periods` is "auto", "none", or a list of ints."""
    params = params or Params()
    methods = list(methods or METHODS)
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SeriesError(
            f"unknown method(s): {', '.join(unknown)}; choose from {', '.join(METHODS)}"
        )

    estimate: PeriodEstimate | None = None
    if isinstance(periods, list):
        plist, source = periods, "given"
    elif periods == "none":
        plist, source = [], "none"
    else:
        source = "auto"
        estimate = detect_period(ts.values) if len(ts.values) >= 12 else None
        plist = [estimate.period] if estimate else []

    out = Analysis(ts, params, source, plist, estimate)
    for m in methods:
        if m in STL_METHODS and not plist:
            out.skipped[m] = (
                "no seasonal period given"
                if source == "none"
                else "no seasonality detected (pass --period N to force one)"
            )
            continue
        try:
            det = run_method(m, ts.values, periods=plist, params=params)
        except DetectorError as exc:
            out.skipped[m] = str(exc)
            continue
        det.flags = det.flags & ~ts.filled  # never flag an interpolated sample
        out.detections[m] = det
    return out


@dataclass
class Evaluation:
    windows: list[tuple[int, int]]
    point_precision: float
    point_recall: float
    point_f1: float
    nab: NabResult


def evaluate_detection(
    ts: TimeSeries, det: Detection, windows: list[tuple[str, str]]
) -> Evaluation:
    idx = windows_to_index(ts.timestamps, windows)
    truth = windows_to_truth(len(ts.values), idx)
    pc = point_counts(det.flags, truth)
    return Evaluation(idx, pc.precision, pc.recall, pc.f1, nab_score(det.flags, idx))


# --------------------------------------------------------------------- report


def _esc(value: object) -> str:
    return html.escape(str(value))


def _fmt_params(det: Detection) -> str:
    keep = {k: v for k, v in det.params.items() if k != "n_outliers"}
    return ", ".join(f"{k}={v}" for k, v in keep.items())


def _quality_table(ts: TimeSeries, an: Analysis) -> str:
    q = ts.quality
    rows: list[tuple[str, object]] = [
        ("Rows read", q.rows_read),
        ("Dropped: missing timestamp", q.null_timestamp),
        ("Dropped: unparseable timestamp", q.unparseable_timestamp),
        ("Dropped: missing value", q.null_value),
        ("Dropped: non-numeric value", q.non_numeric_value),
        ("Dropped: NaN / infinite value", q.non_finite_value),
        ("Dropped: duplicate timestamp", q.duplicate_timestamps),
        ("Rows were out of time order (sorted)", "yes" if q.out_of_order else "no"),
        ("Samples analysed", len(ts.values)),
        ("Sampling step", f"{q.step_seconds:g} s" if q.step_seconds else "-"),
        ("Gaps filled by interpolation", q.gaps_filled),
        ("First / last", f"{ts.timestamps[0]} / {ts.timestamps[-1]}"),
    ]
    if an.period_source == "auto":
        est = an.period_estimate
        rows.append(
            (
                "Seasonal period (auto)",
                f"{est.period} (ACF {est.acf:.2f})" if est else "none found",
            )
        )
    elif an.period_source == "given":
        rows.append(("Seasonal period (given)", ", ".join(map(str, an.periods))))
    body = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in rows)
    notes = "".join(f"<li>{_esc(n)}</li>" for n in q.notes)
    return f"<table>{body}</table>" + (
        f"<ul class='note'>{notes}</ul>" if notes else ""
    )


def _results_table(an: Analysis, windows: list[tuple[str, str]] | None) -> str:
    ts = an.series
    n = len(ts.values)
    head = ["Method", "Parameters", "Flagged", "Events", "Flag rate"]
    if windows is not None:
        head += ["Windows hit", "Point P / R / F1", "NAB score", "False-alarm events"]
    rows = []
    for name, det in an.detections.items():
        cells = [
            _esc(name),
            _esc(_fmt_params(det)),
            str(det.count),
            str(len(runs(det.flags))),
            f"{100 * det.count / n:.2f}%",
        ]
        if windows is not None:
            ev = evaluate_detection(ts, det, windows)
            nab = "n/a" if np.isnan(ev.nab.normalized) else f"{ev.nab.normalized:.1f}"
            cells += [
                f"{ev.nab.windows_hit}/{ev.nab.windows_total}",
                f"{ev.point_precision:.2f} / {ev.point_recall:.2f} / {ev.point_f1:.2f}",
                nab,
                str(ev.nab.false_positive_events),
            ]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    for name, why in an.skipped.items():
        span = len(head) - 1
        rows.append(
            f"<tr class='skip'><td>{_esc(name)}</td><td colspan='{span}'>skipped: {_esc(why)}</td></tr>"
        )
    th = "".join(f"<th>{h}</th>" for h in head)
    return f"<table class='grid'><thead><tr>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _figure(
    an: Analysis, windows: list[tuple[str, str]] | None, visible: list[str]
) -> go.Figure:
    ts = an.series
    x = ts.timestamps.astype("datetime64[ms]").astype(object)
    scatter = go.Scattergl if len(x) > _WEBGL_ABOVE else go.Scatter
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.06,
        subplot_titles=(
            "Series and flagged points",
            "Anomaly score of the first shown method",
        ),
    )
    fig.add_trace(
        scatter(
            x=x,
            y=ts.values,
            mode="lines",
            name="series",
            line={"color": "#374151", "width": 1},
        ),
        row=1,
        col=1,
    )
    if windows:
        for lo, hi in windows_to_index(ts.timestamps, windows):
            fig.add_vrect(
                x0=x[lo],
                x1=x[hi],
                fillcolor="#f59e0b",
                opacity=0.18,
                line_width=0,
                layer="below",
                row=1,
                col=1,
            )
    palette = [
        "#dc2626",
        "#2563eb",
        "#059669",
        "#7c3aed",
        "#d97706",
        "#0891b2",
        "#be185d",
        "#4d7c0f",
        "#475569",
        "#a16207",
    ]
    for i, (name, det) in enumerate(an.detections.items()):
        idx = np.flatnonzero(det.flags)
        show = True if name in visible else "legendonly"
        fig.add_trace(
            go.Scatter(
                x=[x[j] for j in idx],
                y=ts.values[idx],
                mode="markers",
                name=f"{name} ({len(idx)})",
                marker={
                    "color": palette[i % len(palette)],
                    "size": 7,
                    "symbol": "circle-open",
                    "line": {"width": 2},
                },
                visible=show,
            ),
            row=1,
            col=1,
        )
    first = next((d for n, d in an.detections.items() if n in visible), None)
    if first is not None:
        finite = np.where(np.isfinite(first.scores), first.scores, np.nan)
        fig.add_trace(
            scatter(
                x=x,
                y=finite,
                mode="lines",
                name=f"{first.method} score",
                line={"color": "#2563eb", "width": 1},
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        if first.threshold is not None:
            fig.add_hline(
                y=first.threshold, line_dash="dash", line_color="#dc2626", row=2, col=1
            )
        if first.components is not None:
            fit = first.components["trend"] + first.components["seasonal"]
            fig.add_trace(
                scatter(
                    x=x,
                    y=fit,
                    mode="lines",
                    name="STL trend + seasonal",
                    line={"color": "#059669", "width": 1, "dash": "dot"},
                    visible="legendonly",
                ),
                row=1,
                col=1,
            )
    fig.update_layout(
        height=680,
        margin={"t": 50, "r": 20, "l": 60, "b": 40},
        legend={"orientation": "h", "y": -0.12},
        hovermode="x unified",
    )
    return fig


def build_report(an: Analysis, windows: list[tuple[str, str]] | None = None) -> str:
    ts = an.series
    name = ts.name
    default_visible = [m for m in ("stl-mad", "mad") if m in an.detections][:1] or list(
        an.detections
    )[:1]
    if an.detections:
        chart = f"<script>{get_plotlyjs()}</script>" + _figure(
            an, windows, default_visible
        ).to_html(full_html=False, include_plotlyjs=False)
        chart_note = (
            "Click a method in the legend to show or hide its flagged points. "
            + ("Shaded bands are the labelled anomaly windows. " if windows else "")
        )
    else:
        chart = "<p class='warn'>No detector produced a result; see the skipped list above.</p>"
        chart_note = ""
    win_note = (
        "<p class='note'>Point P/R/F1 treat every sample inside a labelled window as an anomaly. "
        "The NAB score (100 = perfect, 0 = flags nothing) collapses each run of flags to one "
        "detection and rewards a hit early in a window.</p>"
        if windows is not None
        else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anomalies: {_esc(name)}</title>
<style>
body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; color: #111827; }}
table {{ border-collapse: collapse; }}
th, td {{ text-align: left; padding: 2px 14px 2px 0; vertical-align: top; }}
table.grid th {{ border-bottom: 1px solid #d1d5db; }}
tr.skip td {{ color: #6b7280; font-style: italic; }}
.note {{ color: #4b5563; }}
.warn {{ color: #b45309; font-weight: 600; }}
</style></head><body>
<h1>Anomalies: {_esc(name)}</h1>
<h2>Data quality</h2>
{_quality_table(ts, an)}
<h2>Detector results</h2>
{_results_table(an, windows)}
{win_note}
<h2>Chart</h2>
<p class="note">{chart_note}</p>
{chart}
</body></html>
"""


# ------------------------------------------------------------------------ CLI


def _parse_periods(text: str) -> str | list[int]:
    t = text.strip().lower()
    if t in ("auto", "none"):
        return t
    try:
        vals = [int(part) for part in t.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--period must be 'auto', 'none', or integers like 48 or 48,336"
        ) from exc
    if any(v < 2 for v in vals):
        raise argparse.ArgumentTypeError("periods must be >= 2")
    return vals


def _write_flags(path: Path, an: Analysis) -> None:
    cols: dict[str, object] = {
        "timestamp": [str(t) for t in an.series.timestamps],
        "value": an.series.values,
        "interpolated": an.series.filled,
    }
    for name, det in an.detections.items():
        cols[name.replace("-", "_")] = det.flags
    pl.DataFrame(cols).write_csv(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Statistical anomaly detection (Z-score, IQR, MAD, Generalized ESD, rolling, STL) on a time-series CSV."
    )
    p.add_argument(
        "csv", type=Path, help="CSV with a timestamp column and a numeric value column"
    )
    p.add_argument("-o", "--output", type=Path, default=Path("anomaly_report.html"))
    p.add_argument("--time-col", default="timestamp")
    p.add_argument("--value-col", default="value")
    p.add_argument(
        "--methods",
        default="all",
        help=f"comma-separated subset of: {', '.join(METHODS)} (default all)",
    )
    p.add_argument(
        "--period",
        type=_parse_periods,
        default="auto",
        help="'auto' (default), 'none', or period(s) in samples, e.g. 48 or 48,336 (MSTL)",
    )
    p.add_argument("--z-threshold", type=float, default=3.0)
    p.add_argument("--iqr-k", type=float, default=1.5)
    p.add_argument("--mad-threshold", type=float, default=3.5)
    p.add_argument(
        "--window",
        type=int,
        default=100,
        help="look-back window for the rolling methods",
    )
    p.add_argument(
        "--alpha", type=float, default=0.05, help="Generalized ESD significance level"
    )
    p.add_argument(
        "--max-outliers",
        type=int,
        default=None,
        help="Generalized ESD upper bound (default 5%% of n)",
    )
    p.add_argument(
        "--stl-seasonal",
        type=int,
        default=25,
        help="STL seasonal smoother window (odd)",
    )
    p.add_argument(
        "--no-stl-robust", action="store_true", help="plain (non-robust) STL fits"
    )
    p.add_argument(
        "--labels",
        type=Path,
        help="NAB-style labels JSON to score the detectors against",
    )
    p.add_argument(
        "--label-key", help="entry in the labels JSON (default: match by file name)"
    )
    p.add_argument(
        "--flags-out",
        type=Path,
        help="also write a CSV with one flag column per method",
    )
    args = p.parse_args(argv)

    methods = (
        list(METHODS)
        if args.methods == "all"
        else [m.strip() for m in args.methods.split(",") if m.strip()]
    )
    params = Params(
        z_threshold=args.z_threshold,
        iqr_k=args.iqr_k,
        mad_threshold=args.mad_threshold,
        window=args.window,
        alpha=args.alpha,
        max_outliers=args.max_outliers,
        stl_seasonal=args.stl_seasonal,
        stl_robust=not args.no_stl_robust,
    )
    try:
        ts = load_series(args.csv, args.time_col, args.value_col)
        windows = (
            load_windows(args.labels, args.csv.name, args.label_key)
            if args.labels
            else None
        )
        an = analyze(ts, methods, args.period, params)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(an, windows), encoding="utf-8")
    if args.flags_out:
        args.flags_out.parent.mkdir(parents=True, exist_ok=True)
        _write_flags(args.flags_out, an)
    print(
        f"{len(ts.values)} samples ({ts.quality.rows_read} rows read); period: {an.periods or 'none'} ({an.period_source})"
    )
    for name, det in an.detections.items():
        line = (
            f"  {name:12s} flagged {det.count:5d} in {len(runs(det.flags)):4d} events"
        )
        if windows is not None:
            ev = evaluate_detection(ts, det, windows)
            line += f"   windows {ev.nab.windows_hit}/{ev.nab.windows_total}  NAB {ev.nab.normalized:6.1f}"
        print(line)
    for name, why in an.skipped.items():
        print(f"  {name:12s} skipped: {why}")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
