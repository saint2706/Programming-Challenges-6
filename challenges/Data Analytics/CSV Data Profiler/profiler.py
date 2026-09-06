"""CSV Data Profiler: auto-detect types/nulls/distributions, emit a self-contained HTML report.

Run with:  uv run --with polars --with plotly python profiler.py data.csv -o report.html
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl

TOP_N_DEFAULT = 10
SNIFF_SAMPLE_BYTES = 65536


def sniff_dialect(path: Path) -> tuple[str, str]:
    """Guess (encoding, delimiter) from a sample of the file.

    Tries utf-8 first (the common case) and falls back to latin-1, which
    never fails to decode -- it just may decode incorrectly for genuinely
    non-Latin-1 encodings, which is a reasonable last resort for a profiler.
    """
    raw = path.read_bytes()[:SNIFF_SAMPLE_BYTES]
    for encoding in ("utf-8", "latin-1"):
        try:
            sample = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 never fails
        sample = raw.decode("latin-1", errors="replace")

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    return encoding, delimiter


@dataclass
class ColumnProfile:
    name: str
    polars_dtype: str
    semantic_type: str
    null_count: int
    null_pct: float
    unique_count: int
    stats: dict[str, Any] = field(default_factory=dict)
    figure_html: str = ""


@dataclass
class DatasetProfile:
    path: Path
    n_rows: int
    n_cols: int
    duplicate_rows: int
    encoding: str
    delimiter: str
    columns: list[ColumnProfile]
    correlation_html: str = ""


def classify_semantic_type(series: pl.Series) -> str:
    dtype = series.dtype
    if dtype == pl.Boolean:
        return "boolean"
    if dtype.is_temporal():
        return "datetime"
    if dtype.is_numeric():
        return "numeric"

    non_null = series.drop_nulls()
    if non_null.is_empty():
        return "text"
    lowered = non_null.cast(pl.Utf8).str.to_lowercase()
    if set(lowered.unique().to_list()) <= {"true", "false", "0", "1", "yes", "no"}:
        return "boolean"

    n = non_null.len()
    n_unique = non_null.n_unique()
    if n_unique / n <= 0.5:
        return "categorical"
    return "text"


def iqr_outlier_count(values: pl.Series) -> int:
    non_null = values.drop_nulls()
    if non_null.len() < 4:
        return 0
    q1 = non_null.quantile(0.25)
    q3 = non_null.quantile(0.75)
    if q1 is None or q3 is None:
        return 0
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(non_null.filter((non_null < lower) | (non_null > upper)).len())


def _fig_to_div(fig: go.Figure) -> str:
    return pio.to_html(
        fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False}
    )


def profile_numeric(series: pl.Series) -> tuple[dict[str, Any], str]:
    non_null = series.drop_nulls()
    stats = {
        "min": non_null.min(),
        "max": non_null.max(),
        "mean": non_null.mean(),
        "median": non_null.median(),
        "stddev": non_null.std(),
        "outliers": iqr_outlier_count(series),
    }
    fig = go.Figure(
        go.Histogram(x=non_null.to_list(), nbinsx=30, marker_color="#4C78A8")
    )
    fig.update_layout(
        title=f"Distribution of {series.name}",
        margin=dict(l=40, r=20, t=40, b=30),
        height=280,
    )
    return stats, _fig_to_div(fig)


def profile_categorical(series: pl.Series, top_n: int) -> tuple[dict[str, Any], str]:
    counts = series.drop_nulls().value_counts(sort=True).head(top_n)
    labels = counts[series.name].cast(pl.Utf8).to_list()
    values = counts["count"].to_list()
    stats = {"top_values": list(zip(labels, values))}
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color="#F58518"))
    fig.update_layout(
        title=f"Top values of {series.name}",
        margin=dict(l=120, r=20, t=40, b=30),
        height=max(200, 24 * len(labels)),
        yaxis=dict(autorange="reversed"),
    )
    return stats, _fig_to_div(fig)


def profile_datetime(series: pl.Series) -> tuple[dict[str, Any], str]:
    non_null = series.drop_nulls()
    stats = {"min": non_null.min(), "max": non_null.max()}
    counts = non_null.dt.truncate("1mo").value_counts(sort=True).sort(series.name)
    fig = go.Figure(
        go.Scatter(
            x=counts[series.name].to_list(),
            y=counts["count"].to_list(),
            mode="lines+markers",
        )
    )
    fig.update_layout(
        title=f"Row count by month for {series.name}",
        margin=dict(l=40, r=20, t=40, b=30),
        height=280,
    )
    return stats, _fig_to_div(fig)


def profile_boolean(series: pl.Series) -> tuple[dict[str, Any], str]:
    counts = series.drop_nulls().cast(pl.Utf8).value_counts(sort=True)
    labels = counts[series.name].to_list()
    values = counts["count"].to_list()
    stats = {"counts": list(zip(labels, values))}
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#54A24B"))
    fig.update_layout(
        title=f"Value counts for {series.name}",
        margin=dict(l=40, r=20, t=40, b=30),
        height=260,
    )
    return stats, _fig_to_div(fig)


def build_correlation_heatmap(df: pl.DataFrame, numeric_cols: list[str]) -> str:
    if len(numeric_cols) < 2:
        return ""
    corr = df.select(numeric_cols).corr()
    matrix = corr.to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=numeric_cols,
            y=numeric_cols,
            colorscale="RdBu",
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in matrix],
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Correlation matrix (numeric columns)",
        margin=dict(l=80, r=20, t=40, b=60),
        height=400,
    )
    return _fig_to_div(fig)


def profile_dataset(path: Path, sep: str | None, top_n: int) -> DatasetProfile:
    encoding, delimiter = sniff_dialect(path)
    delimiter = sep or delimiter
    df = pl.read_csv(
        path,
        separator=delimiter,
        encoding=encoding,
        try_parse_dates=True,
        infer_schema_length=10_000,
    )

    n_rows, n_cols = df.shape
    duplicate_rows = int(df.is_duplicated().sum())

    columns: list[ColumnProfile] = []
    numeric_cols: list[str] = []
    for name in df.columns:
        series = df[name]
        semantic = classify_semantic_type(series)
        null_count = int(series.null_count())

        if semantic == "numeric":
            numeric_cols.append(name)
            stats, fig_html = profile_numeric(series)
        elif semantic == "datetime":
            stats, fig_html = profile_datetime(series)
        elif semantic == "boolean":
            stats, fig_html = profile_boolean(series)
        else:
            stats, fig_html = profile_categorical(series, top_n)

        columns.append(
            ColumnProfile(
                name=name,
                polars_dtype=str(series.dtype),
                semantic_type=semantic,
                null_count=null_count,
                null_pct=round(100 * null_count / n_rows, 2) if n_rows else 0.0,
                unique_count=int(series.n_unique()),
                stats=stats,
                figure_html=fig_html,
            )
        )

    correlation_html = build_correlation_heatmap(df, numeric_cols)

    return DatasetProfile(
        path=path,
        n_rows=n_rows,
        n_cols=n_cols,
        duplicate_rows=duplicate_rows,
        encoding=encoding,
        delimiter=delimiter,
        columns=columns,
        correlation_html=correlation_html,
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.3f}"
    return html.escape(str(value))


def render_column_section(col: ColumnProfile) -> str:
    rows = [
        ("Semantic type", col.semantic_type),
        ("Stored as", col.polars_dtype),
        ("Nulls", f"{col.null_count} ({col.null_pct}%)"),
        ("Unique values", col.unique_count),
    ]
    if col.semantic_type == "numeric":
        rows += [
            ("Min", _fmt(col.stats["min"])),
            ("Max", _fmt(col.stats["max"])),
            ("Mean", _fmt(col.stats["mean"])),
            ("Median", _fmt(col.stats["median"])),
            ("Std dev", _fmt(col.stats["stddev"])),
            ("Outliers (IQR rule)", col.stats["outliers"]),
        ]
    elif col.semantic_type == "datetime":
        rows += [
            ("Earliest", _fmt(col.stats["min"])),
            ("Latest", _fmt(col.stats["max"])),
        ]

    table_rows = "\n".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{_fmt(v) if not isinstance(v, str) else html.escape(v)}</td></tr>"
        for k, v in rows
    )
    return f"""
    <section class="column-card">
      <h3>{html.escape(col.name)}</h3>
      <table class="stats">{table_rows}</table>
      <div class="chart">{col.figure_html}</div>
    </section>
    """


CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 2rem;
       background: #0f1115; color: #e8e8e8; }
h1 { font-size: 1.6rem; }
.summary { display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0 2rem; }
.summary .stat { background: #1a1d24; border-radius: 8px; padding: 0.75rem 1.25rem; }
.summary .stat .value { font-size: 1.4rem; font-weight: 600; }
.summary .stat .label { font-size: 0.8rem; opacity: 0.7; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1.5rem; }
.column-card { background: #1a1d24; border-radius: 10px; padding: 1rem 1.25rem; }
table.stats { width: 100%; border-collapse: collapse; margin: 0.5rem 0; font-size: 0.9rem; }
table.stats th { text-align: left; opacity: 0.7; font-weight: 400; padding: 2px 0; width: 45%; }
table.stats td { text-align: right; padding: 2px 0; }
.correlation { margin-top: 2rem; }
"""


def render_report(profile: DatasetProfile) -> str:
    plotly_js = pio.to_html(go.Figure(), include_plotlyjs="inline", full_html=False)
    # Extract just the <script>...plotly.min.js...</script> block once, reused for every chart.
    script_start = plotly_js.index("<script")
    script_end = plotly_js.index("</script>") + len("</script>")
    plotly_bundle = plotly_js[script_start:script_end]

    columns_html = "\n".join(render_column_section(c) for c in profile.columns)
    correlation_section = (
        f'<section class="correlation"><h2>Correlations</h2>{profile.correlation_html}</section>'
        if profile.correlation_html
        else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CSV Profile: {html.escape(profile.path.name)}</title>
<style>{CSS}</style>
</head>
<body>
{plotly_bundle}
<h1>CSV Data Profile &mdash; {html.escape(profile.path.name)}</h1>
<div class="summary">
  <div class="stat"><div class="value">{profile.n_rows:,}</div><div class="label">rows</div></div>
  <div class="stat"><div class="value">{profile.n_cols}</div><div class="label">columns</div></div>
  <div class="stat"><div class="value">{profile.duplicate_rows:,}</div><div class="label">duplicate rows</div></div>
  <div class="stat"><div class="value">{html.escape(profile.encoding)}</div><div class="label">encoding</div></div>
  <div class="stat"><div class="value">{html.escape(profile.delimiter)}</div><div class="label">delimiter</div></div>
</div>
<div class="grid">
{columns_html}
</div>
{correlation_section}
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to the input CSV file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("report.html"),
        help="Output HTML report path",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=None,
        help="Force a delimiter instead of auto-detecting",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_DEFAULT,
        help="Top-N values for categorical columns",
    )
    args = parser.parse_args(argv)

    if not args.csv_path.exists():
        print(f"error: {args.csv_path} does not exist", file=sys.stderr)
        return 1

    profile = profile_dataset(args.csv_path, args.sep, args.top_n)
    args.output.write_text(render_report(profile), encoding="utf-8")
    print(
        f"Profiled {profile.n_rows:,} rows x {profile.n_cols} columns -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
