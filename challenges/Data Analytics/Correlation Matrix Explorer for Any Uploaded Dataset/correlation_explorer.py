"""Correlation Matrix Explorer: any CSV in, one interactive HTML heatmap out.

Run with:  uv run python correlation_explorer.py data.csv -o report.html
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from scipy import stats

MIN_PAIR_N = 3  # fewer complete pairs than this and r is undefined/meaningless
ALPHA_DEFAULT = 0.05
SNIFF_SAMPLE_BYTES = 65536
METHODS = ("pearson", "spearman")


class CorrelationError(ValueError):
    """Raised when a dataset can't produce a correlation matrix."""


@dataclass
class MatrixResult:
    method: str
    r: np.ndarray  # NaN where undefined
    p: np.ndarray  # NaN where undefined
    n: np.ndarray  # complete pairs per cell
    significant: np.ndarray  # bool, after BH correction across unique pairs


@dataclass
class Analysis:
    columns: list[str]
    skipped: dict[str, str]  # column -> reason
    n_rows: int
    alpha: float
    results: dict[str, MatrixResult]


def sniff_dialect(path: Path) -> tuple[str, str]:
    """Guess (encoding, delimiter); utf-8 first, latin-1 as never-failing fallback."""
    raw = path.read_bytes()[:SNIFF_SAMPLE_BYTES]
    for encoding in ("utf-8", "latin-1"):
        try:
            sample = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    return encoding, delimiter


def load_csv(path: Path, sep: str | None = None) -> pl.DataFrame:
    encoding, guessed = sniff_dialect(path)
    return pl.read_csv(
        path,
        separator=sep or guessed,
        encoding="utf8" if encoding == "utf-8" else "latin1",
        infer_schema_length=10000,
        null_values=["", "NA", "N/A", "NaN", "nan", "null", "NULL"],
    )


def select_numeric(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, str]]:
    """Keep numeric columns that can carry a correlation; record why others were dropped."""
    keep: list[pl.Series] = []
    skipped: dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        if s.dtype == pl.Boolean or not s.dtype.is_numeric():
            skipped[col] = "not numeric"
            continue
        s = s.cast(pl.Float64).fill_nan(None)
        non_null = s.drop_nulls()
        if non_null.len() < MIN_PAIR_N:
            skipped[col] = f"fewer than {MIN_PAIR_N} non-null values"
        elif non_null.n_unique() < 2:
            skipped[col] = "constant (zero variance)"
        else:
            keep.append(s)
    return (pl.DataFrame(keep) if keep else pl.DataFrame()), skipped


def benjamini_hochberg(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Boolean mask of hypotheses rejected by BH at FDR `alpha`. NaN p-values never reject."""
    p = np.asarray(pvalues, dtype=float)
    reject = np.zeros(p.shape, dtype=bool)
    valid = np.flatnonzero(~np.isnan(p))
    m = valid.size
    if m == 0:
        return reject
    order = valid[np.argsort(p[valid])]
    thresholds = alpha * np.arange(1, m + 1) / m
    passed = np.flatnonzero(p[order] <= thresholds)
    if passed.size:
        reject[order[: passed.max() + 1]] = True
    return reject


def _pair_stats(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float, int]:
    mask = ~(np.isnan(x) | np.isnan(y))
    n = int(mask.sum())
    if n < MIN_PAIR_N:
        return np.nan, np.nan, n
    xs, ys = x[mask], y[mask]
    # After pairwise deletion a column can become constant; r is undefined then.
    if np.ptp(xs) == 0 or np.ptp(ys) == 0:
        return np.nan, np.nan, n
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    res = fn(xs, ys)
    return float(res[0]), float(res[1]), n


def compute_matrix(
    df: pl.DataFrame, method: str, alpha: float = ALPHA_DEFAULT
) -> MatrixResult:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    data = df.to_numpy().astype(float)
    k = data.shape[1]
    r = np.full((k, k), np.nan)
    p = np.full((k, k), np.nan)
    n = np.zeros((k, k), dtype=int)
    for i in range(k):
        r[i, i], p[i, i] = 1.0, 0.0
        n[i, i] = int((~np.isnan(data[:, i])).sum())
        for j in range(i + 1, k):
            rij, pij, nij = _pair_stats(data[:, i], data[:, j], method)
            r[i, j] = r[j, i] = rij
            p[i, j] = p[j, i] = pij
            n[i, j] = n[j, i] = nij
    iu = np.triu_indices(k, 1)
    sig = np.zeros((k, k), dtype=bool)
    sig[iu] = benjamini_hochberg(p[iu], alpha)
    return MatrixResult(method, r, p, n, sig | sig.T)


def analyze(df: pl.DataFrame, alpha: float = ALPHA_DEFAULT) -> Analysis:
    numeric, skipped = select_numeric(df)
    if numeric.width < 2:
        raise CorrelationError(
            f"need at least 2 usable numeric columns, found {numeric.width}"
            + (f" (skipped: {skipped})" if skipped else "")
        )
    if numeric.height < MIN_PAIR_N:
        raise CorrelationError(
            f"need at least {MIN_PAIR_N} rows, found {numeric.height}"
        )
    results = {m: compute_matrix(numeric, m, alpha) for m in METHODS}
    return Analysis(numeric.columns, skipped, numeric.height, alpha, results)


def _fmt(v: float, digits: int = 3) -> str:
    return "n/a" if np.isnan(v) else f"{v:.{digits}f}"


def _fmt_p(v: float) -> str:
    if np.isnan(v):
        return "n/a"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def build_figure(analysis: Analysis) -> go.Figure:
    """One heatmap trace per method; buttons toggle which is visible."""
    # Column names are user data and Plotly renders hover/tick text as HTML: escape them.
    labels = [html.escape(c) for c in analysis.columns]
    k = len(labels)
    fig = go.Figure()
    for idx, method in enumerate(METHODS):
        res = analysis.results[method]
        hover = [
            [
                f"<b>{labels[i]}</b> vs <b>{labels[j]}</b><br>"
                f"{method} r = {_fmt(res.r[i, j])}<br>"
                f"p = {_fmt_p(res.p[i, j])}<br>n = {res.n[i, j]}"
                + ("<br>significant (BH)" if res.significant[i, j] else "")
                for j in range(k)
            ]
            for i in range(k)
        ]
        text = [
            [
                ("" if np.isnan(res.r[i, j]) else f"{res.r[i, j]:.2f}")
                + ("*" if res.significant[i, j] else "")
                for j in range(k)
            ]
            for i in range(k)
        ]
        fig.add_trace(
            go.Heatmap(
                z=[[None if np.isnan(v) else float(v) for v in row] for row in res.r],
                x=labels,
                y=labels,
                text=text,
                texttemplate="%{text}",
                hovertext=hover,
                hoverinfo="text",
                zmin=-1,
                zmax=1,
                zmid=0,
                colorscale="RdBu",
                reversescale=True,
                visible=idx == 0,
                colorbar={"title": "r"},
                name=method,
            )
        )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=f"Correlation matrix ({analysis.n_rows} rows; * = significant after BH, "
        f"FDR {analysis.alpha:g})",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.12,
                "buttons": [
                    {
                        "label": m.capitalize(),
                        "method": "update",
                        "args": [{"visible": [j == i for j in range(len(METHODS))]}],
                    }
                    for i, m in enumerate(METHODS)
                ],
            }
        ],
        height=max(500, 60 * k + 200),
    )
    return fig


def _pairs_table(analysis: Analysis) -> str:
    rows: list[str] = []
    cols = analysis.columns
    for m in METHODS:
        res = analysis.results[m]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if res.significant[i, j]:
                    rows.append(
                        f"<tr><td>{m}</td><td>{html.escape(cols[i])}</td>"
                        f"<td>{html.escape(cols[j])}</td><td>{_fmt(res.r[i, j])}</td>"
                        f"<td>{_fmt_p(res.p[i, j])}</td><td>{res.n[i, j]}</td></tr>"
                    )
    if not rows:
        return "<p>No pair is significant after multiple-comparison correction.</p>"
    return (
        "<table><thead><tr><th>method</th><th>a</th><th>b</th><th>r</th><th>p</th>"
        "<th>n</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def render_report(
    analysis: Analysis, title: str = "Correlation Matrix Explorer"
) -> str:
    fig = build_figure(analysis)
    plotly_js = pio.to_html(go.Figure(), include_plotlyjs="inline", full_html=False)
    chart = pio.to_html(
        fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False}
    )
    skipped = "".join(
        f"<li><code>{html.escape(c)}</code>: {html.escape(why)}</li>"
        for c, why in analysis.skipped.items()
    )
    skipped_html = f"<h2>Skipped columns</h2><ul>{skipped}</ul>" if skipped else ""
    safe_title = html.escape(title)
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{safe_title}</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.25rem .6rem}"
        "</style></head><body>"
        f"<h1>{safe_title}</h1>"
        f"<p>{analysis.n_rows} rows, {len(analysis.columns)} numeric columns. Pairwise deletion "
        "for missing values; hover a cell for r, p and the pair's n. Significance flags use "
        "Benjamini-Hochberg across all unique pairs (per method).</p>"
        f"{plotly_js}{chart}<h2>Significant pairs</h2>{_pairs_table(analysis)}{skipped_html}"
        "</body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="input CSV file")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("correlation_report.html")
    )
    parser.add_argument(
        "--sep", default=None, help="delimiter override (default: sniffed)"
    )
    parser.add_argument(
        "--alpha", type=float, default=ALPHA_DEFAULT, help="FDR level (default 0.05)"
    )
    args = parser.parse_args(argv)
    if not args.csv.is_file():
        print(f"error: {args.csv} not found", file=sys.stderr)
        return 2
    try:
        analysis = analyze(load_csv(args.csv, args.sep), args.alpha)
    except (CorrelationError, pl.exceptions.PolarsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    args.output.write_text(render_report(analysis, args.csv.name), encoding="utf-8")
    print(
        f"wrote {args.output} ({len(analysis.columns)} columns, {analysis.n_rows} rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
