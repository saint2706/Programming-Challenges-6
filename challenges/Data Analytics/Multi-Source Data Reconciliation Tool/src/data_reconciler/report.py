"""Write the results: CSV/JSON tables and one self-contained HTML report.

The report embeds Plotly's JavaScript once and nothing else from outside, so it opens offline. Every
value that came from the data (names, codes, notes) is HTML-escaped: a source file must not be able to
inject markup into the report.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.offline import get_plotlyjs

from data_reconciler import compare as cmp
from data_reconciler.reconcile import VERDICTS, Reconciliation, group_rates, unmatched

COLORS = {
    cmp.MATCH: "#2f9e5b",
    cmp.FORMAT_ONLY: "#7a8fa6",
    cmp.WITHIN: "#6cb0f0",
    cmp.BOTH_MISSING: "#c9d1dc",
    cmp.UNVERIFIABLE: "#9a9a9a",
    cmp.NEAR: "#e6c229",
    cmp.LEFT_MISSING: "#f0a43a",
    cmp.RIGHT_MISSING: "#d97706",
    cmp.INVALID: "#8b1e1e",
    cmp.MISMATCH: "#d6336c",
}
STATUS_COLORS = {
    "matched": "#2f9e5b",
    "only_left": "#f0a43a",
    "only_right": "#d97706",
    "duplicate_key": "#d6336c",
    "unkeyed": "#9a9a9a",
}
STYLE = """
:root{--bg:#fff;--fg:#1c2430;--muted:#5b6678;--card:#f5f7fa;--line:#c9d1dc}
@media (prefers-color-scheme: dark){:root{--bg:#10151c;--fg:#e6ebf2;--muted:#9aa6b8;--card:#19212b;--line:#334052}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1150px;padding:16px}
h1{font-size:1.4rem;margin:.2em 0}h2{font-size:1.1rem;margin:1.8em 0 .4em}p.note{color:var(--muted);margin:.2em 0 .8em}
.cards{display:flex;flex-wrap:wrap;gap:10px}.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 14px}
.card b{display:block;font-size:1.25rem}.card span{color:var(--muted);font-size:.8rem}
.plot{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px;margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:.88rem}td,th{padding:3px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}
.tablewrap{overflow:auto;max-height:420px;border:1px solid var(--line);border-radius:6px}
"""
LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#8a96a8"},
    "margin": {"l": 10, "r": 10, "t": 10, "b": 50},
    "legend": {"orientation": "h", "y": -0.35},
}


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.3f}" if abs(value) < 100 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def table_html(frame: pl.DataFrame, limit: int = 15) -> str:
    if frame.is_empty():
        return "<p class=note>none</p>"
    head = "".join(f"<th>{escape(c)}</th>" for c in frame.columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td{' class=n' if isinstance(v, int | float) else ''}>{escape(_fmt(v))}</td>"
            for v in row
        )
        + "</tr>"
        for row in frame.head(limit).iter_rows()
    )
    return f"<div class=tablewrap><table><tr>{head}</tr>{body}</table></div>"


def _plot(fig: go.Figure, height: int) -> str:
    fig.update_layout(height=height, **LAYOUT)
    return f"<div class=plot>{fig.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False})}</div>"


def status_figure(rec: Reconciliation) -> go.Figure:
    fig = go.Figure()
    sides = (("left", rec.summary["left"]), ("right", rec.summary["right"]))
    for status, color in STATUS_COLORS.items():
        counts = [info["status"].get(status, 0) for _, info in sides]
        if any(counts):
            fig.add_bar(
                y=[info["name"] for _, info in sides],
                x=counts,
                name=status,
                orientation="h",
                marker_color=color,
            )
    fig.update_layout(barmode="stack", xaxis_title="rows in scope")
    return fig


def field_figure(rec: Reconciliation) -> go.Figure:
    fig = go.Figure()
    fields = list(rec.summary["fields_same_entity"])
    for category in reversed(cmp.CATEGORIES):
        counts = [rec.summary["fields_same_entity"][f][category] for f in fields]
        if any(counts):
            fig.add_bar(
                y=fields,
                x=counts,
                name=category,
                orientation="h",
                marker_color=COLORS[category],
            )
    fig.update_layout(
        barmode="stack", xaxis_title="matched pairs", yaxis={"autorange": "reversed"}
    )
    return fig


def distance_figure(rec: Reconciliation, field: str) -> go.Figure:
    km = rec.differences.filter(
        (pl.col("field") == field)
        & (pl.col("verdict") != "different_entity")
        & (pl.col("metric") > 0)
    )["metric"]
    # bin log10(km): Plotly bins linearly even on a log axis, which would pile everything into one bar
    fig = go.Figure(
        go.Histogram(
            x=np.log10(km.to_numpy()).tolist(),
            nbinsx=30,
            marker_color=COLORS[cmp.MISMATCH],
        )
    )
    low, high = (
        (np.floor(np.log10(km.min())), np.ceil(np.log10(km.max())))
        if km.len()
        else (0, 1)
    )
    ticks = [float(t) for t in np.arange(low, high + 1)]
    fig.update_layout(
        xaxis={
            "tickvals": ticks,
            "ticktext": [f"{10**t:,.4g}" for t in ticks],
            "title": f"{field}: km apart (log scale)",
        },
        yaxis_title="pairs",
    )
    return fig


def build_html(rec: Reconciliation, title: str | None = None) -> str:
    s, cfg = rec.summary, rec.config
    title = title or f"{cfg.left.name} vs {cfg.right.name}"
    cards = [
        (f"{s['left']['rows']:,}", f"{cfg.left.name} rows in scope"),
        (f"{s['right']['rows']:,}", f"{cfg.right.name} rows in scope"),
        (
            f"{s['matched']:,}",
            "matched pairs ("
            + ", ".join(f"{n:,} by {k}" for k, n in s["matched_by_key"].items())
            + ")",
        ),
        *((f"{s['verdicts'][v]:,}", f"pairs {v.replace('_', ' ')}") for v in VERDICTS),
    ]
    parts = [
        f"<h1>{escape(title)}</h1>",
        "<div class=cards>"
        + "".join(
            f"<div class=card><b>{escape(v)}</b><span>{escape(k)}</span></div>"
            for v, k in cards
        )
        + "</div>",
        "<h2>Where every row went</h2><p class=note>Each in-scope row ends in exactly one place.</p>",
        _plot(status_figure(rec), 200),
        "<h2>Field by field</h2><p class=note>Matched pairs whose key was not reused for a different entity.</p>",
        _plot(field_figure(rec), 60 + 38 * len(cfg.fields)),
    ]
    for spec in cfg.fields:
        if (
            spec.compare == "geo"
            and rec.differences.filter(pl.col("field") == spec.name).height
        ):
            parts.append(
                f"<h2>How far apart: {escape(spec.name)}</h2><p class=note>Pairs whose location differs by more than the {spec.tolerance_km:g} km tolerance, excluding keys reused for a different entity.</p>"
            )
            parts.append(_plot(distance_figure(rec, spec.name), 260))
    parts.append(
        f"<h2>Keys reused for a different entity ({s['verdicts']['different_entity']:,})</h2>"
    )
    parts.append(
        f"<p class=note>Pairs where {escape(', '.join(cfg.identity))} all disagree: the same code now names something else.</p>"
    )
    reused = rec.pairs.filter(pl.col("verdict") == "different_entity")
    show = [
        "key",
        *[f"{n}__{side}" for n in cfg.identity for side in ("left", "right")],
    ]
    parts.append(
        table_html(reused.select(show), 30)
        if cfg.identity
        else "<p class=note>identity not configured</p>"
    )
    for name, xw in rec.crosswalks.items():
        parts.append(
            f"<h2>Learned crosswalk: {escape(name)}</h2><p class=note>The right value each left value maps to, learned from the pairs. Mismatches are pairs that disagree with it.</p>"
        )
        parts.append(table_html(xw, 12))
    if cfg.group_by:
        parts.append(
            f"<h2>Difference rate by {escape(cfg.group_by)}</h2><p class=note>Groups with 20+ matched pairs, highest share of pairs with a difference first.</p>"
        )
        parts.append(table_html(group_rates(rec), 15))
    for side, name in (("left", cfg.left.name), ("right", cfg.right.name)):
        un = unmatched(rec, side)
        parts.append(f"<h2>Unmatched in {escape(name)} ({un.height:,})</h2>")
        counts = (
            un.group_by("status")
            .agg(rows=pl.len(), found_out_of_scope=pl.col("found_out_of_scope").sum())
            .sort("rows", descending=True)
        )
        parts.append(
            "<p class=note>found_out_of_scope: the other source does have this key, but the row is outside its scope.</p>"
            + table_html(counts)
        )
    body = "".join(parts)
    return (
        f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{escape(title)}</title><style>{STYLE}</style><script>{get_plotlyjs()}</script></head><body>{body}</body></html>"
    )


def write_outputs(rec: Reconciliation, out: Path) -> dict[str, Path]:
    """Write every artifact into ``out`` and return their paths."""
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "pairs": out / "pairs.csv",
        "differences": out / "differences.csv",
        "unmatched_left": out / "unmatched_left.csv",
        "unmatched_right": out / "unmatched_right.csv",
        "summary": out / "summary.json",
        "report": out / "report.html",
    }
    rec.pairs.write_csv(paths["pairs"])
    rec.differences.write_csv(paths["differences"])
    unmatched(rec, "left").write_csv(paths["unmatched_left"])
    unmatched(rec, "right").write_csv(paths["unmatched_right"])
    for name, frame in rec.crosswalks.items():
        frame.write_csv(out / f"crosswalk_{name}.csv")
    paths["summary"].write_text(
        json.dumps(rec.summary, indent=2) + "\n", encoding="utf-8"
    )
    paths["report"].write_text(build_html(rec), encoding="utf-8")
    return paths
