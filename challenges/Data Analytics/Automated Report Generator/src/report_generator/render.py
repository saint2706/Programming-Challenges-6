"""Writing a report as Markdown, a self-contained HTML page, and JSON findings.

The HTML page inlines plotly.js once, escapes every piece of text that comes from the data, and puts
each finding's evidence and checks behind a ``<details>`` so the prose stays short.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import asdict
from itertools import pairwise

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from report_generator.engine import SECTION_ORDER, Report
from report_generator.findings import Finding

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
INK, MUTED, GRID = "#1f2933", "#5f6b7a", "#e4e7eb"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def e(text) -> str:
    return html.escape(str(text), quote=True)


# ---- JSON -------------------------------------------------------------------------------------
def to_dict(report: Report) -> dict:
    checks = {
        fid: [
            {
                "name": c.name,
                "claimed": c.claimed,
                "recomputed": c.recomputed,
                "ok": c.ok,
            }
            for c in cs
        ]
        for fid, cs in report.checks.items()
    }
    n_checks = sum(len(v) for v in checks.values())
    return {
        "title": report.title,
        "overview": report.overview.to_json(),
        "findings": [f.to_json() for f in report.findings],
        "notes": [f.to_json() for f in report.notes],
        "candidates": report.candidates,
        "dropped": report.dropped,
        "verification": {
            "ran": report.verified,
            "engine": "DuckDB SQL aggregation plus scipy/statsmodels recomputation",
            "checks": n_checks,
            "failed": sum(1 for v in checks.values() for c in v if not c["ok"]),
            "by_finding": checks,
        },
        "columns": {c.name: {"role": c.role, "n_unique": c.n_unique, "n_missing": c.n_missing} for c in report.profile.columns.values()},
        "config": asdict(report.config),
    }  # fmt: skip


def clean(obj):
    """JSON-safe copy: numpy scalars become Python numbers, NaN and infinity become null."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def to_json(report: Report) -> str:
    return (
        json.dumps(
            clean(to_dict(report)), indent=4, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    )


# ---- Markdown ---------------------------------------------------------------------------------
def _check_line(report: Report) -> str:
    if not report.verified:
        return (
            "Verification was skipped; numbers were audited against their facts only."
        )
    total = sum(len(v) for v in report.checks.values())
    kept = len(report.findings) + len(report.notes) + 1
    line = (
        f"Every number in this report was recomputed from the raw table with DuckDB and scipy: "
        f"{total} checks across {kept} statements"
    )
    if report.dropped:
        line += f"; {len(report.dropped)} candidate statements failed and were left out"
    return line + "."


def to_markdown(report: Report) -> str:
    out = [f"# {report.title}", "", report.overview.text, ""]
    if report.takeaways:
        out += ["## At a glance", ""]
        out += [f"- {f.text}" for f in report.takeaways]
        out.append("")
    for section in SECTION_ORDER:
        items = [f for f in report.findings if f.section == section]
        if not items:
            continue
        out += [f"## {section}", ""]
        for f in items:
            out += [f"- {f.text}", f"  *{f.detail}*" if f.detail else ""]
        out.append("")
    if report.notes:
        out += ["## Data notes", ""]
        out += [f"- {f.text}" for f in report.notes]
        out.append("")
    out += ["## How this was checked", "", _check_line(report), ""]
    return (
        "\n".join(line for line in out if line is not None)
        .replace("\n\n\n", "\n\n")
        .rstrip()
        + "\n"
    )


# ---- HTML -------------------------------------------------------------------------------------
CSS = """
:root { color-scheme: light; --ink: #1f2933; --muted: #5f6b7a; --line: #e4e7eb; --bg: #f6f7f9;
        --card: #ffffff; --accent: #0072B2; --ok: #0a7a4b; --bad: #b4540a; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 16px/1.6 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 900px; margin: 0 auto; padding: 28px 16px 72px; }
h1 { font-size: 1.7rem; margin: 0 0 6px; }
h2 { font-size: 1.1rem; margin: 28px 0 10px; color: var(--muted); text-transform: uppercase;
     letter-spacing: .04em; font-weight: 650; }
.lede { color: var(--muted); margin: 0 0 18px; }
.glance { background: var(--card); border: 1px solid var(--line); border-left: 4px solid var(--accent);
          border-radius: 10px; padding: 10px 18px; }
.glance li { margin: 8px 0; }
article { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
          padding: 14px 18px 8px; margin: 0 0 12px; }
article p { margin: 0 0 8px; font-size: 1.02rem; }
details { margin: 4px 0 8px; font-size: .86rem; color: var(--muted); }
summary { cursor: pointer; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; margin-top: 6px; }
th, td { text-align: left; padding: 3px 8px; border-bottom: 1px solid var(--line); }
.ok { color: var(--ok); } .bad { color: var(--bad); font-weight: 600; }
ul.notes { padding-left: 20px; } .check { background: #eef7f2; border: 1px solid #cfe7da;
  border-radius: 10px; padding: 10px 16px; color: #1d4d38; font-size: .92rem; }
@media (max-width: 600px) { main { padding: 14px 10px 48px; } article { padding: 10px 12px 4px; } }
"""


def _layout(fig: go.Figure, height: int = 260) -> go.Figure:
    fig.update_layout(
        height=height, margin={"l": 52, "r": 12, "t": 28, "b": 38}, paper_bgcolor="#fff", plot_bgcolor="#fff",
        font={"family": FONT, "size": 12, "color": INK}, showlegend=False,
        title={"text": "", "x": 0},
    )  # fmt: skip
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def figure(f: Finding) -> go.Figure | None:
    c = f.chart
    if not c:
        return None
    kind = c["kind"]
    fig = go.Figure()
    if kind == "series":
        fig.add_trace(
            go.Scatter(
                x=c["dates"],
                y=c["values"],
                mode="lines",
                line={"color": PALETTE[0], "width": 2},
                name="actual",
            )
        )
        if c.get("adjusted"):
            fig.add_trace(
                go.Scatter(
                    x=c["dates"],
                    y=c["adjusted"],
                    mode="lines",
                    line={"color": MUTED, "width": 1.4, "dash": "dot"},
                    name="seasonally adjusted",
                )
            )
        if c.get("trend"):
            fig.add_trace(
                go.Scatter(
                    x=c["dates"],
                    y=c["trend"],
                    mode="lines",
                    line={"color": PALETTE[1], "width": 2.4, "dash": "dash"},
                    name="trend",
                )
            )
        for m in c.get("marks", []):
            idx = c["dates"].index(m["date"])
            fig.add_trace(
                go.Scatter(
                    x=[m["date"]],
                    y=[c["values"][idx]],
                    mode="markers",
                    marker={
                        "color": PALETTE[1],
                        "size": 9,
                        "line": {"color": "#fff", "width": 1},
                    },
                    name=m["label"],
                )
            )
        fig.update_xaxes(type="date")
        fig.update_layout(showlegend=False)
    elif kind == "season":
        fig.add_trace(
            go.Bar(
                x=[str(x) for x in c["labels"]],
                y=c["values"],
                marker={
                    "color": [
                        PALETTE[1] if v == max(c["values"]) else PALETTE[0]
                        for v in c["values"]
                    ]
                },
            )
        )
        fig.update_yaxes(title_text="seasonal effect")
    elif kind == "bars":
        vals = c["values"]
        fig.add_trace(
            go.Bar(
                x=vals[::-1],
                y=[str(x) for x in c["labels"]][::-1],
                orientation="h",
                marker={"color": PALETTE[0]},
            )
        )
        if c.get("percent"):
            fig.update_xaxes(tickformat=".0%")
        fig.update_layout(
            margin={"l": 140, "r": 12, "t": 20, "b": 30},
            height=max(200, 38 + 24 * len(vals)),
        )
    elif kind == "lorenz":
        fig.add_trace(
            go.Scatter(
                x=c["x"],
                y=c["y"],
                mode="lines",
                line={"color": PALETTE[0], "width": 2.4},
                fill="tozeroy",
                fillcolor="rgba(0,114,178,.12)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"color": MUTED, "dash": "dash", "width": 1},
            )
        )
        fig.update_xaxes(title_text="share of groups (largest first)", tickformat=".0%")
        fig.update_yaxes(title_text="share of total", tickformat=".0%")
    elif kind == "scatter":
        fig.add_trace(
            go.Scattergl(
                x=c["x"],
                y=c["y"],
                mode="markers",
                marker={"size": 4, "color": "rgba(0,114,178,.35)"},
            )
        )
        fig.update_xaxes(title_text=e(c["xlabel"]))
        fig.update_yaxes(title_text=e(c["ylabel"]))
    elif kind == "hist":
        edges = c["edges"]
        mids = [(a + b) / 2 for a, b in pairwise(edges)]
        fig.add_trace(go.Bar(x=mids, y=c["counts"], marker={"color": PALETTE[0]}))
        fig.update_yaxes(title_text="rows")
    else:
        return None
    return _layout(fig, fig.layout.height or 260)


def _evidence(report: Report, f: Finding) -> str:
    rows = "".join(
        f"<tr><td>{e(c.name)}</td><td>{c.claimed:,.6g}</td><td>{c.recomputed:,.6g}</td>"
        f"<td class='{'ok' if c.ok else 'bad'}'>{'matches' if c.ok else 'DIFFERS'}</td></tr>"
        for c in report.checks.get(f.id, [])
    )
    table = (
        f"<table><thead><tr><th>Check</th><th>Report</th><th>Recomputed</th><th></th></tr></thead><tbody>{rows}</tbody></table>"
        if rows
        else ""
    )
    return (
        f"<details><summary>Evidence</summary><div>{e(f.detail)}</div>{table}</details>"
        if (f.detail or table)
        else ""
    )


def _article(report: Report, f: Finding, render) -> str:
    fig = figure(f)
    chart = render(fig) if fig is not None else ""
    return f"<article><p>{e(f.text)}</p>{chart}{_evidence(report, f)}</article>"


def to_html(report: Report) -> str:
    def render(fig: go.Figure) -> str:
        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"responsive": True, "displaylogo": False},
        )

    parts = [
        f"<h1>{e(report.title)}</h1>",
        f"<p class='lede'>{e(report.overview.text)}</p>",
    ]
    if report.takeaways:
        parts.append(
            "<h2>At a glance</h2><ul class='glance'>"
            + "".join(f"<li>{e(f.text)}</li>" for f in report.takeaways)
            + "</ul>"
        )
    for section in SECTION_ORDER:
        items = [f for f in report.findings if f.section == section]
        if items:
            parts.append(
                f"<h2>{e(section)}</h2>"
                + "".join(_article(report, f, render) for f in items)
            )
    if report.notes:
        parts.append(
            "<h2>Data notes</h2><article><ul class='notes'>"
            + "".join(f"<li>{e(f.text)}</li>" for f in report.notes)
            + "</ul></article>"
        )
    parts.append(
        f"<h2>How this was checked</h2><div class='check'>{e(_check_line(report))}</div>"
    )
    if report.dropped:
        parts.append(
            "<details><summary>Statements that failed a check</summary><ul>"
            + "".join(
                f"<li>{e(d['id'])}: {e(d['reason'])}</li>" for d in report.dropped
            )
            + "</ul></details>"
        )
    return (
        f"<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<link rel='icon' href='data:,'><title>{e(report.title)}</title><style>{CSS}</style>"
        f"<script>{get_plotlyjs()}</script></head><body><main>{''.join(parts)}</main></body></html>\n"
    )
