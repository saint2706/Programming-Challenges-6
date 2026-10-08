"""Self-contained HTML report for a `CheckResult`.

Everything derived from the CSVs (column names, category values, messages, example
values) goes through `html.escape`. Plotly figures are embedded as JSON by Plotly
itself, which escapes `<`, `>` and `&`; a test checks that a hostile column name
cannot break out of the script block.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from data_quality.dq_monitor import CheckResult, coerce

_SEV_COLOR = {
    "critical": "#c62828",
    "warn": "#ef6c00",
    "info": "#0277bd",
    "ok": "#2e7d32",
}
_BASE, _BATCH = "#9e9e9e", "#d81b60"
_MAX_CHARTS = 8
_CHART_CHECKS = {
    "drift.numeric",
    "drift.categorical",
    "values.out_of_range",
    "values.unseen_categories",
    "values.missing_categories",
    "nulls.rate_shift",
}


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.01},
        height=340,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        template="plotly_white",
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def _volume_figure(result: CheckResult, baseline: dict[str, Any]) -> go.Figure:
    names = [b["name"] for b in baseline["batches"]]
    rows = [b["rows"] for b in baseline["batches"]]
    fig = go.Figure()
    fig.add_bar(x=names, y=rows, name="baseline batches", marker_color=_BASE)
    fig.add_bar(
        x=[result.batch], y=[result.rows], name="this batch", marker_color=_BATCH
    )
    return _layout(fig, "Rows per batch")


def _numeric_figure(spec: dict[str, Any], result: CheckResult) -> go.Figure | None:
    col, _ = coerce(result.frame[spec["name"]], spec["type"])
    x = col.drop_nulls().to_numpy().astype(float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    ref = np.asarray(spec["quantiles"])
    lo, hi = np.percentile(np.concatenate([ref, x]), [0.5, 99.5])
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    bins = {"start": float(lo), "end": float(hi), "size": float((hi - lo) / 30)}
    fig = go.Figure()
    fig.add_histogram(
        x=ref,
        xbins=bins,
        histnorm="probability",
        name="baseline",
        marker_color=_BASE,
        opacity=0.7,
    )
    fig.add_histogram(
        x=x,
        xbins=bins,
        histnorm="probability",
        name="this batch",
        marker_color=_BATCH,
        opacity=0.6,
    )
    fig.update_layout(barmode="overlay")
    fig.add_vrect(
        x0=spec["lower_bound"],
        x1=spec["upper_bound"],
        fillcolor="#2e7d32",
        opacity=0.05,
        line_width=0,
    )
    fig.update_xaxes(range=[float(lo), float(hi)])
    return _layout(
        fig,
        f"{spec['name']}: distribution vs baseline (middle 99%; green = learned range)",
    )


def _categorical_figure(spec: dict[str, Any], result: CheckResult) -> go.Figure | None:
    col, _ = coerce(result.frame[spec["name"]], spec["type"])
    vals = col.drop_nulls()
    if vals.len() == 0:
        return None
    vc = vals.value_counts()
    obs = dict(zip(vc[vals.name].to_list(), vc["count"].to_list(), strict=True))
    total = sum(spec["categories"].values())
    ref = {k: v / total for k, v in spec["categories"].items()}
    ranked = sorted(
        set(ref) | set(obs),
        key=lambda k: -max(ref.get(k, 0), obs.get(k, 0) / vals.len()),
    )[:12]
    fig = go.Figure()
    fig.add_bar(
        x=ranked, y=[ref.get(k, 0) for k in ranked], name="baseline", marker_color=_BASE
    )
    fig.add_bar(
        x=ranked,
        y=[obs.get(k, 0) / vals.len() for k in ranked],
        name="this batch",
        marker_color=_BATCH,
    )
    fig.update_layout(barmode="group")
    return _layout(fig, f"{spec['name']}: category share vs baseline (top 12)")


def _figures(
    result: CheckResult, baseline: dict[str, Any]
) -> list[tuple[str, go.Figure]]:
    figs: list[tuple[str, go.Figure]] = []
    if (
        result.rows
        and any(a.check == "volume.row_count" for a in result.alerts)
        or result.rows == 0
    ):
        figs.append(("volume", _volume_figure(result, baseline)))
    if result.frame is None or result.rows == 0:
        return figs
    specs = {c["name"]: c for c in baseline["columns"]}
    seen: list[str] = []
    for a in result.alerts:
        if (
            a.check in _CHART_CHECKS
            and a.column in specs
            and a.column in result.frame.columns
            and a.column not in seen
        ):
            seen.append(a.column)
    for name in seen[:_MAX_CHARTS]:
        spec = specs[name]
        fig = None
        if spec["kind"] == "numeric":
            fig = _numeric_figure(spec, result)
        elif spec["kind"] == "categorical":
            fig = _categorical_figure(spec, result)
        if fig is not None:
            figs.append((name, fig))
    return figs


_CSS = """
:root{--fg:#212121;--muted:#616161;--bg:#fafafa;--card:#fff;--line:#e0e0e0}
@media (prefers-color-scheme: dark){:root{--fg:#eee;--muted:#aaa;--bg:#121212;--card:#1e1e1e;--line:#333}}
body{font-family:system-ui,sans-serif;margin:0;padding:24px 16px;background:var(--bg);color:var(--fg)}
main{max-width:1100px;margin:0 auto}
h1{margin:0 0 4px}h2{margin-top:32px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin:12px 0}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;color:#fff;font-weight:600;font-size:.85rem}
.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
.kpis{display:flex;gap:12px;flex-wrap:wrap}.kpi{flex:1;min-width:120px;text-align:center}
.kpi b{font-size:1.6rem;display:block}
"""


def render_report(result: CheckResult, baseline: dict[str, Any]) -> str:
    status = result.status
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Data quality report: {_e(result.batch)}</title>",
        f"<style>{_CSS}</style></head><body><main>",
        (
            f"<h1>Data quality report</h1><p class='muted'>Batch <code>{_e(result.batch)}</code>: {result.rows} rows. "
            f"Baseline: {len(baseline['batches'])} batches, {sum(baseline['row_counts'])} rows.</p>"
        ),
        f"<p><span class='badge' style='background:{_SEV_COLOR[status]}'>{_e(status.upper())}</span></p>",
        "<div class='kpis'>",
    ]
    for sev in ("critical", "warn", "info"):
        parts.append(
            f"<div class='card kpi'><b style='color:{_SEV_COLOR[sev]}'>{result.count(sev)}</b>{sev}</div>"
        )
    parts.append("</div>")

    parts.append("<h2>Alerts</h2>")
    if result.alerts:
        parts.append(
            "<div class='card'><table><tr><th>Severity</th><th>Check</th><th>Column</th><th>Detail</th></tr>"
        )
        for a in result.alerts:
            parts.append(
                f"<tr><td><span class='badge' style='background:{_SEV_COLOR[a.severity]}'>{_e(a.severity)}</span></td>"
                f"<td><code>{_e(a.check)}</code></td><td>{_e(a.column) or '-'}</td><td>{_e(a.message)}</td></tr>"
            )
        parts.append("</table></div>")
    else:
        parts.append(
            "<div class='card'>No alerts: the batch looks like the baseline.</div>"
        )
    for note in result.notes:
        parts.append(f"<p class='muted'>{_e(note)}</p>")

    figs = _figures(result, baseline)
    if figs:
        parts.append("<h2>Where it differs</h2>")
        parts.append(f"<script>{get_plotlyjs()}</script>")
        for _, fig in figs:
            parts.append(
                f"<div class='card'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"
            )

    parts.append("<h2>Baseline schema</h2><div class='card'><table>")
    parts.append(
        "<tr><th>Column</th><th>Type</th><th>Kind</th><th>Null rate</th><th>Key</th></tr>"
    )
    for c in baseline["columns"]:
        parts.append(
            f"<tr><td>{_e(c['name'])}</td><td>{_e(c['type'])}</td><td>{_e(c['kind'])}</td>"
            f"<td>{c['null_rate']:.1%}</td><td>{'yes' if c['unique'] else ''}</td></tr>"
        )
    parts.append("</table></div></main></body></html>")
    return "".join(parts)
