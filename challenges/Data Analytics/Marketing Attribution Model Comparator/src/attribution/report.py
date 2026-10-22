"""A self-contained HTML report (and a JSON summary) for a comparison and for the simulated trials."""

from __future__ import annotations

import html
import json
import math
import re

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from attribution.compare import LABELS, Comparison
from attribution.evaluate import ScenarioResult

PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7a5195",
    "#555555",
]
INK, MUTED, GRID = "#1f2933", "#5f6b7a", "#e4e7eb"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

CSS = """
:root { color-scheme: light; --ink: #1f2933; --muted: #5f6b7a; --line: #e4e7eb; --bg: #f6f7f9;
        --card: #fff; --accent: #0072B2; --bad: #b4540a; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; } h2 { font-size: 1.15rem; margin: 0 0 8px; }
.sub { color: var(--muted); margin: 0 0 18px; }
section { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
          padding: 16px 18px 12px; margin: 0 0 16px; }
.kpis { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 16px; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 8px 14px; }
.kpi b { display: block; font-size: 1.3rem; font-variant-numeric: tabular-nums; }
.kpi span { color: var(--muted); font-size: .82rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: .88rem; }
th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; } .scroll { overflow-x: auto; }
.note { color: var(--muted); font-size: .86rem; margin: 6px 0 0; } .best { font-weight: 700; color: var(--accent); }
@media (max-width: 600px) { main { padding: 10px 10px 40px; } section { padding: 10px; } }
"""


def e(text) -> str:
    return html.escape(str(text), quote=True)


def pct(x: float, digits: int = 1) -> str:
    return "–" if x is None or not math.isfinite(x) else f"{100 * x:.{digits}f}%"


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin={"l": 60, "r": 16, "t": 30, "b": 50}, paper_bgcolor="#fff", plot_bgcolor="#fff",
        font={"family": FONT, "size": 12, "color": INK},
    )  # fmt: skip
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def short_name(name: str) -> str:
    return re.sub(r"^cat\d+=", "", name)  # "cat9=123" -> "123"; other names stay whole


def share_figure(c: Comparison, names: list[str]) -> go.Figure:
    """One panel per channel group: every model's share of conversions, with bootstrap intervals."""
    order = np.argsort(-c.share.mean(axis=0))
    fig = go.Figure()
    for mi, m in enumerate(c.models):
        err = None
        if c.low is not None:
            err = {
                "type": "data", "symmetric": False,
                "array": (c.high[mi, order] - c.share[mi, order]).tolist(),
                "arrayminus": (c.share[mi, order] - c.low[mi, order]).tolist(),
            }  # fmt: skip
        fig.add_trace(
            go.Bar(
                x=[e(names[i]) for i in order], y=c.share[mi, order].tolist(), name=LABELS[m],
                marker={"color": PALETTE[mi % len(PALETTE)]}, error_y=err,
                hovertemplate="%{x}<br>" + LABELS[m] + ": %{y:.1%}<extra></extra>",
            )
        )  # fmt: skip
    fig.update_layout(barmode="group", legend={"orientation": "h", "y": 1.12})
    fig.update_yaxes(title_text="Share of conversions", tickformat=".0%")
    fig.update_xaxes(title_text="Channel (largest first)", type="category")
    return _layout(fig, 460)


def heatmap(
    z: np.ndarray,
    labels: list[str],
    title: str,
    fmt: str,
    zmin: float,
    zmax: float,
    scale: str,
) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=z.tolist(), x=[e(x) for x in labels], y=[e(x) for x in labels], zmin=zmin, zmax=zmax, colorscale=scale,
            text=[[format(v, fmt) for v in row] for row in z], texttemplate="%{text}", showscale=False,
        )
    )  # fmt: skip
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-35)
    return _layout(fig, 400).update_layout(
        title={"text": title, "x": 0.01, "font": {"size": 13}}
    )


def scenario_figure(r: ScenarioResult, models: tuple[str, ...]) -> go.Figure:
    fig = go.Figure()
    xs = [e(c) for c in r.channels]
    fig.add_trace(
        go.Bar(
            x=xs,
            y=r.truth.tolist(),
            name="True incremental share",
            marker={"color": "#1f2933"},
        )
    )
    for mi, m in enumerate(models):
        fig.add_trace(
            go.Bar(
                x=xs,
                y=r.shares[mi].tolist(),
                name=LABELS[m],
                marker={"color": PALETTE[mi % len(PALETTE)]},
            )
        )
    fig.update_layout(barmode="group", legend={"orientation": "h", "y": 1.2})
    fig.update_yaxes(title_text="Share", tickformat=".0%")
    return _layout(fig, 360)


def _table(headers: list[str], rows: list[list[str]], raw: bool = False) -> str:
    head = "".join(f"<th>{e(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c if raw else e(c)}</td>" for c in r) + "</tr>"
        for r in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def build_html(
    title: str,
    c: Comparison | None,
    scenarios: list[ScenarioResult],
    models: tuple[str, ...],
    notes: list[str] | None = None,
) -> str:
    def render(fig: go.Figure) -> str:
        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"responsive": True, "displaylogo": False},
        )

    body = [f"<h1>{e(title)}</h1>"]
    if c is not None:
        j = c.journeys
        names = [short_name(n) for n in j.channels]
        kp = [
            (f"{j.n_converted:,}", "converting journeys"),
            (f"{int(j.null_count):,}", "journeys without a conversion"),
            (f"{j.k}", "channels"),
            (f"{j.lengths.mean():.2f}", "touches per converting journey"),
            (
                f"{100 * float(np.mean(j.lengths > 1)):.0f}%",
                "converting journeys with 2+ touches",
            ),
        ]
        body.append(
            '<div class="kpis">'
            + "".join(
                f'<div class="kpi"><b>{e(v)}</b><span>{e(k)}</span></div>'
                for v, k in kp
            )
            + "</div>"
        )
        boot = (
            f"; error bars are 95% bootstrap intervals from {c.draws} resamples"
            if c.draws
            else ""
        )
        body.append(
            f"<section><h2>Who gets the credit?</h2>{render(share_figure(c, names))}<p class='note'>Every model divides the same {j.total_value:,.0f} conversions{e(boot)}.</p></section>"
        )
        spread = c.share.max(axis=0) - c.share.min(axis=0)
        rows = []
        for ci in np.argsort(-spread):
            lo, hi = c.share[:, ci].argmin(), c.share[:, ci].argmax()
            rows.append(
                [
                    names[ci],
                    pct(c.share[:, ci].min()),
                    LABELS[c.models[lo]],
                    pct(c.share[:, ci].max()),
                    LABELS[c.models[hi]],
                    pct(spread[ci]),
                ]
            )
        body.append(
            "<section><h2>Where the models disagree most</h2>"
            + _table(
                [
                    "Channel",
                    "Lowest share",
                    "under",
                    "Highest share",
                    "under",
                    "Spread",
                ],
                rows,
            )
            + "</section>"
        )
        labels = [LABELS[m] for m in c.models]
        body.append("<section><h2>How alike are the models?</h2><div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:8px'>"
                    + render(heatmap(c.distance(), labels, "Share that would move (total variation distance)", ".2f", 0, 0.5, "Blues"))
                    + render(heatmap(c.rank_correlation(), labels, "Agreement on channel ranking (Kendall tau)", ".2f", -1, 1, "RdBu"))
                    + "</div></section>")  # fmt: skip
        spend = j.touch_spend()
        if spend.sum() > 0:
            cpc = c.cost_per_conversion()
            head = ["Channel", "Spend share", *[LABELS[m] for m in c.models]]
            rows = [
                [
                    names[ci],
                    pct(spend[ci] / spend.sum()),
                    *[
                        ("–" if not np.isfinite(cpc[mi, ci]) else f"{cpc[mi, ci]:.4g}")
                        for mi in range(len(c.models))
                    ],
                ]
                for ci in np.argsort(-spend)
            ]
            body.append(
                "<section><h2>What a conversion costs, by model</h2>"
                + _table(head, rows)
                + "<p class='note'>Spend on a channel divided by the conversions the model credits it. The cheapest channel changes with the model.</p></section>"
            )
    if scenarios:
        parts = [
            "<h2>Which model is right? Scored against a world with a known answer</h2><p class='note'>Simulated customers, each channel's true effect known; error = the share of conversions a model puts on the wrong channel (0 is perfect). Mean over seeds.</p>"
        ]
        head = ["Scenario", *[LABELS[m] for m in models], "Best"]
        rows = []
        for r in scenarios:
            best = int(np.argmin(r.error))
            rows.append(
                [
                    r.name,
                    *[
                        (f"<span class='best'>{pct(v)}</span>" if i == best else pct(v))
                        for i, v in enumerate(r.error)
                    ],
                    LABELS[models[best]],
                ]
            )
        parts.append(_table(head, rows, raw=True))
        for r in scenarios:
            parts.append(
                f"<h2 style='margin-top:14px'>{e(r.name)}</h2><p class='note'>{e(r.description)}</p>{render(scenario_figure(r, models))}"
            )
        body.append("<section>" + "".join(parts) + "</section>")
    for n in notes or []:
        body.append(f"<p class='note'>{e(n)}</p>")
    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'><link rel='icon' href='data:,'>"
        f"<title>{e(title)}</title><style>{CSS}</style><script>{get_plotlyjs()}</script></head>"
        f"<body><main>{''.join(body)}</main></body></html>\n"
    )


def to_json(
    c: Comparison | None, scenarios: list[ScenarioResult], models: tuple[str, ...]
) -> str:
    def f(x):
        return (
            None
            if x is None or (isinstance(x, float) and not math.isfinite(x))
            else float(x)
        )

    out: dict = {"models": list(models)}
    if c is not None:
        j = c.journeys
        out["comparison"] = {
            "channels": j.channels,
            "converting_journeys": j.n_converted,
            "null_journeys": int(j.null_count),
            "total_value": f(j.total_value),
            "credit": {
                m: [f(v) for v in row]
                for m, row in zip(c.models, c.credit, strict=True)
            },
            "share": {
                m: [f(v) for v in row] for m, row in zip(c.models, c.share, strict=True)
            },
            "share_low": None
            if c.low is None
            else {
                m: [f(v) for v in row] for m, row in zip(c.models, c.low, strict=True)
            },
            "share_high": None
            if c.high is None
            else {
                m: [f(v) for v in row] for m, row in zip(c.models, c.high, strict=True)
            },
            "bootstrap_draws": c.draws,
            "total_variation": [[f(v) for v in row] for row in c.distance()],
            "kendall_tau": [[f(v) for v in row] for row in c.rank_correlation()],
        }
    out["scenarios"] = [
        {
            "name": r.name, "description": r.description, "channels": r.channels, "seeds": r.seeds, "customers": r.customers,
            "true_share": [f(v) for v in r.truth],
            "share": {m: [f(v) for v in row] for m, row in zip(models, r.shares, strict=True)},
            "share_misplaced": {m: f(v) for m, v in zip(models, r.error, strict=True)},
            "share_misplaced_sd": {m: f(v) for m, v in zip(models, r.error_sd, strict=True)},
            "kendall_tau_with_truth": {m: f(v) for m, v in zip(models, r.tau, strict=True)},
        }
        for r in scenarios
    ]  # fmt: skip
    return json.dumps(out, indent=4, ensure_ascii=False, allow_nan=False) + "\n"
