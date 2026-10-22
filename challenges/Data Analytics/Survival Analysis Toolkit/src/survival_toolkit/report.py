"""A self-contained HTML report: curves with a risk table, tests, hazard ratios, PH diagnostics.

One file, no network: plotly.js is inlined once and every figure shares it. All text that comes from
the data (column names, group labels) is HTML-escaped, and the figures carry it as JSON that
plotly escapes for a ``<script>`` element.
"""

from __future__ import annotations

import html
import math

import numpy as np
from plotly import graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from survival_toolkit.analysis import Analysis, CoxFit, GroupSummary

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
        --card: #ffffff; --accent: #0072B2; --warn: #b4540a; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; } h2 { font-size: 1.15rem; margin: 0 0 8px; }
.sub { color: var(--muted); margin: 0 0 20px; }
section { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
          padding: 18px 18px 14px; margin: 0 0 16px; }
.kpis { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 16px; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
       padding: 10px 14px; min-width: 130px; }
.kpi b { display: block; font-size: 1.35rem; font-variant-numeric: tabular-nums; }
.kpi span { color: var(--muted); font-size: .82rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: .9rem; }
th, td { text-align: right; padding: 5px 8px; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; }
.scroll { overflow-x: auto; }
.note { color: var(--muted); font-size: .86rem; margin: 6px 0 0; }
.warn { color: var(--warn); }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
code { background: #eef1f4; padding: 1px 5px; border-radius: 4px; font-size: .85em; }
@media (max-width: 600px) { main { padding: 12px 10px 40px; } section { padding: 12px; } }
"""


def e(text) -> str:
    return html.escape(str(text), quote=True)


def num(v: float | None, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return f"{v:.{digits}f}"


def fmt_p(p: float | None) -> str:
    if p is None or not math.isfinite(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_time(v: float) -> str:
    return "not reached" if not math.isfinite(v) else f"{v:g}"


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 56, "r": 16, "t": 28, "b": 44},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": FONT, "size": 13, "color": INK},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hoverlabel={"font": {"family": FONT}},
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def _step_xy(
    time: np.ndarray, values: np.ndarray, start: float = 1.0
) -> tuple[list, list]:
    x = np.concatenate([[0.0], time])
    y = np.concatenate([[start], values])
    return x.tolist(), y.tolist()


def _band(g: GroupSummary) -> tuple[list, list] | None:
    c = g.curve
    ok = np.isfinite(c.lower)
    if ok.sum() < 2:
        return None
    t = np.concatenate([[0.0], c.time[ok]])
    lo = np.concatenate([[1.0], c.lower[ok]])
    hi = np.concatenate([[1.0], c.upper[ok]])
    # a step polygon: duplicate each x so the band follows the steps
    xs = np.repeat(t, 2)[1:]
    top = np.repeat(hi, 2)[:-1]
    bottom = np.repeat(lo, 2)[:-1]
    return (xs.tolist() + xs[::-1].tolist(), top.tolist() + bottom[::-1].tolist())


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def curves_figure(a: Analysis) -> go.Figure:
    """Survival curves with confidence bands and a risk table underneath."""
    top = max(float(g.curve.data.time.max()) for g in a.groups)
    ticks = _nice_ticks(top)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.04
    )
    for i, g in enumerate(a.groups):
        color = PALETTE[i % len(PALETTE)]
        band = _band(g)
        if band:
            fig.add_trace(
                go.Scatter(x=band[0], y=band[1], fill="toself", mode="none",
                           fillcolor=_rgba(color, 0.16), hoverinfo="skip", showlegend=False),
                row=1, col=1,
            )  # fmt: skip
        x, y = _step_xy(g.curve.time, g.curve.survival)
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines", line={"shape": "hv", "color": color, "width": 2.4},
                name=f"{e(g.label)} (n={g.n:,})",
                hovertemplate=f"{e(g.label)}<br>t=%{{x:g}} {e(a.unit)}<br>S(t)=%{{y:.3f}}<extra></extra>",
            ),
            row=1, col=1,
        )  # fmt: skip
        counts = g.curve.table(np.array(ticks))
        fig.add_trace(
            go.Scatter(
                x=ticks, y=[e(g.label)] * len(ticks), mode="text",
                text=[f"{int(c):,}" for c in counts], textfont={"color": color, "size": 12},
                hoverinfo="skip", showlegend=False,
            ),
            row=2, col=1,
        )  # fmt: skip
    _layout(fig, 560 if len(a.groups) < 6 else 640)
    fig.update_yaxes(title_text="Survival probability", range=[0, 1.02], row=1, col=1)
    fig.update_yaxes(
        showgrid=False, autorange="reversed", row=2, col=1, tickfont={"size": 11}
    )
    fig.update_xaxes(
        title_text=f"Time ({a.unit})", tickmode="array", tickvals=ticks, row=2, col=1
    )
    fig.update_xaxes(range=[0, top * 1.02])
    fig.add_annotation(
        text="Number at risk", xref="paper", yref="paper", x=0, y=0.235, showarrow=False,
        font={"size": 11, "color": MUTED}, xanchor="left",
    )  # fmt: skip
    return fig


def _nice_ticks(top: float) -> list[float]:
    raw = top / 6
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min(
        (m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), default=raw
    )
    return [float(v) for v in np.arange(0, top + 1e-9, step)]


def hazard_figure(a: Analysis) -> go.Figure:
    """Cumulative hazard and the log-log plot, the two classic looks at proportional hazards."""
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.1,
        subplot_titles=("Cumulative hazard (Nelson-Aalen)", "log(-log S) against log time"),
    )  # fmt: skip
    for i, g in enumerate(a.groups):
        color = PALETTE[i % len(PALETTE)]
        c = g.curve
        x, y = _step_xy(c.time, c.cumulative_hazard, 0.0)
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", line={"shape": "hv", "color": color, "width": 2},
                       name=e(g.label), legendgroup=e(g.label)),
            row=1, col=1,
        )  # fmt: skip
        ok = (c.survival > 0) & (c.survival < 1)
        fig.add_trace(
            go.Scatter(
                x=np.log(c.time[ok]).tolist(), y=np.log(-np.log(c.survival[ok])).tolist(),
                mode="lines", line={"shape": "hv", "color": color, "width": 2},
                name=e(g.label), legendgroup=e(g.label), showlegend=False,
            ),
            row=1, col=2,
        )  # fmt: skip
    _layout(fig, 430)
    fig.update_layout(legend={"orientation": "h", "y": -0.28, "x": 0})
    fig.update_xaxes(title_text=f"Time ({a.unit})", row=1, col=1)
    fig.update_xaxes(title_text=f"log time ({a.unit})", row=1, col=2)
    fig.update_yaxes(title_text="H(t)", row=1, col=1)
    fig.update_yaxes(title_text="log(-log S(t))", row=1, col=2)
    fig.update_annotations(font={"size": 13, "color": INK})
    return fig


def forest_figure(fits: list[tuple[str, CoxFit]]) -> go.Figure:
    fig = go.Figure()
    names = [r["name"] for r in fits[0][1].model.summary()]
    for i, (label, fit) in enumerate(fits):
        rows = {r["name"]: r for r in fit.model.summary()}
        color = PALETTE[i % len(PALETTE)]
        use = [n for n in names if n in rows]
        fig.add_trace(
            go.Scatter(
                x=[rows[n]["hr"] for n in use], y=[e(n) for n in use], mode="markers", name=e(label),
                marker={"color": color, "size": 8},
                error_x={
                    "type": "data", "symmetric": False, "color": color, "thickness": 1.6,
                    "array": [rows[n]["hr_upper"] - rows[n]["hr"] for n in use],
                    "arrayminus": [rows[n]["hr"] - rows[n]["hr_lower"] for n in use],
                },
                hovertemplate="%{y}<br>HR %{x:.3f}<extra></extra>",
            )
        )  # fmt: skip
    _layout(fig, max(320, 60 + 30 * len(names)))
    fig.add_vline(x=1, line={"color": MUTED, "dash": "dash", "width": 1})
    fig.update_xaxes(type="log", title_text="Hazard ratio (log scale; 1 = no effect)")
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_layout(margin={"l": 220, "r": 16, "t": 28, "b": 44})
    return fig


def schoenfeld_figure(fit: CoxFit, top: int = 4) -> go.Figure | None:
    model = fit.model
    t, scaled = model.scaled_schoenfeld()
    ranked = sorted(fit.ph["terms"], key=lambda r: -r["chi2"])[:top]
    if not ranked:
        return None
    fig = make_subplots(
        rows=1,
        cols=len(ranked),
        subplot_titles=[e(r["name"]) for r in ranked],
        horizontal_spacing=0.07,
    )
    order = np.argsort(t)
    for col, term in enumerate(ranked, start=1):
        j = model.names.index(term["name"])
        y = scaled[order, j]
        x = t[order]
        fig.add_trace(
            go.Scatter(x=x.tolist(), y=y.tolist(), mode="markers", marker={"size": 3, "color": _rgba(PALETTE[0], 0.28)},
                       showlegend=False, hoverinfo="skip"),
            row=1, col=col,
        )  # fmt: skip
        bins = np.array_split(np.arange(x.size), min(12, max(2, x.size // 25)))
        mx = [float(x[b].mean()) for b in bins]
        my = [float(y[b].mean()) for b in bins]
        se = [
            float(y[b].std(ddof=1) / math.sqrt(b.size)) if b.size > 1 else 0.0
            for b in bins
        ]
        fig.add_trace(
            go.Scatter(x=mx, y=my, mode="lines+markers", line={"color": PALETTE[1], "width": 2.4},
                       error_y={"type": "data", "array": [1.96 * s for s in se], "color": PALETTE[1]},
                       showlegend=False,
                       hovertemplate="t=%{x:g}<br>mean %{y:.3f}<extra></extra>"),
            row=1, col=col,
        )  # fmt: skip
        fig.add_hline(
            y=float(model.beta[j]),
            line={"color": MUTED, "dash": "dash", "width": 1},
            row=1,
            col=col,
        )
    _layout(fig, 340)
    fig.update_annotations(font={"size": 12, "color": INK})
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Scaled residual", col=1)
    return fig


def _table(headers: list[str], rows: list[list[str]], *, raw: bool = False) -> str:
    head = "".join(f"<th>{e(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c if raw else e(c)}</td>" for c in r) + "</tr>"
        for r in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _group_table(a: Analysis) -> str:
    headers = ["Group", "n", "Events", "Censored", f"Median ({a.unit})"]
    headers += [f"S({h:g})" for h in a.horizons]
    if a.tau is not None:
        headers.append(f"RMST to {a.tau:g}")
    rows = []
    for i, g in enumerate(a.groups):
        sw = f'<span class="swatch" style="background:{PALETTE[i % len(PALETTE)]}"></span>'
        lo, hi = g.median_ci
        med = fmt_time(g.median)
        if math.isfinite(g.median):
            med += f" ({fmt_time(lo)} to {fmt_time(hi)})"
        cells = [
            sw + e(g.label),
            f"{g.n:,}",
            f"{g.events:,}",
            f"{g.censored:,}",
            e(med),
        ]
        for r in g.retention:
            cells.append(
                "–" if r["s"] is None
                else f"{r['s']:.3f} ({r['lower']:.3f}–{r['upper']:.3f})"
            )  # fmt: skip
        if a.tau is not None:
            cells.append(
                "–" if g.rmst is None else f"{g.rmst[0]:.1f} ± {g.rmst[1]:.1f}"
            )
        rows.append(cells)
    return _table(headers, rows, raw=True)


def _tests_section(a: Analysis) -> str:
    if not a.tests:
        return ""
    names = {
        "logrank": "Log-rank (all times equal)",
        "wilcoxon": "Gehan-Breslow-Wilcoxon (early times)",
        "tarone-ware": "Tarone-Ware (in between)",
        "peto": "Peto-Peto (early times, robust to censoring)",
        "fh:0,1": "Fleming-Harrington (0,1) (late times)",
    }
    rows = [
        [names.get(k, k), f"{v.statistic:.2f}", str(v.df), fmt_p(v.p_value)]
        for k, v in a.tests.items()
    ]
    out = "<h2>Do the curves differ?</h2>" + _table(["Test", "χ²", "df", "p"], rows)
    if a.trend is not None:
        out += (
            f'<p class="note">Trend across the groups in order: χ² {a.trend.statistic:.2f} on 1 df, '
            f"p {fmt_p(a.trend.p_value)}.</p>"
        )
    if a.pairwise:
        prow = [
            [
                f"{r['a']} vs {r['b']}",
                f"{r['statistic']:.2f}",
                fmt_p(r["p"]),
                fmt_p(r["p_adjusted"]),
            ]
            for r in a.pairwise
        ]
        out += "<h2 style='margin-top:14px'>Pairs (log-rank, Holm-adjusted)</h2>"
        out += _table(["Pair", "χ²", "p", "adjusted p"], prow)
    if a.rmst_pairs:
        rrow = [
            [f"{r['a']} − {r['b']}", f"{r['difference']:.2f}",
             f"{r['lower']:.2f} to {r['upper']:.2f}", fmt_p(r["p"])]
            for r in a.rmst_pairs
        ]  # fmt: skip
        out += f"<h2 style='margin-top:14px'>Restricted mean survival time to {a.tau:g} {e(a.unit)}</h2>"
        out += _table(["Difference", f"{e(a.unit)} gained", "95% CI", "p"], rrow)
    return f"<section>{out}</section>"


def _cox_table(fit: CoxFit) -> str:
    rows = [
        [r["name"], f"{r['hr']:.3f}", f"{r['hr_lower']:.3f}–{r['hr_upper']:.3f}",
         f"{r['coef']:.3f}", f"{r['se']:.3f}", fmt_p(r["p"])]
        for r in fit.model.summary()
    ]  # fmt: skip
    return _table(["Covariate", "HR", "95% CI", "coef", "se", "p"], rows)


def _ph_table(fit: CoxFit) -> str:
    rows = [
        [
            t["name"],
            f"{t['chi2']:.2f}",
            fmt_p(t["p"]),
            "violated" if t["p"] < 0.05 else "ok",
        ]
        for t in fit.ph["terms"]
    ]
    g = fit.ph["global"]
    rows.append(
        [
            "GLOBAL",
            f"{g['chi2']:.2f}",
            fmt_p(g["p"]),
            "violated" if g["p"] < 0.05 else "ok",
        ]
    )
    return _table(["Covariate", "χ²", "p", "verdict"], rows)


def _cox_section(a: Analysis, figs: dict[str, str]) -> str:
    if a.cox is None:
        return ""
    fits = [("Cox", a.cox)] + (
        [(f"stratified by {a.cox_stratified.strata_column}", a.cox_stratified)]
        if a.cox_stratified
        else []
    )
    parts = []
    for label, fit in fits:
        m = fit.model
        lr_stat, df, p = m.likelihood_ratio()
        c = m.concordance()["c"]
        g = fit.ph["global"]
        head = (
            f"<h2>{e(label)}</h2><p class='note'>n = {m.n:,}, events = {m.n_events:,}, ties: {e(m.ties)}; "
            f"likelihood-ratio χ² {lr_stat:.1f} on {df} df (p {fmt_p(p)}); concordance {c:.3f}. "
            f"Proportional-hazards test (global, {e(fit.ph['transform'])} time): χ² {g['chi2']:.1f}, p {fmt_p(g['p'])}"
            f"{' — <span class=warn>the assumption fails</span>' if g['p'] < 0.05 else ''}.</p>"
        )
        if fit.dropped:
            head += f"<p class='note'>{fit.dropped:,} rows with a missing covariate were left out.</p>"
        parts.append(
            head
            + _cox_table(fit)
            + "<h2 style='margin-top:14px'>Proportional-hazards check</h2>"
            + _ph_table(fit)
        )
    body = "".join(f"<div>{p}</div>" for p in parts)
    return (
        "<section><h2>Hazard ratios</h2>" + figs.get("forest", "") + "</section>"
        + "<section>" + body + "</section>"
        + (f"<section><h2>Scaled Schoenfeld residuals (largest departures)</h2>{figs['schoenfeld']}"
           "<p class='note'>Each point is one event; the orange line is the mean in time bins, the dashed line the fitted coefficient. A slope means the effect changes with time.</p></section>"
           if "schoenfeld" in figs else "")
    )  # fmt: skip


def build_html(a: Analysis) -> str:
    def render(fig: go.Figure) -> str:
        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"responsive": True, "displaylogo": False},
        )

    figs = {"curves": render(curves_figure(a)), "hazard": render(hazard_figure(a))}
    if a.cox is not None:
        fits = [("Cox", a.cox)]
        if a.cox_stratified is not None:
            fits.append(
                (f"stratified by {a.cox_stratified.strata_column}", a.cox_stratified)
            )
        figs["forest"] = render(forest_figure(fits))
        sch = schoenfeld_figure(a.cox)
        if sch is not None:
            figs["schoenfeld"] = render(sch)
    censored = a.n - a.events
    kpis = [
        (f"{a.n:,}", "subjects"),
        (f"{a.events:,}", f"events ({100 * a.events / a.n:.1f}%)"),
        (f"{censored:,}", "censored"),
    ]
    if a.follow_up_median is not None and math.isfinite(a.follow_up_median):
        kpis.append((f"{a.follow_up_median:g}", f"median follow-up, {a.unit}"))
    kpi_html = "".join(
        f'<div class="kpi"><b>{e(v)}</b><span>{e(k)}</span></div>' for v, k in kpis
    )
    dropped = "; ".join(f"{v:,} rows: {e(k)}" for k, v in a.dropped.items())
    notes = "".join(f"<p class='note'>{e(n)}</p>" for n in a.notes)
    by = f" by {e(a.group_column)}" if a.group_column else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>{e(a.title)} – survival report</title>
<style>{CSS}</style>
<script>{get_plotlyjs()}</script>
</head><body><main>
<h1>{e(a.title)}</h1>
<p class="sub">Survival analysis{by}; confidence intervals: {e(a.ci_method)}, 95%.{(" Cleaning dropped " + dropped + ".") if dropped else ""}</p>
<div class="kpis">{kpi_html}</div>
<section><h2>Survival curves</h2>{figs["curves"]}
<p class="note">Shaded bands are 95% confidence intervals. Numbers under the plot are subjects still at risk.</p></section>
<section><h2>Groups</h2>{_group_table(a)}
<p class="note">S(t) is the probability of still being event-free at t, with its confidence interval. RMST is the average event-free time within the horizon.</p></section>
{_tests_section(a)}
<section><h2>Hazard shape</h2>{figs["hazard"]}
<p class="note">Roughly parallel curves on the right are consistent with proportional hazards.</p></section>
{_cox_section(a, figs)}
{notes}
</main></body></html>
"""
