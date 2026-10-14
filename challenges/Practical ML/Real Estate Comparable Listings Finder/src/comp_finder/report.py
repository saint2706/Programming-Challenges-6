"""Charts (Altair, exported to PNG by ``vl-convert``) and the written report."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import polars as pl

from comp_finder.engine import METHODS
from comp_finder.pipeline import Result

REFERENCE = "GBM AVM (no comps)"
NAMES = {**METHODS, "avm": REFERENCE}
SHORT = {
    "knn": "Scaled kNN",
    "ridge": "Ridge-weighted",
    "hedonic": "Hedonic-weighted",
    "leaf": "Leaf proximity",
    "avm": "GBM AVM",
}


def mdape_chart(metrics: dict) -> alt.Chart:
    rows = [
        {
            "method": SHORT[m],
            "MdAPE": v["mdape"],
            "low": v["ci"]["mdape"][0],
            "high": v["ci"]["mdape"][1],
        }
        for m, v in metrics["main"].items()
    ]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("method:N", sort=None, title=None)
    )
    bars = base.mark_bar(color="#2a9d8f").encode(
        x=alt.X(
            "MdAPE:Q", axis=alt.Axis(format="%"), title="Median absolute % error (test)"
        )
    )
    err = base.mark_errorbar(color="#264653").encode(
        x=alt.X("low:Q", title=None), x2="high:Q"
    )
    return (bars + err).properties(width=420, height=170)


def k_chart(metrics: dict) -> alt.Chart:
    rows = [
        {"method": SHORT[m], "k": int(k), "MdAPE": v}
        for m, ks in metrics["k_sweep"].items()
        for k, v in ks.items()
    ]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_line(point=True)
        .encode(
            x=alt.X("k:O", title="Comps used (k)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("MdAPE:Q", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
            color=alt.Color("method:N", title=None),
        )
        .properties(width=360, height=200)
    )


def _pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}%"


def _ci(ci: tuple[float, float]) -> str:
    return f"[{_pct(ci[0])}, {_pct(ci[1])}]"


def markdown(metrics: dict) -> str:
    d, s = metrics["data"], metrics["settings"]
    lines = [
        "# Comparable listings: results",
        "",
        (
            f"{d['sales']:,} King County sales, {d['first']} to {d['last']}. Fit on {d['fit']:,}, "
            f"validation {d['validation']:,}, test {d['test']:,} ({d['test_first_sales']:,} first "
            f"sales of a property) from {metrics['windows']['test_start']}. Settings chosen on "
            f"validation: k={s['k']}, market-index extrapolation `{s['extrapolate']}`, "
            f"comps combined by `{s['combine']}`."
        ),
        "",
        "## Test window",
        "",
        "| Method | MdAPE | 95% CI | Within 10% | Within 20% | Median bias |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for m, v in metrics["main"].items():
        lines.append(
            f"| {NAMES[m]} | {_pct(v['mdape'])} | {_ci(v['ci']['mdape'])} | {_pct(v['within10'])} | "
            f"{_pct(v['within20'])} | {_pct(v['bias'])} |"
        )
    lines += [
        "",
        "Paired bootstrap, first minus second (negative MdAPE = first is better):",
        "",
        "| Pair | MdAPE diff (pts) | 95% CI | Within-10% diff (pts) | 95% CI |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for pair, v in metrics["paired"].items():
        a, b = pair.split("-")
        m, w = v["mdape"], v["within10"]
        lines.append(
            f"| {SHORT[a]} vs {SHORT[b]} | {m[0] * 100:+.2f} | [{m[1] * 100:+.2f}, {m[2] * 100:+.2f}] | "
            f"{w[0] * 100:+.2f} | [{w[1] * 100:+.2f}, {w[2] * 100:+.2f}] |"
        )
    lines += [
        "",
        "## Adjustments (MdAPE)",
        "",
        "| Method | none | + market time | + attributes | GBM-adjusted |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for m, v in metrics["adjust_ablation"].items():
        lines.append(
            f"| {SHORT[m]} | {_pct(v['none'])} | {_pct(v['time'])} | {_pct(v['full'])} | {_pct(v['gbm'])} |"
        )
    lines += [
        "",
        "## Number of comps (MdAPE)",
        "",
        "| Method | "
        + " | ".join(f"k={k}" for k in next(iter(metrics["k_sweep"].values())))
        + " |",
        "| --- | "
        + " | ".join("---:" for _ in next(iter(metrics["k_sweep"].values())))
        + " |",
    ]
    for m, v in metrics["k_sweep"].items():
        lines.append(
            f"| {SHORT[m]} | " + " | ".join(_pct(x) for x in v.values()) + " |"
        )
    lines += [
        "",
        "## Where the comps come from",
        "",
        "| Method | Median km | Same zip | Median days old | Median gross adjustment |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for m, v in metrics["comp_quality"].items():
        lines.append(
            f"| {SHORT[m]} | {v['median_km']:.2f} | {_pct(v['same_zip'], 0)} | {v['median_days_old']:.0f} | {_pct(v['median_gross_adjustment'])} |"
        )
    lines += [
        "",
        "## Pool and extrapolation (MdAPE)",
        "",
        "| Method | pool = all earlier sales | pool = fit window only |",
        "| --- | ---: | ---: |",
    ]
    for m, v in metrics["pool"].items():
        lines.append(f"| {SHORT[m]} | {_pct(v['prior'])} | {_pct(v['fit'])} |")
    lines += [
        "",
        "| Index beyond fit window | "
        + " | ".join(SHORT[m] for m in next(iter(metrics["extrapolation"].values())))
        + " |",
        "| --- | "
        + " | ".join("---:" for _ in next(iter(metrics["extrapolation"].values())))
        + " |",
    ]
    for e, v in metrics["extrapolation"].items():
        lines.append(f"| {e} | " + " | ".join(_pct(x) for x in v.values()) + " |")
    lines += [
        "",
        "## Bracketing",
        "",
        "| Method | Bracketed (top-k) | Bracketed (bracket-aware) | MdAPE top-k | MdAPE bracket-aware | MdAPE if bracketed | MdAPE if not |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m, v in metrics["bracketing"].items():
        lines.append(
            f"| {SHORT[m]} | {_pct(v['bracketed_share_top'], 0)} | {_pct(v['bracketed_share_selected'], 0)} | {_pct(v['mdape_top'])} | "
            f"{_pct(v['mdape_selected'])} | {_pct(v['mdape_when_bracketed'])} | {_pct(v['mdape_when_not'])} |"
        )
    for name, segs in metrics["segments"].items():
        lines += [
            "",
            f"## By {name} (MdAPE)",
            "",
            "| Segment | n | " + " | ".join(SHORT[m] for m in METHODS) + " | GBM AVM |",
            "| --- | ---: | "
            + " | ".join("---:" for _ in range(len(METHODS) + 1))
            + " |",
        ]
        for label, v in segs.items():
            lines.append(
                f"| {label} | {v['n']:,} | "
                + " | ".join(_pct(v[m]) for m in [*METHODS, "avm"])
                + " |"
            )
    if "repeat_sales" in metrics:
        r = metrics["repeat_sales"]
        lines += [
            "",
            f"## Resales in the test window (n={r['n']})",
            "",
            "| Estimator | MdAPE |",
            "| --- | ---: |",
            f"| GBM AVM | {_pct(r['avm'])} |",
        ]
        for k, v in r.items():
            if k.startswith(("knn", "leaf")):
                lines.append(f"| {k.replace('_', ' ')} | {_pct(v)} |")
    z = metrics["zipcode"]
    lines += [
        "",
        "## GBM AVM: zipcode",
        "",
        f"lat/long only {_pct(z['latlong_only'])}, plus zipcode as a categorical {_pct(z['with_zip_categorical'])}.",
        "",
    ]
    lines += [
        "## Speed",
        "",
        "Milliseconds per subject (find comps and estimate, single process):",
        "",
    ]
    lines += [f"- {SHORT[m]}: {v:.2f}" for m, v in metrics["ms_per_subject"].items()]
    lines += [
        "",
        "## Market index (log, flat vs trend extrapolation)",
        "",
        "| Month | Fitted | Flat | Trend |",
        "| --- | --- | ---: | ---: |",
    ]
    for mth, v in metrics["market_index"].items():
        y, mo = divmod(int(mth) - 1, 12)
        lines.append(
            f"| {y}-{mo + 1:02d} | {'yes' if v['fitted'] else 'no'} | {v['flat']:+.3f} | {v['trend']:+.3f} |"
        )
    lines += [
        "",
        "## GBM AVM signed error by sale month (test)",
        "",
        "| Month | n | Median error |",
        "| --- | ---: | ---: |",
    ]
    for mth, v in metrics["monthly_bias"].items():
        y, mo = divmod(int(mth) - 1, 12)
        lines.append(f"| {y}-{mo + 1:02d} | {v['n']} | {_pct(v['avm_median_error'])} |")
    return "\n".join(lines) + "\n"


def write_report(result: Result, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, default=str), encoding="utf-8"
    )
    for name, chart in (
        ("mdape", mdape_chart(result.metrics)),
        ("k_sweep", k_chart(result.metrics)),
    ):
        chart.save(out_dir / f"{name}.png", scale_factor=2)
    path = out_dir / "report.md"
    path.write_text(markdown(result.metrics), encoding="utf-8")
    return path
