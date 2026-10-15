"""``results/report.md`` and the four charts, written from the experiment's ``result`` dict."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_curve

PALETTE = {
    "LightGBM (calibrated)": "#1b6ca8",
    "LightGBM (raw)": "#7fb3d5",
    "Logistic regression": "#e08a1e",
    "14-day mean out-of-stock hours": "#6b8e23",
    "Yesterday": "#999999",
}


def _table(headers: list[str], rows: list[list]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _f(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def reliability_chart(result: dict) -> alt.Chart:
    rows = [
        {"model": name, **r}
        for name, key in (("raw", "raw"), ("isotonic-calibrated", "calibrated"))
        for r in result["reliability"][key]
    ]
    df = pl.DataFrame(rows)
    diag = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    line = (
        alt.Chart(diag)
        .mark_line(color="#999", strokeDash=[4, 4])
        .encode(
            x=alt.X("x:Q", title="mean predicted risk"),
            y=alt.Y("y:Q", title="observed stockout rate"),
        )
    )
    pts = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "mean_p:Q", title="mean predicted risk", scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y(
                "rate:Q", title="observed stockout rate", scale=alt.Scale(domain=[0, 1])
            ),
            color=alt.Color("model:N", title=None),
        )
    )
    return (line + pts).properties(
        width=380, height=320, title="Reliability on the test window"
    )


def pr_points(
    y: np.ndarray, p: np.ndarray, n: int = 600
) -> tuple[np.ndarray, np.ndarray]:
    """The precision-recall curve thinned to about ``n`` points spaced evenly in *recall*.

    ``precision_recall_curve`` returns a point per distinct score (hundreds of thousands here).
    Thinning them evenly by position drops the few hundred at the top-left, where the curve is
    steepest, so the grid is on recall instead.
    """
    precision, recall, _ = precision_recall_curve(y, p)
    # recall falls as the index rises, so -recall is increasing and searchsorted applies
    idx = np.searchsorted(-recall, -np.linspace(0, 1, n))
    idx = np.unique(np.clip(idx, 0, len(recall) - 1))
    return recall[idx], precision[idx]


def pr_chart(y: np.ndarray, probs: dict[str, np.ndarray]) -> alt.Chart:
    frames = []
    for name, p in probs.items():
        recall, precision = pr_points(y, p)
        frames.append(
            pl.DataFrame({"recall": recall, "precision": precision, "model": name})
        )
    df = pl.concat(frames)
    base = pl.DataFrame({"x": [0.0, 1.0], "y": [float(y.mean())] * 2})
    line = (
        alt.Chart(base)
        .mark_line(color="#999", strokeDash=[4, 4])
        .encode(x=alt.X("x:Q", title="recall"), y=alt.Y("y:Q", title="precision"))
    )
    curves = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("recall:Q", title="recall", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("precision:Q", title="precision", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "model:N",
                title=None,
                scale=alt.Scale(domain=list(probs), range=[PALETTE[k] for k in probs]),
            ),
        )
    )
    return (line + curves).properties(
        width=420, height=320, title="Precision-recall, test window"
    )


def cost_chart(result: dict) -> alt.Chart:
    rows = [
        {"ratio": r["ratio"], "model": name, "saving": r["saving_vs_best_trivial"]}
        for name, key in (
            ("LightGBM (calibrated)", "lightgbm"),
            ("14-day mean out-of-stock hours", "mean_oos_hours_14d"),
        )
        for r in result["cost"][key]
    ]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_bar()
        .encode(
            x=alt.X("ratio:O", title="cost of a missed stockout / cost of one check"),
            y=alt.Y(
                "saving:Q",
                title="saving vs best of never / always check",
                axis=alt.Axis(format="%"),
            ),
            xOffset="model:N",
            color=alt.Color(
                "model:N",
                title=None,
                scale=alt.Scale(
                    domain=["LightGBM (calibrated)", "14-day mean out-of-stock hours"],
                    range=[
                        PALETTE["LightGBM (calibrated)"],
                        PALETTE["14-day mean out-of-stock hours"],
                    ],
                ),
            ),
        )
        .properties(
            width=380,
            height=300,
            title="What a threshold chosen on validation saves on test",
        )
    )


def importance_chart(result: dict, top: int = 15) -> alt.Chart:
    df = pl.DataFrame(result["importance"][:top])
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("feature:N", sort="-x", title=None),
            x=alt.X("mean_abs_shap:Q", title="mean |SHAP| (log-odds)"),
            color=alt.Color("group:N", title="feature group"),
        )
        .properties(width=380, height=340, title="What the model uses")
    )


def markdown(result: dict) -> str:
    r = result
    base = r["base_rate"]
    lgb, raw = r["lightgbm"], r["lightgbm_raw"]
    names = {
        "yesterday": "Yesterday's state",
        "same_weekday_last_week": "Same weekday last week",
        "heavy_rate_7d": "Heavy-day rate, 7 days",
        "mean_oos_hours_14d": "Mean out-of-stock hours, 14 days",
    }
    metric_rows = [
        [names[k], _f(v["pr_auc"]), _f(v["roc_auc"]), "-", "-"]
        for k, v in r["baselines"].items()
    ]
    metric_rows += [
        [
            "Logistic regression",
            *(_f(r["logistic"][m]) for m in ("pr_auc", "roc_auc", "log_loss", "brier")),
        ],
        [
            "LightGBM (raw)",
            *(_f(raw[m]) for m in ("pr_auc", "roc_auc", "log_loss", "brier")),
        ],
        [
            "**LightGBM (isotonic)**",
            *(f"**{_f(lgb[m])}**" for m in ("pr_auc", "roc_auc", "log_loss", "brier")),
        ],
    ]
    out = [
        "# Retail Stockout Risk Classifier: results",
        "",
        (
            f"Label: {r['label_hours']}+ out-of-stock hours (of 16) the next day. Base rate "
            f"{_pct(base['train'])} train, {_pct(base['validation'])} validation, {_pct(base['test'])} test."
        ),
        (
            f"Label days: train {r['label_days']['train'][0]} to {r['label_days']['train'][1]}, validation "
            f"{r['label_days']['validation'][0]} to {r['label_days']['validation'][1]}, test "
            f"{r['label_days']['test'][0]} to {r['label_days']['test'][1]} ({r['rows']['test']:,} test rows)."
        ),
        "",
        "## Models on the test window",
        "",
        _table(["Model", "PR-AUC", "ROC-AUC", "Log-loss", "Brier"], metric_rows),
        "",
        f"Calibration (ECE): raw {_f(raw['ece'], 4)}, isotonic {_f(lgb['ece'], 4)}.",
        "",
        "## Worklist: check the top share of each day's store-products",
        "",
    ]
    rows = []
    for i, frac in enumerate(r["worklist"]["lightgbm"]):
        ref = r["worklist"]["mean_oos_hours_14d"][i]
        rows.append(
            [
                _pct(frac["fraction"]),
                _pct(frac["precision"]),
                _f(frac["lift"], 2),
                _pct(frac["recall"]),
                _pct(ref["precision"]),
                _f(ref["lift"], 2),
            ]
        )
    out += [
        _table(
            [
                "Top share",
                "Precision",
                "Lift",
                "Recall",
                "14-day baseline precision",
                "Baseline lift",
            ],
            rows,
        ),
        "",
        "## Cost of a threshold",
        "",
        (
            "One check costs 1, a missed stockout costs `ratio`; a check is assumed to prevent the stockout "
            "it finds, which makes the saving an upper bound."
        ),
        "",
    ]
    rows = [
        [
            c["ratio"],
            _f(c["threshold"]),
            _f(c["bayes"]),
            _pct(c["checked_share"]),
            _f(c["cost"]),
            _f(min(c["never"], c["always"])),
            _pct(c["saving_vs_best_trivial"]),
        ]
        for c in r["cost"]["lightgbm"]
    ]
    out += [
        _table(
            [
                "Ratio",
                "Val-picked threshold",
                "1/ratio",
                "Checked",
                "Cost",
                "Best trivial",
                "Saving",
            ],
            rows,
        ),
        "",
        "## Newly at risk",
        "",
        (
            f"Rows that were not a heavy stockout day today: {r['newly_at_risk']['rows']:,}, "
            f"base rate {_pct(r['newly_at_risk']['base_rate'])}. LightGBM PR-AUC "
            f"{_f(r['newly_at_risk']['lightgbm']['pr_auc'])} / ROC-AUC {_f(r['newly_at_risk']['lightgbm']['roc_auc'])}; "
            f"14-day baseline {_f(r['newly_at_risk']['mean_oos_hours_14d']['pr_auc'])} / "
            f"{_f(r['newly_at_risk']['mean_oos_hours_14d']['roc_auc'])}."
        ),
        "",
    ]
    cats = r["per_category"]
    wins = sum(c["pr_auc"] > c["baseline_pr_auc"] for c in cats)
    best = max(cats, key=lambda c: c["pr_auc"] - c["baseline_pr_auc"])
    worst = min(cats, key=lambda c: c["pr_auc"] - c["baseline_pr_auc"])
    out += [
        "## Categories",
        "",
        (
            f"The model beats the 14-day baseline's PR-AUC in {wins} of {len(cats)} first-level "
            f"categories. Largest gain: category {best['group']} "
            f"({_f(best['pr_auc'])} vs {_f(best['baseline_pr_auc'])}); smallest: category "
            f"{worst['group']} ({_f(worst['pr_auc'])} vs {_f(worst['baseline_pr_auc'])})."
        ),
        "",
    ]
    if "ablation" in r:
        out += [
            "## Feature groups (drop one, refit)",
            "",
            _table(
                ["Dropped", "PR-AUC", "Change vs full"],
                [
                    [a["dropped"], _f(a["pr_auc"]), f"{a['delta_pr_auc']:+.4f}"]
                    for a in r["ablation"]
                ],
            ),
            "",
        ]
    if "label_definitions" in r:
        out += [
            "## Label definitions",
            "",
            _table(
                [
                    "Stockout means",
                    "Base rate",
                    "Yesterday PR-AUC",
                    "14-day PR-AUC",
                    "LightGBM PR-AUC",
                    "LightGBM ROC-AUC",
                ],
                [
                    [
                        f"{d['hours']}+ hours",
                        _pct(d["base_rate"]),
                        _f(d["yesterday"]["pr_auc"]),
                        _f(d["mean_oos_hours_14d"]["pr_auc"]),
                        _f(d["lightgbm"]["pr_auc"]),
                        _f(d["lightgbm"]["roc_auc"]),
                    ]
                    for d in r["label_definitions"]
                ],
            ),
            "",
        ]
    out += [
        "## Censored sales",
        "",
        _table(
            [
                "Out-of-stock hours",
                "Rows",
                "Mean units sold",
                "In-stock sales rate x 16 h",
            ],
            [
                [
                    c["oos_hours"],
                    f"{c['rows']:,}",
                    _f(c["mean_sale"], 2),
                    _f(c["mean_demand_16h"], 2),
                ]
                for c in r["censoring"]
            ],
        ),
        "",
        "## Top features (mean |SHAP|, log-odds)",
        "",
        _table(
            ["Feature", "Group", "Mean abs SHAP", "Gain share"],
            [
                [
                    i["feature"],
                    i["group"],
                    _f(i["mean_abs_shap"], 4),
                    _pct(i["gain_share"]),
                ]
                for i in r["importance"][:12]
            ],
        ),
        "",
        "## Drift",
        "",
        "PSI of features, train to test: "
        + ", ".join(
            f"{k} {_f(v, 3)}" for k, v in r["drift"]["psi_train_to_test"].items()
        )
        + ".",
        "",
    ]
    return "\n".join(out)


def write(out: dict, directory: Path) -> None:
    result, arrays = out["result"], out["arrays"]
    (directory / "report.md").write_text(markdown(result), encoding="utf-8")
    charts = {
        "reliability.png": reliability_chart(result),
        "pr_curve.png": pr_chart(arrays["y_test"], arrays["p"]),
        "cost.png": cost_chart(result),
        "importance.png": importance_chart(result),
    }
    for name, chart in charts.items():
        chart.save(str(directory / name), scale_factor=2)
