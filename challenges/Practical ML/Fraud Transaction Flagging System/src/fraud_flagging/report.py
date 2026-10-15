"""``results/report.md`` and the charts, written from the experiment's ``result`` dict."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import polars as pl

from fraud_flagging import evaluate

NAMES = {
    "isolation_forest": "Isolation forest (no labels)",
    "logistic": "Logistic regression",
    "logistic_balanced": "Logistic regression, balanced weights",
    "lightgbm_none": "LightGBM, no weighting",
    "lightgbm_sqrt": "LightGBM, sqrt weighting",
    "lightgbm_balanced": "LightGBM, balanced weighting",
    "lightgbm_undersample": "LightGBM, undersampled 20:1",
}


def _table(headers: list[str], rows: list[list]) -> str:
    """A markdown table with padded columns, laid out the way the repo's formatter (dprint) does."""
    cells = [[str(c) for c in headers], *[[str(c) for c in row] for row in rows]]
    widths = [max(3, *(len(r[i]) for r in cells)) for i in range(len(headers))]

    def line(row: list[str]) -> str:
        return (
            "| "
            + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True))
            + " |"
        )

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([line(cells[0]), sep, *[line(r) for r in cells[1:]]])


def _f(x, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _ci(pair, pct: bool = False) -> str:
    lo, hi = pair
    if lo is None:
        return "-"
    return f"{_pct(lo)} to {_pct(hi)}" if pct else f"{lo:,.0f} to {hi:,.0f}"


def _money(x) -> str:
    return f"{x:,.0f}"


def operating_points(result: dict) -> pl.DataFrame:
    """Every operating point as (recall, precision, kind, label), for the PR chart."""
    rows = []
    for target, p in result["fixed_precision"].items():
        if p and p["precision"] is not None and float(target) >= 0.8:
            rows.append(
                (p["recall"], p["precision"], "precision target", f"P>={target}")
            )
    for c in result["cost"]:
        a = c["amount_aware"]
        if a["precision"] is not None and c["review_cost"] in (5.0, 25.0, 100.0):
            rows.append(
                (
                    a["recall"],
                    a["precision"],
                    "cost (amount-aware)",
                    f"review {c['review_cost']:g}",
                )
            )
    for cap in result["capacity"]:
        b = cap["by_expected_loss"]
        if b["precision"] is not None:
            rows.append(
                (b["recall"], b["precision"], "capacity", f"{cap['per_hour']}/hour")
            )
    return pl.DataFrame(
        rows, schema=["recall", "precision", "kind", "label"], orient="row"
    )


def pr_chart(result: dict, arrays: dict) -> alt.Chart:
    y = arrays["y_test"]
    series = {
        "LightGBM, calibrated (main)": arrays["p"],
        "Logistic regression": arrays["scores"]["logistic"],
        "Isolation forest": arrays["scores"]["isolation_forest"],
    }
    frames = []
    for name, s in series.items():
        recall, precision = evaluate.pr_points(y, s)
        frames.append(
            pl.DataFrame({"recall": recall, "precision": precision, "model": name})
        )
    curves = (
        alt.Chart(pl.concat(frames))
        .mark_line()
        .encode(
            x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("precision:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model:N", title=None),
        )
    )
    pts = operating_points(result)
    marks = (
        alt.Chart(pts)
        .mark_point(size=90, filled=True, opacity=0.9)
        .encode(
            x="recall:Q",
            y="precision:Q",
            shape=alt.Shape("kind:N", title="operating point"),
            tooltip=["label:N"],
        )
    )
    labels = (
        alt.Chart(pts)
        .mark_text(dx=8, dy=-8, fontSize=10)
        .encode(x="recall:Q", y="precision:Q", text="label:N")
    )
    return (curves + marks + labels).properties(
        width=520,
        height=380,
        title="Precision-recall on the test window, with the operating points",
    )


def cost_chart(result: dict) -> alt.Chart:
    rows = []
    for r in result["cost_sweep"]:
        for key, name in (
            ("flag_nothing", "flag nothing"),
            ("flag_everything", "flag everything"),
            ("static", "static threshold"),
            ("amount_aware", "amount-aware"),
            ("oracle", "oracle (best possible)"),
        ):
            rows.append(
                {"review_cost": r["review_cost"], "policy": name, "cost": r[key]}
            )
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "review_cost:Q", scale=alt.Scale(type="log"), title="cost of one review"
            ),
            y=alt.Y(
                "cost:Q",
                scale=alt.Scale(type="log"),
                title="total cost on the test window",
            ),
            color=alt.Color("policy:N", title=None),
        )
        .properties(
            width=520,
            height=340,
            title="What each rule costs as the review cost changes",
        )
    )


def stability_chart(result: dict) -> alt.Chart:
    rows = []
    for s in result["threshold_stability"]:
        if s.get("resamples"):
            for metric in ("test_precision", "test_recall"):
                rows.append(
                    {
                        "target": f"P >= {s['target']}",
                        "metric": metric.replace("test_", ""),
                        **s[metric],
                    }
                )
    df = pl.DataFrame(rows)
    base = alt.Chart(df).encode(
        y=alt.Y("target:N", title="validation precision target"), color="metric:N"
    )
    return (
        base.mark_rule(strokeWidth=4).encode(
            x=alt.X(
                "p5:Q",
                title="on the test window (5th to 95th percentile)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            x2="p95:Q",
        )
        + base.mark_point(filled=True, size=80).encode(x="median:Q")
    ).properties(width=480, height=200, title="If validation had been drawn again")


def reliability_chart(result: dict) -> alt.Chart:
    rows = [
        {"model": name, "mean_p": r["mean_p"], "rate": r["rate"], "n": r["n"]}
        for name, key in (
            ("raw (undersampled)", "reliability_raw"),
            ("after Platt scaling", "reliability_platt"),
        )
        for r in result["calibration"][key]
    ]
    diag = (
        alt.Chart(pl.DataFrame({"x": [1e-4, 1.0], "y": [1e-4, 1.0]}))
        .mark_line(color="#999", strokeDash=[4, 4])
        .encode(
            x=alt.X(
                "x:Q", title="mean predicted probability", scale=alt.Scale(type="log")
            ),
            y=alt.Y("y:Q", title="observed fraud rate", scale=alt.Scale(type="log")),
        )
    )
    pts = (
        alt.Chart(pl.DataFrame(rows).filter(pl.col("rate") > 0))
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "mean_p:Q",
                title="mean predicted probability",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y("rate:Q", title="observed fraud rate", scale=alt.Scale(type="log")),
            color=alt.Color("model:N", title=None),
        )
    )
    return (diag + pts).properties(
        width=380, height=320, title="Calibration on the test window"
    )


def importance_chart(result: dict, top: int = 15) -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame(result["importance"][:top]))
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
    out = [
        "# Fraud Transaction Flagging System: results",
        "",
        (
            f"Train on hours 0-{r['windows']['validation_from_hour']:g}, validate on "
            f"{r['windows']['validation_from_hour']:g}-{r['windows']['test_from_hour']:g}, test on "
            f"{r['windows']['test_from_hour']:g}-48."
        ),
        "",
        _table(
            ["Window", "Transactions", "Frauds", "Fraud rate", "Fraud amount"],
            [
                [
                    w,
                    f"{r['rows'][w]:,}",
                    r["frauds"][w],
                    f"{100 * r['fraud_rate'][w]:.3f}%",
                    _money(r["fraud_amount"][w]),
                ]
                for w in ("train", "validation", "test")
            ],
        ),
        "",
        "## Models (PR-AUC; 95% interval for the test value in brackets)",
        "",
        _table(
            ["Model", "Validation PR-AUC", "Test PR-AUC", "Test ROC-AUC"],
            [
                [
                    NAMES[k],
                    _f(c["validation"]["pr_auc"]),
                    f"{_f(c['test']['pr_auc'])} ({_f(c['test_pr_auc_ci'][0], 2)}-{_f(c['test_pr_auc_ci'][1], 2)})",
                    _f(c["test"]["roc_auc"]),
                ]
                for k, c in r["candidates"].items()
            ]
            + [
                [
                    "Amount only",
                    "-",
                    _f(r["baselines"]["amount_only"]["pr_auc"]),
                    _f(r["baselines"]["amount_only"]["roc_auc"]),
                ]
            ]
            + [["Random", "-", _f(r["baselines"]["random"]["pr_auc"]), "0.500"]],
        ),
        "",
        f"Main model, chosen on validation PR-AUC among the LightGBM variants: **{NAMES['lightgbm_' + r['main_strategy']]}**.",
        "",
        "## Calibration",
        "",
        _table(
            ["", "Log-loss", "Brier"],
            [
                [
                    "Raw",
                    _f(r["calibration"]["raw"]["log_loss"], 5),
                    _f(r["calibration"]["raw"]["brier"], 5),
                ],
                [
                    "Platt scaled",
                    _f(r["calibration"]["platt_scaled"]["log_loss"], 5),
                    _f(r["calibration"]["platt_scaled"]["brier"], 5),
                ],
            ],
        ),
        "",
        "## Operating points chosen on validation, judged on test",
        "",
        "### Fixed precision",
        "",
        _table(
            [
                "Target",
                "Threshold",
                "Val precision / recall",
                "Test precision (95%)",
                "Test recall (95%)",
                "Flagged",
                "Fraud amount caught",
            ],
            [
                [
                    t,
                    _f(p["threshold"], 4),
                    f"{_pct(p['validation']['precision'])} / {_pct(p['validation']['recall'])}",
                    f"{_pct(p['precision'])} ({_ci(p['precision_ci'], True)})",
                    f"{_pct(p['recall'])} ({_ci(p['recall_ci'], True)})",
                    p["flagged"],
                    _money(p["fraud_amount_caught"]),
                ]
                for t, p in r["fixed_precision"].items()
                if p
            ],
        ),
        "",
        "### Fixed recall",
        "",
        _table(
            [
                "Target",
                "Threshold",
                "Test precision",
                "Test recall",
                "Flagged",
                "Alerts per 10k",
            ],
            [
                [
                    t,
                    _f(p["threshold"], 4),
                    _pct(p["precision"]),
                    _pct(p["recall"]),
                    p["flagged"],
                    f"{p['alerts_per_10k']:.1f}",
                ]
                for t, p in r["fixed_recall"].items()
                if p
            ],
        ),
        "",
        "### Cost (review cost c per flagged transaction; a missed fraud loses its amount)",
        "",
        _table(
            [
                "c",
                "Flag nothing",
                "Flag everything",
                "Static threshold (flagged)",
                "Amount-aware (flagged)",
                "Oracle",
                "Amount-aware 95%",
            ],
            [
                [
                    f"{c['review_cost']:g}",
                    _money(c["trivial_flag_nothing"]),
                    _money(c["trivial_flag_everything"]),
                    f"{_money(c['static']['total'])} ({c['static']['flagged']})",
                    f"{_money(c['amount_aware']['total'])} ({c['amount_aware']['flagged']})",
                    _money(c["trivial_oracle"]),
                    _ci(c["amount_aware"]["cost_ci"]),
                ]
                for c in r["cost"]
            ],
        ),
        "",
        "### Review capacity (reviews per hour, highest expected loss first)",
        "",
        _table(
            [
                "Per hour",
                "Flagged",
                "Frauds caught",
                "Precision",
                "Recall",
                "Fraud amount caught",
            ],
            [
                [
                    c["per_hour"],
                    c["by_expected_loss"]["flagged"],
                    c["by_expected_loss"]["tp"],
                    _pct(c["by_expected_loss"]["precision"]),
                    _pct(c["by_expected_loss"]["recall"]),
                    _money(c["by_expected_loss"]["fraud_amount_caught"]),
                ]
                for c in r["capacity"]
            ],
        ),
        "",
        "## How much a threshold moves with the validation draw",
        "",
        _table(
            [
                "Precision target",
                "Threshold (5-95%)",
                "Test precision (5-95%)",
                "Test recall (5-95%)",
            ],
            [
                [
                    s["target"],
                    f"{_f(s['threshold']['p5'], 3)} to {_f(s['threshold']['p95'], 3)}",
                    f"{_pct(s['test_precision']['p5'])} to {_pct(s['test_precision']['p95'])}",
                    f"{_pct(s['test_recall']['p5'])} to {_pct(s['test_recall']['p95'])}",
                ]
                for s in r["threshold_stability"]
                if s.get("resamples")
            ],
        ),
        "",
    ]
    if "rolling_origin" in r:
        out += [
            "## Rolling origin (train on the past, score the next 6 hours)",
            "",
            _table(
                ["Hours", "Frauds", "LightGBM PR-AUC", "Logistic PR-AUC"],
                [
                    [
                        f"{x['from_hour']:g}-{x['to_hour']:g}",
                        x["frauds"],
                        _f(x["lightgbm"]["pr_auc"]),
                        _f(x["logistic"]["pr_auc"]),
                    ]
                    for x in r["rolling_origin"]
                ],
            ),
            "",
        ]
    if "ablation" in r:
        out += [
            "## Feature groups (drop one, refit)",
            "",
            _table(
                ["Dropped", "Test PR-AUC", "Change"],
                [
                    [a["dropped"], _f(a["pr_auc"]), f"{a['delta_pr_auc']:+.3f}"]
                    for a in r["ablation"]
                ],
            ),
            "",
        ]
    err = r.get("errors_at_precision_0.9")
    if err:
        out += [
            "## Errors at the precision-0.9 point",
            "",
            (
                f"{err['caught']} frauds caught, {err['missed']} missed. Median amount caught {_f(err['median_amount_caught'], 2)}, "
                f"missed {_f(err['median_amount_missed'], 2)}; the median score of a missed fraud is {_f(err['median_score_missed'], 4)}."
            ),
            "",
        ]
    out += [
        "## Top features (mean |SHAP|)",
        "",
        _table(
            ["Feature", "Group", "Mean abs SHAP", "Gain share"],
            [
                [
                    i["feature"],
                    i["group"],
                    _f(i["mean_abs_shap"], 3),
                    _pct(i["gain_share"]),
                ]
                for i in r["importance"][:10]
            ],
        ),
        "",
        "## Drift (PSI, train to test)",
        "",
        ", ".join(f"{k} {_f(v, 3)}" for k, v in r["drift"]["psi_train_to_test"].items())
        + ".",
        "",
    ]
    return "\n".join(out)


def write(out: dict, directory: Path) -> None:
    result, arrays = out["result"], out["arrays"]
    (directory / "report.md").write_text(
        markdown(result).rstrip() + "\n", encoding="utf-8", newline="\n"
    )
    charts = {
        "pr_curve.png": pr_chart(result, arrays),
        "cost.png": cost_chart(result),
        "stability.png": stability_chart(result),
        "reliability.png": reliability_chart(result),
        "importance.png": importance_chart(result),
    }
    for name, chart in charts.items():
        chart.save(str(directory / name), scale_factor=2)
