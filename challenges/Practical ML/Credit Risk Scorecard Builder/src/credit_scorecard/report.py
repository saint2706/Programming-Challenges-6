"""Charts (Altair / Vega-Lite) and the written report.

Altair renders in Streamlit directly and exports to PNG through ``vl-convert`` (Rust, no browser).
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
from sklearn.metrics import roc_curve

from credit_scorecard import data
from credit_scorecard.pipeline import Result
from credit_scorecard.scorecard import Scorecard


def woe_chart(card: Scorecard) -> alt.Chart:
    rows = [
        {
            "feature": data.LABELS[f],
            "bin": b.label,
            "order": i,
            "WOE": b.woe,
            "bad rate": b.bad_rate,
            "n": b.count,
        }
        for f in card.features
        for i, b in enumerate(card.binnings[f].bins)
    ]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_bar()
        .encode(
            x=alt.X(
                "bin:N",
                sort=alt.EncodingSortField("order"),
                title=None,
                axis=alt.Axis(labelAngle=-40),
            ),
            y=alt.Y("WOE:Q"),
            color=alt.condition(
                alt.datum.WOE > 0, alt.value("#2a9d8f"), alt.value("#e76f51")
            ),
            tooltip=[
                "feature",
                "bin",
                "WOE",
                alt.Tooltip("bad rate", format=".1%"),
                "n",
            ],
        )
        .properties(width=170, height=130)
        .facet(
            facet=alt.Facet(
                "feature:N", title=None, sort=[data.LABELS[f] for f in card.features]
            ),
            columns=4,
        )
        .resolve_scale(x="independent")
    )


def roc_chart(result: Result) -> alt.Chart:
    y = result.test[data.TARGET].to_numpy()
    frames = []
    for name, risk in result.risk.items():
        fpr, tpr, _ = roc_curve(y, risk)
        frames.append(
            pl.DataFrame(
                {"model": name, "false positive rate": fpr, "true positive rate": tpr}
            )
        )
    diagonal = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    return alt.layer(
        alt.Chart(pl.concat(frames))
        .mark_line()
        .encode(x="false positive rate:Q", y="true positive rate:Q", color="model:N"),
        alt.Chart(diagonal)
        .mark_line(strokeDash=[4, 4], color="gray")
        .encode(x="x:Q", y="y:Q"),
    ).properties(width=320, height=320, title="ROC on the test set")


def score_distribution_chart(result: Result) -> alt.Chart:
    score = result.scorecard.score(result.test)
    frame = pl.DataFrame(
        {
            "score": score,
            "outcome": np.where(
                result.test[data.TARGET].to_numpy() == 1, "bad", "good"
            ),
        }
    )
    return (
        alt.Chart(frame)
        .transform_density(
            "score",
            groupby=["outcome"],
            as_=["score", "density"],
            extent=[score.min(), score.max()],
        )
        .mark_area(opacity=0.5)
        .encode(
            x="score:Q",
            y="density:Q",
            color=alt.Color("outcome:N", scale=alt.Scale(range=["#2a9d8f", "#e76f51"])),
        )
        .properties(width=420, height=220, title="Test-set score by outcome")
    )


def calibration_chart(result: Result) -> alt.Chart:
    cal = result.calibration.select(
        "band",
        pl.col("observed_bad_rate").alias("observed"),
        pl.col("predicted_pd").alias("predicted"),
    ).unpivot(index="band", variable_name="series", value_name="bad rate")
    return (
        alt.Chart(cal)
        .mark_line(point=True)
        .encode(
            x=alt.X("band:O", title="score decile (0 = riskiest)"),
            y=alt.Y(
                "bad rate:Q", scale=alt.Scale(type="log"), axis=alt.Axis(format="%")
            ),
            color="series:N",
        )
        .properties(
            width=420,
            height=220,
            title="Observed bad rate vs scorecard PD, by score decile",
        )
    )


def points_chart(card: Scorecard, applicant_points: dict[str, int]) -> alt.Chart:
    rows = [
        {
            "feature": data.LABELS[f],
            "points": applicant_points[f],
            "best possible": max(card.points[f]),
        }
        for f in card.features
    ]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("feature:N", sort="-x", title=None)
    )
    return (
        base.mark_bar(color="#cbd5e1").encode(
            x=alt.X("best possible:Q", title="points")
        )
        + base.mark_bar(color="#2a9d8f", size=10).encode(x="points:Q")
    ).properties(width=420, height=220)


def markdown_table(df: pl.DataFrame, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(df.columns) + " |\n|" + " --- |" * len(df.columns)
    body = [
        "| "
        + " | ".join(
            format(v, fmt.get(c, "")) if v is not None else ""
            for c, v in zip(df.columns, row, strict=True)
        )
        + " |"
        for row in df.iter_rows()
    ]
    return "\n".join([head, *body])


def scorecard_table(card: Scorecard) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "feature": data.LABELS[f],
                "bin": b.label,
                "n": b.count,
                "bad rate": b.bad_rate,
                "WOE": b.woe,
                "points": card.points[f][i],
            }
            for f in card.features
            for i, b in enumerate(card.binnings[f].bins)
        ]
    )


def write_report(result: Result, out_dir: Path) -> Path:
    """Write ``scorecard.json``, ``metrics.json``, four PNG charts and ``report.md`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    card, m = result.scorecard, result.metrics
    card.save(out_dir / "scorecard.json")
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    charts = {
        "woe": woe_chart(card),
        "roc": roc_chart(result),
        "score_distribution": score_distribution_chart(result),
        "calibration": calibration_chart(result),
    }
    for name, chart in charts.items():
        chart.save(out_dir / f"{name}.png", scale_factor=2)

    sc, gb = m["scorecard_test"], m["gbm_test"]
    lo, hi = m["scorecard_test_auc_ci95"]
    selection = pl.DataFrame(card.selection).with_columns(
        pl.col("feature").replace(data.LABELS)
    )
    report = f"""# Credit risk scorecard: results

Give Me Some Credit, {m["n_train"]:,} train / {
        m["n_test"]:,} test rows (stratified random split),
bad rate {m["bad_rate_train"]:.2%}.

## Discrimination (test set)

{
        markdown_table(
            pl.DataFrame(
                {
                    "model": ["Scorecard", "Gradient boosting (untuned)"],
                    "AUC": [sc["auc"], gb["auc"]],
                    "Gini": [sc["gini"], gb["gini"]],
                    "KS": [sc["ks"], gb["ks"]],
                }
            ),
            {"AUC": ".4f", "Gini": ".4f", "KS": ".4f"},
        )
    }

Scorecard AUC 95% bootstrap interval: [{lo:.4f}, {hi:.4f}]. Train AUC {
        m["scorecard_train"]["auc"]:.4f}.

![ROC](roc.png)

## Stability

Score PSI, train vs test: **{m["score_psi_train_vs_test"]:.4f}** (below 0.1 is stable).

{
        markdown_table(
            pl.DataFrame(
                {
                    "feature": [data.LABELS[f] for f in result.feature_psi],
                    "PSI": list(result.feature_psi.values()),
                }
            ),
            {"PSI": ".5f"},
        )
    }

## Calibration

Mean predicted PD {m["mean_predicted_pd_test"]:.2%} against observed {
        m["bad_rate_test"]:.2%}; Brier {m["brier_test"]:.4f}.

{
        markdown_table(
            result.calibration, {"observed_bad_rate": ".2%", "predicted_pd": ".2%"}
        )
    }

![Calibration](calibration.png)
![Score distribution](score_distribution.png)

## Feature selection

{markdown_table(selection, {"iv": ".3f"})}

## The scorecard

Base score {card.base_score:.0f} at {card.base_odds:.0f}:1 good:bad odds, {
        card.pdo:.0f} points to double the odds.

{markdown_table(scorecard_table(card), {"bad rate": ".2%", "WOE": "+.3f"})}

![WOE by bin](woe.png)
"""
    path = out_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    return path
