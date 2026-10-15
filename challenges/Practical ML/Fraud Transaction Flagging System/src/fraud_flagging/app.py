"""Streamlit explorer: choose how to flag, see precision, recall, cost and the review load it implies.

Run with ``uv run streamlit run src/fraud_flagging/app.py`` after ``uv run fraud evaluate``.
"""

from __future__ import annotations

import json

import altair as alt
import numpy as np
import polars as pl
import streamlit as st

from fraud_flagging import evaluate
from fraud_flagging.paths import data_dir, results_dir

POLICY_LABELS = {
    "precision": "Precision target",
    "recall": "Recall target",
    "cost": "Cost (amount-aware)",
    "capacity": "Review capacity",
}


@st.cache_data
def load(path: str, mtime: float) -> pl.DataFrame:
    return pl.read_parquet(path)


def main() -> None:
    st.set_page_config(page_title="Fraud flagging", layout="wide")
    st.title("Fraud Transaction Flagging System")
    pred_path = data_dir() / "predictions.parquet"
    meta_path = data_dir() / "model" / "model.json"
    metrics_path = results_dir() / "metrics.json"
    if not (pred_path.exists() and meta_path.exists() and metrics_path.exists()):
        st.error(
            "No results yet. Run `uv run fraud evaluate` in the challenge folder first."
        )
        st.stop()
    preds = load(str(pred_path), pred_path.stat().st_mtime)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.caption(
        "ULB credit-card fraud, 48 hours. Thresholds are chosen on the validation window (hours 24-36) "
        "and shown on the test window (36-48). Amounts are in the data's own currency units."
    )

    policy = st.sidebar.radio(
        "How to flag", list(POLICY_LABELS), format_func=POLICY_LABELS.get
    )
    test = preds.filter(pl.col("part") == "test")
    y, p = test["y"].to_numpy(), test["p"].to_numpy()
    amount, hour = test["Amount"].to_numpy(), test["hour"].to_numpy()
    review_cost = st.sidebar.slider(
        "Cost of one review",
        0.5,
        200.0,
        25.0,
        0.5,
        help="Used by the cost policy and for the total cost shown for every policy.",
    )

    if policy == "precision":
        target = st.sidebar.select_slider(
            "Precision target",
            [float(k) for k in meta["thresholds"]["precision"]],
            value=0.9,
        )
        thr = meta["thresholds"]["precision"][str(target)]
        flag = np.zeros(len(y), bool) if thr is None else p >= thr
    elif policy == "recall":
        target = st.sidebar.select_slider(
            "Recall target", [float(k) for k in meta["thresholds"]["recall"]], value=0.8
        )
        thr = meta["thresholds"]["recall"][str(target)]
        flag = np.zeros(len(y), bool) if thr is None else p >= thr
    elif policy == "cost":
        flag = evaluate.amount_aware_flag(p, amount, review_cost)
    else:
        per_hour = st.sidebar.slider("Reviews per hour", 1, 30, 3)
        flag = evaluate.capacity_flag(p * amount, hour, per_hour)

    c = evaluate.confusion(y, flag)
    cost = evaluate.policy_cost(y, amount, flag, review_cost)
    triv = evaluate.trivial_costs(y, amount, review_cost)
    cols = st.columns(6)
    cols[0].metric(
        "Flagged",
        f"{c['flagged']:,}",
        f"{c['alerts_per_10k']:.1f} per 10k",
        delta_color="off",
    )
    cols[1].metric(
        "Precision", "-" if c["precision"] is None else f"{c['precision']:.1%}"
    )
    cols[2].metric("Recall", "-" if c["recall"] is None else f"{c['recall']:.1%}")
    cols[3].metric(
        "Fraud amount caught",
        f"{cost['caught_amount']:,.0f}",
        f"of {amount[y == 1].sum():,.0f}",
        delta_color="off",
    )
    cols[4].metric(
        "Total cost",
        f"{cost['total']:,.0f}",
        help="Missed fraud amounts plus one review per flagged transaction.",
    )
    cols[5].metric(
        "Saving vs flag nothing",
        f"{1 - cost['total'] / triv['flag_nothing']:.1%}",
        f"flag everything: {triv['flag_everything']:,.0f}",
        delta_color="off",
    )

    left, right = st.columns(2)
    recall, precision = evaluate.pr_points(y, p)
    curve = pl.DataFrame({"recall": recall, "precision": precision})
    here = pl.DataFrame(
        {"recall": [c["recall"] or 0.0], "precision": [c["precision"] or 0.0]}
    )
    line = (
        alt.Chart(curve)
        .mark_line()
        .encode(
            x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("precision:Q", scale=alt.Scale(domain=[0, 1])),
        )
    )
    dot = (
        alt.Chart(here)
        .mark_point(size=160, filled=True, color="crimson")
        .encode(x="recall:Q", y="precision:Q")
    )
    left.subheader("Precision-recall (test)")
    left.altair_chart(line + dot, width="stretch")

    hourly = (
        pl.DataFrame({"hour": hour, "flagged": flag.astype(int), "fraud": y})
        .group_by("hour")
        .agg(pl.col("flagged").sum(), pl.col("fraud").sum())
        .sort("hour")
        .unpivot(index="hour", variable_name="what", value_name="count")
    )
    right.subheader("Alerts and frauds per hour (test)")
    right.altair_chart(
        alt.Chart(hourly)
        .mark_bar()
        .encode(x="hour:O", y="count:Q", color="what:N", xOffset="what:N"),
        width="stretch",
    )

    st.subheader("Highest-risk transactions (test)")
    shown = test.with_columns(
        pl.Series("flagged", flag),
        (pl.col("p") * pl.col("Amount")).alias("expected_loss"),
    )
    st.dataframe(
        shown.sort("p", descending=True)
        .head(50)
        .select(
            "tx_id",
            "hour",
            "Amount",
            pl.col("p").alias("risk"),
            "expected_loss",
            "flagged",
            pl.col("y").alias("was fraud"),
        ),
        hide_index=True,
        column_config={
            "risk": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.3f"
            )
        },
    )
    stab = [s for s in metrics["threshold_stability"] if s.get("resamples")]
    if stab:
        st.subheader("How firmly a threshold is known")
        st.dataframe(
            pl.DataFrame(
                [
                    {
                        "precision target": s["target"],
                        "threshold 5-95%": f"{s['threshold']['p5']:.3f} to {s['threshold']['p95']:.3f}",
                        "test recall 5-95%": f"{s['test_recall']['p5']:.0%} to {s['test_recall']['p95']:.0%}",
                    }
                    for s in stab
                ]
            ),
            hide_index=True,
        )


main()
