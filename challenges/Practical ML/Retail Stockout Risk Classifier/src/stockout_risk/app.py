"""Streamlit explorer: pick what a missed stockout costs, see the threshold, the worklist and the fit.

Run with ``uv run streamlit run src/stockout_risk/app.py`` after ``uv run stockout evaluate``.
"""

from __future__ import annotations

import json

import altair as alt
import numpy as np
import polars as pl
import streamlit as st

from stockout_risk import evaluate
from stockout_risk.paths import data_dir, results_dir


@st.cache_data
def load(path: str, mtime: float) -> pl.DataFrame:
    return pl.read_parquet(path)


def main() -> None:
    st.set_page_config(page_title="Stockout Risk", layout="wide")
    st.title("Retail Stockout Risk Classifier")
    pred_path = data_dir() / "predictions.parquet"
    metrics_path = results_dir() / "metrics.json"
    if not pred_path.exists() or not metrics_path.exists():
        st.error(
            "No results yet. Run `uv run stockout evaluate` in the challenge folder first."
        )
        st.stop()
    preds = load(str(pred_path), pred_path.stat().st_mtime)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    hours = metrics["label_hours"]
    st.caption(
        f"Risk of {hours}+ out-of-stock hours (of 16) the next day, FreshRetailNet-50K. "
        "Scores are isotonic-calibrated on the validation window; the threshold below is picked "
        "there too and judged on the test window."
    )

    ratio = st.sidebar.slider(
        "A missed stockout costs this many checks", 1.0, 50.0, 10.0, 0.5
    )
    val = preds.filter(pl.col("part") == "validation")
    test = preds.filter(pl.col("part") == "test")
    y_val, p_val = val["y"].to_numpy(), val["p"].to_numpy()
    y, p = test["y"].to_numpy(), test["p"].to_numpy()

    threshold = evaluate.best_threshold(y_val, p_val, ratio)
    flag = p >= threshold
    tp = int((flag & (y == 1)).sum())
    spent = evaluate.cost(y, p, threshold, ratio)
    trivial = min(ratio * float(y.mean()), 1.0)
    cols = st.columns(5)
    cols[0].metric(
        "Threshold", f"{threshold:.2f}", f"1/ratio = {1 / ratio:.2f}", delta_color="off"
    )
    cols[1].metric("Checked", f"{flag.mean():.1%}")
    cols[2].metric("Precision", f"{tp / max(int(flag.sum()), 1):.1%}")
    cols[3].metric("Recall", f"{tp / max(int(y.sum()), 1):.1%}")
    cols[4].metric(
        "Saving",
        f"{1 - spent / trivial:.1%}",
        help="Against the cheaper of never / always checking.",
    )

    left, right = st.columns(2)
    grid = np.linspace(0.0, 1.0, 101)
    curve = pl.DataFrame(
        {"threshold": grid, "cost": [evaluate.cost(y, p, t, ratio) for t in grid]}
    )
    chart = (
        alt.Chart(curve)
        .mark_line()
        .encode(x="threshold:Q", y=alt.Y("cost:Q", title="cost per row"))
    )
    mark = (
        alt.Chart(pl.DataFrame({"threshold": [threshold]}))
        .mark_rule(color="crimson")
        .encode(x="threshold:Q")
    )
    left.subheader("Cost against threshold (test)")
    left.altair_chart(chart + mark, width="stretch")

    rel = pl.DataFrame(evaluate.reliability(y, p))
    diag = (
        alt.Chart(pl.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[4, 4])
        .encode(x=alt.X("x:Q", title=None), y=alt.Y("y:Q", title=None))
    )
    points = (
        alt.Chart(rel)
        .mark_line(point=True)
        .encode(
            x=alt.X("mean_p:Q", title="mean predicted risk"),
            y=alt.Y("rate:Q", title="observed rate"),
        )
    )
    right.subheader("Reliability (test)")
    right.altair_chart(diag + points, width="stretch")

    st.subheader("Worklist")
    days = sorted(test["label_day"].unique().to_list())
    day = st.selectbox("Day to prepare for", days, index=len(days) - 1)
    top = st.slider("Rows", 5, 200, 25)
    rows = test.filter(pl.col("label_day") == day).sort("p", descending=True).head(top)
    st.dataframe(
        rows.select(
            "store_id",
            "product_id",
            "cat1",
            pl.col("p").alias("risk"),
            pl.col("y").alias("stocked out"),
        ),
        hide_index=True,
        column_config={
            "risk": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )
    st.caption(
        f"{int(rows['y'].sum())} of {rows.height} stocked out ({rows['y'].mean():.0%}) against a base rate of {y.mean():.0%}."
    )

    st.subheader("By category")
    st.dataframe(
        pl.DataFrame(metrics["per_category"]).rename({"group": "category"}),
        hide_index=True,
    )


main()
