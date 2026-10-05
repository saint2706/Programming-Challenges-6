"""Streamlit dashboard: drift over time, what drifted in a window, and what it cost in accuracy.

Run:  uv run streamlit run app.py
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import pipeline
from data import FEATURES
from monitor import columns_of

st.set_page_config(page_title="Model Drift Monitor", layout="wide")


@st.cache_resource(show_spinner="Loading the model and scanning the live stream...")
def get_state():
    art, live = pipeline.load_context()
    return art, live, pipeline.run_monitor(art, live)


art, live, res = get_state()
table, alerts = res.table, res.alerts
thresholds = art["thresholds"]
signals = ["score", *FEATURES]

st.title("Model Drift Monitor")
st.caption(
    f"Weekly windows of {art['window']} rows against the training baseline; labels arrive "
    f"{art['delay']} windows late, so accuracy lags. An alert means the inputs or scores moved, "
    "not that the model is wrong: compare with the accuracy panel."
)

with st.sidebar:
    signal = st.selectbox("Signal", signals, key="signal")
    stats = (
        ["psi", "js", "chi2"] if signal == "day" else ["psi", "ks", "js", "wasserstein"]
    )
    stat = st.selectbox("Statistic", stats, key="stat")
    window = st.selectbox(
        "Window to inspect", list(table["window"]), index=len(table) // 2, key="window"
    )

col = f"{signal}_{stat}"
series = table.select("window", col).to_pandas().rename(columns={col: "value"})
limit = thresholds.get((signal, stat))
hits = alerts.filter(
    (alerts["signal"] == signal) & (alerts["stat"] == stat)
).to_pandas()

st.subheader(f"{signal} / {stat} per window")
line = (
    alt.Chart(series).mark_line().encode(x="window:Q", y=alt.Y("value:Q", title=stat))
)
layers = [line]
if limit is not None:
    layers.append(
        alt.Chart(pd.DataFrame({"y": [limit]}))
        .mark_rule(strokeDash=[4, 4], color="firebrick")
        .encode(y="y:Q")
    )
if not hits.empty:
    layers.append(
        alt.Chart(hits)
        .mark_point(color="firebrick", size=60, filled=True)
        .encode(x="window:Q", y="value:Q")
    )
st.altair_chart(alt.layer(*layers), use_container_width=True)
st.caption(
    "Dashed line: calibrated threshold (about 1% of no-drift windows exceed it). Dots: alerts."
)

left, right = st.columns(2)
with left:
    st.subheader(f"Accuracy (released with a {art['delay']}-window delay)")
    perf = (
        table.filter(table["perf_window"].is_not_null())
        .select("perf_window", "accuracy")
        .to_pandas()
    )
    acc_line = (
        alt.Chart(perf)
        .mark_line()
        .encode(
            x=alt.X("perf_window:Q", title="window"),
            y=alt.Y("accuracy:Q", scale=alt.Scale(zero=False)),
        )
    )
    seq = res.sequential.to_pandas()
    if seq.empty:
        st.altair_chart(acc_line, use_container_width=True)
    else:
        marks = (
            alt.Chart(seq)
            .mark_tick(thickness=3, size=14)
            .encode(x="window:Q", y=alt.Y("detector:N", title=None), color="detector:N")
        )
        st.altair_chart(
            alt.vconcat(acc_line, marks).resolve_scale(x="shared"),
            use_container_width=True,
        )
    st.caption(
        "Ticks below: windows in which a sequential detector (ADWIN / Page-Hinkley) alarmed."
    )

with right:
    st.subheader(f"What drifted in window {window}")
    row = table.filter(table["window"] == window).row(0, named=True)
    ranking = []
    for s in signals:
        for st_name in (
            ("psi", "js", "chi2") if s == "day" else ("psi", "ks", "js", "wasserstein")
        ):
            lim = thresholds.get((s, st_name))
            if lim:
                ranking.append(
                    {
                        "signal": s,
                        "stat": st_name,
                        "value": row[f"{s}_{st_name}"],
                        "x threshold": row[f"{s}_{st_name}"] / lim,
                    }
                )
    rank_df = pd.DataFrame(ranking).sort_values("x threshold", ascending=False).head(10)
    st.dataframe(rank_df, hide_index=True, use_container_width=True)
    st.caption("Ratio to the calibrated threshold; above 1 is an alert.")

st.subheader(f"Baseline vs window {window}: {signal}")
start, end = row["start"], row["end"]
cols = columns_of(art["clf"], live[start:end])
base_col = art["baseline"].columns[signal]
baseline_vals = base_col.reference if hasattr(base_col, "reference") else None
if baseline_vals is not None:
    both = pd.DataFrame(
        {
            "value": np.concatenate([baseline_vals, cols[signal]]),
            "set": ["baseline"] * len(baseline_vals)
            + [f"window {window}"] * len(cols[signal]),
        }
    )
    hist = (
        alt.Chart(both)
        .transform_density("value", groupby=["set"], as_=["value", "density"])
        .mark_area(opacity=0.45)
        .encode(x="value:Q", y="density:Q", color="set:N")
    )
    st.altair_chart(hist, use_container_width=True)
else:
    counts = pd.concat(
        [
            pd.DataFrame(
                {
                    "category": base_col.categories.tolist(),
                    "share": base_col.proportions[:-1],
                    "set": "baseline",
                }
            ),
            pd.DataFrame(
                {
                    "category": base_col.categories.tolist(),
                    "share": base_col.bin_counts(cols[signal])[:-1]
                    / max(len(cols[signal]), 1),
                    "set": f"window {window}",
                }
            ),
        ]
    )
    st.altair_chart(
        alt.Chart(counts)
        .mark_bar()
        .encode(x="category:N", y="share:Q", color="set:N", xOffset="set:N"),
        use_container_width=True,
    )
