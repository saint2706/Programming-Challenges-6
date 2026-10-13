"""Streamlit app: score an applicant, inspect the scorecard, check performance and device speed.

Run with ``uv run streamlit run src/credit_scorecard/app.py``.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import streamlit as st

from credit_scorecard import data, ov_scoring, pipeline, report

st.set_page_config(page_title="Credit Risk Scorecard", layout="wide")

# Widget settings for the raw features: (min, max, default, step).
RANGES = {
    "RevolvingUtilizationOfUnsecuredLines": (0.0, 5.0, 0.3, 0.01),
    "age": (18, 100, 45, 1),
    "NumberOfTime30-59DaysPastDueNotWorse": (0, 20, 0, 1),
    "DebtRatio": (0.0, 5.0, 0.35, 0.01),
    "MonthlyIncome": (0, 50_000, 5_400, 100),
    "NumberOfOpenCreditLinesAndLoans": (0, 40, 8, 1),
    "NumberOfTimes90DaysLate": (0, 20, 0, 1),
    "NumberRealEstateLoansOrLines": (0, 10, 1, 1),
    "NumberOfTime60-89DaysPastDueNotWorse": (0, 20, 0, 1),
    "NumberOfDependents": (0, 10, 0, 1),
}
OPTIONAL = {
    "MonthlyIncome",
    "NumberOfDependents",
}  # missing in the training data, so "unknown" is a bin


@st.cache_resource
def load() -> pipeline.Result:
    return pipeline.run()


result = load()
card, m = result.scorecard, result.metrics

st.title("Credit Risk Scorecard")
st.caption(
    f"Give Me Some Credit ({m['n_train']:,} train / {m['n_test']:,} test). Monotonic WOE bins, "
    f"logistic regression on WOE, {card.pdo:.0f} points to double the odds from {card.base_score:.0f} "
    f"at {card.base_odds:.0f}:1."
)

st.sidebar.header("Applicant")
applicant: dict[str, float | None] = {}
for f in card.features:
    lo, hi, default, step = RANGES[f]
    label = data.LABELS[f]
    if f in OPTIONAL and st.sidebar.checkbox(f"{label}: unknown", key=f"unknown-{f}"):
        applicant[f] = None
        continue
    if f in data.SPECIAL_VALUES and st.sidebar.checkbox(
        f"{label}: code 98 (not a count)", key=f"code-{f}"
    ):
        applicant[f] = 98.0
        continue
    applicant[f] = float(st.sidebar.number_input(label, lo, hi, default, step, key=f))

frame = pl.DataFrame(
    {f: [applicant[f]] for f in card.features},
    schema={f: pl.Float64 for f in card.features},
)
points = {f: int(p[0]) for f, p in card.feature_points(frame).items()}
score = sum(points.values())
pd_ = float(card.probability_bad(score))

tab_applicant, tab_card, tab_bins, tab_perf, tab_device = st.tabs(
    ["Applicant", "Scorecard", "Bins and WOE", "Performance", "OpenVINO"]
)

with tab_applicant:
    a, b, c = st.columns(3)
    a.metric("Score", score)
    b.metric("Probability of default", f"{pd_:.2%}")
    c.metric(
        "Odds good : bad",
        f"{card.base_odds * 2 ** ((score - card.base_score) / card.pdo):.1f} : 1",
    )
    st.altair_chart(report.points_chart(card, points), width="stretch")
    st.caption("Dark bar: points earned. Grey: the most this feature can give.")
    reasons = card.reason_codes(applicant)
    if reasons:
        st.subheader("Reasons for the score")
        for f, lost in reasons:
            st.write(f"- **{data.LABELS[f]}**: {lost} points below the best bin")

with tab_card:
    st.dataframe(
        report.scorecard_table(card),
        column_config={
            "bad rate": st.column_config.NumberColumn(format="percent"),
            "WOE": st.column_config.NumberColumn(format="%+.3f"),
        },
        hide_index=True,
        width="stretch",
    )
    st.subheader("Feature selection")
    st.dataframe(pl.DataFrame(card.selection), hide_index=True, width="stretch")

with tab_bins:
    st.altair_chart(report.woe_chart(card), width="content")

with tab_perf:
    sc, gb = m["scorecard_test"], m["gbm_test"]
    lo, hi = m["scorecard_test_auc_ci95"]
    st.dataframe(
        pl.DataFrame(
            {
                "model": ["Scorecard", "Gradient boosting (untuned)"],
                "AUC": [sc["auc"], gb["auc"]],
                "Gini": [sc["gini"], gb["gini"]],
                "KS": [sc["ks"], gb["ks"]],
            }
        ),
        hide_index=True,
    )
    st.write(
        f"Scorecard AUC 95% interval [{lo:.3f}, {hi:.3f}]; score PSI train vs test "
        f"{m['score_psi_train_vs_test']:.4f}."
    )
    left, right = st.columns(2)
    left.altair_chart(report.roc_chart(result))
    right.altair_chart(report.calibration_chart(result))
    st.altair_chart(report.score_distribution_chart(result))

with tab_device:
    devices = ov_scoring.available_devices()
    st.write({name: full for name, full in devices.items()})
    st.caption(
        "The NPU needs static float16 graphs, so only CPU and GPU can run this scorecard."
    )
    choices = [d for d in devices if d in ("CPU", "GPU")]
    device = st.selectbox("Device", choices)
    rows = st.select_slider(
        "Rows to score", [100_000, 1_000_000, 4_000_000], value=1_000_000
    )
    if st.button("Benchmark"):
        big = pl.concat([result.test] * max(1, rows // result.test.height))
        start = time.perf_counter()
        exact = card.score(big)
        numpy_ms = (time.perf_counter() - start) * 1000
        scorer = ov_scoring.OvScorer(card, device)
        x = scorer.matrix(big)
        scorer.score(x)  # compile and warm up
        start = time.perf_counter()
        got = scorer.score(x)
        ov_ms = (time.perf_counter() - start) * 1000
        a, b, c = st.columns(3)
        a.metric("NumPy", f"{numpy_ms:.0f} ms")
        b.metric(f"OpenVINO {device}", f"{ov_ms:.0f} ms", f"{numpy_ms / ov_ms:.1f}x")
        c.metric("Identical scores", f"{np.mean(got == exact):.3%}")
