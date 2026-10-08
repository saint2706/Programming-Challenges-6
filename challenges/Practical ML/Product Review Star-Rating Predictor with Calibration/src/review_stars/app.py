"""Streamlit demo: paste a review, see calibrated star probabilities, a conformal set and an
abstain decision for each of the three framings, and browse the calibration study.

Run:  uv run streamlit run src/review_stars/app.py

It reads ``results/`` (or ``$REVIEW_STARS_HOME/results``), written by ``review-stars benchmark``.
Absolute imports only: Streamlit runs this file as a script, so relative imports would fail.
"""

from __future__ import annotations

import json
from itertools import pairwise

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from review_stars import config, predict
from review_stars import report as report_mod
from review_stars.pipeline import FIXED_TAUS

st.set_page_config(page_title="Star-rating calibration", layout="wide")
st.title("Product review star ratings, with calibrated confidence")

results = config.results_dir()
report_path = results / "report.json"
if not report_path.exists():
    st.info(
        "No finished benchmark in this folder yet. Run `uv run review-stars benchmark` "
        "(see the README), then reload this page."
    )
    st.stop()
report = json.loads(report_path.read_text(encoding="utf-8"))
stages = report.get("stages", {})


@st.cache_resource(show_spinner=False)
def cached_bundle(path: str):
    return predict.load_bundle(path)


def stage_or_note(name: str):
    if name not in stages:
        st.info(
            f"The `{name}` stage is not computed yet: run `review-stars benchmark --stage {name}`."
        )
        return None
    return stages[name]


tab_predict, tab_cal, tab_sel = st.tabs(
    ["Predict", "Calibration", "Selective prediction"]
)

# ---------------------------------------------------------------- predict

with tab_predict:
    left, right = st.columns([3, 2])
    with left:
        st.text_area(
            "Review text",
            key="review",
            height=140,
            placeholder="Works great, fits my fridge.",
        )
        st.text_input("Title (optional)", key="title")
    with right:
        st.slider(
            "Abstain below this confidence",
            0.0,
            1.0,
            0.8,
            0.05,
            key="threshold",
            help="A calibrated 0.8 means about 80% of such reviews are rated exactly this star.",
        )
        go_button = st.button("Predict", key="predict", type="primary")

    if go_button:
        try:
            bundle = cached_bundle(str(results))
            with st.spinner("Embedding the review (the first call loads the model)..."):
                st.session_state["prediction"] = predict.predict_review(
                    bundle,
                    predict.get_encoder(),
                    st.session_state["review"],
                    st.session_state["title"],
                )
        except (ValueError, FileNotFoundError) as exc:
            st.session_state.pop("prediction", None)
            st.error(str(exc))

    out = st.session_state.get("prediction")
    if out:
        threshold = st.session_state["threshold"]
        if out["truncated"]:
            st.caption(
                "This review is longer than 512 tokens: only the first 512 were read."
            )
        st.caption(
            f"{out['n_tokens']} tokens. Text read by the model: {out['text'][:200]}"
        )
        cols = st.columns(len(out["framings"]))
        for col, (framing, r) in zip(cols, out["framings"].items(), strict=True):
            with col:
                st.subheader(framing)
                st.metric(
                    f"{framing}: stars",
                    r["star"],
                    f"expected {r['expected']:.2f}",
                    delta_color="off",
                )
                st.metric(f"{framing}: confidence", f"{r['confidence']:.0%}")
                stars_in_set = ", ".join(map(str, r["set"]))
                st.markdown(f"**{1 - out['alpha']:.0%} set:** {{{stars_in_set}}}")
                if r["confidence"] < threshold:
                    st.warning(
                        f"Abstain: confidence {r['confidence']:.0%} is below {threshold:.0%}."
                    )
                else:
                    st.success(f"Answer: {r['star']} stars.")
                fig = go.Figure()
                fig.add_bar(
                    x=[1, 2, 3, 4, 5], y=r["raw_probs"], name="as trained", opacity=0.5
                )
                fig.add_bar(
                    x=[1, 2, 3, 4, 5], y=r["probs"], name=f"calibrated (T={r['T']:.2f})"
                )
                fig.update_layout(
                    barmode="group",
                    height=260,
                    margin={"t": 10, "b": 10},
                    xaxis_title="stars",
                    yaxis_title="probability",
                    legend={"orientation": "h"},
                )
                st.plotly_chart(fig, use_container_width=True, key=f"bars-{framing}")

# ---------------------------------------------------------------- calibration

with tab_cal:
    ev = stage_or_note("evaluate")
    if ev:
        c1, c2 = st.columns(2)
        split = c1.selectbox(
            "Evaluated on",
            list(ev["splits"]),
            key="cal_split",
            format_func=lambda s: report_mod.SPLIT_TITLES[s],
        )
        model = c2.selectbox(
            "Model",
            report["models"],
            index=report["models"].index("ens-classification")
            if "ens-classification" in report["models"]
            else 0,
            key="cal_model",
        )
        rows = ev["splits"][split]["rows"]
        kinds = [k for k in report_mod.KINDS if f"{model}|{k}" in rows]
        fig = go.Figure()
        fig.add_scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="perfect",
        )
        hist = go.Figure()
        for kind in kinds:
            rel = rows[f"{model}|{kind}"]["reliability_fixed"]
            edges = [float(e) for e in rel["edges"]]
            centers = [(a + b) / 2 for a, b in pairwise(edges)]
            keep = [i for i, a in enumerate(rel["acc"]) if a is not None]
            acc = [rel["acc"][i] for i in keep]
            above = [
                (rel["hi"][i] - rel["acc"][i]) if rel["hi"][i] is not None else 0
                for i in keep
            ]
            below = [
                (rel["acc"][i] - rel["lo"][i]) if rel["lo"][i] is not None else 0
                for i in keep
            ]
            fig.add_scatter(
                x=[centers[i] for i in keep],
                y=acc,
                mode="lines+markers",
                name=kind,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": above,
                    "arrayminus": below,
                },
            )
            hist.add_bar(x=centers, y=rel["n"], name=kind)
        fig.update_layout(
            height=380,
            xaxis_title="confidence",
            yaxis_title="accuracy",
            title=f"{model}: reliability on {report_mod.SPLIT_TITLES[split]}",
        )
        st.plotly_chart(fig, use_container_width=True, key="reliability")
        hist.update_layout(
            barmode="group",
            height=200,
            xaxis_title="confidence",
            yaxis_title="reviews",
            title="How sharp is it? Reviews per confidence bin",
        )
        st.plotly_chart(hist, use_container_width=True, key="sharpness")
        st.dataframe(
            pl.DataFrame(
                [
                    {
                        "calibrator": k,
                        **{
                            name: report_mod.fmt_ci(rows[f"{model}|{k}"]["ci"][key])
                            for key, name in report_mod.MAIN_COLUMNS
                        },
                    }
                    for k in kinds
                ]
            ),
            hide_index=True,
        )
        cov = go.Figure()
        cov.add_scatter(
            x=[0.4, 1],
            y=[0.4, 1],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="nominal",
        )
        for kind in kinds:
            c = rows[f"{model}|{kind}"]["coverage"]
            cov.add_scatter(
                x=[p["level"] for p in c],
                y=[p["coverage"] for p in c],
                mode="lines+markers",
                name=kind,
            )
        cov.update_layout(
            height=320,
            xaxis_title="nominal mass of the smallest set",
            yaxis_title="empirical coverage",
        )
        st.plotly_chart(cov, use_container_width=True, key="coverage")
    shift_stage = stages.get("shift")
    if shift_stage:
        rec = shift_stage["models"][
            st.session_state.get("cal_model", report["models"][0])
        ]
        st.caption(
            "Recalibration budget: Software labels used to refit the temperature"
        )
        st.dataframe(
            pl.DataFrame(
                [
                    {
                        "labels": r["n"],
                        "ECE": report_mod.fmt_ci(r["ece"]),
                        "T": report_mod.fmt_ci(r["T"]),
                    }
                    for r in rec["rows"]
                ]
            ),
            hide_index=True,
        )

# ---------------------------------------------------------------- selective

with tab_sel:
    sel = stage_or_note("selective")
    if sel:
        tau = st.select_slider(
            "Keep reviews the model is at least this sure of",
            options=[f"{t:.1f}" for t in FIXED_TAUS],
            value="0.8",
            key="sel_tau",
        )
        model = st.selectbox(
            "Model",
            [m for m in report["models"] if m.startswith("ens-")] or report["models"],
            key="sel_model",
        )
        table = []
        for kind in ("none", "temperature"):
            for split in ("test", "ood_test"):
                for f in sel["models"][model][kind][split]["fixed"]:
                    if f"{f['tau']:.1f}" == tau:
                        table.append(
                            {
                                "confidence": kind,
                                "evaluated on": report_mod.SPLIT_TITLES[split],
                                "kept": report_mod.fmt_pct(f["coverage"]),
                                "accuracy of kept": report_mod.fmt_pct(
                                    None if f["risk"] is None else 1 - f["risk"]
                                ),
                            }
                        )
        st.dataframe(pl.DataFrame(table), hide_index=True)
        st.caption(
            f"With calibrated confidence, the accuracy of what is kept is about {float(tau):.0%} or more."
        )
        fig = go.Figure()
        for kind in ("none", "temperature"):
            curve = sel["models"][model][kind]["test"]["summary"]["error"]["curve"]
            fig.add_scatter(
                x=curve["coverage"],
                y=curve["risk"],
                mode="lines",
                name=f"{kind} (test)",
            )
        fig.update_layout(
            height=320,
            xaxis_title="coverage (share kept)",
            yaxis_title="error rate of what is kept",
        )
        st.plotly_chart(fig, use_container_width=True, key="risk")
