"""Streamlit labeling UI over a project folder created with ``uv run active-labeling init``.

Run:  AL_PROJECT=projects/demo uv run streamlit run src/active_labeling/app.py
"""

from __future__ import annotations

import os

import streamlit as st

from active_labeling import explain, strategies
from active_labeling.project import Project

st.set_page_config(page_title="Active-learning labeler", layout="wide")
st.title("Active-learning labeler")

path = st.sidebar.text_input("Project folder", os.environ.get("AL_PROJECT", "")).strip()
if not path:
    st.info(
        "Create a project with `uv run active-labeling init <folder> --demo` (or `--csv`), then enter its folder here."
    )
    st.stop()
try:
    proj = Project(path)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if (
    st.session_state.get("batch_for", path) != path
):  # a pending batch belongs to the old project
    st.session_state.pop("batch", None)
    for key in [k for k in st.session_state if k.startswith("pick-")]:
        del st.session_state[key]
st.session_state["batch_for"] = path

strategy = st.sidebar.selectbox(
    "Strategy", strategies.NAMES, index=strategies.NAMES.index("margin"), key="strategy"
)
batch_size = int(
    st.sidebar.number_input(
        "Batch size", min_value=1, max_value=200, value=10, key="batch_size"
    )
)

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

status = proj.status()
c1, c2, c3 = st.columns(3)
c1.metric("Labeled", f"{status['labeled']:,} / {status['items']:,}")
c2.metric(
    "Intents with a label", f"{status['classes_with_label']} / {status['classes']}"
)
rule = status["rule"]
c3.metric(
    "Stopping signal",
    "reached" if status["stop"]["fired"] else "not yet",
    help=(
        f"Share of pool predictions that change between rounds stays below {rule['threshold']} "
        f"for {rule['k']} rounds in a row (Bloodgood & Vijay-Shanker 2009)."
    ),
)


def accept(item: int, name: str) -> None:
    st.session_state[f"pick-{item}"] = name


if st.button("Suggest a batch", key="suggest"):
    _, reasons, _ = proj.suggest(strategy, batch_size, seed=len(status["rounds"]))
    st.session_state["batch"] = [(r.idx, r) for r in reasons]
    if not reasons:
        st.success("Every item is labeled.")

batch = st.session_state.get("batch", [])
for item, reason in batch:
    with st.container(border=True):
        st.markdown(f"**#{item}** {proj.texts[item]}")
        left, right = st.columns([3, 1])
        left.selectbox(
            "Intent",
            proj.classes,
            index=None,
            placeholder="Choose an intent",
            key=f"pick-{item}",
            label_visibility="collapsed",
        )
        if (
            not reason.cold
        ):  # with no labels the "top class" is arbitrary: offer no guess
            guess = proj.classes[reason.top[0][0]]
            right.button(
                f"Accept: {guess}",
                key=f"guess-{item}",
                on_click=accept,
                args=(item, guess),
            )
        with st.expander("Why this item?"):
            st.text(explain.render(reason, proj.classes, proj.texts))

if batch:
    submit_col, sim_col = st.columns(2)
    if submit_col.button("Submit labels", key="submit", type="primary"):
        chosen = {i: st.session_state.get(f"pick-{i}") for i, _ in batch}
        chosen = {i: name for i, name in chosen.items() if name}
        if not chosen:
            st.warning("Choose an intent for at least one item first.")
        else:
            proj.submit(chosen)
            st.session_state.pop("batch")
            st.session_state["flash"] = f"Saved {len(chosen)} labels."
            st.rerun()
    if proj.has_gold and sim_col.button(
        "Simulate the annotator on this batch", key="simulate"
    ):
        proj.simulate([i for i, _ in batch])
        st.session_state.pop("batch")
        st.session_state["flash"] = (
            f"Labeled {len(batch)} items with their gold labels (simulated)."
        )
        st.rerun()

rounds = status["rounds"]
if rounds:
    st.subheader("Learning so far")
    measured = [r for r in rounds if r["accuracy"] is not None]
    left, right = st.columns(2)
    if measured:
        left.caption("Accuracy on the held-out evaluation set")
        left.line_chart(
            {
                "labels": [r["n_labeled"] for r in measured],
                "accuracy": [r["accuracy"] for r in measured],
            },
            x="labels",
            y="accuracy",
        )
    changes = [r for r in rounds if r["change"] is not None]
    if changes:
        right.caption("Share of pool predictions that changed since the previous round")
        right.line_chart(
            {
                "labels": [r["n_labeled"] for r in changes],
                "change": [r["change"] for r in changes],
            },
            x="labels",
            y="change",
        )
    st.caption("Labels per intent")
    st.bar_chart({"labels": status["counts"]}, y="labels")
