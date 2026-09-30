"""Streamlit review UI: candidate duplicate pairs side by side, with live scoring.

    uv run streamlit run app.py

Run `cli.py fetch`, `embed` and `evaluate` first; this page reads what they
cached. Two sliders (text/image fusion weight and decision threshold) re-score
the cached candidate pairs instantly -- nothing is re-embedded or re-indexed.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import data
import embed
import evaluate
import pipeline
import streamlit as st

DATA_DIR = Path(os.environ.get("DUPLICATE_DETECTOR_DATA_DIR", data.DATA_DIR))


@dataclass
class State:
    report: dict
    splits: dict[str, evaluate.SplitData]


@st.cache_resource(show_spinner="Loading embeddings and building the LanceDB index...")
def load_state(data_dir: str) -> State:
    root = Path(data_dir)
    report = pipeline.load_report(root / pipeline.REPORT_FILE)
    df = pipeline.load_slice(root)
    emb = embed.load_embeddings(
        root / pipeline.EMBEDDINGS_FILE, df["posting_id"].to_list()
    )
    if emb is None:
        raise FileNotFoundError(
            "cached embeddings are missing or stale -- run the `embed` command first"
        )
    s = report["settings"]
    val, test = pipeline.prepare_splits(
        df, emb, root, s["k"], s["val_fraction"], s["seed"]
    )
    return State(report, {"test": test, "val": val})


def image_path(root: Path, name: str) -> Path | None:
    """Path to a listing image, refusing names that could point outside train_images/."""
    if name != Path(name).name:
        return None
    path = root / "train_images" / name
    return path if path.exists() else None


def show_listing(column, root: Path, title: str, image_name: str) -> None:
    with column:
        path = image_path(root, image_name)
        if path is None:
            st.caption("image unavailable")
        else:
            st.image(str(path), width=220)
        st.text(
            title
        )  # plain text: titles are untrusted strings, never rendered as markdown/HTML


st.set_page_config(page_title="Duplicate Listing Review", layout="wide")
st.title("Duplicate Product Listing Review")

try:
    state = load_state(str(DATA_DIR))
except FileNotFoundError as exc:
    st.error(f"{exc}")
    st.stop()

fused = state.report["modes"]["fused"]

with st.sidebar:
    st.header("Operating point")
    split_name = st.selectbox(
        "Split",
        ["test", "val"],
        key="split",
        help="Held-out groups; test is what `evaluate` reports.",
    )
    weight = st.slider(
        "Text weight (image = 1 - weight)",
        0.0,
        1.0,
        float(fused["weight"]),
        0.05,
        key="weight",
    )
    threshold = st.slider(
        "Decision threshold on fused score",
        -0.2,
        1.0,
        float(min(max(fused["threshold"], -0.2), 1.0)),
        0.005,
        key="threshold",
    )
    st.header("Review")
    borderline = st.checkbox("Borderline pairs only", value=False, key="borderline")
    band = st.slider(
        "Borderline band (+/-)",
        0.01,
        0.30,
        0.05,
        0.01,
        key="band",
        disabled=not borderline,
    )
    limit = st.slider("Pairs to show", 5, 50, 10, key="limit")
    show_truth = st.checkbox("Show ground-truth label", value=False, key="show_truth")

split = state.splits[split_name]
live = evaluate.pair_level(split, weight, threshold)

cols = st.columns(5)
cols[0].metric("Precision", f"{live.precision:.3f}")
cols[1].metric("Recall", f"{live.recall:.3f}")
cols[2].metric("F1", f"{live.f1:.3f}")
cols[3].metric("Flagged pairs", f"{live.n_flagged:,}")
cols[4].metric("Candidate recall", f"{split.candidate_recall:.3f}")
st.caption(
    f"{split.df.height:,} listings, {len(split.pairs):,} candidate pairs retrieved from LanceDB. "
    "Recall counts every ground-truth pair, including ones retrieval never surfaced."
)

pairs = evaluate.review_pairs(
    split, weight, threshold, band=band if borderline else None, limit=limit
)
st.subheader(
    "Highest-scoring candidate pairs"
    if not borderline
    else "Borderline candidate pairs"
)
if pairs.is_empty():
    st.info("No candidate pairs in this range.")

for i, row in enumerate(pairs.iter_rows(named=True)):
    with st.container(border=True):
        left, right, scores = st.columns([2, 2, 1.4])
        show_listing(left, DATA_DIR, row["title_a"], row["image_a"])
        show_listing(right, DATA_DIR, row["title_b"], row["image_b"])
        with scores:
            st.metric("Fused", f"{row['fused']:.3f}")
            st.caption(f"text {row['text']:.3f} | image {row['image']:.3f}")
            st.markdown("**Flagged as duplicate**" if row["flagged"] else "Not flagged")
            if show_truth:
                st.caption(
                    "Ground truth: true duplicate"
                    if row["is_duplicate"]
                    else "Ground truth: different products"
                )
