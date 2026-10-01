"""Streamlit page: paste a resume, see ranked jobs and why each one matched.

Run:  uv run streamlit run app.py
"""

from __future__ import annotations

import json

import ranker as rk
import streamlit as st

st.set_page_config(page_title="Resume-to-Job Matching Scorer", layout="wide")


@st.cache_resource(
    show_spinner="Loading the embedding model and building the job index..."
)
def get_ranker() -> rk.Ranker:
    return rk.load_default_ranker()


def benchmark_line(scorer: str) -> str | None:
    if not rk.RESULTS.is_file():
        return None
    r = json.loads(rk.RESULTS.read_text(encoding="utf-8"))
    m = r["results"]["full"]["resume->jobs"].get(scorer)
    if m is None:
        return None
    return f"Benchmark nDCG@10 for {scorer}: {m['ndcg@10']['mean']:.3f} (full resume text, held-out test split)"


st.title("Resume-to-Job Matching Scorer")
ranker = get_ranker()

with st.sidebar:
    scorer = st.selectbox(
        "Scorer",
        rk.SCORER_NAMES,
        key="scorer",
        help="fusion = z-scored embedding + BM25",
    )
    top = st.slider("Jobs to show", 3, 25, 10, key="top")
    line = benchmark_line(scorer)
    if line:
        st.caption(line)
    st.caption(
        "tfidf and bm25 explanations are exact: the listed terms plus the stated remainder add up to the score. "
        "For embedding and fusion the terms are only shared words; use the post-hoc probe for the model."
    )

tab_jobs, tab_resumes = st.tabs(["Resume to jobs", "Job to resumes"])

with tab_jobs:
    if st.button("Load a sample resume", key="sample"):
        st.session_state["resume_text"] = ranker.resumes["text"][0]
    st.text_area(
        "Resume text",
        height=220,
        key="resume_text",
        placeholder="Paste a plain-text resume here",
    )
    if st.button("Rank jobs", key="rank_jobs", type="primary"):
        text = st.session_state.get("resume_text", "").strip()
        if not text:
            st.warning("Paste a resume first.")
            st.session_state.pop("matches", None)
        else:
            st.session_state["matches"] = ranker.rank_jobs(text, top=top, scorer=scorer)
            st.session_state["matches_text"] = text

    for i, m in enumerate(st.session_state.get("matches", []), 1):
        with st.expander(
            f"{i}. {m.title} [{m.category}] - score {m.score:.3f}", expanded=i == 1
        ):
            st.markdown("**Terms behind this score**")
            st.markdown(
                ", ".join(f"`{t}` {w:.3f}" for t, w in m.terms) or "_nothing in common_"
            )
            st.markdown(f"_{m.evidence_note()}_")
            if m.gaps:
                st.markdown(
                    "**Common in postings like this but missing from your resume**"
                )
                st.markdown(", ".join(f"`{g}`" for g in m.gaps))
            if st.button(
                "Probe the embedding match (slow, post-hoc)", key=f"dense-{m.id}"
            ):
                st.caption(
                    "Resume sentences whose removal lowers the embedding similarity most"
                )
                for sentence, drop in ranker.explain_dense(
                    st.session_state["matches_text"], m.id
                ):
                    st.markdown(f"`{drop:+.4f}` {sentence}")

with tab_resumes:
    job_text = st.text_area("Job description", height=220, key="job_text")
    if st.button("Rank resumes", key="rank_resumes"):
        if not job_text.strip():
            st.warning("Paste a job description first.")
        else:
            for i, m in enumerate(
                ranker.rank_resumes(job_text, top=top, scorer=scorer), 1
            ):
                with st.expander(
                    f"{i}. {m.title} [{m.category}] - score {m.score:.3f}"
                ):
                    st.markdown(
                        ", ".join(f"`{t}` {w:.3f}" for t, w in m.terms)
                        or "_nothing in common_"
                    )
