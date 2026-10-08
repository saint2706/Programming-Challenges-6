from pathlib import Path

import pytest
import resume_matcher
from helpers import HashEncoder, _frames
from resume_matcher import ranker as ranker_module
from resume_matcher.ranker import Ranker
from streamlit.testing.v1 import AppTest

RESUME = "nurse patient ward triage dosage clinic nurse patient team worked"


@pytest.fixture
def at(monkeypatch, tmp_path):
    import streamlit as st

    st.cache_resource.clear()
    jobs, resumes = _frames()
    r = Ranker.build(
        jobs, resumes, HashEncoder(), index_dir=tmp_path / "idx", fusion_weight=0.5
    )
    monkeypatch.setattr(ranker_module, "load_default_ranker", lambda: r)
    app = AppTest.from_file(
        str(Path(resume_matcher.__file__).parent / "app.py"), default_timeout=60
    )
    app.run()
    return app


def test_page_renders_without_error(at):
    assert not at.exception
    assert "Resume-to-Job" in at.title[0].value


def test_ranking_a_resume_shows_same_category_jobs_with_evidence(at):
    at.text_area(key="resume_text").set_value(RESUME)
    at.selectbox(key="scorer").set_value("tfidf")
    at.button(key="rank_jobs").click().run()
    assert not at.exception
    titles = [e.label for e in at.expander]
    assert len(titles) == 10
    assert all("HEALTHCARE" in t for t in titles[:3])
    assert any("nurse" in m.value for m in at.markdown)
    assert any("the other" in m.value for m in at.markdown)


def test_empty_resume_shows_a_warning_not_an_exception(at):
    at.button(key="rank_jobs").click().run()
    assert not at.exception
    assert any("paste" in w.value.lower() for w in at.warning)
