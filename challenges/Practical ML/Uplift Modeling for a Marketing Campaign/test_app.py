import metrics
import pipeline
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def at(monkeypatch, tiny):
    import streamlit as st

    st.cache_resource.clear()
    art = tiny[0]
    monkeypatch.setattr(pipeline, "load_artifacts", lambda *a, **k: art)
    app = AppTest.from_file("app.py", default_timeout=120)
    app.run()
    return app


def test_page_renders_without_error(at):
    assert not at.exception
    assert "Uplift" in at.title[0].value
    assert at.selectbox(key="outcome").value == "visit"
    assert at.selectbox(key="learner").value == "T"


def test_the_learner_picker_never_offers_the_random_baseline(at):
    assert "random" not in at.selectbox(key="learner").options


def test_budget_of_100_percent_equals_treating_everyone(at, tiny):
    at.slider(key="budget").set_value(100).run()
    assert not at.exception
    ate = metrics.ate(pipeline.ranked_scores(tiny[0], "visit")["T"])
    assert at.metric[0].label == "Incremental outcomes per 1,000 customers"
    assert at.metric[0].value == f"{1000 * ate:.2f}"


def test_changing_the_budget_changes_the_headline_number(at):
    small = at.slider(key="budget").set_value(5).run().metric[0].value
    large = at.slider(key="budget").set_value(60).run().metric[0].value
    assert small != large


def test_switching_outcome_and_learner_works(at):
    at.selectbox(key="outcome").set_value("conversion").run()
    assert not at.exception
    at.selectbox(key="learner").set_value("X").run()
    assert not at.exception


def test_a_profitable_setting_shows_the_best_share(at):
    at.number_input(key="value").set_value(50.0).run()
    at.number_input(key="cost").set_value(0.01).run()
    assert not at.exception
    assert any("best share" in el.value.lower() for el in [*at.caption, *at.markdown])
