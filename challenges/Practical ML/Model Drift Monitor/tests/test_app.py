from pathlib import Path

import drift_monitor
import pytest
from drift_monitor import pipeline
from streamlit.testing.v1 import AppTest


@pytest.fixture
def at(monkeypatch, tiny_context):
    import streamlit as st

    st.cache_resource.clear()
    monkeypatch.setattr(pipeline, "load_context", lambda: tiny_context)
    app = AppTest.from_file(
        str(Path(drift_monitor.__file__).parent / "app.py"), default_timeout=120
    )
    app.run()
    return app


def test_page_renders_without_error(at):
    assert not at.exception
    assert "Model Drift Monitor" in at.title[0].value
    assert at.selectbox(key="signal").value == "score"


def test_the_feature_ranking_puts_the_shifted_feature_first_after_the_step(at):
    at.selectbox(key="window").set_value(30).run()
    assert not at.exception
    ranking = at.dataframe[0].value
    assert ranking.iloc[0]["x threshold"] > 1
    assert "nswprice" in set(ranking["signal"].head(3))


def test_changing_the_window_changes_the_ranking(at):
    before = (
        at.selectbox(key="window")
        .set_value(3)
        .run()
        .dataframe[0]
        .value["x threshold"]
        .iloc[0]
    )
    after = (
        at.selectbox(key="window")
        .set_value(30)
        .run()
        .dataframe[0]
        .value["x threshold"]
        .iloc[0]
    )
    assert after > before


def test_the_categorical_signal_offers_chi_square_and_renders(at):
    at.selectbox(key="signal").set_value("day").run()
    assert not at.exception
    assert "chi2" in at.selectbox(key="stat").options
    assert "ks" not in at.selectbox(key="stat").options
