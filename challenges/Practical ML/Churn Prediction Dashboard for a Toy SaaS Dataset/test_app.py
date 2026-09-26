"""Headless smoke tests for the Streamlit dashboard via
`streamlit.testing.v1.AppTest` -- this repo's precedent for testing a
non-HTTP UI framework headlessly (the Encrypted Diary/Journal challenge
does the same for its Textual TUI with `Pilot`). These run the real
`app.py` end to end (real data fetch/cache, real model training, real
SHAP) rather than mocking the app away, so the first run in particular
is slow -- hence the generous timeout.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

APP_TIMEOUT = 180  # first run trains two real models and computes real SHAP values.


def _run_app() -> AppTest:
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    assert not at.exception
    return at


def test_app_loads_without_exception():
    _run_app()


def test_app_shows_title_and_threshold_dependent_metrics():
    at = _run_app()
    assert "Churn Prediction Dashboard" in at.title[0].value
    assert len(at.metric) >= 4  # accuracy, precision, recall, f1 score


def test_default_threshold_is_not_a_naive_half():
    at = _run_app()
    slider = at.sidebar.slider[0]
    # Started at the F1-optimal threshold for an imbalanced (~26.5%
    # positive) dataset, not a naive 0.5 -- see model.best_f1_threshold.
    assert slider.value != 0.5


def test_dragging_the_threshold_slider_changes_displayed_metrics():
    at = _run_app()
    precision_before = at.metric[
        1
    ].value  # column order: accuracy, precision, recall, f1
    at.sidebar.slider[0].set_value(0.95).run(timeout=APP_TIMEOUT)
    assert not at.exception
    precision_after = at.metric[1].value
    assert precision_after != precision_before


def test_switching_model_still_renders_without_exception():
    at = _run_app()
    at.sidebar.selectbox[0].set_value("LightGBM").run(timeout=APP_TIMEOUT)
    assert not at.exception
    assert len(at.metric) >= 4


def test_changing_the_drill_down_row_index_updates_without_exception():
    at = _run_app()
    at.number_input[0].set_value(1).run(timeout=APP_TIMEOUT)
    assert not at.exception
