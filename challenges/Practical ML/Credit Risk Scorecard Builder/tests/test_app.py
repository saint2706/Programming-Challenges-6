"""Drives the Streamlit app headlessly. The first run trains the scorecard and the GBM benchmark."""

from __future__ import annotations

from pathlib import Path

import credit_scorecard
from streamlit.testing.v1 import AppTest

APP_TIMEOUT = 180


def _run() -> AppTest:
    at = AppTest.from_file(str(Path(credit_scorecard.__file__).parent / "app.py"))
    at.run(timeout=APP_TIMEOUT)
    assert not at.exception
    return at


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_app_loads_and_shows_score_probability_and_odds():
    at = _run()
    assert at.title[0].value == "Credit Risk Scorecard"
    assert int(_metric(at, "Score")) > 300
    assert _metric(at, "Probability of default").endswith("%")


def test_raising_utilization_lowers_the_score():
    at = _run()
    before = int(_metric(at, "Score"))
    next(
        w for w in at.sidebar.number_input if w.label == "Revolving utilization"
    ).set_value(1.5).run(timeout=APP_TIMEOUT)
    assert not at.exception
    assert int(_metric(at, "Score")) < before


def test_unknown_income_and_sentinel_code_still_render():
    at = _run()
    next(
        c for c in at.sidebar.checkbox if c.label == "Monthly income: unknown"
    ).check().run(timeout=APP_TIMEOUT)
    assert not at.exception
    before = int(_metric(at, "Score"))
    next(
        c for c in at.sidebar.checkbox if c.label.startswith("90+ days late")
    ).check().run(timeout=APP_TIMEOUT)
    assert not at.exception
    # Code 98 swaps the "never late" bin (86 points) for the special bin (34 points).
    assert before - int(_metric(at, "Score")) > 40
