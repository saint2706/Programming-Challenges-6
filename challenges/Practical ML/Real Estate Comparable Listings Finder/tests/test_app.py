"""Drives the Streamlit app headlessly on the real data (the first run fits the models)."""

from __future__ import annotations

from pathlib import Path

import comp_finder
from streamlit.testing.v1 import AppTest

APP_TIMEOUT = 180


def _run() -> AppTest:
    at = AppTest.from_file(str(Path(comp_finder.__file__).parent / "app.py"))
    at.run(timeout=APP_TIMEOUT)
    assert not at.exception
    return at


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_app_shows_estimates_a_map_and_one_row_per_comp():
    at = _run()
    assert at.title[0].value == "Comparable Listings Finder"
    assert _metric(at, "Comps estimate").startswith("$")
    assert _metric(at, "Actual sale price").startswith("$")
    assert len(at.get("deck_gl_json_chart")) == 1
    assert len(at.dataframe[0].value) == 5
    assert len(at.expander) >= 5


def test_more_comps_and_another_method_still_render():
    at = _run()
    at.sidebar.slider[0].set_value(9).run(timeout=APP_TIMEOUT)
    assert not at.exception
    assert len(at.dataframe[0].value) == 9
    at.sidebar.selectbox[0].select("knn").run(timeout=APP_TIMEOUT)
    assert not at.exception
    assert _metric(at, "Comps estimate").startswith("$")


def test_a_custom_property_has_no_actual_price():
    at = _run()
    at.sidebar.radio[0].set_value("A custom property").run(timeout=APP_TIMEOUT)
    assert not at.exception
    assert all(m.label != "Actual sale price" for m in at.metric)
    assert _metric(at, "Comps estimate").startswith("$")
    assert len(at.dataframe[0].value) == 5
