"""Drives the Streamlit app headlessly on predictions from the synthetic table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import stockout_risk
from stockout_risk import pipeline
from streamlit.testing.v1 import AppTest

APP = str(Path(stockout_risk.__file__).parent / "app.py")


@pytest.fixture
def results(home, out):
    (home / "data").mkdir()
    (home / "results").mkdir()
    out["predictions"].write_parquet(home / "data" / "predictions.parquet")
    (home / "results" / "metrics.json").write_text(
        json.dumps(out["result"]), encoding="utf-8"
    )
    return home


def _run() -> AppTest:
    at = AppTest.from_file(APP)
    at.run(timeout=60)
    assert not at.exception, at.exception
    return at


def test_app_shows_the_threshold_metrics_and_a_worklist(results):
    at = _run()
    assert at.title[0].value == "Retail Stockout Risk Classifier"
    labels = [m.label for m in at.metric]
    assert labels == ["Threshold", "Checked", "Precision", "Recall", "Saving"]
    assert len(at.dataframe) == 2  # the worklist and the per-category table
    assert len(at.dataframe[0].value) == 25


def test_a_costlier_miss_lowers_the_threshold_and_checks_more(results):
    at = _run()
    cheap_threshold = float(at.metric[0].value)
    cheap_checked = float(at.metric[1].value.rstrip("%"))
    at.sidebar.slider[0].set_value(40.0).run(timeout=60)
    assert float(at.metric[0].value) <= cheap_threshold
    assert float(at.metric[1].value.rstrip("%")) >= cheap_checked


def test_the_worklist_follows_the_day_and_size_controls(results):
    at = _run()
    at.slider[0].set_value(10).run(timeout=60)
    assert len(at.dataframe[0].value) == 10
    days = at.selectbox[0].options
    at.selectbox[0].select(days[0]).run(timeout=60)
    assert not at.exception


def test_without_results_the_app_says_what_to_run(home):
    at = AppTest.from_file(APP)
    at.run(timeout=60)
    assert "stockout evaluate" in at.error[0].value
    assert pipeline.data_dir() == home / "data"
