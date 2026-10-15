"""Drives the Streamlit app headlessly on results from the synthetic table."""

from __future__ import annotations

import json
from pathlib import Path

import fraud_flagging
import pytest
from fraud_flagging import models
from streamlit.testing.v1 import AppTest

APP = str(Path(fraud_flagging.__file__).parent / "app.py")


@pytest.fixture
def results(home, out):
    (home / "results").mkdir()
    models.save_model(home / "data" / "model", out["booster"], out["meta"])
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


def test_app_shows_the_metrics_a_worklist_and_the_stability_table(results):
    at = _run()
    assert at.title[0].value == "Fraud Transaction Flagging System"
    assert [m.label for m in at.metric] == [
        "Flagged",
        "Precision",
        "Recall",
        "Fraud amount caught",
        "Total cost",
        "Saving vs flag nothing",
    ]
    assert len(at.dataframe[0].value) == 50 and len(at.dataframe) == 2


def test_every_policy_runs_and_a_higher_precision_target_flags_fewer(results):
    at = _run()
    flagged = {}
    for target in (0.5, 0.95):
        at.sidebar.radio[0].set_value("precision").run(timeout=60)
        at.sidebar.select_slider[0].set_value(target).run(timeout=60)
        flagged[target] = int(at.metric[0].value.replace(",", ""))
    assert flagged[0.95] <= flagged[0.5]
    for policy in ("recall", "cost", "capacity"):
        at.sidebar.radio[0].set_value(policy).run(timeout=60)
        assert not at.exception, (policy, at.exception)


def test_a_cheaper_review_flags_more_under_the_cost_policy(results):
    at = _run()
    at.sidebar.radio[0].set_value("cost").run(timeout=60)
    at.sidebar.slider[0].set_value(100.0).run(timeout=60)
    dear = int(at.metric[0].value.replace(",", ""))
    at.sidebar.slider[0].set_value(1.0).run(timeout=60)
    assert int(at.metric[0].value.replace(",", "")) >= dear


def test_capacity_policy_respects_the_hourly_budget(results):
    at = _run()
    at.sidebar.radio[0].set_value("capacity").run(timeout=60)
    at.sidebar.slider[1].set_value(1).run(timeout=60)
    assert int(at.metric[0].value.replace(",", "")) <= 12


def test_without_results_the_app_says_what_to_run(home):
    at = AppTest.from_file(APP)
    at.run(timeout=60)
    assert "fraud evaluate" in at.error[0].value
