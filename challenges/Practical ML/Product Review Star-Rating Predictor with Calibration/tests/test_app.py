import shutil
from pathlib import Path

import pytest
from helpers import fake_choice, make_features, tiny_config
from review_stars import pipeline, predict
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "src" / "review_stars" / "app.py")


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    pipeline.run_all(make_features(seed=0), home / "results", tiny_config())
    return home


@pytest.fixture
def home(finished, tmp_path, monkeypatch):
    dest = tmp_path / "app-home"
    shutil.copytree(finished, dest)
    monkeypatch.setenv("REVIEW_STARS_HOME", str(dest))
    monkeypatch.setattr(predict, "get_encoder", lambda: fake_choice(dim=12).encoder)
    return dest


def open_app():
    return AppTest.from_file(APP, default_timeout=120).run()


def test_without_a_finished_benchmark_the_app_says_what_to_run(tmp_path, monkeypatch):
    monkeypatch.setenv("REVIEW_STARS_HOME", str(tmp_path / "empty"))
    at = open_app()
    assert not at.exception and "review-stars benchmark" in at.info[0].value
    assert len(at.tabs) == 0


def test_a_finished_benchmark_opens_three_tabs_without_errors(home):
    at = open_app()
    assert not at.exception and not at.error
    assert [t.label for t in at.tabs] == [
        "Predict",
        "Calibration",
        "Selective prediction",
    ]
    assert "calibrated" in at.title[0].value.lower()


def test_predicting_a_review_shows_every_framing_with_its_set_and_a_decision(home):
    at = open_app()
    at.text_area(key="review").set_value("great item works perfect")
    at.slider(key="threshold").set_value(0.0)
    at.button(key="predict").click().run()
    assert not at.exception and not at.error
    labels = [m.label for m in at.metric]
    for framing in ("classification", "regression", "ordinal"):
        assert f"{framing}: stars" in labels and f"{framing}: confidence" in labels
    assert len(at.success) == 3 and not at.warning  # threshold 0: nobody abstains
    assert (
        sum("set:**" in m.value for m in at.markdown) == 3
    )  # one conformal set per framing


def test_raising_the_threshold_makes_every_framing_abstain_without_re_predicting(home):
    at = open_app()
    at.text_area(key="review").set_value("the item")
    at.button(key="predict").click().run()
    at.slider(key="threshold").set_value(1.0).run()
    assert not at.exception and len(at.warning) == 3 and not at.success
    at.slider(key="threshold").set_value(0.0).run()
    assert len(at.success) == 3 and not at.warning


def test_an_empty_review_is_an_error_message_not_a_traceback(home):
    at = open_app()
    at.button(key="predict").click().run()
    assert not at.exception and "empty" in at.error[0].value


def test_a_missing_head_file_is_reported_as_an_error_message(home):
    (home / "results" / "models" / "mlp-ordinal-s0.pt").unlink()
    at = open_app()
    at.text_area(key="review").set_value("anything at all")
    at.button(key="predict").click().run()
    assert not at.exception and "mlp-ordinal-s0.pt" in at.error[0].value


def test_the_calibration_tab_switches_split_and_model_and_lists_the_metrics(home):
    at = open_app()
    at.selectbox(key="cal_split").select("ood_test").run()
    at.selectbox(key="cal_model").select("linear-ordinal").run()
    assert not at.exception and not at.error
    assert len(at.dataframe) >= 1


def test_the_selective_tab_shows_the_effect_of_a_stated_confidence_threshold(home):
    at = open_app()
    at.select_slider(key="sel_tau").set_value("0.9").run()
    assert not at.exception and not at.error
    assert len(at.dataframe) >= 1


def test_a_partial_run_still_opens_and_the_missing_tabs_explain_themselves(
    tmp_path, monkeypatch
):
    home = tmp_path / "partial"
    pipeline.run_all(
        make_features(seed=1),
        home / "results",
        tiny_config(),
        stages=("train", "calibrate"),
    )
    monkeypatch.setenv("REVIEW_STARS_HOME", str(home))
    at = open_app()
    assert not at.exception
    assert any("not computed" in i.value for i in at.info)
