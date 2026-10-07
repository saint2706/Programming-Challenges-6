import json

import data
import evaluate
import metrics
import numpy as np
import pipeline
import pytest
from helpers import make_frame

TINY = {"n_estimators": 40, "min_child_samples": 30, "num_leaves": 15}


def test_report_has_every_section(tiny):
    _, report, _ = tiny
    assert set(report) >= {"config", "randomization", "rows", "outcomes", "synthetic"}
    assert set(report["outcomes"]) == {"visit", "conversion"}
    assert set(report["synthetic"]) == {"heterogeneous", "constant", "none"}
    assert len(report["randomization"]) == 12
    assert sum(report["rows"].values()) == 20000


def test_artifacts_round_trip(tiny):
    art, report, results = tiny
    assert art.report == json.loads((results / "report.json").read_text())
    assert set(art.scores) == {"visit", "conversion"}
    frame = art.scores["visit"]
    scorers = {"T", "S", "X", "TO", "DR", "response", "random"}
    assert {"t", "y", *scorers} <= set(frame.columns)
    assert frame.height == report["outcomes"]["visit"]["n_test"]
    assert set(art.models["visit"]) == scorers


def test_saved_models_reproduce_the_saved_test_scores(tiny):
    art, _, _ = tiny
    _, _, test = data.split(make_frame(20000, strength=2.0), seed=0)
    X, _, _ = data.xy(test, "visit")
    saved = art.scores["visit"]["T"].to_numpy()
    assert np.allclose(art.models["visit"]["T"].predict(X), saved, atol=1e-6)


def test_ranked_scores_gives_one_ranking_per_scorer(tiny):
    art, _, _ = tiny
    ranked = pipeline.ranked_scores(art, "visit")
    assert "T" in ranked and "random" in ranked
    reported = art.report["outcomes"]["visit"]["learners"]["T"]["qini"]["est"]
    assert metrics.qini_coefficient(ranked["T"]) == pytest.approx(reported)


def test_report_is_strict_json(tiny):
    _, _, results = tiny

    def reject(constant):
        pytest.fail(f"non-standard JSON constant {constant}")

    json.loads((results / "report.json").read_text(), parse_constant=reject)


def test_clean_turns_nan_and_numpy_into_json_values():
    out = pipeline.clean({"a": np.float32(1.5), "b": float("nan"), "c": [np.int64(3)]})
    assert out == {"a": 1.5, "b": None, "c": [3]}


def test_finished_stages_are_reused_and_fresh_recomputes(tmp_path, monkeypatch):
    kw = {
        "seed": 0,
        "n_boot": 20,
        "n_seeds": 1,
        "df": make_frame(8000),
        "grid": [TINY],
        "synth_n": 2000,
        "synth_params": TINY,
        "stages": ("visit",),
    }
    first = pipeline.run_all(tmp_path, tmp_path / "r", **kw)

    def boom(*args, **kwargs):
        raise AssertionError("a finished stage was recomputed")

    monkeypatch.setattr(evaluate, "evaluate_real", boom)
    again = pipeline.run_all(tmp_path, tmp_path / "r", **kw)
    assert again["outcomes"]["visit"] == first["outcomes"]["visit"]
    with pytest.raises(AssertionError, match="recomputed"):
        pipeline.run_all(tmp_path, tmp_path / "r", fresh=True, **kw)


def test_a_failed_randomization_check_aborts_before_any_stage(tmp_path):
    with pytest.raises(data.RandomizationError):
        pipeline.run_all(
            tmp_path,
            tmp_path / "r",
            df=make_frame(20000, confounded=True),
            grid=[TINY],
            n_boot=10,
            n_seeds=1,
            synth_n=1000,
            synth_params=TINY,
        )
    assert not list((tmp_path / "r").glob("stages/*"))


def test_load_artifacts_before_a_benchmark_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="benchmark"):
        pipeline.load_artifacts(tmp_path)
