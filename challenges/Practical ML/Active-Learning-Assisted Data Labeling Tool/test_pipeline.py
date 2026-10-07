import json
import shutil
from dataclasses import replace

import evaluate
import numpy as np
import pipeline
import pytest
from helpers import fake_choice, make_problem, tiny_config
from model import C_GRID
from pipeline import Config

CFG = tiny_config()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    prob = make_problem(n=300)
    results = tmp_path_factory.mktemp("results")
    return prob, results, pipeline.run_all(prob, results, CFG, n_jobs=1)


def copy_of(results, tmp_path):
    """Mutating tests work on their own copy of the shared results directory."""
    return shutil.copytree(results, tmp_path / "results")


def counting(monkeypatch):
    calls = []
    real = evaluate.run_job

    def wrapped(*a, **k):
        calls.append(a[1:4])
        return real(*a, **k)

    monkeypatch.setattr(evaluate, "run_job", wrapped)
    return calls


def test_report_has_every_stage_with_paired_summaries(tiny):
    _, _, report = tiny
    assert report["missing"] == []
    main = report["main"]["summary"]
    assert set(main) == set(pipeline.strategies.NAMES)
    assert main["random"]["alc_vs_random"] is None
    ci = main["margin"]["alc_vs_random"]
    assert ci["lo"] <= ci["est"] <= ci["hi"] and ci["n"] == 2
    assert set(main["margin"]["targets"]) == {"0.90", "0.95"}
    assert report["main"]["curves"]["margin"]["n"][-1] == 42
    assert set(report["batch"]) == {"5", "10"}
    assert set(report["batch"]["5"]["summary"]) == {"random", "margin"}
    assert {"threshold", "k", "labels", "acc_gap"} <= set(report["stop"]["rule"])
    assert "stop" in main["margin"] and 0 < report["ceiling"]["accuracy"] <= 1
    assert report["data"]["pool"] == 300 and report["data"]["classes"] == 6


def test_report_json_is_written_and_strict(tiny):
    _, results, report = tiny

    def reject(constant):
        raise ValueError(f"non-finite {constant} in report.json")

    on_disk = json.loads((results / "report.json").read_text(), parse_constant=reject)
    assert (
        on_disk["main"]["summary"]["random"]["alc"]["est"]
        == report["main"]["summary"]["random"]["alc"]["est"]
    )


def test_a_rerun_reuses_every_finished_job(tiny, monkeypatch, tmp_path):
    prob, results, report = tiny
    results = copy_of(results, tmp_path)
    calls = counting(monkeypatch)
    again = pipeline.run_all(prob, results, CFG, n_jobs=1)
    assert calls == [] and again["main"] == report["main"]


def test_fresh_recomputes_only_the_requested_stage_and_keeps_the_others(
    tiny, monkeypatch, tmp_path
):
    prob, results, report = tiny
    results = copy_of(results, tmp_path)
    calls = counting(monkeypatch)
    again = pipeline.run_all(prob, results, CFG, stages=("stop",), fresh=True, n_jobs=1)
    assert len(calls) == CFG.tune_seeds
    assert again["missing"] == [] and again["main"] == report["main"]


def test_more_seeds_compute_only_the_new_seeds(tiny, monkeypatch, tmp_path):
    prob, results, _ = tiny
    results = copy_of(results, tmp_path)
    calls = counting(monkeypatch)
    more = pipeline.run_all(
        prob, results, replace(CFG, seeds=3), stages=("main",), n_jobs=1
    )
    assert len(calls) == len(pipeline.strategies.NAMES)  # seed 2 for each strategy
    assert more["main"]["summary"]["random"]["alc"]["n"] == 3


def test_a_changed_budget_or_changed_data_never_reuses_stale_jobs(
    tiny, monkeypatch, tmp_path
):
    prob, results, _ = tiny
    results = copy_of(results, tmp_path)
    calls = counting(monkeypatch)
    longer = pipeline.run_all(
        prob, results, replace(CFG, budget=40), stages=("stop",), n_jobs=1
    )
    assert len(calls) == CFG.tune_seeds  # the new stop jobs only
    assert longer["missing"] == ["main", "batch"] and "main" not in longer
    calls.clear()
    other = replace(prob, X=prob.X * 0.999)
    pipeline.run_all(other, results, CFG, stages=("stop",), n_jobs=1)
    assert len(calls) == CFG.tune_seeds


def test_a_partial_run_lists_what_is_missing_and_a_later_run_completes_it(tmp_path):
    prob = make_problem(n=300)
    first = pipeline.run_all(prob, tmp_path, CFG, stages=("main",), n_jobs=1)
    assert first["missing"] == ["batch", "stop"] and "batch" not in first
    assert "stop" not in first["main"]["summary"]["margin"]  # no rule yet
    second = pipeline.run_all(prob, tmp_path, CFG, stages=("batch", "stop"), n_jobs=1)
    assert second["missing"] == [] and "stop" in second["main"]["summary"]["margin"]


def test_config_rejects_a_sensitivity_set_without_random_or_with_unknown_names():
    with pytest.raises(ValueError, match="random"):
        Config(sens_strategies=("margin",))
    with pytest.raises(ValueError, match="unknown strategy"):
        Config(sens_strategies=("random", "nope"))


def test_job_keys_change_with_everything_a_curve_depends_on():
    job = {"strategy": "margin", "seed": 0, "b": 10, "eval": "test"}
    base = pipeline.job_key("fp", 10.0, CFG, job)
    assert base == pipeline.job_key("fp", 10.0, CFG, dict(job))
    for other in (
        pipeline.job_key("fp2", 10.0, CFG, job),
        pipeline.job_key("fp", 30.0, CFG, job),
        pipeline.job_key("fp", 10.0, replace(CFG, budget=31), job),
        pipeline.job_key("fp", 10.0, replace(CFG, init=13), job),
        pipeline.job_key("fp", 10.0, CFG, job | {"seed": 1}),
        pipeline.job_key("fp", 10.0, CFG, job | {"b": 5}),
        pipeline.job_key("fp", 10.0, CFG, job | {"eval": "val"}),
    ):
        assert other != base
    assert pipeline.job_key("fp", 10.0, replace(CFG, seeds=9), job) == base


def test_load_problem_embeds_once_picks_C_and_leaves_test_alone(banking_dir):
    calls = []

    def choice():
        calls.append(1)
        return fake_choice()

    problem, splits, info = pipeline.load_problem(
        banking_dir, get_choice=choice, val_size=30
    )
    assert len(calls) == 1  # one backend decision for pool, validation and test
    assert (
        problem.X.shape == (len(splits.pool_texts), 16)
        and problem.X.dtype == np.float64
    )
    assert problem.n_classes == 5 and problem.C in C_GRID
    assert np.array_equal(problem.y_eval, splits.test_y) and problem.X_val.shape[
        0
    ] == len(splits.val_texts)
    assert info["backend"] == "fake" and info["duplicates_collapsed"] == 2
    again, _, _ = pipeline.load_problem(banking_dir, get_choice=choice, val_size=30)
    assert len(calls) == 1 and np.array_equal(again.X, problem.X)
    assert (banking_dir / "embeddings").is_dir()


def test_load_problem_without_data_says_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch"):
        pipeline.load_problem(tmp_path, get_choice=fake_choice)
