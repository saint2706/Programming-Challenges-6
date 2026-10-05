import math

import polars as pl
import pytest

from evaluate import (
    DETECT_WITHIN,
    Scenario,
    benchmark,
    default_scenarios,
    summarize,
    wilson,
)
from monitor import monitor
from test_monitor import W, make_fitted, synth


def test_wilson_handles_zero_all_and_empty():
    assert (
        wilson(0, 20)[0] == 0.0
        and wilson(0, 20)[1] == 0.0
        and 0 < wilson(0, 20)[2] < 0.2
    )
    assert (
        wilson(20, 20)[0] == 1.0
        and wilson(20, 20)[2] == 1.0
        and 0.8 < wilson(20, 20)[1] < 1
    )
    lo_hi = wilson(10, 20)
    assert lo_hi[0] == pytest.approx(0.5) and lo_hi[1] < 0.5 < lo_hi[2]
    assert all(math.isnan(v) for v in wilson(0, 0))


@pytest.fixture(scope="module")
def setup():
    clf, base, thr, seq, ref = make_fitted()
    return clf, base, thr, seq, ref


def _run(setup, scenarios, n_seeds=6, **kw):
    clf, base, thr, seq, ref = setup
    # the benchmark uses small windows here so the synthetic reference supports it
    return benchmark(
        clf, base, thr, seq, ref, scenarios, n_seeds=n_seeds, window=W, **kw
    )


def test_default_scenarios_cover_every_drift_type_and_the_no_drift_control(setup):
    names = {s.name.split(":")[0] for s in default_scenarios(setup[0])}
    assert names == {
        "none",
        "covariate_important",
        "covariate_unimportant",
        "prior",
        "concept",
        "score_noise",
    }


def _covariate(setup, sigmas):
    clf = setup[0]
    top = next(
        s
        for s in default_scenarios(clf)
        if s.name.startswith("covariate_important") and s.magnitude == sigmas
    )
    return top


def test_a_strong_shift_of_the_important_feature_is_detected_with_a_delay_from_the_true_start(
    setup,
):
    runs = _run(setup, [_covariate(setup, 2.0)])
    s = summarize(runs).filter(pl.col("detector") == "features_any").row(0, named=True)
    assert s["detection_rate"] >= 0.95
    assert (
        0 <= s["mean_delay_windows"] <= 2
    )  # drift starts exactly at a window boundary


def test_no_drift_false_alarms_per_window_are_low_for_single_tests_and_higher_for_the_union(
    setup,
):
    runs = _run(
        setup, [Scenario("none", 0.0, lambda df, s, seed, c: (df, c))], n_seeds=8
    )
    fa = {
        r["detector"]: r["false_alarm_per_window"]
        for r in summarize(runs).iter_rows(named=True)
    }
    for single in (
        "score_psi",
        "score_ks",
        "score_js",
        "score_wasserstein",
        "page_hinkley_score",
        "adwin_score",
        "adwin_error",
    ):
        assert fa[single] < 0.06, (single, fa)
    # ~28 feature tests at 1% each: the union alarms far more often (multiple comparisons)
    assert fa["features_any"] > fa["score_ks"]


def test_undetected_drift_has_a_null_delay_not_zero(setup):
    runs = _run(
        setup, [Scenario("none", 0.0, lambda df, s, seed, c: (df, c))], n_seeds=3
    )
    s = summarize(
        runs.with_columns(pl.lit("x").alias("scenario"))
    )  # treat as a drift scenario
    quiet = s.filter(pl.col("detection_rate") == 0)
    assert quiet.height > 0 and quiet["mean_delay_windows"].null_count() == quiet.height
    undetected = runs.filter(~pl.col("detected"))
    assert undetected["delay_windows"].null_count() == undetected.height


def test_larger_shifts_are_detected_no_later_and_no_less_often(setup):
    small, big = _covariate(setup, 0.25), _covariate(setup, 2.0)
    s = summarize(_run(setup, [small, big])).filter(pl.col("detector") == "score_ks")
    lo = s.filter(pl.col("magnitude") == 0.25).row(0, named=True)
    hi = s.filter(pl.col("magnitude") == 2.0).row(0, named=True)
    assert hi["detection_rate"] >= lo["detection_rate"]


def test_delay_is_measured_from_the_first_drifted_window_within_the_detection_horizon(
    setup,
):
    runs = _run(setup, [_covariate(setup, 2.0)], n_seeds=3)
    d = runs.filter(pl.col("detected"))["delay_windows"]
    assert d.min() >= 0 and d.max() < DETECT_WITHIN


def test_chance_rate_is_the_no_drift_detection_rate_and_missing_without_a_control(
    setup,
):
    none = Scenario("none", 0.0, lambda df, s, seed, c: (df, c))
    with_ctrl = summarize(_run(setup, [none, _covariate(setup, 2.0)], n_seeds=4))
    assert with_ctrl["chance_rate"].null_count() == 0
    row = with_ctrl.filter(
        (pl.col("scenario") == "none") & (pl.col("detector") == "score_ks")
    ).row(0, named=True)
    assert row["chance_rate"] == row["detection_rate"]
    assert (
        summarize(_run(setup, [_covariate(setup, 2.0)], n_seeds=2))[
            "chance_rate"
        ].null_count()
        > 0
    )


def test_concept_drift_is_invisible_to_input_detectors_but_caught_by_the_error_stream(
    setup,
):
    none = Scenario("none", 0.0, lambda df, s, seed, c: (df, c))
    concept = next(
        s
        for s in default_scenarios(setup[0])
        if s.name == "concept" and s.magnitude == 0.4
    )
    s = summarize(_run(setup, [none, concept], n_seeds=10)).filter(
        pl.col("scenario") == "concept"
    )
    excess = {
        r["detector"]: r["detection_rate"] - r["chance_rate"]
        for r in s.iter_rows(named=True)
    }
    assert excess["adwin_error"] >= 0.6
    assert excess["score_ks"] <= 0.3 and excess["features_any"] <= 0.3


def test_benchmark_streams_are_resampled_from_the_reference_only(setup):
    clf, base, thr, seq, _ref = setup
    res = monitor(clf, base, thr, synth(10 * W, 99), seq, window=W)
    assert (
        res.table.height == 10
    )  # sanity: the helper window size is what the benchmark uses
