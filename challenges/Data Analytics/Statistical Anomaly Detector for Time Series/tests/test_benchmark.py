"""The claims the README makes about *why* each method fails, as executable checks."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from ts_anomaly import benchmark as bm
from ts_anomaly.detectors import Params, run_method, stl_detect
from ts_anomaly.evaluate import event_counts, point_counts
from ts_anomaly.synth import base_series, clean_noise, make_scenarios


def test_scenarios_are_deterministic_and_well_formed():
    a, b = make_scenarios(4), make_scenarios(4)
    assert [s.name for s in a] == [s.name for s in b]
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x.values, y.values) and np.array_equal(x.truth, y.truth)
        assert x.truth.any() and x.truth.shape == x.values.shape
    assert not np.array_equal(make_scenarios(5)[0].values, a[0].values)
    assert [s.period for s in a] == [24, 24, 24, 24, None]


def test_spikes_hidden_inside_the_seasonal_range_defeat_every_global_and_rolling_method():
    for seed in range(3):
        sc = make_scenarios(seed)[0]
        for m in ("zscore", "iqr", "mad", "gesd", "rolling-z", "rolling-mad"):
            det = run_method(m, sc.values, params=Params(window=48))
            assert event_counts(det.flags, sc.truth, bm.TOL).recall <= 0.3, m
        det = stl_detect(sc.values, 24, "gesd")
        assert event_counts(det.flags, sc.truth, bm.TOL).recall == 1.0


def test_zscore_is_masked_by_the_outliers_it_is_looking_for():
    recalls = {"zscore": [], "mad": [], "iqr": []}
    for seed in range(10):
        sc = make_scenarios(seed)[4]
        assert sc.values.std(ddof=1) > 2 * sc.values[~sc.truth].std(ddof=1)
        for m, r in recalls.items():
            r.append(point_counts(run_method(m, sc.values).flags, sc.truth).recall)
    assert np.mean(recalls["zscore"]) < 0.2
    assert np.mean(recalls["mad"]) > 0.95
    assert np.mean(recalls["iqr"]) > 0.9


def test_gesd_needs_max_outliers_to_be_a_real_upper_bound():
    small, large = [], []
    for seed in range(10):
        sc = make_scenarios(seed)[4]  # 10 true outliers
        small.append(
            point_counts(
                run_method("gesd", sc.values, params=Params(max_outliers=5)).flags,
                sc.truth,
            ).recall
        )
        large.append(
            point_counts(
                run_method("gesd", sc.values, params=Params(max_outliers=20)).flags,
                sc.truth,
            ).recall
        )
    assert np.mean(small) < 0.7  # can only ever report 5 of the 10
    assert np.mean(large) > 0.95


def test_stl_with_a_wrong_period_stops_finding_the_spikes():
    sc = make_scenarios(0)[0]
    true = event_counts(stl_detect(sc.values, 24, "z").flags, sc.truth, bm.TOL)
    wrong = event_counts(stl_detect(sc.values, 18, "z").flags, sc.truth, bm.TOL)
    multiple = event_counts(stl_detect(sc.values, 48, "z").flags, sc.truth, bm.TOL)
    assert true.recall == 1.0 and multiple.recall == 1.0
    assert wrong.recall < 0.3


def test_robust_stl_with_the_default_smoother_leaves_a_heavy_tailed_residual():
    robust7, plain7, robust25 = [], [], []
    for seed in range(3):
        x = base_series(1200, 24, rng=np.random.default_rng(seed))
        for store, p in (
            (robust7, Params(stl_robust=True, stl_seasonal=7)),
            (plain7, Params(stl_robust=False, stl_seasonal=7)),
            (robust25, Params(stl_robust=True, stl_seasonal=25)),
        ):
            store.append(stl_detect(x, 24, "mad", p).count)
    assert (
        np.mean(plain7) < 3
    )  # theory for a 3.5-sigma cut on 1200 Gaussian points: 0.6
    assert np.mean(robust7) > 10 * max(np.mean(plain7), 1)
    assert np.mean(robust25) < np.mean(robust7) / 3


@pytest.mark.parametrize(
    ("method", "theory", "tolerance"),
    [
        ("zscore", 2 * stats.norm.sf(3.0), 0.15),
        ("iqr", 2 * stats.norm.sf(stats.norm.ppf(0.75) * (1 + 1.5 * 2)), 0.15),
        ("mad", 2 * stats.norm.sf(3.5), 0.4),
    ],
)
def test_point_detector_false_positive_rates_match_gaussian_theory(
    method, theory, tolerance
):
    flagged = total = 0
    for seed in range(300):
        x = clean_noise(1000, seed)
        flagged += run_method(method, x).count
        total += len(x)
    assert flagged / total == pytest.approx(theory, rel=tolerance)


def test_rolling_z_false_positive_rate_follows_a_t_distribution_not_the_normal():
    window, flagged, total = 50, 0, 0
    for seed in range(300):
        det = run_method(
            "rolling-z", clean_noise(1000, seed), params=Params(window=window)
        )
        flagged += det.flags[window:].sum()
        total += 1000 - window
    t_theory = 2 * stats.t.sf(3.0 / np.sqrt(1 + 1 / window), window - 1)
    assert t_theory > 1.5 * 2 * stats.norm.sf(
        3.0
    )  # the normal would under-predict badly
    assert flagged / total == pytest.approx(t_theory, rel=0.15)


def test_samuelson_table_marks_n_10_as_impossible_and_n_11_as_possible():
    table = bm.samuelson_table()
    row10 = next(line for line in table.splitlines() if line.startswith("| 10 |"))
    row11 = next(line for line in table.splitlines() if line.startswith("| 11 |"))
    assert "**no**" in row10 and "**no**" not in row11


def test_period_detection_table_recovers_every_period_in_the_fixed_sample():
    table = bm.period_detection_table(cases=20)
    assert "| exact period | 20 |" in table
    assert "White noise called seasonal: 0 of 200" in table


def test_markdown_table_formatting():
    assert (
        bm.md_table(["a", "b"], [[1, 0.5], ["x", "y"]])
        == "| a | b |\n|---|---|\n| 1 | 0.50 |\n| x | y |"
    )
