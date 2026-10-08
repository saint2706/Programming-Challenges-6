import numpy as np
import pytest
from helpers import confounded_rct
from uplift import metrics, synth

S = np.array([6, 5, 4, 3, 2, 1.0])
T = np.array([1, 0, 1, 0, 1, 0])
Y = np.array([1, 0, 1, 0, 0, 1])


def hand():
    return metrics.rank(S, T, Y)


def test_curve_matches_hand_computed_values():
    x, q, u = metrics.curve(hand())
    assert x == pytest.approx(np.arange(7) / 6)
    assert q == pytest.approx([0, 0, 1 / 6, 2 / 6, 2 / 6, 2 / 6, 1 / 6])
    assert u == pytest.approx([0, 0, 1 / 3, 1 / 2, 2 / 3, 5 / 9, 1 / 3])


def test_coefficients_match_hand_computed_values():
    assert metrics.qini_coefficient(hand()) == pytest.approx(1 / 8)
    assert metrics.auuc(hand()) == pytest.approx(11 / 54)
    assert metrics.ate(hand()) == pytest.approx(1 / 3)


def test_uplift_at_k_is_the_observed_rate_difference_in_the_top_k():
    assert metrics.uplift_at(hand(), 0.5) == pytest.approx(1.0)
    assert metrics.uplift_at(hand(), 1.0) == pytest.approx(1 / 3)


def test_rank_does_not_depend_on_input_order():
    perm = np.random.default_rng(0).permutation(6)
    a = metrics.qini_coefficient(hand())
    b = metrics.qini_coefficient(metrics.rank(S[perm], T[perm], Y[perm]))
    assert a == pytest.approx(b)


def test_all_tied_scores_give_exactly_zero():
    r = metrics.rank(np.zeros(6), T, Y)
    assert metrics.qini_coefficient(r) == 0.0
    assert metrics.auuc(r) == 0.0
    x, _, _ = metrics.curve(r)
    assert len(x) == 2  # one tie group: the curve is the straight random line


def test_reversed_truth_is_negative_and_truth_is_positive(rct):
    _, t, y, tau = rct
    assert metrics.qini_coefficient(metrics.rank(tau, t, y)) > 0
    assert metrics.qini_coefficient(metrics.rank(-tau, t, y)) < 0


def test_random_scores_have_a_qini_ci_that_covers_zero(rct):
    _, t, y, _ = rct
    score = np.random.default_rng(1).random(len(y))
    boot = metrics.bootstrap({"r": metrics.rank(score, t, y)}, n_boot=100, seed=0)
    ci = metrics.interval(boot, "r", "qini")
    assert ci["lo"] < 0 < ci["hi"]


def test_no_effect_data_gives_a_qini_ci_covering_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 12)).astype(np.float32)
    t = (rng.random(20000) < 0.85).astype(np.int8)
    y, _ = synth.simulate(X, t, seed=0, scenario="none", base_rate=0.1)
    boot = metrics.bootstrap({"o": metrics.rank(X[:, 0], t, y)}, n_boot=100, seed=0)
    ci = metrics.interval(boot, "o", "qini")
    assert ci["lo"] < 0 < ci["hi"]


def test_bootstrap_resamples_keep_the_curve_finite_with_few_controls():
    rng = np.random.default_rng(0)
    t = (rng.random(300) < 0.97).astype(int)  # ~9 controls: prefixes often have none
    y = (rng.random(300) < 0.3).astype(int)
    r = metrics.rank(rng.random(300), t, y)
    boot = metrics.bootstrap({"a": r}, n_boot=50, seed=0)
    assert all(np.isfinite(v).all() for v in boot.draws["a"].values())


def test_bootstrap_uses_the_same_resamples_for_every_scorer(rct):
    _, t, y, tau = rct
    r = metrics.rank(tau, t, y)
    boot = metrics.bootstrap({"a": r, "b": r}, n_boot=20, seed=0)
    assert np.array_equal(boot.draws["a"]["qini"], boot.draws["b"]["qini"])
    d = metrics.diff_interval(boot, "a", "b", "qini")
    assert d["est"] == 0 and d["lo"] == 0 and d["hi"] == 0


def test_interval_brackets_the_point_estimate(rct):
    _, t, y, tau = rct
    boot = metrics.bootstrap({"o": metrics.rank(tau, t, y)}, n_boot=100, seed=0)
    ci = metrics.interval(boot, "o", "qini")
    assert ci["lo"] <= ci["est"] <= ci["hi"]


def test_summarize_has_the_expected_keys():
    keys = metrics.summarize(hand()).keys()
    assert {"qini", "auuc", "ate", "uplift@10", "incremental@50"} <= set(keys)


def test_decile_calibration_recovers_a_step_effect():
    rng = np.random.default_rng(0)
    n = 40000
    score = rng.random(n)
    t = (rng.random(n) < 0.5).astype(int)
    effect = np.where(score > 0.5, 0.2, 0.0)
    y = (rng.random(n) < 0.1 + effect * t).astype(int)
    rows = metrics.decile_calibration(score, t, y, n_bins=10)
    assert [r["decile"] for r in rows] == list(range(1, 11))
    top, bottom = rows[0], rows[-1]
    assert top["observed"] == pytest.approx(0.2, abs=3 * top["se"])
    assert bottom["observed"] == pytest.approx(0.0, abs=3 * bottom["se"])
    assert top["mean_pred"] > bottom["mean_pred"]


def test_a_plain_difference_in_means_is_biased_by_confounding():
    _, t, y, _ = confounded_rct()
    score = np.random.default_rng(1).random(len(y))
    assert metrics.ate(metrics.rank(score, t, y)) > 0.05  # true effect is 0.02


def test_inverse_propensity_arm_weights_remove_the_bias():
    _, t, y, e = confounded_rct()
    score = np.random.default_rng(1).random(len(y))
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    assert metrics.ate(metrics.rank(score, t, y, arm_weights=w)) == pytest.approx(
        0.02, abs=0.004
    )


def test_unit_arm_weights_change_nothing():
    plain = metrics.summarize(hand())
    weighted = metrics.summarize(metrics.rank(S, T, Y, arm_weights=np.ones(6)))
    assert weighted == pytest.approx(plain)


def test_arm_weights_flow_through_the_bootstrap():
    _, t, y, e = confounded_rct(n=60_000)
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    r = metrics.rank(np.random.default_rng(1).random(len(y)), t, y, arm_weights=w)
    boot = metrics.bootstrap({"r": r}, n_boot=40, seed=0)
    ci = metrics.interval(boot, "r", "ate")
    assert ci["lo"] <= 0.02 + 0.01 and ci["hi"] >= 0.02 - 0.01
    assert ci["est"] < 0.04  # not the confounded 0.05+


def test_weighted_calibration_in_one_bin_equals_the_weighted_ate():
    _, t, y, e = confounded_rct(n=100_000)
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    score = np.random.default_rng(1).random(len(y))
    row = metrics.decile_calibration(score, t, y, n_bins=1, arm_weights=w)[0]
    assert row["observed"] == pytest.approx(
        metrics.ate(metrics.rank(score, t, y, arm_weights=w)), abs=1e-9
    )
    assert row["observed"] == pytest.approx(0.02, abs=0.008)
    assert row["se"] > 0


def test_weighted_calibration_se_matches_the_unweighted_formula_at_unit_weights():
    rng = np.random.default_rng(0)
    t = (rng.random(5000) < 0.5).astype(int)
    y = (rng.random(5000) < 0.2).astype(int)
    score = rng.random(5000)
    a = metrics.decile_calibration(score, t, y, n_bins=4)
    b = metrics.decile_calibration(score, t, y, n_bins=4, arm_weights=np.ones(5000))
    assert [r["se"] for r in a] == pytest.approx([r["se"] for r in b])
