import numpy as np
import pytest
from drift_monitor.stats import Baseline, class_rate_shift, js, ks, psi, wasserstein
from scipy import stats as sps

RNG = np.random.default_rng(0)


def test_psi_is_zero_for_identical_distributions_and_nonnegative():
    p = np.array([0.2, 0.3, 0.5])
    assert psi(p, p) == pytest.approx(0.0, abs=1e-12)
    assert psi(p, np.array([0.5, 0.3, 0.2])) > 0
    assert psi(np.array([0.5, 0.3, 0.2]), p) > 0


def test_psi_matches_a_hand_computed_two_bin_example():
    # (0.5-0.8)*ln(0.5/0.8) + (0.5-0.2)*ln(0.5/0.2)
    expected = (0.5 - 0.8) * np.log(0.5 / 0.8) + (0.5 - 0.2) * np.log(0.5 / 0.2)
    assert psi(np.array([0.5, 0.5]), np.array([0.8, 0.2])) == pytest.approx(
        expected, rel=1e-6
    )


def test_js_distance_is_zero_for_identical_and_one_for_disjoint_support():
    assert js(np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.5, 0.0])) == pytest.approx(
        0.0, abs=1e-9
    )
    assert js(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(
        1.0, abs=1e-3
    )


def test_empty_bins_keep_psi_and_js_finite():
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    assert np.isfinite(psi(a, b)) and np.isfinite(js(a, b))


def test_ks_and_wasserstein_match_scipy():
    ref, live = RNG.normal(0, 1, 2000), RNG.normal(0.5, 1.2, 336)
    assert ks(ref, live) == pytest.approx(sps.ks_2samp(ref, live).statistic)
    assert wasserstein(ref, live) == pytest.approx(
        sps.wasserstein_distance(ref, live) / ref.std()
    )


def _baseline(n=5000):
    cols = {
        "score": RNG.beta(2, 5, n),
        "x": RNG.normal(10, 2, n),
        "k": RNG.integers(1, 4, n),
    }
    return Baseline.fit(cols, categorical=["k"]), cols


def test_window_from_the_same_distribution_has_small_statistics():
    base, _ = _baseline()
    out = base.window_stats("x", RNG.normal(10, 2, 336))
    assert set(out) == {"psi", "ks", "js", "wasserstein"}
    assert out["psi"] < 0.1 and out["ks"] < 0.15 and out["wasserstein"] < 0.2


def test_a_shifted_window_scores_much_higher_on_every_statistic():
    base, _ = _baseline()
    same = base.window_stats("x", RNG.normal(10, 2, 336))
    moved = base.window_stats("x", RNG.normal(14, 2, 336))
    for stat in same:
        assert moved[stat] > 3 * same[stat], stat


def test_values_beyond_the_baseline_range_land_in_the_outer_bins():
    base, _ = _baseline()
    far = base.window_stats("x", np.full(336, 1e6))
    assert (
        far["psi"] > 1
        and far["ks"] == pytest.approx(1.0)
        and np.isfinite(far["wasserstein"])
    )


def test_constant_columns_give_zero_drift_and_stay_finite():
    base = Baseline.fit({"c": np.full(1000, 3.0)})
    same = base.window_stats("c", np.full(336, 3.0))
    assert all(v == 0 and np.isfinite(v) for v in same.values())
    moved = base.window_stats("c", np.full(336, 4.0))
    assert moved["ks"] == pytest.approx(1.0) and np.isfinite(moved["psi"])


def test_a_constant_window_against_a_varying_baseline_is_finite():
    base, _ = _baseline()
    out = base.window_stats("score", np.full(336, 0.2))
    assert all(np.isfinite(v) for v in out.values()) and out["ks"] > 0.3


def test_categorical_window_stats_use_chi_square_and_flag_unseen_categories():
    base, _ = _baseline()
    same = base.window_stats("k", RNG.integers(1, 4, 336))
    assert set(same) == {"psi", "js", "chi2"}
    skewed = base.window_stats("k", np.ones(336, dtype=int))
    unseen = base.window_stats("k", np.full(336, 9))
    assert skewed["chi2"] > 10 * same["chi2"] and skewed["psi"] > same["psi"]
    assert unseen["psi"] > 1 and np.isfinite(unseen["chi2"])


def test_class_rate_shift_is_the_absolute_rate_difference():
    assert class_rate_shift(0.4, np.array([1, 0, 0, 0])) == pytest.approx(0.15)
    assert class_rate_shift(0.5, np.array([], dtype=int)) == 0.0


def test_unknown_column_raises_a_clear_error():
    base, _ = _baseline()
    with pytest.raises(KeyError, match="nope"):
        base.window_stats("nope", np.array([1.0]))
