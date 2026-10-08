import numpy as np
import pytest
from drift_monitor.stats import Baseline
from drift_monitor.thresholds import RULE_OF_THUMB, calibrate, null_distribution


def _setup(n_train=5000, n_ref=4000, seed=1):
    rng = np.random.default_rng(seed)

    def cols(n):
        return {
            "score": rng.beta(2, 5, n),
            "x": rng.normal(0, 1, n),
            "k": rng.integers(1, 5, n),
        }

    train, ref = cols(n_train), cols(n_ref)
    return Baseline.fit(train, categorical=["k"]), ref


def _fresh_exceedance(base, thresholds, window=336, n=2000, seed=99):
    rng = np.random.default_rng(seed)
    hits = {key: 0 for key in thresholds}
    for _ in range(n):
        live = {
            "score": rng.beta(2, 5, window),
            "x": rng.normal(0, 1, window),
            "k": rng.integers(1, 5, window),
        }
        for col, arr in live.items():
            for stat, value in base.window_stats(col, arr).items():
                if value > thresholds[(col, stat)]:
                    hits[(col, stat)] += 1
    return {k: v / n for k, v in hits.items()}


def test_null_distribution_has_one_array_per_column_and_statistic():
    base, ref = _setup()
    null = null_distribution(base, ref, n_windows=50, window=200, seed=0)
    assert ("x", "ks") in null and ("k", "chi2") in null and ("score", "psi") in null
    assert all(len(v) == 50 for v in null.values())
    assert ("k", "ks") not in null


def test_calibrated_thresholds_hit_the_target_false_alarm_rate_on_fresh_windows():
    base, ref = _setup()
    thr = calibrate(
        null_distribution(base, ref, n_windows=800, seed=0, mode="iid"), alpha=0.01
    )
    rates = _fresh_exceedance(base, thr)
    assert all(0.003 <= r <= 0.03 for r in rates.values()), rates


def test_block_thresholds_on_stationary_data_stay_near_the_target_with_a_wider_band():
    # overlapping windows from a 4000-row reference describe the tail less precisely
    base, ref = _setup()
    thr = calibrate(
        null_distribution(base, ref, n_windows=800, seed=0, mode="blocks"), alpha=0.01
    )
    rates = _fresh_exceedance(base, thr)
    assert all(r <= 0.06 for r in rates.values()), rates
    assert np.mean(list(rates.values())) < 0.03


def test_smaller_windows_need_larger_ks_and_psi_thresholds():
    base, ref = _setup()
    big = calibrate(null_distribution(base, ref, n_windows=300, window=800, seed=0))
    small = calibrate(null_distribution(base, ref, n_windows=300, window=100, seed=0))
    for key in (("x", "ks"), ("x", "psi"), ("score", "ks")):
        assert small[key] > big[key], key


def test_thresholds_are_deterministic_per_seed_and_differ_across_seeds():
    base, ref = _setup()
    a = calibrate(null_distribution(base, ref, n_windows=100, seed=3))
    b = calibrate(null_distribution(base, ref, n_windows=100, seed=3))
    c = calibrate(null_distribution(base, ref, n_windows=100, seed=4))
    assert a == b and a != c


def test_the_rule_of_thumb_is_far_too_lax_at_one_week_windows():
    base, ref = _setup()
    thr = calibrate(null_distribution(base, ref, n_windows=500, seed=0))
    assert RULE_OF_THUMB["psi"] == 0.2
    assert (
        thr[("x", "psi")] < RULE_OF_THUMB["psi"] / 2
    )  # a real shift would hide below 0.2


def _drifting_level_series(n, seed, scale=0.5):
    """A slowly wandering level plus noise: neighbouring rows share their level."""
    rng = np.random.default_rng(seed)
    level = np.cumsum(rng.normal(0, 0.05, n // 50 + 1)).repeat(50)[:n] * scale
    return level + rng.normal(0, 1, n)


def test_block_null_is_wider_than_iid_on_serially_correlated_data():
    train = {"x": _drifting_level_series(6000, 1)}
    ref = {"x": _drifting_level_series(4000, 2)}
    base = Baseline.fit(train)
    iid = calibrate(
        null_distribution(base, ref, n_windows=400, window=336, seed=0, mode="iid")
    )
    blocks = calibrate(
        null_distribution(base, ref, n_windows=400, window=336, seed=0, mode="blocks")
    )
    assert blocks[("x", "ks")] > iid[("x", "ks")]
    assert blocks[("x", "wasserstein")] > iid[("x", "wasserstein")]


def test_block_windows_are_contiguous_slices_of_the_reference():
    base, ref = _setup()
    null = null_distribution(base, ref, n_windows=30, window=200, seed=1, mode="blocks")
    assert all(len(v) == 30 for v in null.values())
    with pytest.raises(ValueError, match="window"):
        null_distribution(
            base, {"x": np.arange(50.0)}, n_windows=5, window=200, mode="blocks"
        )


def test_unknown_null_mode_is_rejected():
    base, ref = _setup()
    with pytest.raises(ValueError, match="mode"):
        null_distribution(base, ref, n_windows=5, mode="nope")
