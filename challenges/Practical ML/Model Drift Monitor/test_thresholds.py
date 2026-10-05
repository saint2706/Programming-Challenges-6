import numpy as np

from stats import Baseline
from thresholds import RULE_OF_THUMB, calibrate, null_distribution

RNG = np.random.default_rng(1)


def _setup(n_train=5000, n_ref=4000):
    cols = lambda n: {
        "score": RNG.beta(2, 5, n),
        "x": RNG.normal(0, 1, n),
        "k": RNG.integers(1, 5, n),
    }
    train, ref = cols(n_train), cols(n_ref)
    return Baseline.fit(train, categorical=["k"]), ref


def _fresh_exceedance(base, thresholds, window=336, n=600):
    hits = {key: 0 for key in thresholds}
    for _ in range(n):
        live = {
            "score": RNG.beta(2, 5, window),
            "x": RNG.normal(0, 1, window),
            "k": RNG.integers(1, 5, window),
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
    thr = calibrate(null_distribution(base, ref, n_windows=800, seed=0), alpha=0.01)
    rates = _fresh_exceedance(base, thr)
    assert all(0.002 <= r <= 0.03 for r in rates.values()), rates


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
