from __future__ import annotations

import numpy as np
import pytest
from credit_scorecard import metrics
from scipy.stats import ks_2samp


@pytest.mark.parametrize("integer_scores", [False, True])
def test_ks_matches_scipy_including_tied_scores(integer_scores):
    rng = np.random.default_rng(0)
    y = (rng.random(5_000) < 0.2).astype(int)
    score = rng.normal(size=5_000) + 0.8 * y
    if integer_scores:
        score = np.round(score * 3)  # lots of ties
    expected = ks_2samp(score[y == 1], score[y == 0]).statistic
    assert metrics.ks(y, score) == pytest.approx(expected)


def test_perfect_and_useless_rankings():
    y = np.array([0, 0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert metrics.auc(y, perfect) == 1.0 and metrics.gini(y, perfect) == 1.0
    assert metrics.ks(y, perfect) == 1.0
    assert metrics.gini(y, np.zeros(5)) == 0.0


def test_psi_is_zero_for_identical_samples_and_matches_a_hand_value():
    expected = np.arange(1000, dtype=float)
    assert metrics.psi(expected, expected) == pytest.approx(0.0, abs=1e-12)
    shares = [150] + [100] * 8 + [50]
    actual = np.concatenate([np.full(c, 100 * k + 50.0) for k, c in enumerate(shares)])
    hand = 0.05 * np.log(1.5) - 0.05 * np.log(0.5)
    assert metrics.psi(expected, actual) == pytest.approx(hand)


def test_psi_grows_with_a_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(size=20_000)
    small = metrics.psi(base, rng.normal(0.1, 1, 20_000))
    large = metrics.psi(base, rng.normal(1.0, 1, 20_000))
    assert small < 0.1  # stable
    assert large > 0.25  # a different population


def test_bootstrap_interval_brackets_the_auc():
    rng = np.random.default_rng(2)
    y = (rng.random(4_000) < 0.15).astype(int)
    score = rng.normal(size=4_000) + y
    lo, hi = metrics.bootstrap_auc_interval(y, score, n=100)
    assert lo < metrics.auc(y, score) < hi


def test_calibration_table_partitions_the_rows():
    rng = np.random.default_rng(3)
    score = rng.integers(300, 700, 10_000).astype(float)
    pd_ = 1 / (1 + 50 * 2 ** ((score - 600) / 20))
    y = (rng.random(10_000) < pd_).astype(int)
    table = metrics.calibration_table(y, score, pd_, n_bins=10)
    assert table["n"].sum() == 10_000
    assert table["band"].to_list() == sorted(table["band"].to_list())
    assert table["score_max"].to_list()[:-1] < table["score_min"].to_list()[1:]
    assert table["observed_bad_rate"].is_between(0, 1).all()
