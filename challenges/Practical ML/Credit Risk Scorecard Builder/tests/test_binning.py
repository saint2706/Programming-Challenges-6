from __future__ import annotations

import math

import numpy as np
import pytest
from credit_scorecard.binning import Binning, fit_binning


def _range_bad_rates(binning: Binning) -> list[float]:
    return [b.bad_rate for b in binning.bins if b.kind == "range"]


def _logistic_sample(slope: float, n: int = 30_000, seed: int = 1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.5 + slope * x)))).astype(int)
    return x, y


def test_two_bin_woe_and_iv_match_hand_computation():
    x = np.array([0.0] * 1000 + [1.0] * 1000)
    y = np.array([1] * 100 + [0] * 900 + [1] * 300 + [0] * 700)
    b = fit_binning("f", x, y, min_bin_frac=0.05)
    assert [(bin_.good, bin_.bad) for bin_ in b.bins] == [(900, 100), (700, 300)]
    pg = np.array([900.5, 700.5]) / 1601
    pb = np.array([100.5, 300.5]) / 401
    assert [bin_.woe for bin_ in b.bins] == pytest.approx(np.log(pg / pb))
    assert b.iv == pytest.approx(float(np.sum((pg - pb) * np.log(pg / pb))))


@pytest.mark.parametrize(
    ("slope", "direction"), [(1.0, "ascending"), (-1.0, "descending")]
)
def test_bad_rate_is_monotone_in_the_detected_direction(slope, direction):
    x, y = _logistic_sample(slope)
    b = fit_binning("f", x, y)
    rates = _range_bad_rates(b)
    assert b.direction == direction
    assert rates == sorted(rates, reverse=direction == "descending")
    assert len(rates) >= 4  # real signal should not collapse into one or two bins


def test_u_shaped_signal_still_yields_monotone_bins():
    rng = np.random.default_rng(2)
    x = rng.normal(size=30_000)
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-(-3 + 0.9 * x**2)))).astype(int)
    rates = _range_bad_rates(fit_binning("f", x, y))
    assert rates == sorted(rates) or rates == sorted(rates, reverse=True)


def test_special_and_missing_values_get_their_own_bins():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 6, 10_000).astype(float)
    y = (rng.random(len(x)) < 0.04 + 0.03 * x).astype(int)
    x[:300] = 98.0
    y[:300] = (rng.random(300) < 0.6).astype(int)
    x[300:700] = np.nan
    y[300:700] = (rng.random(400) < 0.01).astype(int)
    b = fit_binning("f", x, y, specials=[96.0, 98.0])
    kinds = [bin_.kind for bin_ in b.bins]
    assert kinds[-2:] == ["special", "missing"]
    special, missing = b.bins[-2], b.bins[-1]
    assert special.count == 300 and missing.count == 400
    assert special.bad_rate > 0.5 and missing.bad_rate < 0.05
    assert sum(bin_.count for bin_ in b.bins) == len(x)
    assert (
        max(bin_.bad_rate for bin_ in b.bins if bin_.kind == "range") < 0.3
    )  # 98 did not leak in


def test_every_range_bin_meets_the_minimum_size_and_has_both_classes():
    x, y = _logistic_sample(0.8)
    b = fit_binning("f", x, y, min_bin_frac=0.1)
    for bin_ in b.bins:
        assert bin_.count >= math.ceil(0.1 * len(x))
        assert bin_.good > 0 and bin_.bad > 0


def test_max_bins_is_respected():
    x, y = _logistic_sample(1.5)
    assert len(fit_binning("f", x, y, max_bins=3).bins) <= 3


def test_cut_points_are_right_closed():
    x = np.repeat(np.arange(5, dtype=float), 2000)
    rng = np.random.default_rng(4)
    y = (rng.random(len(x)) < 0.02 + 0.05 * x).astype(int)
    b = fit_binning("f", x, y, max_bins=3)
    edge = b.edges[0]
    idx = b.index(np.array([edge, edge + 1e-9]))
    assert list(idx) == [0, 1]


def test_unseen_missing_value_scores_neutral_woe():
    x, y = _logistic_sample(1.0)
    b = fit_binning("f", x, y)
    assert not any(bin_.kind == "missing" for bin_ in b.bins)
    assert b.transform(np.array([np.nan]))[0] == 0.0


def test_constant_feature_is_one_bin_with_no_information():
    x = np.full(5_000, 7.0)
    y = (np.random.default_rng(5).random(5_000) < 0.1).astype(int)
    b = fit_binning("f", x, y)
    assert len(b.bins) == 1 and b.bins[0].label == "all values"
    assert b.iv == pytest.approx(0.0, abs=1e-12)


def test_serialisation_roundtrip_preserves_transform():
    x, y = _logistic_sample(1.0)
    x[:50] = np.nan
    b = fit_binning("f", x, y, specials=[0.12345])
    again = Binning.from_dict(b.to_dict())
    assert again == b
    assert np.array_equal(again.transform(x), b.transform(x))
