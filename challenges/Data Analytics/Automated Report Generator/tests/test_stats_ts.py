"""The statistics under the findings, each against an independent implementation or oracle."""

import numpy as np
import pymannkendall as pmk
import pytest
from report_generator import stats_ts as ts


def ar1(rng, n, phi):
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("phi", [0.0, 0.6])
def test_mann_kendall_matches_pymannkendall(seed, phi):
    rng = np.random.default_rng(seed)
    y = ar1(rng, 80, phi) + 0.02 * np.arange(80)
    ours, ref = ts.mann_kendall(y), pmk.hamed_rao_modification_test(y)
    assert ours.s == pytest.approx(ref.s)
    assert ours.var_s == pytest.approx(ref.var_s, rel=1e-9)
    assert ours.z == pytest.approx(ref.z, rel=1e-9)
    assert ours.p_hr == pytest.approx(ref.p, rel=1e-9)
    assert ours.p >= ours.p_hr
    assert ours.slope == pytest.approx(ref.slope)
    assert ours.tau == pytest.approx(ref.Tau)


def test_mann_kendall_with_ties_and_short_input():
    y = np.repeat(np.arange(10.0), 3)
    ref = pmk.hamed_rao_modification_test(y)
    assert ts.mann_kendall(y).var_s == pytest.approx(ref.var_s, rel=1e-9)
    with pytest.raises(ValueError, match="at least 8"):
        ts.mann_kendall(np.arange(5.0))


def test_the_trend_test_survives_autocorrelated_noise_better_than_the_plain_one():
    """On a smooth stationary series the plain test sees a trend half the time; this one far less."""
    rng = np.random.default_rng(0)
    naive = hamed_rao = ours = 0
    reps = 80
    for _ in range(reps):
        y = ar1(rng, 100, 0.85)
        naive += pmk.original_test(y).p < 0.01
        hamed_rao += pmk.hamed_rao_modification_test(y).p < 0.01
        ours += ts.mann_kendall(y).p < 0.01
    assert naive / reps > 0.3
    assert ours <= hamed_rao
    # smooth noise leaves few independent points (about 8 of 100), so some false trends survive;
    # the README reports the measured rate and the findings are worded as trends, not causes
    assert ours / reps <= 0.25


def test_the_trend_test_keeps_its_power():
    rng = np.random.default_rng(1)
    hits = sum(
        ts.mann_kendall(ar1(rng, 100, 0.5) + 0.08 * np.arange(100)).p < 0.01
        for _ in range(30)
    )
    assert hits >= 28


def test_bootstrap_is_reproducible_and_skipped_when_unneeded():
    y = np.random.default_rng(2).normal(size=60) + 0.05 * np.arange(60)
    assert ts.mann_kendall(y) == ts.mann_kendall(y)
    flat = ts.mann_kendall(np.random.default_rng(3).normal(size=60))
    assert flat.p_bootstrap is None and flat.p == flat.p_hr


@pytest.mark.parametrize("seed", range(12))
def test_pelt_equals_the_exact_dynamic_program(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 70))
    y = (
        rng.normal(size=n)
        + np.repeat(rng.normal(0, 2, 4), np.ceil(n / 4).astype(int))[:n]
    )
    penalty = float(rng.uniform(2, 12))
    fast, exact = ts.pelt(y, penalty, 3), ts.pelt_exact(y, penalty, 3)
    assert fast == exact
    assert ts.segment_sse(y, fast) + penalty * len(fast) <= ts.segment_sse(y, []) + 1e-9


def test_pelt_finds_a_planted_shift_and_nothing_in_noise():
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(0, 1, 60), rng.normal(4, 1, 60)])
    sigma = ts.robust_sigma(y)
    cuts = ts.pelt(y, 3 * sigma**2 * np.log(y.size), 5)
    assert len(cuts) == 1 and abs(cuts[0] - 60) <= 2
    noise = rng.normal(size=120)
    assert ts.pelt(noise, 3 * ts.robust_sigma(noise) ** 2 * np.log(120), 5) == []


def test_robust_sigma_ignores_a_level_shift():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 2, 400)
    shifted = y + np.where(np.arange(400) > 200, 50, 0)
    assert ts.robust_sigma(shifted) == pytest.approx(ts.robust_sigma(y), rel=0.1)
    assert ts.robust_sigma(y) == pytest.approx(2, rel=0.2)


def test_stl_strength_separates_seasonal_from_noise():
    rng = np.random.default_rng(3)
    t = np.arange(120)
    seasonal = ts.stl(10 + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.3, 120), 12)
    flat = ts.stl(10 + rng.normal(0, 1, 120), 12)
    assert seasonal.seasonal_strength > 0.9
    assert flat.seasonal_strength < 0.5
    trended = ts.stl(0.1 * t + rng.normal(0, 0.3, 120), 12)
    assert trended.trend_strength > 0.9
    with pytest.raises(ValueError, match="two full periods"):
        ts.stl(np.arange(20.0), 12)


def season(rng, n, period, amp, kind="white"):
    t = np.arange(n)
    base = {
        "white": rng.normal(size=n),
        "rw": np.cumsum(rng.normal(size=n)),
        "ar": ar1(rng, n, 0.7),
    }[kind]
    return base + amp * (
        np.sin(2 * np.pi * t / period) + 0.5 * np.sin(4 * np.pi * t / period + 1)
    )


@pytest.mark.parametrize(("period", "n"), [(52, 104), (12, 60), (7, 70)])
def test_repeatability_has_low_false_positive_rate_and_high_power(period, n):
    rng = np.random.default_rng(period)
    gate = lambda r: r is not None and r.p <= 0.01 and r.rho >= 0.3
    false = {
        kind: np.mean(
            [
                gate(ts.repeatability(season(rng, n, period, 0, kind), period, 199))
                for _ in range(60)
            ]
        )
        for kind in ("white", "ar", "rw")
    }
    assert max(false.values()) <= 0.08, false
    power = np.mean(
        [
            gate(ts.repeatability(season(rng, n, period, 6), period, 199))
            for _ in range(40)
        ]
    )
    assert power >= 0.9


def test_repeatability_is_immune_to_a_wandering_level_and_needs_two_cycles():
    rng = np.random.default_rng(5)
    rep = ts.repeatability(season(rng, 120, 12, 6, "rw"), 12)
    assert rep is not None and rep.p < 0.01 and rep.cycles == 10
    assert ts.repeatability(np.arange(20.0), 12) is None
    y = season(
        rng, 60, 12, 3
    )  # the permutation test is seeded, so a report is reproducible
    assert ts.repeatability(y, 12, seed=1) == ts.repeatability(y, 12, seed=1)


def test_periodic_decomposition_recovers_the_pattern():
    rng = np.random.default_rng(6)
    t = np.arange(104)
    y = 5 + 0.05 * t + 4 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 0.5, 104)
    dec = ts.periodic_decompose(y, 52)
    assert abs(dec.seasonal.mean()) < 0.5
    assert np.corrcoef(dec.seasonal, 4 * np.sin(2 * np.pi * t / 52))[0, 1] > 0.95
    assert np.allclose(dec.trend + dec.seasonal + dec.resid, y)
    assert ts.block_size(52) == 4 and ts.block_size(12) == 1
