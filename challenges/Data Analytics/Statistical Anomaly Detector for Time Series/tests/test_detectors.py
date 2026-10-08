from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from ts_anomaly.detectors import (
    DetectorError,
    Params,
    autocorrelation,
    detect_period,
    generalized_esd,
    gesd,
    gesd_critical_values,
    iqr,
    iqr_scores,
    mad,
    modified_z_scores,
    rolling_mad,
    rolling_z,
    run_method,
    stl_decompose,
    stl_detect,
    z_scores,
    zscore,
)
from ts_anomaly.synth import base_series, clean_noise, inject_spikes

# The worked example from the NIST/SEMATECH e-Handbook, section 1.3.5.17.3
# (Rosner's 1983 data): 54 values, 3 outliers (6.01, 5.42, 5.34) at alpha 0.05.
NIST = np.array(
    [
        -0.25,
        0.68,
        0.94,
        1.15,
        1.20,
        1.26,
        1.26,
        1.34,
        1.38,
        1.43,
        1.49,
        1.49,
        1.55,
        1.56,
    ]
    + [
        1.58,
        1.65,
        1.69,
        1.70,
        1.76,
        1.77,
        1.81,
        1.91,
        1.94,
        1.96,
        1.99,
        2.06,
        2.09,
        2.10,
    ]
    + [
        2.14,
        2.15,
        2.23,
        2.24,
        2.26,
        2.35,
        2.37,
        2.40,
        2.47,
        2.54,
        2.62,
        2.64,
        2.90,
        2.92,
    ]
    + [2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68, 4.30, 4.64, 5.34, 5.42, 6.01]
)


# ---------------------------------------------------------------- point scores


def test_z_scores_match_the_definition():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    expected = np.abs(x - x.mean()) / x.std(ddof=1)
    assert np.allclose(z_scores(x), expected)
    assert zscore(x).flags.tolist() == [False] * 5  # n=5: max |z| is 1.79 < 3


def test_constant_series_has_no_anomalies_under_any_point_detector():
    x = np.full(20, 7.0)
    for det in (zscore(x), iqr(x), mad(x), gesd(x)):
        assert det.count == 0


def test_samuelson_bound_means_zscore_cannot_fire_for_n_le_10():
    # max |z| <= (n-1)/sqrt(n): 2.846 at n=10, 3.015 at n=11.
    for n, fires in ((10, False), (11, True)):
        x = np.zeros(n)
        x[0] = 1e9
        x = x + np.random.default_rng(0).normal(0, 1, n)
        z = z_scores(x).max()
        assert z <= (n - 1) / np.sqrt(n) + 1e-9
        assert (zscore(x).count > 0) is fires
        assert mad(x).flags[0]  # the robust score has no such ceiling


def test_iqr_fences_by_hand():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 20], dtype=float)
    q1, q3 = np.percentile(x, [25, 75])  # 3.25, 7.75
    assert (q1, q3) == (3.25, 7.75)
    det = iqr(x, 1.5)  # upper fence 7.75 + 1.5 * 4.5 = 14.5
    assert det.flags.tolist() == [False] * 9 + [True]
    assert det.scores[-1] == pytest.approx((20 - 7.75) / 4.5)
    assert iqr(x, 3.0).count == 0  # fence moves out to 21.25 with k=3
    assert iqr(x, 1.5).params == {"k": 1.5}


def test_iqr_with_zero_spread_flags_anything_off_the_tied_value():
    x = np.array([5.0] * 11 + [9.0])
    assert np.isinf(iqr_scores(x)[-1])
    assert iqr(x).flags.tolist() == [False] * 11 + [True]


def test_modified_z_by_hand():
    x = np.array([2, 3, 3, 4, 4, 4, 5, 5, 6, 50], dtype=float)
    # median 4; |x - 4| = 2,1,1,0,0,0,1,1,2,46 -> MAD 1
    scores = modified_z_scores(x)
    assert scores[-1] == pytest.approx(0.6745 * 46)
    assert scores[0] == pytest.approx(0.6745 * 2)
    assert mad(x).flags.tolist() == [False] * 9 + [True]


def test_modified_z_falls_back_to_mean_absolute_deviation_when_mad_is_zero():
    x = np.array([5.0] * 10 + [9.0])
    scores = modified_z_scores(x)  # MAD = 0; mean abs dev = 4/11
    assert scores[-1] == pytest.approx(4 / (1.253314 * 4 / 11))
    assert mad(x).flags[-1]
    assert not mad(x).flags[:-1].any()


def test_bad_input_is_rejected():
    for bad in ([1.0, 2.0], [1.0, np.nan, 3.0, 4.0], np.ones((3, 3))):
        with pytest.raises(DetectorError):
            zscore(bad)


# ------------------------------------------------------------------------ GESD


def test_gesd_reproduces_the_nist_worked_example():
    res = generalized_esd(NIST, 10, 0.05)
    assert res.n_outliers == 3
    assert sorted(NIST[res.indices]) == [5.34, 5.42, 6.01]
    assert list(NIST[res.indices]) == [6.01, 5.42, 5.34]  # most extreme first
    # NIST table: R1..R3 = 3.118, 2.942, 3.179 ; lambda1..3 = 3.158, 3.151, 3.143
    assert res.r[:3] == pytest.approx([3.118, 2.942, 3.179], abs=2e-3)
    assert res.lam[:3] == pytest.approx([3.158, 3.151, 3.143], abs=2e-3)
    # the fourth is not significant even though it is the next most extreme
    assert res.r[3] < res.lam[3]
    # note R2 < lambda2: the test still keeps it, because R3 > lambda3 (that is
    # the point of testing "largest i with R_i > lambda_i")
    assert res.r[1] < res.lam[1]


def test_first_gesd_critical_value_is_the_grubbs_critical_value():
    for n in (20, 54, 200):
        t = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
        grubbs = (n - 1) / np.sqrt(n) * np.sqrt(t**2 / (n - 2 + t**2))
        assert gesd_critical_values(n, 1, 0.05)[0] == pytest.approx(grubbs, rel=1e-12)


def test_gesd_finds_two_outliers_that_mask_each_other():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 60)
    x[10], x[40] = 7.0, 7.5
    # With one outlier assumed the pair inflates the sd; with k >= 2 both go.
    assert generalized_esd(x, 5, 0.05).n_outliers >= 2
    assert set(generalized_esd(x, 5, 0.05).indices[:2]) == {10, 40}
    # and the specified upper bound really is a bound
    assert generalized_esd(x, 1, 0.05).n_outliers <= 1


def test_gesd_argument_validation():
    x = clean_noise(30, 0)
    with pytest.raises(DetectorError):
        generalized_esd(x, 0)
    with pytest.raises(DetectorError):
        generalized_esd(x, 28)  # > n - 3
    with pytest.raises(DetectorError):
        generalized_esd(x, 5, alpha=1.5)
    with pytest.raises(DetectorError):
        gesd(np.arange(5.0))


def test_gesd_rarely_cries_wolf_on_clean_noise():
    alarms = sum(
        generalized_esd(clean_noise(200, s), 10, 0.05).n_outliers > 0
        for s in range(400)
    )
    assert (
        alarms / 400 < 0.08
    )  # nominal 0.05; 3 sigma of sampling noise is 0.033/sqrt-ish


def test_gesd_detection_wrapper_exposes_the_count():
    x = NIST
    det = gesd(x, Params(max_outliers=10))
    assert det.count == 3
    assert det.params["n_outliers"] == 3


# --------------------------------------------------------------------- rolling


def test_rolling_scores_compare_each_point_only_to_the_past():
    x = np.zeros(40)
    x += np.random.default_rng(0).normal(0, 1, 40)
    x[30] += 25
    det = rolling_z(x, window=20)
    assert det.flags[30]
    assert not det.flags[:20].any()  # no full window yet
    # the *next* points are not flagged just because a spike sits in their window
    assert not det.flags[31]


def test_rolling_with_flat_window_flags_any_change():
    x = np.concatenate([np.ones(30), [2.0], np.ones(5)])
    for det in (rolling_z(x, 10), rolling_mad(x, 10)):
        assert det.flags[30] and det.count == 1
        assert np.isinf(det.scores[30])


def test_rolling_window_validation():
    x = clean_noise(30, 0)
    with pytest.raises(DetectorError):
        rolling_z(x, window=2)
    with pytest.raises(DetectorError):
        rolling_mad(x, window=30)


def test_rolling_z_is_masked_by_a_burst_but_rolling_mad_is_not():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 400)
    x[300:310] += 8.0
    z = rolling_z(x, 50).flags[300:310].sum()
    m = rolling_mad(x, 50).flags[300:310].sum()
    assert m == 10
    assert z < m


def test_rolling_mad_adapts_to_a_sustained_shift_and_stops_flagging():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 500)
    x[250:330] += 8.0  # much longer than the window
    flags = rolling_mad(x, 50).flags
    assert flags[250:255].all()  # the onset is caught...
    assert not flags[290:330].any()  # ...then the new level became "normal"
    assert flags[250:330].sum() < 0.5 * 80


# ------------------------------------------------------------------------- STL


def _seasonal_with_spikes(seed=0, n=600, period=24):
    rng = np.random.default_rng(seed)
    x = base_series(n, period, rng=rng)
    truth = inject_spikes(x, np.array([100, 210, 330, 450]), 8.0, rng)
    return x, truth


def test_stl_finds_spikes_hidden_inside_the_seasonal_range():
    x, truth = _seasonal_with_spikes()
    assert zscore(x).count == 0  # +-8 on a +-10 swing is invisible to a global test
    det = stl_detect(x, 24, "gesd")
    assert det.flags[truth].all()
    assert det.count <= 8  # the 4 spikes plus at most a few borderline noise points
    assert set(det.components) == {"trend", "seasonal", "resid"}
    assert det.params["periods"] == [24]


def test_stl_decomposition_is_additive():
    x, _ = _seasonal_with_spikes()
    c = stl_decompose(x, 24)
    assert np.allclose(c["trend"] + c["seasonal"] + c["resid"], x)


def test_stl_scorers_all_run_and_agree_on_the_big_spikes():
    x, truth = _seasonal_with_spikes()
    for name in ("mad", "z", "iqr", "gesd"):
        det = stl_detect(x, 24, name)
        assert det.method == f"stl-{name}"
        assert det.flags[truth].all(), name


def test_stl_argument_validation():
    x, _ = _seasonal_with_spikes()
    with pytest.raises(DetectorError):
        stl_decompose(x[:30], 24)  # under two cycles
    with pytest.raises(DetectorError):
        stl_decompose(x, 1)
    with pytest.raises(DetectorError):
        stl_decompose(x, 24, seasonal=8)  # must be odd
    with pytest.raises(DetectorError):
        stl_detect(x, 24, "nope")


def test_mstl_accepts_two_periods():
    rng = np.random.default_rng(0)
    t = np.arange(720)
    x = (
        20
        + 5 * np.sin(2 * np.pi * t / 12)
        + 8 * np.sin(2 * np.pi * t / 120)
        + rng.normal(0, 0.5, 720)
    )
    x[400] += 12
    det = stl_detect(x, [12, 120], "gesd")
    assert det.flags[400]
    assert det.params["periods"] == [12, 120]
    assert det.components["resid"].std() < 1.5


# ---------------------------------------------------------------------- period


def test_autocorrelation_of_a_sine_peaks_at_its_period():
    t = np.arange(600)
    ac = autocorrelation(np.sin(2 * np.pi * t / 20), 60)
    assert ac[0] == pytest.approx(1.0)
    assert ac[20] > 0.9
    assert ac[10] < -0.9
    assert autocorrelation(np.ones(10), 3).tolist() == [0.0] * 4


@pytest.mark.parametrize("period", [7, 24, 48, 97])
def test_detect_period_recovers_exact_periods(period):
    rng = np.random.default_rng(period)
    n = period * 15
    x = base_series(n, period, noise_sd=1.5, trend_per_step=25 / n, rng=rng)
    inject_spikes(x, rng.integers(0, n, 6), 20.0, rng)
    est = detect_period(x)
    assert est is not None and est.period == period


def test_detect_period_prefers_the_fundamental_over_its_multiples():
    t = np.arange(1200)
    x = np.sin(2 * np.pi * t / 12) + 0.4 * np.sin(2 * np.pi * t / 6)  # harmonics only
    x += np.random.default_rng(0).normal(0, 0.2, 1200)
    assert detect_period(x).period == 12


def test_detect_period_on_a_non_sinusoidal_shape():
    x = np.tile(np.arange(20, dtype=float), 40)  # sawtooth
    x += np.random.default_rng(0).normal(0, 0.5, len(x))
    assert detect_period(x).period == 20


def test_detect_period_rejects_noise_constants_and_smooth_drift():
    for seed in range(20):
        assert detect_period(clean_noise(500, seed)) is None
    assert detect_period(np.full(200, 3.0)) is None
    rng = np.random.default_rng(0)
    ar1 = np.zeros(2000)
    for i in range(1, 2000):
        ar1[i] = 0.98 * ar1[i - 1] + rng.normal()
    assert detect_period(ar1) is None  # highly autocorrelated, but nothing repeats


def test_detect_period_ignores_a_short_series():
    assert detect_period(np.arange(5.0)) is None


# -------------------------------------------------------------------- dispatch


def test_run_method_dispatch_and_errors():
    x, _ = _seasonal_with_spikes()
    assert run_method("mad", x).method == "mad"
    assert run_method("stl-mad", x, periods=[24]).method == "stl-mad"
    with pytest.raises(DetectorError, match="needs a seasonal period"):
        run_method("stl-mad", x)
    with pytest.raises(DetectorError, match="unknown method"):
        run_method("prophet", x)


def test_params_change_the_outcome():
    x = np.concatenate([np.random.default_rng(0).normal(0, 1, 200), [4.0]])
    assert zscore(x, 3.0).flags[-1]
    assert not run_method("zscore", x, params=Params(z_threshold=6.0)).flags[-1]
