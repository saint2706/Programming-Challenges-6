import numpy as np
import pytest
from spc_stream.charts import CUSUM, EWMA
from spc_stream.monitor import Monitor, Prewhitener, estimate_baseline


def test_moving_range_sigma_estimate():
    x = np.random.default_rng(0).normal(10, 2, 5000).tolist()
    b = estimate_baseline(x)
    assert b.mean == pytest.approx(10, abs=0.1)
    assert b.sigma == pytest.approx(2, rel=0.05)


def test_baseline_rejects_flat_or_short():
    with pytest.raises(ValueError):
        estimate_baseline([1.0])
    with pytest.raises(ValueError, match="no variation"):
        estimate_baseline([3.0] * 10)


def ar1(n, phi, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def test_prewhitener_recovers_ar1_and_whitens():
    x = ar1(4000, 0.8).tolist()
    p = Prewhitener(ar1=True)
    res = p.fit(x)
    assert p.phi == pytest.approx(0.8, abs=0.03)
    r = np.array(res)
    assert abs(np.corrcoef(r[:-1], r[1:])[0, 1]) < 0.05


def test_prewhitener_seasonal_difference():
    p = Prewhitener(season=3)
    assert p.fit([1, 2, 3, 11, 12, 13]) == [10, 10, 10]
    assert p(21) == 10  # continues from the fitted state


def test_monitor_finds_planted_step_and_respects_phase1():
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(0, 1, 300), rng.normal(1.5, 1, 100)]
    events = list(Monitor(CUSUM(), baseline_n=200).run(x))
    assert all(e.point is None for e in events[:200])
    alarms = [e.i for e in events if e.alarm]
    assert alarms and 300 <= alarms[0] < 320


def test_autocorrelation_inflates_false_alarms_without_prewhitening():
    x = ar1(6000, 0.9, seed=3).tolist()
    raw = sum(bool(e.alarm) for e in Monitor(EWMA(), 1000).run(x))
    white = sum(bool(e.alarm) for e in Monitor(EWMA(), 1000, ar1=True).run(x))
    assert raw > 5 * max(white, 1)


def test_cooldown_holds_back_repeat_alarms():
    x = [0.0, 1.0] * 50 + [50.0] * 30
    loud = sum(bool(e.alarm) for e in Monitor(CUSUM(), 100).run(x))
    quiet = sum(bool(e.alarm) for e in Monitor(CUSUM(), 100, cooldown=100).run(x))
    assert loud > 5 and quiet == 1


def test_baseline_must_cover_season():
    with pytest.raises(ValueError):
        Monitor(EWMA(), baseline_n=10, season=48)
