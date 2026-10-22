"""Kaplan-Meier against R's survfit, a textbook example, a naive loop and the edge cases."""

import numpy as np
import pytest
from helpers import arr, gbsg, lung, naive_km, r_cases, rng_data
from survival_toolkit import kaplan_meier as km
from survival_toolkit.types import Survival, SurvivalError, n_at_risk

KM_CASES = list(r_cases()["km"])


def _survival_for(name: str) -> Survival:
    ref = r_cases()["km"][name]
    df = lung()
    if name.startswith("gbsg"):
        g = gbsg()
        return Survival(g["rfstime"].to_numpy(), g["event"].to_numpy() == 1)
    time, event = df["time"].to_numpy(), df["event"].to_numpy() == 1
    if name == "lung_male":
        mask = df["sex"].to_numpy() == 1
        return Survival(time[mask], event[mask])
    if name == "lung_female":
        mask = df["sex"].to_numpy() == 2
        return Survival(time[mask], event[mask])
    if name == "lung_late_entry":
        return Survival(time, event, arr(ref["entry"]))
    return Survival(time, event)


@pytest.mark.parametrize("name", KM_CASES)
def test_curve_matches_r(name):
    ref = r_cases()["km"][name]
    fit = km.fit(_survival_for(name), ci=ref["conf_type"])
    np.testing.assert_allclose(fit.time, arr(ref["time"]))
    np.testing.assert_array_equal(fit.n_risk, arr(ref["n_risk"]))
    np.testing.assert_array_equal(fit.n_event, arr(ref["n_event"]))
    np.testing.assert_allclose(fit.survival, arr(ref["surv"]), rtol=1e-12)
    # summary(survfit)$std.err is the Greenwood standard error of S(t) itself
    finite = np.isfinite(fit.se)
    np.testing.assert_allclose(fit.se[finite], arr(ref["std_err"])[finite], rtol=1e-10)
    for ours, theirs in ((fit.lower, ref["lower"]), (fit.upper, ref["upper"])):
        want = arr(theirs)
        ok = np.isfinite(want) & np.isfinite(ours)
        assert ok.sum() > 10
        np.testing.assert_allclose(ours[ok], want[ok], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", KM_CASES)
def test_quantiles_and_median_interval_match_r(name):
    ref = r_cases()["km"][name]["quantile"]
    fit = km.fit(_survival_for(name), ci=r_cases()["km"][name]["conf_type"])
    for p, want in zip(ref["p"], ref["value"], strict=True):
        got = fit.quantile(p)
        if want is None:
            assert np.isinf(got)
        else:
            assert got == pytest.approx(want)
    lo, hi = fit.median_ci()
    for got, want in ((lo, ref["lower"][1]), (hi, ref["upper"][1])):
        if want is None:
            assert np.isinf(got)
        else:
            assert got == pytest.approx(want)


@pytest.mark.parametrize("name", [n for n in KM_CASES if "rmst" in r_cases()["km"][n]])
def test_restricted_mean_matches_r(name):
    ref = r_cases()["km"][name]["rmst"]
    mean, se = km.fit(_survival_for(name)).restricted_mean(ref["tau"])
    assert mean == pytest.approx(ref["mean"], rel=1e-10)
    assert se == pytest.approx(ref["se"], rel=1e-8)


def test_six_mp_leukaemia_textbook_values():
    """Freireich et al. 1963, 6-MP arm (Klein & Moeschberger table 1.1): 21 patients, 9 relapses."""
    events = [6, 6, 6, 7, 10, 13, 16, 22, 23]
    censored = [6, 9, 10, 11, 17, 19, 20, 25, 32, 32, 34, 35]
    data = Survival(
        np.array(events + censored, dtype=float),
        np.array([True] * len(events) + [False] * len(censored)),
    )
    fit = km.fit(data)
    np.testing.assert_allclose(
        fit.at([6, 7, 10, 13, 16, 22, 23]),
        [0.857, 0.807, 0.753, 0.690, 0.627, 0.538, 0.448],
        atol=1e-3,
    )
    assert fit.n_risk[:3].tolist() == [21, 17, 15]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("entry", [False, True])
def test_matches_naive_loop_with_ties_and_late_entry(seed, entry):
    data = rng_data(seed, entry=entry, ties=not entry)
    times, surv = naive_km(data)
    fit = km.fit(data)
    np.testing.assert_allclose(fit.time, times)
    np.testing.assert_allclose(fit.survival, surv, rtol=1e-12)


def test_survival_function_is_a_right_continuous_step():
    fit = km.fit(rng_data(1))
    t = fit.time[3]
    assert fit.at(t) == fit.survival[3]
    assert fit.at(t - 1e-9) == fit.survival[2]
    assert fit.at(0.0) == 1.0
    assert fit.at(1e9) == fit.survival[-1]
    lo, hi = fit.ci_at([0.0, t])
    assert lo[0] == hi[0] == 1.0
    assert lo[1] < fit.survival[3] < hi[1]


def test_intervals_nest_and_stay_in_unit_range():
    data = rng_data(3, n=120)
    for method in km.CI_METHODS:
        fit = km.fit(data, ci=method)
        ok = np.isfinite(fit.lower)
        assert np.all(fit.lower[ok] >= 0) and np.all(fit.upper[ok] <= 1)
        assert np.all(fit.lower[ok] <= fit.survival[ok] + 1e-12)
        assert np.all(fit.upper[ok] >= fit.survival[ok] - 1e-12)
    wide, narrow = km.fit(data, alpha=0.01), km.fit(data, alpha=0.2)
    ok = np.isfinite(wide.lower) & (wide.survival < 1)
    assert np.all(wide.lower[ok] <= narrow.lower[ok])


def test_curve_that_reaches_zero_has_no_variance_or_interval_there():
    fit = km.fit(Survival(np.array([1.0, 2.0, 3.0]), np.array([True, True, True])))
    assert fit.survival[-1] == 0.0
    assert np.isnan(fit.se[-1]) and np.isnan(fit.lower[-1])
    assert fit.median() == 2.0
    assert fit.median_ci()[1] <= 3.0


def test_median_on_exact_half_uses_the_midpoint_rule():
    fit = km.fit(
        Survival(np.array([1.0, 2.0, 3.0, 4.0]), np.array([True, True, False, True]))
    )
    # S: 0.75, 0.5, 0.5, 0 ... S hits 0.5 exactly at t=2 and stays until t=4
    assert fit.median() == pytest.approx(3.0)


def test_nelson_aalen_and_fleming_harrington():
    data = rng_data(5)
    fit = km.fit(data)
    h = fit.cumulative_hazard
    np.testing.assert_allclose(h, np.cumsum(fit.n_event / fit.n_risk))
    assert np.all(np.diff(h) > 0)
    assert np.all(fit.fleming_harrington >= fit.survival - 1e-12)
    assert np.all(fit.cumulative_hazard_se > 0)


def test_risk_table_counts():
    data = Survival(np.array([2.0, 4.0, 6.0]), np.array([True, False, True]))
    fit = km.fit(data)
    np.testing.assert_array_equal(fit.table(np.array([0, 2, 3, 6, 7])), [3, 3, 2, 1, 0])
    late = Survival(np.array([5.0, 6.0]), np.array([True, True]), np.array([0.0, 3.0]))
    np.testing.assert_array_equal(n_at_risk(late, np.array([1.0, 3.0, 4.0])), [1, 1, 2])


def test_restricted_mean_is_the_area_under_the_curve():
    data = Survival(np.array([1.0, 2.0, 4.0]), np.array([True, False, True]))
    mean, se = km.fit(data).restricted_mean(3.0)
    # S = 1 on [0,1), 2/3 on [1,3] -> 1 + 2/3 * 2
    assert mean == pytest.approx(1 + 2 / 3 * 2)
    assert se > 0
    with pytest.raises(SurvivalError, match="tau"):
        km.fit(data).restricted_mean(5.0)


@pytest.mark.parametrize(
    ("time", "event", "message"),
    [
        ([1.0, 2.0], [False, False], "no events"),
    ],
)
def test_no_events_is_an_error(time, event, message):
    with pytest.raises(SurvivalError, match=message):
        km.fit(Survival(np.array(time), np.array(event)))


def test_validation_rejects_bad_input():
    with pytest.raises(SurvivalError, match="> 0"):
        Survival(np.array([0.0, 1.0]), np.array([True, True]))
    with pytest.raises(SurvivalError, match="missing or infinite"):
        Survival(np.array([np.nan, 1.0]), np.array([True, True]))
    with pytest.raises(SurvivalError, match="entry < time"):
        Survival(np.array([1.0, 2.0]), np.array([True, True]), np.array([1.0, 0.0]))
    with pytest.raises(SurvivalError, match="same length"):
        Survival(np.array([1.0, 2.0]), np.array([True]))
    with pytest.raises(SurvivalError, match="no observations"):
        Survival(np.array([]), np.array([], dtype=bool))
    with pytest.raises(SurvivalError, match="ci must"):
        km.fit(rng_data(0), ci="wald")  # type: ignore[arg-type]
    with pytest.raises(SurvivalError, match="alpha"):
        km.fit(rng_data(0), alpha=1.5)
