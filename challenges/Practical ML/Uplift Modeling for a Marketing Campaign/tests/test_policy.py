import numpy as np
import pytest
from uplift import metrics, policy

S = np.array([6, 5, 4, 3, 2, 1.0])
T = np.array([1, 0, 1, 0, 1, 0])
Y = np.array([1, 0, 1, 0, 0, 1])


def hand():
    return metrics.rank(S, T, Y)


def test_contacting_everyone_gives_the_ate():
    assert policy.incremental_at(hand(), 1.0) == pytest.approx(1 / 3)


def test_incremental_is_interpolated_between_cutoffs():
    assert policy.incremental_at(hand(), 0.5) == pytest.approx(1 / 2)
    assert policy.incremental_at(hand(), 0.0) == 0.0


def test_budget_table_scales_with_the_population():
    rows = policy.budget_table(hand(), [0.5, 1.0], n_customers=1000)
    assert [r["share"] for r in rows] == [0.5, 1.0]
    assert rows[1]["contacts"] == 1000
    assert rows[1]["incremental"] == pytest.approx(1000 / 3)
    assert rows[0]["per_contact"] == pytest.approx(rows[0]["incremental"] / 500)


def test_profit_and_break_even_cost_are_consistent():
    r = hand()
    be = policy.break_even_cost(r, 0.5, value=10.0)
    assert be == pytest.approx(10.0 * 0.5 / 0.5)
    assert policy.profit(r, 0.5, 10.0, be, 1000) == pytest.approx(0.0, abs=1e-9)
    assert policy.profit(r, 0.5, 10.0, be * 0.5, 1000) > 0


def test_best_fraction_never_grows_as_contact_cost_rises(rct):
    _, t, y, tau = rct
    r = metrics.rank(tau, t, y)
    costs = (0.0, 0.01, 0.03, 0.06, 1.0)
    fracs = [policy.best_fraction(r, 1.0, c, len(y))[0] for c in costs]
    assert fracs == sorted(fracs, reverse=True)


def test_free_contact_targets_the_peak_of_the_uplift_curve(rct):
    _, t, y, tau = rct
    r = metrics.rank(tau, t, y)
    frac, _ = policy.best_fraction(r, 1.0, 0.0, len(y))
    x, _, u = metrics.curve(r)
    assert np.interp(frac, x, u) == pytest.approx(u.max(), rel=0.02)


def test_prohibitive_cost_means_contact_nobody(rct):
    _, t, y, tau = rct
    frac, prof = policy.best_fraction(metrics.rank(tau, t, y), 1.0, 10.0, len(y))
    assert frac == 0.0 and prof == 0.0
