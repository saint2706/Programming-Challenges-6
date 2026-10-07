import numpy as np
import pytest
import synth


def features(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 12)).astype(np.float32), (rng.random(n) < 0.85).astype(
        np.int8
    )


def test_none_scenario_has_exactly_zero_effect():
    X, t = features()
    _, tau = synth.simulate(X, t, seed=0, scenario="none")
    assert np.all(tau == 0)


def test_constant_scenario_helps_everyone():
    X, t = features()
    _, tau = synth.simulate(X, t, seed=0, scenario="constant")
    assert (tau > 0).all()


def test_heterogeneous_scenario_has_winners_and_sleeping_dogs():
    X, t = features()
    _, tau = synth.simulate(X, t, seed=0, scenario="heterogeneous")
    assert (tau > 0).mean() > 0.1
    assert (tau < 0).mean() > 0.1


def test_base_rate_is_respected():
    X, t = features(20000)
    y, _ = synth.simulate(X, t, seed=0, scenario="none", base_rate=0.05)
    assert 0.03 < y.mean() < 0.09


def test_observed_difference_in_means_matches_mean_tau():
    X, t = features(200_000)
    y, tau = synth.simulate(X, t, seed=3, scenario="heterogeneous", strength=1.5)
    observed = y[t == 1].mean() - y[t == 0].mean()
    assert observed == pytest.approx(tau.mean(), abs=0.006)


def test_deterministic_given_seed_and_different_across_seeds():
    X, t = features()
    a = synth.simulate(X, t, seed=1)
    b = synth.simulate(X, t, seed=1)
    c = synth.simulate(X, t, seed=2)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_unknown_scenario_raises():
    X, t = features(100)
    with pytest.raises(ValueError, match="scenario"):
        synth.simulate(X, t, seed=0, scenario="nope")
