"""Each model against a hand calculation, a slow reference, or an independent brute-force computation."""

from itertools import permutations

import numpy as np
import pytest
from attribution import logistic, markov, rules, shapley
from attribution.journeys import AttributionError, from_paths, resample_weights
from helpers import random_journeys, reference_credit, toy


def test_the_rules_on_one_journey_by_hand():
    j = from_paths([["a", "b", "c", "a"]], [1])
    names = j.channels  # a, b, c
    first = rules.credit(j, "first_touch")
    assert dict(zip(names, first, strict=True)) == {"a": 1, "b": 0, "c": 0}
    assert dict(zip(names, rules.credit(j, "last_touch"), strict=True)) == {
        "a": 1,
        "b": 0,
        "c": 0,
    }
    assert dict(zip(names, rules.credit(j, "linear"), strict=True)) == pytest.approx(
        {"a": 0.5, "b": 0.25, "c": 0.25}
    )
    pos = dict(zip(names, rules.credit(j, "position_based"), strict=True))
    assert pos == pytest.approx(
        {"a": 0.8, "b": 0.1, "c": 0.1}
    )  # 40 + 40 on a, the 20 split over b and c
    # time decay: touches at steps 0..3, conversion at step 4, half-life 1: weights 1/16, 1/8, 1/4, 1/2
    decay = dict(zip(names, rules.credit(j, "time_decay", half_life=1.0), strict=True))
    total = 1 / 16 + 1 / 8 + 1 / 4 + 1 / 2
    assert decay == pytest.approx(
        {"a": (1 / 16 + 1 / 2) / total, "b": (1 / 8) / total, "c": (1 / 4) / total}
    )


@pytest.mark.parametrize("rule", rules.RULES)
@pytest.mark.parametrize("seed", range(4))
def test_rules_match_the_slow_reference(rule, seed):
    j = random_journeys(seed)
    j.clicked[:] = np.random.default_rng(seed).random(j.clicked.size) < 0.3
    j.value[:] = np.random.default_rng(seed + 9).uniform(1, 5, j.n_converted)
    np.testing.assert_allclose(
        rules.credit(j, rule, half_life=2.0), reference_credit(j, rule), rtol=1e-12
    )


@pytest.mark.parametrize("rule", rules.RULES)
def test_every_rule_hands_out_exactly_the_conversions(rule):
    j = random_journeys(1)
    assert rules.credit(j, rule, half_life=3.0).sum() == pytest.approx(j.total_value)
    frac = rules.touch_fractions(j, rule, half_life=3.0)
    np.testing.assert_allclose(np.add.reduceat(frac, j.offsets[:-1]), 1.0)


def test_last_click_falls_back_to_the_last_touch():
    j = from_paths([["a", "b", "c"]], [1])
    assert rules.credit(j, "last_click").tolist() == [0, 0, 1]
    j.clicked[:] = [True, False, False]
    assert rules.credit(j, "last_click").tolist() == [1, 0, 0]


def test_a_rule_does_not_care_about_the_other_journeys():
    j = random_journeys(2)
    w = np.zeros(j.n_converted)
    w[0] = 3
    assert rules.credit(j, "linear", w).sum() == pytest.approx(3.0)


def test_rule_errors():
    j = toy()
    with pytest.raises(AttributionError, match="half_life"):
        rules.credit(j, "time_decay")
    with pytest.raises(AttributionError, match="unknown rule"):
        rules.credit(j, "middle_touch")


# ---- Markov chain --------------------------------------------------------------------------------
def walk_conversion_rate(
    counts: np.ndarray, removed: int | None, n: int, seed: int
) -> float:
    """Monte Carlo of the same chain: the fraction of walks from START that end in conversion."""
    k = counts.shape[0] - 1
    c = counts.copy()
    if removed is not None:
        c[removed] = 0
        c[removed, k + 1] = 1
    p = c / c.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n):
        s = k
        while True:
            s = rng.choice(k + 2, p=p[s])
            if s == k:
                wins += 1
                break
            if s == k + 1:
                break
    return wins / n


def test_markov_conversion_probability_by_hand_and_by_simulation():
    j = toy()
    counts = markov.transition_counts(j)
    # START -> A 6/8, B 2/8. A: ->B 2/6? no: A has 2 (to B) + 1 (conv) + 3 (lost) = 6 exits. B: 2 conv + 2 lost + 2 (from A) ...
    assert markov.conversion_probability(counts) == pytest.approx(0.375)
    for removed in (None, 0, 1):
        sim = walk_conversion_rate(counts, removed, 6000, seed=1)
        assert markov.conversion_probability(counts, removed) == pytest.approx(
            sim, abs=0.02
        )


def test_markov_removal_effect_closed_form():
    # one channel path only: removing it removes every conversion, so its effect is 1 and it gets everything
    j = from_paths([["a"]], [4], [6])
    assert markov.credit(j).tolist() == pytest.approx([4.0])
    # a channel nobody converts through still has an effect through the paths it sits on
    j = from_paths([["a", "b"], ["b"]], [5, 0], [0, 5])
    credit = markov.credit(j)
    assert credit.sum() == pytest.approx(5.0) and credit[0] > 0


def test_markov_credit_conserves_conversions_and_ignores_unused_channels():
    j = random_journeys(3)
    assert markov.credit(j).sum() == pytest.approx(j.total_value)
    j2 = from_paths([["a", "b"], ["a"]], [3, 2], [4, 4])
    extra = from_paths([["a", "b"], ["a"], ["z"]], [3, 2, 0], [4, 4, 0])
    np.testing.assert_allclose(markov.credit(extra)[:2], markov.credit(j2), rtol=1e-9)


def test_transition_counts_add_up():
    j = random_journeys(4)
    c = markov.transition_counts(j)
    k = j.k
    assert c[k, :k].sum() == pytest.approx(
        j.n_converted + j.null_count
    )  # every journey starts once
    assert c[:, k].sum() == pytest.approx(j.n_converted)  # and each conversion ends one
    assert c[:k, k + 1].sum() == pytest.approx(j.null_count)


# ---- logistic and Shapley ------------------------------------------------------------------------
def test_logistic_fit_is_a_stationary_point_and_recovers_coefficients():
    rng = np.random.default_rng(0)
    masks = np.arange(1, 16)
    bits = ((masks[:, None] >> np.arange(4)) & 1).astype(float)
    truth = np.array([-2.0, 0.8, 0.0, -0.4, 1.2])
    total = np.full(masks.size, 40000.0)
    p = 1 / (1 + np.exp(-(truth[0] + bits @ truth[1:])))
    conv = rng.binomial(total.astype(int), p).astype(float)
    x = np.hstack([np.ones((masks.size, 1)), bits])
    beta = logistic.fit(x, conv, total, ridge=0.0)
    np.testing.assert_allclose(beta, truth, atol=0.05)
    grad = x.T @ (conv - total / (1 + np.exp(-(x @ beta))))
    assert np.abs(grad).max() < 1e-3


def test_shapley_matches_brute_force_over_all_orders():
    rng = np.random.default_rng(1)
    n = 4
    v = rng.uniform(0, 1, 1 << n)
    v[0] = 0.0
    fast = shapley.shapley_values(v, n)
    slow = np.zeros(n)
    orders = list(permutations(range(n)))
    for order in orders:
        s = 0
        for c in order:
            slow[c] += v[s | 1 << c] - v[s]
            s |= 1 << c
    np.testing.assert_allclose(fast, slow / len(orders), atol=1e-12)
    assert fast.sum() == pytest.approx(v[-1] - v[0])  # efficiency


def test_shapley_game_properties_on_a_fitted_model():
    j = random_journeys(5, n=500, k=4)
    model = logistic.fit_presence(j)
    v = shapley.coalition_values(model)
    assert v[(1 << j.k) - 1] == pytest.approx(
        float(model.total @ model.probability(model.bits))
    )
    phi = shapley.shapley_values(v, j.k)
    assert phi.sum() == pytest.approx(v[-1] - v[0])
    assert shapley.credit(j).sum() == pytest.approx(j.total_value)


def test_a_channel_the_model_gives_no_weight_gets_no_credit():
    j = from_paths([["a"], ["b"], ["a", "b"], ["c"]], [10, 0, 10, 10], [90, 90, 80, 90])
    model = logistic.fit_presence(j)
    assert (
        model.beta[1:].min() < 0.2 and model.beta[1] > 1
    )  # channel a lifts conversion; b does not
    credit = shapley.credit(j)
    assert credit[0] > credit[1] and credit.sum() == pytest.approx(j.total_value)


def test_model_size_limits():
    big = from_paths([[f"c{i}"] for i in range(13)], [1] * 13, [1] * 13)
    with pytest.raises(AttributionError, match="at most 12"):
        shapley.credit(big)


def test_bootstrap_weights_resample_the_converted_journeys():
    j = random_journeys(6)
    w = resample_weights(j, np.random.default_rng(0))
    assert w.sum() == j.n_converted and w.min() >= 0 and (w == 0).any()
