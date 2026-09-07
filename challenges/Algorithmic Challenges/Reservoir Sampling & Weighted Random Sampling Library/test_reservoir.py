"""Tests for reservoir.py: edge cases, determinism, true streaming, statistics."""

from __future__ import annotations

import math
import random

import pytest
from reservoir import (
    _proportion_tolerance,
    a_res,
    reservoir_chao,
    reservoir_l,
    reservoir_r,
    verify_uniform,
    verify_weighted_k1,
    verify_weighted_k_gt_1,
)

UNIFORM_SAMPLERS = [reservoir_r, reservoir_l]
WEIGHTED_SAMPLERS = [a_res, reservoir_chao]


# ---------------------------------------------------------------------------
# A single-pass iterator that raises if the streaming contract is broken.
# ---------------------------------------------------------------------------


class OneShotIterator:
    """Wraps an iterable so calling `len()` on it is a hard failure.

    `iter()` may legitimately be called on an iterator any number of times
    (the protocol requires it to return itself unchanged), so that alone
    isn't a streaming violation -- what a correct implementation must never
    do is find out the length up front or otherwise treat the stream as
    something other than a single forward pass of `__next__` calls. This
    class only exposes `__iter__`/`__next__`, matching a bare generator, so
    any attempt to call `len()` -- the tell that code fell back to
    materializing a list -- raises immediately instead of failing silently.
    """

    def __init__(self, data):
        self._it = iter(data)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def __len__(self):  # pragma: no cover - the point is that this must never run
        raise AssertionError("len() was called on the stream -- not streaming")


# ---------------------------------------------------------------------------
# Edge cases: k = 0, k >= n
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", UNIFORM_SAMPLERS)
def test_k_zero_returns_empty(fn):
    assert fn(range(100), 0) == []


@pytest.mark.parametrize("fn", WEIGHTED_SAMPLERS)
def test_weighted_k_zero_returns_empty(fn):
    assert fn(((i, 1.0) for i in range(100)), 0) == []


@pytest.mark.parametrize("fn", UNIFORM_SAMPLERS)
def test_k_zero_does_not_consume_stream(fn):
    # A correct k=0 fill via islice(it, 0) touches nothing; OneShotIterator
    # would still tolerate a single re-iteration attempt, so check directly
    # that no items were pulled by exhausting the iterator afterwards.
    it = iter(range(5))
    fn(it, 0)
    assert list(it) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("fn", UNIFORM_SAMPLERS)
@pytest.mark.parametrize("n,k", [(5, 5), (3, 10), (0, 0), (0, 5)])
def test_k_at_least_n_returns_full_stream_in_order(fn, n, k):
    assert fn(iter(range(n)), k) == list(range(n))


@pytest.mark.parametrize("fn", WEIGHTED_SAMPLERS)
@pytest.mark.parametrize("n,k", [(5, 5), (3, 10), (0, 5)])
def test_weighted_k_at_least_n_returns_every_item(fn, n, k):
    items = [(i, i + 1.0) for i in range(n)]
    got = fn(iter(items), k)
    assert sorted(got) == list(range(n))


# ---------------------------------------------------------------------------
# Determinism under a fixed seed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", UNIFORM_SAMPLERS)
def test_determinism_same_seed_same_result(fn):
    a = fn(range(10_000), 20, rng=random.Random(7))
    b = fn(range(10_000), 20, rng=random.Random(7))
    assert a == b


@pytest.mark.parametrize("fn", UNIFORM_SAMPLERS)
def test_determinism_different_seed_usually_differs(fn):
    a = fn(range(10_000), 20, rng=random.Random(1))
    b = fn(range(10_000), 20, rng=random.Random(2))
    assert a != b


@pytest.mark.parametrize("fn", WEIGHTED_SAMPLERS)
def test_weighted_determinism_same_seed_same_result(fn):
    items = [(i, (i % 13) + 1.0) for i in range(5000)]
    a = fn(iter(items), 10, rng=random.Random(3))
    b = fn(iter(items), 10, rng=random.Random(3))
    assert a == b


@pytest.mark.parametrize("fn", WEIGHTED_SAMPLERS)
def test_weighted_seed_accepts_plain_int(fn):
    a = fn(((i, 1.0 + i) for i in range(500)), 5, rng=123)
    b = fn(((i, 1.0 + i) for i in range(500)), 5, rng=123)
    assert a == b


# ---------------------------------------------------------------------------
# True streaming: a generator, and an iterator that forbids len()/re-iteration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", UNIFORM_SAMPLERS)
def test_uniform_accepts_a_generator(fn):
    stream = (i for i in range(2000))
    sample = fn(stream, 15, rng=random.Random(0))
    assert len(sample) == 15
    assert all(0 <= x < 2000 for x in sample)
    assert len(set(sample)) == 15  # no duplicate slots


@pytest.mark.parametrize("fn", UNIFORM_SAMPLERS)
def test_uniform_accepts_one_shot_iterator(fn):
    sample = fn(OneShotIterator(range(3000)), 12, rng=random.Random(0))
    assert len(sample) == 12
    assert all(0 <= x < 3000 for x in sample)


@pytest.mark.parametrize("fn", WEIGHTED_SAMPLERS)
def test_weighted_accepts_a_generator(fn):
    stream = ((i, (i % 7) + 1.0) for i in range(2000))
    sample = fn(stream, 10, rng=random.Random(0))
    assert len(sample) == 10
    assert len(set(sample)) == 10


@pytest.mark.parametrize("fn", WEIGHTED_SAMPLERS)
def test_weighted_accepts_one_shot_iterator(fn):
    data = [(i, (i % 7) + 1.0) for i in range(2000)]
    sample = fn(OneShotIterator(data), 10, rng=random.Random(0))
    assert len(sample) == 10


@pytest.mark.parametrize("fn", WEIGHTED_SAMPLERS)
def test_weighted_never_calls_len_on_a_lazy_stream(fn):
    # OneShotIterator.__len__ raises; if it's ever invoked the test fails.
    fn(OneShotIterator([(i, 1.0 + i) for i in range(500)]), 5, rng=random.Random(0))


@pytest.mark.parametrize("fn", UNIFORM_SAMPLERS)
def test_never_calls_len_on_a_lazy_stream(fn):
    # OneShotIterator.__len__ raises; if it's ever invoked the test fails.
    fn(OneShotIterator(range(500)), 5, rng=random.Random(0))


# ---------------------------------------------------------------------------
# Exact statistical checks -- uniform sampling's k/n guarantee holds for any k
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 4, 10])
def test_uniform_selection_probability_converges_to_k_over_n(k):
    result = verify_uniform(n=20, k=k, trials=15_000, seed=42)
    assert result["passed"], result


@pytest.mark.slow
@pytest.mark.parametrize("sampler", WEIGHTED_SAMPLERS)
def test_weighted_k1_selection_probability_converges_to_weight_share(sampler):
    result = verify_weighted_k1(trials=15_000, seed=42, sampler=sampler)
    assert result["passed"], result


@pytest.mark.slow
def test_weighted_chao_k2_monotonicity_holds():
    # Same weak check a_res gets via --verify: no closed-form marginal for
    # k>1, so assert the one thing any correct weighted-without-replacement
    # scheme must have -- higher weight implies >= inclusion frequency.
    # reservoir_chao's own docstring documents a real caveat about items
    # that arrive during the initial k-item fill (they share a group
    # probability rather than an individually-weighted one); these default
    # weights are ascending in stream order, which keeps that caveat from
    # producing a violation here (see verify_weighted_k_gt_1's docstring).
    result = verify_weighted_k_gt_1(k=2, trials=15_000, sampler=reservoir_chao, seed=42)
    assert result["passed"], result


# ---------------------------------------------------------------------------
# k=1 exactness on both uniform and weighted, spelled out directly
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_uniform_k1_is_exactly_uniform_over_items():
    n, trials = 6, 12_000
    counts = [0] * n
    r = random.Random(11)
    for _ in range(trials):
        (winner,) = reservoir_r(range(n), 1, rng=r)
        counts[winner] += 1
    expected = 1 / n
    tol = _proportion_tolerance(expected, trials)
    for c in counts:
        assert abs(c / trials - expected) <= tol


@pytest.mark.slow
@pytest.mark.parametrize("fn", WEIGHTED_SAMPLERS)
def test_weighted_k1_matches_w_over_sum_w_exactly(fn):
    weights = [1.0, 5.0, 10.0]
    total = sum(weights)
    trials = 12_000
    counts = [0] * len(weights)
    r = random.Random(11)
    for _ in range(trials):
        (winner,) = fn(list(enumerate(weights)), 1, rng=r)
        counts[winner] += 1
    for w, c in zip(weights, counts):
        expected = w / total
        tol = _proportion_tolerance(expected, trials)
        assert abs(c / trials - expected) <= tol


@pytest.mark.slow
def test_reservoir_chao_with_unit_weights_matches_reservoir_r():
    # reservoir_chao's docstring claims k*w_i/W reduces to Algorithm R's k/i
    # rule when every weight is 1 -- so with unit weights it should be
    # statistically indistinguishable from the uniform k/n guarantee that
    # verify_uniform checks for reservoir_r and reservoir_l.
    n, k, trials = 25, 6, 15_000
    expected = k / n
    tol = _proportion_tolerance(expected, trials)
    counts = [0] * n
    r = random.Random(13)
    for _ in range(trials):
        items = [(i, 1.0) for i in range(n)]
        for winner in reservoir_chao(items, k, rng=r):
            counts[winner] += 1
    proportions = [c / trials for c in counts]
    assert max(abs(p - expected) for p in proportions) <= tol


# ---------------------------------------------------------------------------
# Sanity: reservoir_r and reservoir_l agree in distribution shape (both
# uniform), even though their random-number-consumption patterns differ.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_reservoir_r_and_l_agree_within_each_others_tolerance():
    result = verify_uniform(n=25, k=6, trials=15_000, seed=99)
    assert result["reservoir_r"]["passed"]
    assert result["reservoir_l"]["passed"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", UNIFORM_SAMPLERS)
def test_negative_k_raises(fn):
    with pytest.raises(ValueError):
        fn(range(10), -1)


@pytest.mark.parametrize("fn", WEIGHTED_SAMPLERS)
def test_weighted_negative_k_raises(fn):
    with pytest.raises(ValueError):
        fn([(1, 1.0)], -1)


@pytest.mark.parametrize("fn", WEIGHTED_SAMPLERS)
def test_weighted_nonpositive_weight_raises(fn):
    with pytest.raises(ValueError):
        fn([(1, 0.0), (2, 1.0)], 1)
    with pytest.raises(ValueError):
        fn([(1, -3.0)], 1)


@pytest.mark.parametrize("fn", WEIGHTED_SAMPLERS)
def test_weighted_no_duplicate_items_in_result(fn):
    items = [(i, 1.0 + i) for i in range(200)]
    sample = fn(iter(items), 20, rng=random.Random(5))
    assert len(sample) == len(set(sample)) == 20


@pytest.mark.parametrize("fn", UNIFORM_SAMPLERS)
def test_uniform_no_duplicate_slots_across_many_seeds(fn):
    for seed in range(20):
        sample = fn(range(500), 30, rng=random.Random(seed))
        assert len(sample) == len(set(sample)) == 30


def test_tolerance_helper_shrinks_with_more_trials():
    p = 0.3
    assert _proportion_tolerance(p, 1000) > _proportion_tolerance(p, 100_000)


def test_tolerance_helper_matches_binomial_formula():
    p, trials, sigmas = 0.25, 4000, 3.0
    expected = sigmas * math.sqrt(p * (1 - p) / trials)
    assert _proportion_tolerance(p, trials, sigmas) == pytest.approx(expected)
