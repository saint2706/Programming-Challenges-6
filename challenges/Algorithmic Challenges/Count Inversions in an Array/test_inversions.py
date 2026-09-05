"""Tests for inversion counting.

Run with:  uv run --with pytest --with numpy pytest -q
Skip the slow ones with:  -m "not slow"
"""

from __future__ import annotations

import itertools
import math
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from inversions import (
    METHODS,
    Fenwick,
    bubble_sort_swaps,
    count_brute,
    count_fenwick,
    count_greater_to_left,
    count_insort,
    count_inversions,
    count_mergesort,
    count_numpy,
    count_numpy_radix,
    count_significant_inversions,
    count_smaller_to_right,
    from_inversion_table,
    inversion_polynomial,
    inversion_table,
    kendall_tau_b,
    kendall_tau_distance,
    main,
    max_inversions,
    verify,
)

HERE = Path(__file__).parent
np = pytest.importorskip("numpy", reason="the vectorised method needs numpy")

IMPLS = [count_brute, count_insort, count_mergesort, count_fenwick, count_numpy,
         count_numpy_radix]
FAST_IMPLS = IMPLS[1:]
VECTORISED = [count_numpy, count_numpy_radix]


# ---------------------------------------------------------------------------
# Every method agrees with the definition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", FAST_IMPLS)
def test_random_arrays_match_brute_force(impl):
    rng = random.Random(2024)
    for _ in range(400):
        n = rng.randint(0, 70)
        # A small value spread forces ties; a large one forces distinctness.
        spread = rng.choice([1, 2, 3, 10, 10**6])
        a = [rng.randrange(spread) for _ in range(n)]
        assert impl(a) == count_brute(a), a


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("n", range(0, 8))
def test_exhaustive_over_all_permutations(impl, n):
    """For n <= 7 there is no need to sample: check every permutation."""
    for perm in itertools.permutations(range(n)):
        expected = sum(1 for i in range(n) for j in range(i + 1, n) if perm[j] < perm[i])
        assert impl(list(perm)) == expected


@pytest.mark.parametrize("impl", IMPLS)
def test_exhaustive_over_all_binary_arrays(impl):
    """Length-12 0/1 arrays: the maximum-ties case, where off-by-ones live."""
    for bits in itertools.product((0, 1), repeat=12):
        a = list(bits)
        # inversions of a 0/1 array = sum over each 1 of the zeros after it
        expected = 0
        zeros_after = a.count(0)
        for x in a:
            if x == 0:
                zeros_after -= 1
            else:
                expected += zeros_after
        assert impl(a) == expected


def test_verify_helper_passes():
    assert verify(trials=60, verbose=False)


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("a", [[], [1], [1, 1], [1, 2], [2, 1], [0] * 50])
def test_small_and_degenerate(impl, a):
    assert impl(a) == count_brute(a)


@pytest.mark.parametrize("impl", IMPLS)
def test_all_equal_has_no_inversions(impl):
    """Inversions are strict: a[i] > a[j], never a[i] >= a[j]."""
    assert impl([7] * 200) == 0


@pytest.mark.parametrize("impl", IMPLS)
def test_sorted_has_none_and_reversed_has_all(impl):
    n = 200
    assert impl(list(range(n))) == 0
    assert impl(list(range(n))[::-1]) == max_inversions(n)


@pytest.mark.parametrize("method", [m for m in METHODS if m != "brute"])
def test_lengths_around_powers_of_two(method):
    """The vectorised path pads to a power of two; check both sides of each."""
    rng = random.Random(5)
    for n in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]:
        a = [rng.randrange(50) for _ in range(n)]
        assert count_inversions(a, method=method) == count_brute(a), (method, n)


def test_generators_are_accepted():
    assert count_inversions(x for x in [3, 2, 1]) == 3


def test_tuples_and_other_sequences():
    assert count_inversions((3, 2, 1)) == 3
    assert count_inversions(range(10, 0, -1)) == 45


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="method must be one of"):
        count_inversions([1], method="telepathy")


def test_max_inversions_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        max_inversions(-1)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 10])
def test_max_inversions_is_achieved_by_the_reversed_array(n):
    assert count_brute(list(range(n))[::-1]) == max_inversions(n)


# ---------------------------------------------------------------------------
# Types other than small ints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", FAST_IMPLS)
def test_strings(impl):
    a = ["pear", "apple", "fig", "banana"]
    assert impl(a) == count_brute(a) == 4


@pytest.mark.parametrize("impl", FAST_IMPLS)
def test_tuples_compare_lexicographically(impl):
    a = [(2, 1), (1, 9), (2, 0), (1, 9)]
    assert impl(a) == count_brute(a)


@pytest.mark.parametrize("impl", FAST_IMPLS)
def test_unhashable_elements(impl):
    """Rank compression is sort-based, so lists work despite being unhashable."""
    a = [[3], [1, 2], [3], [0]]
    assert impl(a) == count_brute(a)


@pytest.mark.parametrize("impl", FAST_IMPLS)
def test_fractions_and_mixed_numeric_types(impl):
    a = [Fraction(3, 2), 1, 2.5, Fraction(1, 3), 2]
    assert impl(a) == count_brute(a)


@pytest.mark.parametrize("impl", FAST_IMPLS)
def test_huge_integers(impl):
    a = [10**400, 10**200, 10**600, 0]
    assert impl(a) == count_brute(a) == 4


def test_mixed_incomparable_types_raise():
    with pytest.raises(TypeError):
        count_inversions([1, "two", 3], method="mergesort")


# ---------------------------------------------------------------------------
# NaN: the input that has no answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", IMPLS)
def test_nan_is_rejected_with_an_explanation(impl):
    with pytest.raises(ValueError, match="NaN"):
        impl([1.0, math.nan, 2.0])


@pytest.mark.parametrize("impl", IMPLS)
def test_nan_check_can_be_disabled(impl):
    """Opting out is allowed; the answer is then algorithm-dependent, as documented."""
    impl([1.0, math.nan, 2.0], validate=False)  # must not raise


def test_infinities_are_fine():
    a = [math.inf, 1.0, -math.inf, 0.0]
    assert count_inversions(a) == count_brute(a) == 5


def test_negative_zero_ties_with_zero():
    """-0.0 == 0.0, so the pair is tied, not inverted, in both directions."""
    assert count_inversions([0.0, -0.0]) == 0
    assert count_inversions([-0.0, 0.0]) == 0


# ---------------------------------------------------------------------------
# key= and reverse=
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", FAST_IMPLS)
def test_key_is_applied_before_comparing(impl):
    words = ["bbb", "a", "cc"]
    assert impl(words, key=len) == 2          # lengths 3, 1, 2
    assert impl(words) == count_brute(words) == 1  # only "bbb" > "a" alphabetically


def test_key_is_called_once_per_element():
    calls = []

    def key(x):
        calls.append(x)
        return x

    count_mergesort([3, 1, 2], key=key)
    assert sorted(calls) == [1, 2, 3]


@pytest.mark.parametrize("impl", FAST_IMPLS)
def test_reverse_counts_the_complementary_pairs(impl):
    """inv(a) + inv_desc(a) = C(n,2) - (tied pairs), for every array."""
    rng = random.Random(8)
    for _ in range(100):
        a = [rng.randrange(4) for _ in range(rng.randint(0, 20))]
        ties = sum(1 for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] == a[j])
        assert impl(a) + impl(a, reverse=True) == max_inversions(len(a)) - ties


def test_reverse_on_a_descending_array_is_zero():
    assert count_inversions([5, 4, 3, 2, 1], reverse=True) == 0
    assert count_inversions([1, 2, 3, 4, 5], reverse=True) == 10


def test_key_and_reverse_compose():
    """Lengths 1, 3, 2: two ascending pairs, which is what reverse counts."""
    words = ["a", "bbb", "cc"]
    assert count_inversions(words, key=len, reverse=True) == 2
    assert count_inversions(words, key=len) == 1


# ---------------------------------------------------------------------------
# Per-element counts
# ---------------------------------------------------------------------------


def test_count_smaller_to_right_matches_the_definition():
    rng = random.Random(12)
    for _ in range(200):
        a = [rng.randrange(6) for _ in range(rng.randint(0, 25))]
        expected = [sum(1 for j in range(i + 1, len(a)) if a[j] < a[i]) for i in range(len(a))]
        assert count_smaller_to_right(a) == expected


def test_count_greater_to_left_matches_the_definition():
    rng = random.Random(13)
    for _ in range(200):
        a = [rng.randrange(6) for _ in range(rng.randint(0, 25))]
        expected = [sum(1 for i in range(j) if a[i] > a[j]) for j in range(len(a))]
        assert count_greater_to_left(a) == expected


def test_both_per_element_views_sum_to_the_total():
    """The same pairs, attributed to the earlier vs the later index."""
    rng = random.Random(14)
    for _ in range(200):
        a = [rng.randrange(8) for _ in range(rng.randint(0, 30))]
        total = count_brute(a)
        assert sum(count_smaller_to_right(a)) == total
        assert sum(count_greater_to_left(a)) == total


def test_per_element_views_on_empty_input():
    assert count_smaller_to_right([]) == []
    assert count_greater_to_left([]) == []


def test_leetcode_count_of_smaller_after_self():
    assert count_smaller_to_right([5, 2, 6, 1]) == [2, 1, 1, 0]
    assert count_smaller_to_right([-1, -1]) == [0, 0]


# ---------------------------------------------------------------------------
# Fenwick tree in isolation
# ---------------------------------------------------------------------------


def test_fenwick_matches_a_plain_list():
    rng = random.Random(16)
    n = 64
    tree = Fenwick(n)
    plain = [0] * n
    for _ in range(500):
        i = rng.randrange(n)
        d = rng.randint(-5, 5)
        tree.add(i, d)
        plain[i] += d
        c = rng.randrange(n + 1)
        assert tree.prefix(c) == sum(plain[:c])
    assert tree.total() == sum(plain)


def test_fenwick_from_counts_matches_repeated_add():
    counts = [3, 0, 1, 4, 1, 5, 9, 2, 6]
    built = Fenwick.from_counts(counts)
    added = Fenwick(len(counts))
    for i, c in enumerate(counts):
        added.add(i, c)
    assert [built.prefix(k) for k in range(len(counts) + 1)] == \
           [added.prefix(k) for k in range(len(counts) + 1)]


def test_fenwick_bounds():
    tree = Fenwick(4)
    with pytest.raises(IndexError):
        tree.add(4)
    with pytest.raises(IndexError):
        tree.add(-1)
    with pytest.raises(IndexError):
        tree.prefix(-1)
    assert tree.prefix(99) == 0  # clamped, not an error: "everything" is well defined
    with pytest.raises(ValueError, match="non-negative"):
        Fenwick(-1)


def test_fenwick_of_size_zero():
    tree = Fenwick(0)
    assert len(tree) == 0
    assert tree.prefix(0) == tree.total() == 0


# ---------------------------------------------------------------------------
# Mathematical identities -- the reason the number means anything
# ---------------------------------------------------------------------------


def test_inversions_equal_bubble_sort_swaps():
    """The count *is* the minimum number of adjacent transpositions that sort."""
    rng = random.Random(18)
    for _ in range(300):
        a = [rng.randrange(6) for _ in range(rng.randint(0, 20))]
        assert bubble_sort_swaps(a) == count_brute(a)


def test_inversion_polynomial_matches_exhaustive_enumeration():
    """sum over S_n of q^inv = [n]_q!, checked against every permutation."""
    for n in range(0, 8):
        counts = [0] * (max_inversions(n) + 1)
        for perm in itertools.permutations(range(n)):
            counts[count_brute(list(perm))] += 1
        assert counts == inversion_polynomial(n), n


def test_inversion_polynomial_sums_to_n_factorial():
    for n in range(0, 10):
        assert sum(inversion_polynomial(n)) == math.factorial(n)


def test_inversion_polynomial_is_palindromic():
    """inv(pi) + inv(reverse of pi) = C(n,2), so the coefficients mirror."""
    for n in range(0, 10):
        poly = inversion_polynomial(n)
        assert poly == poly[::-1]


def test_inversion_polynomial_is_the_mahonian_triangle():
    """OEIS A008302, rows 1..6, quoted independently of this implementation."""
    assert inversion_polynomial(1) == [1]
    assert inversion_polynomial(2) == [1, 1]
    assert inversion_polynomial(3) == [1, 2, 2, 1]
    assert inversion_polynomial(4) == [1, 3, 5, 6, 5, 3, 1]
    assert inversion_polynomial(5) == [1, 4, 9, 15, 20, 22, 20, 15, 9, 4, 1]
    assert inversion_polynomial(6) == [
        1, 5, 14, 29, 49, 71, 90, 101, 101, 90, 71, 49, 29, 14, 5, 1
    ]


def test_inversion_polynomial_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        inversion_polynomial(-1)


def test_mean_and_variance_of_a_random_permutation():
    """E[inv] = n(n-1)/4 and Var = n(n-1)(2n+5)/72, checked exactly over S_7."""
    n = 7
    counts = inversion_polynomial(n)
    total = math.factorial(n)
    mean = sum(k * c for k, c in enumerate(counts)) / total
    second = sum(k * k * c for k, c in enumerate(counts)) / total
    assert mean == pytest.approx(n * (n - 1) / 4)
    assert second - mean**2 == pytest.approx(n * (n - 1) * (2 * n + 5) / 72)


def test_lehmer_code_is_a_bijection():
    """Every permutation has one table, every valid table one permutation."""
    for n in range(0, 7):
        tables = set()
        for perm in itertools.permutations(range(n)):
            table = inversion_table(list(perm))
            assert all(0 <= table[v] <= n - 1 - v for v in range(n))
            assert from_inversion_table(table) == list(perm)
            tables.add(tuple(table))
        assert len(tables) == math.factorial(n)


def test_lehmer_code_entries_sum_to_the_inversion_count():
    rng = random.Random(20)
    for _ in range(200):
        perm = list(range(rng.randint(0, 15)))
        rng.shuffle(perm)
        assert sum(inversion_table(perm)) == count_brute(perm)


def test_from_inversion_table_rejects_impossible_tables():
    with pytest.raises(ValueError, match=r"outside the allowed range"):
        from_inversion_table([0, 5])
    with pytest.raises(ValueError, match=r"outside the allowed range"):
        from_inversion_table([0, -1])
    # b[1] may be at most n-1-1 = 0 for n = 2, so 1 is out of range.
    with pytest.raises(ValueError, match=r"outside the allowed range"):
        from_inversion_table([0, 1])


def test_from_inversion_table_on_the_extremes():
    assert from_inversion_table([0, 0, 0, 0]) == [0, 1, 2, 3]
    assert from_inversion_table([3, 2, 1, 0]) == [3, 2, 1, 0]
    assert from_inversion_table([]) == []


def test_inversion_table_is_indexed_by_value_not_position():
    """b[v] counts values above v that precede it, so b[v] <= n-1-v."""
    perm = [2, 0, 3, 1]
    assert inversion_table(perm) == [1, 2, 0, 0]
    assert count_greater_to_left(perm) == [0, 1, 0, 2]  # the position-indexed cousin
    assert sum(inversion_table(perm)) == sum(count_greater_to_left(perm)) == count_brute(perm)


def test_inversion_table_rejects_non_permutations():
    with pytest.raises(ValueError, match="permutation of 0"):
        inversion_table([1, 1])
    with pytest.raises(ValueError, match="permutation of 0"):
        inversion_table([5, 2])


def test_inversions_are_invariant_under_order_preserving_relabelling():
    """Only the relative order matters, so any strictly increasing map is free."""
    rng = random.Random(22)
    for _ in range(100):
        a = [rng.randrange(20) for _ in range(rng.randint(0, 25))]
        relabelled = [x**3 - 7 for x in a]  # strictly increasing on the integers
        assert count_inversions(a) == count_inversions(relabelled)


def test_concatenation_inequality():
    """inv(x + y) >= inv(x) + inv(y), with the cross pairs making up the rest."""
    rng = random.Random(24)
    for _ in range(100):
        x = [rng.randrange(10) for _ in range(rng.randint(0, 12))]
        y = [rng.randrange(10) for _ in range(rng.randint(0, 12))]
        cross = sum(1 for a in x for b in y if a > b)
        assert count_inversions(x + y) == count_inversions(x) + count_inversions(y) + cross


# ---------------------------------------------------------------------------
# Significant inversions
# ---------------------------------------------------------------------------


def test_significant_inversions_match_brute_force():
    rng = random.Random(26)
    for factor in (0.0, 0.5, 1.0, 2.0, 3.0):
        for _ in range(60):
            a = [rng.randrange(-20, 20) for _ in range(rng.randint(0, 30))]
            expected = sum(
                1 for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] > factor * a[j]
            )
            assert count_significant_inversions(a, factor) == expected, (factor, a)


def test_significant_with_factor_one_is_the_ordinary_count():
    rng = random.Random(28)
    for _ in range(100):
        a = [rng.randrange(15) for _ in range(rng.randint(0, 25))]
        assert count_significant_inversions(a, 1.0) == count_brute(a)


def test_significant_handles_negative_factors():
    """The bisect-based counting step stays correct where a two-pointer would not."""
    a = [-5, 3, -1, 4, -2]
    for factor in (-1.0, -2.5):
        expected = sum(
            1 for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] > factor * a[j]
        )
        assert count_significant_inversions(a, factor) == expected


def test_significant_rejects_non_numbers():
    with pytest.raises(TypeError, match="needs numbers"):
        count_significant_inversions(["a", "b"])


def test_significant_on_degenerate_input():
    assert count_significant_inversions([], 2) == 0
    assert count_significant_inversions([1], 2) == 0


def test_significant_does_not_mutate_its_input():
    a = [3, 1, 2]
    count_significant_inversions(a, 2)
    assert a == [3, 1, 2]


# ---------------------------------------------------------------------------
# Kendall tau
# ---------------------------------------------------------------------------


def test_tau_distance_is_the_inversion_count_of_the_relabelling():
    rng = random.Random(30)
    for _ in range(200):
        n = rng.randint(0, 12)
        a = list(range(n))
        rng.shuffle(a)
        b = list(range(n))
        rng.shuffle(b)
        pos = {x: i for i, x in enumerate(b)}
        expected = sum(
            1 for i in range(n) for j in range(i + 1, n) if pos[a[i]] > pos[a[j]]
        )
        assert kendall_tau_distance(a, b) == expected


def test_tau_distance_extremes():
    a = [1, 2, 3, 4]
    assert kendall_tau_distance(a, a) == 0
    assert kendall_tau_distance(a, a[::-1]) == 6
    assert kendall_tau_distance(a, a, normalize=True) == 0.0
    assert kendall_tau_distance(a, a[::-1], normalize=True) == 1.0


def test_tau_distance_is_a_metric():
    """Symmetric, zero only on equality, and obeys the triangle inequality."""
    rng = random.Random(32)
    perms = []
    for _ in range(12):
        p = list(range(6))
        rng.shuffle(p)
        perms.append(p)
    for a in perms:
        assert kendall_tau_distance(a, a) == 0
        for b in perms:
            assert kendall_tau_distance(a, b) == kendall_tau_distance(b, a)
            assert (kendall_tau_distance(a, b) == 0) == (a == b)
            for c in perms:
                assert (kendall_tau_distance(a, c)
                        <= kendall_tau_distance(a, b) + kendall_tau_distance(b, c))


def test_tau_distance_on_empty_and_singleton():
    assert kendall_tau_distance([], []) == 0
    assert kendall_tau_distance([], [], normalize=True) == 0.0
    assert kendall_tau_distance([5], [5]) == 0


def test_tau_distance_rejects_mismatched_input():
    with pytest.raises(ValueError, match="same length"):
        kendall_tau_distance([1, 2], [1])
    with pytest.raises(ValueError, match="distinct elements"):
        kendall_tau_distance([1, 1], [1, 1])
    with pytest.raises(ValueError, match="but not the second"):
        kendall_tau_distance([1, 2], [1, 3])


def brute_tau_b(x, y):
    """tau-b straight from its definition, in O(n^2)."""
    n = len(x)
    con_minus_dis = 0
    n0 = n * (n - 1) // 2
    n1 = n2 = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0:
                n1 += 1
            if dy == 0:
                n2 += 1
            con_minus_dis += (dx > 0) - (dx < 0) if False else 0
            s = (1 if dx > 0 else -1 if dx < 0 else 0) * (1 if dy > 0 else -1 if dy < 0 else 0)
            con_minus_dis += s
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    return con_minus_dis / denom if denom else math.nan


def test_tau_b_matches_its_own_definition_with_ties():
    rng = random.Random(34)
    for _ in range(300):
        n = rng.randint(2, 25)
        # A small value range guarantees plenty of ties in both variables.
        x = [rng.randrange(4) for _ in range(n)]
        y = [rng.randrange(4) for _ in range(n)]
        expected = brute_tau_b(x, y)
        got = kendall_tau_b(x, y)
        if math.isnan(expected):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(expected), (x, y)


def test_tau_b_matches_definition_without_ties():
    rng = random.Random(36)
    for _ in range(200):
        n = rng.randint(2, 20)
        x = random.Random(n).sample(range(1000), n)
        y = rng.sample(range(1000), n)
        assert kendall_tau_b(x, y) == pytest.approx(brute_tau_b(x, y))


def test_tau_b_extremes():
    assert kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert kendall_tau_b([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_tau_b_is_nan_when_a_variable_is_constant():
    assert math.isnan(kendall_tau_b([1, 1, 1], [1, 2, 3]))
    assert math.isnan(kendall_tau_b([1, 2, 3], [5, 5, 5]))


def test_tau_b_needs_at_least_two_points():
    assert math.isnan(kendall_tau_b([], []))
    assert math.isnan(kendall_tau_b([1], [1]))


def test_tau_b_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        kendall_tau_b([1, 2], [1])


def test_tau_b_is_symmetric():
    rng = random.Random(38)
    for _ in range(100):
        n = rng.randint(2, 15)
        x = [rng.randrange(5) for _ in range(n)]
        y = [rng.randrange(5) for _ in range(n)]
        a, b = kendall_tau_b(x, y), kendall_tau_b(y, x)
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b)


def test_tau_b_matches_scipy_when_available():
    stats = pytest.importorskip("scipy.stats")
    rng = random.Random(40)
    for _ in range(50):
        n = rng.randint(3, 30)
        x = [rng.randrange(5) for _ in range(n)]
        y = [rng.randrange(5) for _ in range(n)]
        expected = stats.kendalltau(x, y).statistic
        got = kendall_tau_b(x, y)
        if math.isnan(expected):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# The vectorised method's specific hazards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", VECTORISED)
def test_vectorised_padding_and_bit_widths(impl):
    """count_numpy pads to a power of two, count_numpy_radix picks a bit width.

    Both derive a size from n, so every n near a power of two is a chance to
    be off by one in a way that silently changes the answer.
    """
    for n in range(1, 200):
        a = list(range(n))[::-1]
        assert impl(a) == max_inversions(n), n


@pytest.mark.parametrize("impl", VECTORISED)
def test_vectorised_agree_with_mergesort_on_dense_ties(impl):
    """If the internal merge or partition were wrong the count would drift."""
    rng = random.Random(42)
    for _ in range(50):
        a = [rng.randrange(3) for _ in range(rng.randint(200, 400))]
        assert impl(a) == count_mergesort(a)


@pytest.mark.parametrize("impl", VECTORISED)
def test_vectorised_return_a_python_int(impl):
    """int64 would silently wrap for a large enough n; the total must not."""
    assert type(impl([3, 2, 1])) is int


def test_radix_group_invariant_holds_for_every_rank_multiplicity():
    """Equal ranks never split, so ties must contribute exactly zero."""
    for distinct in range(1, 6):
        for n in (distinct, 2 * distinct, 3 * distinct):
            rng = random.Random(n * 10 + distinct)
            a = [rng.randrange(distinct) for _ in range(n)]
            assert count_numpy_radix(a) == count_brute(a)


def test_numpy_ranks_fast_path_and_fallback_agree():
    """Plain numeric input takes the numpy ranking path; anything else falls back."""
    from fractions import Fraction as F

    plain = [5, 2, 9, 2, 7]
    exotic = [F(5), F(2), F(9), F(2), F(7)]        # object dtype -> fallback
    huge = [10**30, 10**20, 10**40, 10**20, 10**35]  # object dtype -> fallback
    for impl in VECTORISED:
        assert impl(plain) == impl(exotic) == impl(huge) == count_brute(plain)


@pytest.mark.slow
@pytest.mark.parametrize("impl", VECTORISED)
def test_vectorised_total_exceeds_int32_range(impl):
    """n = 200k reversed gives 2e10 inversions -- past 2^31, well inside int64."""
    n = 200_000
    assert impl(list(range(n))[::-1]) == max_inversions(n) == 19_999_900_000


@pytest.mark.slow
def test_vectorised_agree_with_fenwick_at_a_million():
    rng = random.Random(44)
    a = [rng.randrange(10**6) for _ in range(1_000_000)]
    expected = count_fenwick(a)
    assert count_numpy(a) == expected
    assert count_numpy_radix(a) == expected


@pytest.mark.slow
def test_no_recursion_limit_on_a_long_array():
    """The merge sort is iterative, so depth is not a function of n."""
    n = 2_000_000
    assert count_mergesort(list(range(n))[::-1]) == max_inversions(n)


# ---------------------------------------------------------------------------
# Non-mutation and return_sorted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", IMPLS)
def test_input_is_never_mutated(impl):
    a = [3, 1, 4, 1, 5]
    original = list(a)
    impl(a)
    assert a == original


def test_mergesort_can_return_the_sorted_array():
    total, ordered = count_mergesort([3, 1, 4, 1, 5], return_sorted=True)
    assert total == 3
    assert ordered == [1, 1, 3, 4, 5]


def test_mergesort_return_sorted_on_short_input():
    assert count_mergesort([], return_sorted=True) == (0, [])
    assert count_mergesort([7], return_sorted=True) == (0, [7])


def test_mergesort_sort_is_stable():
    """Ties keep their input order -- which is also what makes the count right."""
    rng = random.Random(46)
    a = [(rng.randrange(4), i) for i in range(200)]
    _total, ordered = count_mergesort(a, key=lambda p: p[0], return_sorted=True)
    assert ordered == sorted(a, key=lambda p: p[0])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_demo(capsys):
    assert main(["--demo"]) == 0
    assert "Lehmer code" in capsys.readouterr().out


def test_cli_verify(capsys):
    assert main(["--verify"]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_counts_numbers(capsys):
    assert main(["3", "2", "1"]) == 0
    out = capsys.readouterr().out
    assert "3 inversions" in out
    assert "max 3" in out


def test_cli_detail(capsys):
    assert main(["--detail", "2", "1"]) == 0
    out = capsys.readouterr().out
    assert "smaller to the right: [1, 0]" in out
    assert "greater to the left:  [0, 1]" in out


@pytest.mark.parametrize("method", METHODS)
def test_cli_honours_every_method(method, capsys):
    assert main(["--method", method, "4", "3", "2", "1"]) == 0
    assert "6 inversions" in capsys.readouterr().out


def test_module_runs_as_a_script():
    result = subprocess.run(
        [sys.executable, "inversions.py", "--verify"],
        cwd=HERE, capture_output=True, text=True, check=True,
    )
    assert "OK" in result.stdout
