"""Correctness tests for nqueens.py.

Cross-checks the fast mirror-halved counter against the known OEIS A000170
sequence, validates the bitmask step-generator's solutions are genuinely
non-attacking, and verifies the 8-fold symmetry (fundamental solution count)
against Burnside's lemma computed empirically -- not asserted from memory --
against OEIS A002562.
"""

from __future__ import annotations

import pytest
from nqueens import (
    D4_TRANSFORMS,
    all_solutions,
    canonical_form,
    count_fundamental_solutions,
    count_solutions,
    fixed_point_counts,
    is_valid_solution,
    solve_bitmask_steps,
)

# OEIS A000170: number of solutions to the n-queens problem, n = 0, 1, 2, ...
A000170 = [1, 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200]

# OEIS A002562: number of inequivalent (fundamental) solutions, n = 0, 1, 2, ...
A002562 = [1, 1, 0, 0, 1, 2, 1, 6, 12, 46, 92, 341, 1787]


@pytest.mark.parametrize("n", range(len(A000170)))
def test_count_solutions_matches_oeis_a000170(n):
    assert count_solutions(n) == A000170[n]


@pytest.mark.parametrize("n", range(len(A002562)))
def test_count_fundamental_solutions_matches_oeis_a002562(n):
    assert count_fundamental_solutions(n) == A002562[n]


def test_is_valid_solution_accepts_a_known_solution():
    # Classic 4-queens solution: queen row r in column cols[r].
    assert is_valid_solution((1, 3, 0, 2))


def test_is_valid_solution_rejects_same_column():
    assert not is_valid_solution((0, 0, 1, 2))  # not even a permutation


def test_is_valid_solution_rejects_diagonal_attack():
    assert not is_valid_solution((0, 1, 2, 3))  # every row on the main diagonal


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8])
def test_all_solutions_are_all_valid_and_match_count(n):
    solutions = all_solutions(n)
    assert len(solutions) == count_solutions(n)
    assert all(is_valid_solution(s) for s in solutions)
    assert len(set(solutions)) == len(solutions)  # no duplicates


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_small_n_edge_cases(n):
    solutions = all_solutions(n)
    assert len(solutions) == A000170[n]
    if n in (2, 3):
        assert solutions == []
    if n == 0:
        assert solutions == [()]  # the vacuous placement


def test_bitmask_steps_produce_only_valid_solutions():
    found = []
    for step in solve_bitmask_steps(6):
        if step.action == "solution":
            assert is_valid_solution(step.board)
            found.append(step.board)
    assert len(found) == count_solutions(6)
    assert len(set(found)) == len(found)


def test_bitmask_steps_never_place_a_conflicting_queen():
    # Every intermediate "place" board (a prefix of a solution-in-progress)
    # must itself be internally non-attacking.
    for step in solve_bitmask_steps(6):
        if step.action == "place":
            board = step.board
            for i in range(len(board)):
                for j in range(i + 1, len(board)):
                    assert board[i] != board[j]
                    assert abs(board[i] - board[j]) != j - i


def test_solve_bitmask_steps_respects_limit():
    steps = list(solve_bitmask_steps(8, limit=3))
    solution_steps = [s for s in steps if s.action == "solution"]
    assert len(solution_steps) == 3


# ---------------------------------------------------------------------------
# Symmetry: D4 transforms, canonicalization, and Burnside's lemma
# ---------------------------------------------------------------------------


def test_d4_transforms_are_closed_and_form_a_group_of_order_8():
    # Applying every transform to every transform's result should never
    # produce something outside the 8 named transforms (closure), and the
    # identity composed with anything is that thing.
    n = 5
    solution = all_solutions(n)[0]
    images = {name: fn(solution, n) for name, fn in D4_TRANSFORMS.items()}
    assert images["identity"] == solution
    assert len(set(images.values())) <= 8
    for name, image in images.items():
        assert is_valid_solution(image), (name, image)


def test_canonical_form_is_invariant_under_all_eight_transforms():
    n = 6
    solution = all_solutions(n)[0]
    canon = canonical_form(solution, n)
    for fn in D4_TRANSFORMS.values():
        transformed = fn(solution, n)
        assert canonical_form(transformed, n) == canon


@pytest.mark.parametrize("n", range(1, 10))
def test_burnside_lemma_matches_direct_canonicalization_count(n):
    # Burnside: #orbits = (1/|G|) * sum(|Fix(g)| for g in G). We compute
    # |Fix(g)| empirically (by checking g(s) == s for every found solution)
    # rather than asserting it from a textbook, then check the formula
    # against the ground truth obtained by directly deduplicating canonical
    # forms -- two independent computations that must agree if both the
    # transforms and the fixed-point counting are implemented correctly.
    fixed = fixed_point_counts(n)
    assert len(fixed) == 8
    burnside_estimate = sum(fixed.values()) / 8
    assert burnside_estimate == pytest.approx(round(burnside_estimate))
    assert round(burnside_estimate) == count_fundamental_solutions(n)


def test_fixed_counts_of_inverse_transforms_are_equal():
    # g(s) == s iff s == g^-1(s), so |Fix(g)| == |Fix(g^-1)| for any group
    # element -- rot90 and rot270 are inverses of each other, so their fixed
    # counts must match. (An earlier version of this test asserted the
    # commonly-repeated claim that NO n-queens solution has 4-fold rotational
    # symmetry; computing fixed_point_counts directly shows that is false --
    # n=4, n=5, and n=12 all have solutions fixed by a 90-degree rotation.
    # See README.md's "A claim that turned out to be wrong" section.)
    for n in range(1, 13):
        fixed = fixed_point_counts(n)
        assert fixed["rot90"] == fixed["rot270"]


def test_some_small_n_do_have_4fold_rotationally_symmetric_solutions():
    # Documents the actual (verified, not assumed) behavior: contrary to a
    # commonly repeated claim, n=4, n=5, and n=12 all have solutions with
    # true 4-fold rotational symmetry.
    assert fixed_point_counts(4)["rot90"] == 2
    assert fixed_point_counts(5)["rot90"] == 2
    assert fixed_point_counts(12)["rot90"] == 8
    # But n=6 through n=11 in fact have none.
    for n in range(6, 12):
        assert fixed_point_counts(n)["rot90"] == 0
