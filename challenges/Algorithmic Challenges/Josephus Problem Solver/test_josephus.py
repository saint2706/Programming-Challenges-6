"""Tests for the Josephus solver.

Run with:  uv run --with pytest pytest -q
"""

from __future__ import annotations

import random
import time
import subprocess
import sys
from pathlib import Path

import pytest

from josephus import (
    Fenwick,
    elimination_order,
    frames,
    main,
    simulate,
    survivor,
    survivor_fast,
    survivor_pow2,
    survivor_recurrence,
    survivors,
    verify,
)

HERE = Path(__file__).parent
METHODS = [survivor, survivor_recurrence, survivor_fast]


# ---------------------------------------------------------------------------
# Agreement with the simulation oracle
# ---------------------------------------------------------------------------


def test_exhaustive_cross_validation():
    """Every method against the deque oracle for all n<=120, k<=20, every start."""
    assert verify(max_n=120, max_k=20) == []


@pytest.mark.parametrize("fn", METHODS)
def test_trivial_circles(fn):
    for k in range(1, 8):
        assert fn(1, k) == 1
    for k in (1, 2, 3, 4, 100, 101):
        assert fn(2, k) == simulate(2, k)[-1]


@pytest.mark.parametrize("fn", METHODS)
def test_k_of_one_kills_in_order(fn):
    for n in (1, 2, 5, 50):
        assert fn(n, 1) == n
        assert elimination_order(n, 1) == list(range(1, n + 1))


@pytest.mark.parametrize("fn", METHODS)
def test_k_larger_than_n(fn):
    """k may exceed n; counting simply wraps."""
    for n in (1, 3, 7, 20):
        for k in (n + 1, 2 * n + 3, 1000, 10**6 + 1):
            assert fn(n, k) == simulate(n, k)[-1], (n, k)


@pytest.mark.parametrize("fn", METHODS + [lambda n, k: elimination_order(n, k)[-1]])
def test_random_agreement(fn):
    rng = random.Random(20260904)
    for _ in range(400):
        n = rng.randint(1, 300)
        k = rng.randint(1, 60)
        assert fn(n, k) == simulate(n, k)[-1], (n, k)


# ---------------------------------------------------------------------------
# The k = 2 closed form
# ---------------------------------------------------------------------------


def test_pow2_closed_form_matches_recurrence():
    for n in range(1, 3000):
        assert survivor_pow2(n) == survivor_recurrence(n, 2), n


def test_pow2_powers_of_two_survive_at_position_one():
    for m in range(0, 40):
        assert survivor_pow2(1 << m) == 1


def test_pow2_is_a_bit_rotation():
    """J(n) is n with the leading 1 moved to the end."""
    for n in range(1, 5000):
        bits = bin(n)[2:]
        rotated = bits[1:] + "1"
        assert survivor_pow2(n) == int(rotated, 2)


def test_pow2_handles_astronomical_n():
    n = 10**1000 + 12345
    assert survivor_pow2(n) == 2 * (n - (1 << (n.bit_length() - 1))) + 1


# ---------------------------------------------------------------------------
# The O(k log n) algorithm
# ---------------------------------------------------------------------------


def test_fast_is_actually_fast_for_huge_n():
    """No loop over n is allowed to hide in here."""
    n = 10**18
    assert survivor_fast(n, 3) == survivor(n, 3)
    assert 1 <= survivor_fast(n, 7) <= n


def test_fast_iteration_count_is_logarithmic_in_n(monkeypatch):
    """(k-1)*ln(n*k) steps, so 10^12 must cost only ~2x what 10^6 does."""
    counts = {}
    for n in (10**6, 10**12):
        x, steps, k = 0, 0, 5
        target = n * (k - 1)
        while x < target:
            x += x // (k - 1) + 1
            steps += 1
        counts[n] = steps
    assert counts[10**12] < 2.5 * counts[10**6]


def test_survivor_dispatch_agrees_with_both_backends():
    for n, k in [(10, 3), (10**6, 3), (10, 10**6), (5, 10**7), (10**9, 5)]:
        expected = (
            survivor_recurrence(n, k) if n <= 10**7 else survivor_fast(n, k)
        )
        assert survivor(n, k) == expected, (n, k)


def test_fast_skips_its_linear_prefix():
    """Without the seeding step, n=1 walks k-1 times, one increment each."""
    start = time.perf_counter()
    assert survivor_fast(1, 10**12) == 1
    assert survivor_fast(10**12, 3) == survivor(10**12, 3)
    assert time.perf_counter() - start < 1.0


def test_fast_refuses_intractable_work_instead_of_hanging():
    """n=2, k=10^12 wants ~7e11 iterations. That is not slow, it is never."""
    with pytest.raises(ValueError, match="large relative to n"):
        survivor_fast(2, 10**12)
    with pytest.raises(ValueError, match="survivor"):
        survivor_fast(10, 10**9)
    # The guard is advisory, not a hard limit.
    assert survivor_fast(20, 10**6 + 1, max_steps=None) == simulate(20, 10**6 + 1)[-1]


def test_survivor_is_never_blocked_by_the_guard():
    """survivor() may legitimately want more than max_steps when n is huge."""
    assert survivor(10**12, 3) == survivor_fast(10**12, 3)
    assert 1 <= survivor(10**15, 11) <= 10**15


def test_survivor_never_picks_the_pathological_branch():
    """k >> n is where survivor_fast is asymptotically wrong; dispatch must win."""
    start = time.perf_counter()
    for n, k in [(10, 10**9), (5, 10**10), (100, 10**8)]:
        assert survivor(n, k) == survivor_recurrence(n, k), (n, k)
    assert time.perf_counter() - start < 1.0


def test_fenwick_rejects_an_empty_tree():
    with pytest.raises(ValueError, match="at least one slot"):
        Fenwick(0)


# ---------------------------------------------------------------------------
# Elimination order and the Fenwick tree
# ---------------------------------------------------------------------------


def test_elimination_order_is_a_permutation():
    for n, k in [(1, 1), (10, 3), (41, 3), (100, 7), (997, 13)]:
        order = elimination_order(n, k)
        assert sorted(order) == list(range(1, n + 1))


def test_elimination_order_matches_simulation():
    for n in range(1, 60):
        for k in range(1, 12):
            assert elimination_order(n, k) == simulate(n, k), (n, k)


def test_elimination_order_is_independent_of_k_magnitude():
    """k = 10^9 costs the same as k = 2 -- k is not in the complexity."""
    assert elimination_order(50, 10**9) == simulate(50, 10**9)


def test_fenwick_select_and_remove():
    for n in (1, 2, 5, 33, 64, 65, 100):
        fen = Fenwick(n)
        alive = list(range(1, n + 1))
        rng = random.Random(n)
        while alive:
            for j, expected in enumerate(alive, 1):
                assert fen.select(j) == expected
            victim = rng.choice(alive)
            alive.remove(victim)
            fen.remove(victim)


def test_fenwick_select_is_not_quadratic():
    """Just a smoke test that a large tree stays responsive."""
    fen = Fenwick(200_000)
    assert fen.select(1) == 1
    assert fen.select(200_000) == 200_000
    fen.remove(1)
    assert fen.select(1) == 2


# ---------------------------------------------------------------------------
# Generalizations
# ---------------------------------------------------------------------------


def test_the_historical_case():
    """Josephus and 40 others, every third man. Two were meant to live."""
    assert survivors(41, 3, 2) == [16, 31]
    assert survivor(41, 3) == 31


@pytest.mark.parametrize("m", [1, 2, 3, 5, 8])
def test_survivors_matches_the_tail_of_the_simulation(m):
    for n in range(m, 60):
        for k in (1, 2, 3, 7):
            assert survivors(n, k, m) == sorted(simulate(n, k)[-m:]), (n, k, m)


def test_survivors_of_everyone_is_everyone():
    assert survivors(9, 4, 9) == list(range(1, 10))


def test_start_offset_rotates_the_answer():
    for n in (1, 5, 12, 41):
        for k in (2, 3, 5):
            for s in range(1, n + 1):
                expected = simulate(n, k, start=s)[-1]
                assert survivor(n, k, start=s) == expected
                assert elimination_order(n, k, start=s) == simulate(n, k, start=s)


def test_start_wraps_around_the_circle_consistently():
    """start=0 is start=n, start=-1 is start=n-1, and every method agrees."""
    for n in (1, 2, 5, 7, 13):
        for k in (1, 2, 3, 7):
            for s in range(-2 * n, 3 * n + 1):
                expected = simulate(n, k, start=s)[-1]
                assert survivor(n, k, start=s) == expected, (n, k, s)
                assert survivor_recurrence(n, k, start=s) == expected, (n, k, s)
                assert survivor_fast(n, k, start=s) == expected, (n, k, s)
                assert elimination_order(n, k, start=s)[-1] == expected, (n, k, s)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", METHODS + [simulate, elimination_order])
@pytest.mark.parametrize("n,k", [(0, 3), (-1, 3), (5, 0), (5, -2)])
def test_rejects_nonsense(fn, n, k):
    with pytest.raises(ValueError):
        fn(n, k)


def test_survivors_rejects_impossible_m():
    with pytest.raises(ValueError):
        survivors(5, 3, 0)
    with pytest.raises(ValueError):
        survivors(5, 3, 6)


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def test_frames_shows_one_row_per_elimination_plus_the_start():
    rows = list(frames(8, 3))
    assert len(rows) == 9
    assert "start" in rows[0]
    assert "survivor" in rows[-1]
    assert rows[-1].count("--") == 8


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_reports_the_survivor(capsys):
    assert main(["41", "3"]) == 0
    assert "position 31" in capsys.readouterr().out


def test_cli_survivors_and_order(capsys):
    assert main(["41", "3", "--survivors", "2", "--order"]) == 0
    out = capsys.readouterr().out
    assert "[16, 31]" in out and "elimination order:" in out


def test_cli_methods_agree(capsys):
    outputs = set()
    for method in ("auto", "simulate", "recurrence", "fast", "pow2"):
        assert main(["1000", "2", "--method", method]) == 0
        outputs.add(capsys.readouterr().out.strip())
    assert len(outputs) == 1


def test_cli_pow2_rejects_other_k():
    with pytest.raises(SystemExit):
        main(["10", "3", "--method", "pow2"])


def test_cli_verify_subcommand():
    proc = subprocess.run(
        [sys.executable, str(HERE / "josephus.py"), "--verify"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all methods agree" in proc.stdout


def test_cli_animation_is_skipped_when_too_wide(capsys):
    assert main(["100", "3", "--animate"]) == 0
    assert "animation skipped" in capsys.readouterr().err
