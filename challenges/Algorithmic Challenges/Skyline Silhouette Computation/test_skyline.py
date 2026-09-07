"""Tests for the skyline silhouette computation.

Run with:  uv run --with pytest --with sortedcontainers pytest -q
Skip the slow randomized cross-verification with:  -m "not slow"

``sortedcontainers`` backs ``sweep_line_bst``; the whole module is skipped
(not failed) if it isn't installed, via ``pytest.importorskip`` below.
"""

from __future__ import annotations

import itertools
import random
import subprocess
import sys
from pathlib import Path

import pytest
from skyline import (
    Building,
    _random_buildings,
    brute_force,
    divide_and_conquer,
    render_ascii,
    skyline,
    sweep_line,
    sweep_line_bst,
    verify,
)

pytest.importorskip("sortedcontainers")

HERE = Path(__file__).parent
METHODS = (brute_force, sweep_line, sweep_line_bst, divide_and_conquer)


def all_agree(buildings: list[Building]) -> list[list]:
    return [fn(buildings) for fn in METHODS]


# ---------------------------------------------------------------------------
# The textbook example (LeetCode 218)
# ---------------------------------------------------------------------------


def test_leetcode_example():
    buildings = [(2, 9, 10), (3, 7, 15), (5, 12, 12), (15, 20, 10), (19, 24, 8)]
    expected = [(2, 10), (3, 15), (7, 12), (12, 0), (15, 10), (20, 8), (24, 0)]
    for fn in METHODS:
        assert fn(buildings) == expected


def test_second_leetcode_example():
    buildings = [(0, 2, 3), (2, 5, 3)]
    # Same height across a shared edge: must merge to one flat segment.
    expected = [(0, 3), (5, 0)]
    for fn in METHODS:
        assert fn(buildings) == expected


def test_single_building():
    buildings = [(1, 5, 7)]
    expected = [(1, 7), (5, 0)]
    for fn in METHODS:
        assert fn(buildings) == expected


def test_empty_input():
    for fn in METHODS:
        assert fn([]) == []


# ---------------------------------------------------------------------------
# Edge cases named in the brief
# ---------------------------------------------------------------------------


def test_shared_left_and_right_edges():
    # Two buildings starting at the same x, different heights: the taller
    # one's height must be reported at that x, not the shorter one's.
    buildings = [(0, 5, 3), (0, 8, 9)]
    expected = [(0, 9), (8, 0)]
    for fn in METHODS:
        assert fn(buildings) == expected

    # Two buildings ending at the same x.
    buildings = [(0, 5, 3), (2, 5, 9)]
    expected = [(0, 3), (2, 9), (5, 0)]
    for fn in METHODS:
        assert fn(buildings) == expected


def test_adjacent_equal_height_merges_no_extra_point():
    buildings = [(0, 5, 10), (5, 10, 10)]
    expected = [(0, 10), (10, 0)]
    for fn in METHODS:
        got = fn(buildings)
        assert got == expected, f"{fn.__name__} over-emitted: {got}"
        # The x=5 handoff must not appear at all.
        assert 5 not in [x for x, _ in got]


def test_three_adjacent_equal_height_buildings():
    buildings = [(0, 3, 4), (3, 6, 4), (6, 9, 4)]
    expected = [(0, 4), (9, 0)]
    for fn in METHODS:
        assert fn(buildings) == expected


def test_zero_width_building_is_ignored():
    buildings = [(0, 5, 5), (5, 5, 999), (5, 10, 5)]
    expected = [(0, 5), (10, 0)]
    for fn in METHODS:
        assert fn(buildings) == expected


def test_zero_width_alone():
    for fn in METHODS:
        assert fn([(3, 3, 100)]) == []


def test_reversed_left_right_is_treated_as_malformed_and_dropped():
    # left > right: no valid footprint, dropped like a zero-width building.
    buildings = [(5, 5, 1), (10, 5, 100), (0, 20, 1)]
    expected = brute_force([(0, 20, 1)])
    for fn in METHODS:
        assert fn(buildings) == expected


def test_building_fully_contained_in_another():
    buildings = [(0, 10, 20), (3, 7, 5)]
    expected = [(0, 20), (10, 0)]
    for fn in METHODS:
        assert fn(buildings) == expected


def test_identical_footprints():
    buildings = [(0, 5, 10), (0, 5, 10), (0, 5, 10)]
    expected = [(0, 10), (5, 0)]
    for fn in METHODS:
        assert fn(buildings) == expected


def test_dense_ties_many_buildings_same_x():
    # 20 buildings all starting at 0, all ending at 10, ascending heights.
    buildings = [(0, 10, h) for h in range(1, 21)]
    expected = [(0, 20), (10, 0)]
    for fn in METHODS:
        assert fn(buildings) == expected


def test_zero_height_building_is_invisible():
    buildings = [(0, 5, 0), (5, 10, 0)]
    for fn in METHODS:
        assert fn(buildings) == []

    # A zero-height building next to a real one: the zero one is invisible,
    # not a "dip to zero and back".
    buildings = [(0, 5, 0), (5, 10, 8)]
    expected = [(5, 8), (10, 0)]
    for fn in METHODS:
        assert fn(buildings) == expected


def test_negative_coordinates():
    buildings = [(-5, 0, 4), (-3, 2, 6)]
    expected = brute_force(buildings)
    for fn in (sweep_line, sweep_line_bst, divide_and_conquer):
        assert fn(buildings) == expected


def test_ends_at_height_zero():
    buildings = [(0, 5, 3)]
    for fn in METHODS:
        got = fn(buildings)
        assert got[-1] == (5, 0)


def test_negative_height_raises():
    for fn in METHODS:
        with pytest.raises(ValueError):
            fn([(0, 5, -1)])


def test_output_has_no_consecutive_duplicate_heights():
    rng = random.Random(42)
    for _ in range(50):
        buildings = _random_buildings(rng, rng.randint(0, 20), span=15, max_h=5)
        for fn in METHODS:
            points = fn(buildings)
            heights = [h for _, h in points]
            for a, b in itertools.pairwise(heights):
                assert a != b, (
                    f"{fn.__name__} emitted consecutive equal heights: {points}"
                )


# ---------------------------------------------------------------------------
# skyline() dispatcher
# ---------------------------------------------------------------------------


def test_skyline_dispatch_matches_named_methods():
    buildings = [(2, 9, 10), (3, 7, 15), (5, 12, 12)]
    assert skyline(buildings, method="sweep") == sweep_line(buildings)
    assert skyline(buildings, method="sweep_bst") == sweep_line_bst(buildings)
    assert skyline(buildings, method="dc") == divide_and_conquer(buildings)
    assert skyline(buildings, method="brute") == brute_force(buildings)
    assert skyline(buildings) == skyline(buildings, method="auto")


def test_skyline_bad_method_raises():
    with pytest.raises(ValueError):
        skyline([(0, 1, 1)], method="nope")


# ---------------------------------------------------------------------------
# Randomized cross-verification (small, fast subset)
# ---------------------------------------------------------------------------


def test_verify_quick():
    assert verify(trials=60, verbose=False)


@pytest.mark.slow
def test_verify_many_random_trials():
    assert verify(seed=1234, trials=4000, verbose=False)


@pytest.mark.slow
@pytest.mark.parametrize("span", [5, 15, 60, 300])
@pytest.mark.parametrize("max_h", [1, 3, 20, 500])
def test_cross_verification_grid(span, max_h):
    """Property-based-style sweep over overlap density x height distribution."""
    rng = random.Random(hash((span, max_h)) & 0xFFFF)
    for _ in range(30):
        n = rng.randint(0, 30)
        buildings = _random_buildings(rng, n, span=span, max_h=max_h)
        expected = brute_force(buildings)
        assert sweep_line(buildings) == expected
        assert sweep_line_bst(buildings) == expected
        assert divide_and_conquer(buildings) == expected


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------


def test_render_ascii_smoke():
    buildings = [(2, 9, 10), (3, 7, 15), (5, 12, 12), (15, 20, 10), (19, 24, 8)]
    lines = render_ascii(buildings)
    assert any("#" in line for line in lines)
    assert any("key points" in line for line in lines)


def test_render_ascii_empty():
    assert render_ascii([]) == ["(no buildings)"]


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


def test_cli_demo():
    result = subprocess.run(
        [sys.executable, str(HERE / "skyline.py"), "--demo"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "key points" in result.stdout


def test_cli_verify():
    result = subprocess.run(
        [sys.executable, str(HERE / "skyline.py"), "--verify", "--trials", "20"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "OK" in result.stdout


def test_cli_explicit_buildings():
    result = subprocess.run(
        [sys.executable, str(HERE / "skyline.py"), "0,5,10", "5,10,10"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "2 buildings" in result.stdout
