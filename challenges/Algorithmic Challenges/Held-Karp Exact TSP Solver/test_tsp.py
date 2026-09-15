from __future__ import annotations

import itertools
import math
from typing import ClassVar

import pytest
from tsp import (
    branch_and_bound,
    christofides,
    euclidean_instance,
    greedy_edge,
    held_karp,
    nearest_neighbor,
    tour_length,
    two_opt,
)


def _brute_force_optimal(dist: list[list[float]]) -> float:
    """Independent oracle: try every permutation of cities 1..n-1 after fixing city 0."""
    n = len(dist)
    if n <= 1:
        return 0.0
    best = math.inf
    for perm in itertools.permutations(range(1, n)):
        tour = [0, *perm]
        best = min(best, tour_length(dist, tour))
    return best


def _is_valid_tour(tour: list[int], n: int) -> bool:
    return sorted(tour) == list(range(n))


class TestHeldKarpMatchesBruteForce:
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7])
    @pytest.mark.parametrize("seed", range(5))
    def test_random_instances(self, n, seed):
        _, dist = euclidean_instance(n, seed=seed)
        cost, tour = held_karp(dist)
        assert _is_valid_tour(tour, n)
        assert cost == pytest.approx(tour_length(dist, tour))
        assert cost == pytest.approx(_brute_force_optimal(dist), abs=1e-6)

    def test_two_cities(self):
        dist = [[0, 5], [5, 0]]
        cost, tour = held_karp(dist)
        assert cost == 10
        assert sorted(tour) == [0, 1]

    def test_single_city(self):
        assert held_karp([[0]]) == (0.0, [0])

    def test_empty(self):
        assert held_karp([]) == (0.0, [])


class TestBranchAndBoundMatchesHeldKarp:
    @pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 9, 10])
    @pytest.mark.parametrize("seed", range(5))
    def test_random_instances(self, n, seed):
        _, dist = euclidean_instance(n, seed=seed + 100)
        hk_cost, _ = held_karp(dist)
        bnb_cost, bnb_tour = branch_and_bound(dist)
        assert bnb_cost == pytest.approx(hk_cost, abs=1e-6)
        assert _is_valid_tour(bnb_tour, n)
        assert tour_length(dist, bnb_tour) == pytest.approx(bnb_cost, abs=1e-6)

    def test_small_n_delegates_to_held_karp(self):
        _, dist = euclidean_instance(3, seed=1)
        assert branch_and_bound(dist) == held_karp(dist)


class TestHeuristicsProduceValidTours:
    HEURISTICS: ClassVar = [nearest_neighbor, greedy_edge]

    @pytest.mark.parametrize("heuristic", HEURISTICS)
    @pytest.mark.parametrize("n", [3, 4, 5, 10, 20])
    def test_valid_permutation(self, heuristic, n):
        _, dist = euclidean_instance(n, seed=n)
        tour = heuristic(dist)
        assert _is_valid_tour(tour, n)

    @pytest.mark.parametrize("n", [5, 10, 20, 30])
    def test_two_opt_never_makes_a_tour_worse(self, n):
        _, dist = euclidean_instance(n, seed=n + 1)
        start = nearest_neighbor(dist)
        improved = two_opt(dist, start)
        assert _is_valid_tour(improved, n)
        assert tour_length(dist, improved) <= tour_length(dist, start) + 1e-9

    @pytest.mark.parametrize("n", [5, 10, 15])
    def test_christofides_valid_and_within_1_5x_on_metric_instances(self, n):
        _, dist = euclidean_instance(n, seed=n + 50)
        tour = christofides(dist)
        assert _is_valid_tour(tour, n)
        opt_cost, _ = held_karp(dist)
        assert tour_length(dist, tour) <= 1.5 * opt_cost + 1e-6

    @pytest.mark.parametrize("seed", range(15))
    def test_heuristics_never_beat_the_true_optimum(self, seed):
        n = 8
        _, dist = euclidean_instance(n, seed=seed)
        opt_cost, _ = held_karp(dist)
        for heuristic in (nearest_neighbor, greedy_edge):
            tour = heuristic(dist)
            assert tour_length(dist, tour) >= opt_cost - 1e-6
        two_opt_tour = two_opt(dist, nearest_neighbor(dist))
        assert tour_length(dist, two_opt_tour) >= opt_cost - 1e-6


class TestTourLength:
    def test_triangle(self):
        dist = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        assert tour_length(dist, [0, 1, 2]) == 3

    def test_wraps_around(self):
        dist = [[0, 2, 5], [2, 0, 3], [5, 3, 0]]
        assert tour_length(dist, [0, 1, 2]) == 2 + 3 + 5


class TestEuclideanInstanceSatisfiesTriangleInequality:
    @pytest.mark.parametrize("seed", range(5))
    def test_triangle_inequality_holds(self, seed):
        _, dist = euclidean_instance(12, seed=seed)
        n = len(dist)
        for i, j, k in itertools.combinations(range(n), 3):
            assert dist[i][k] <= dist[i][j] + dist[j][k] + 1e-9
