from __future__ import annotations

import itertools

import networkx as nx
import pytest
from coloring import (
    crown_graph,
    crown_interleaved_order,
    dsatur,
    exact_chromatic_number,
    from_networkx,
    greedy_color,
    is_proper_coloring,
    num_colors,
    queen_graph,
    rlf,
    welsh_powell,
)

HEURISTICS = [greedy_color, welsh_powell, dsatur, rlf]


def _brute_force_chromatic_number(graph: dict) -> int:
    """Independent oracle: try every k from 1 up, brute-force every k^n coloring.

    Only used on graphs with <= 10 vertices in these tests -- this is the
    reference exact_chromatic_number's own clique-lower-bound reasoning is
    checked against, not a re-derivation of the same algorithm.
    """
    nodes = list(graph)
    n = len(nodes)
    if n == 0:
        return 0
    edges = [(u, v) for u in graph for v in graph[u] if u < v]
    for k in range(1, n + 1):
        for coloring_tuple in itertools.product(range(k), repeat=n):
            coloring = dict(zip(nodes, coloring_tuple, strict=True))
            if all(coloring[u] != coloring[v] for u, v in edges):
                return k
    return n


class TestHeuristicsProduceProperColorings:
    @pytest.mark.parametrize("heuristic", HEURISTICS)
    @pytest.mark.parametrize(
        "graph",
        [
            {},
            {1: []},
            {1: [2], 2: [1]},
            {1: [2, 3], 2: [1, 3], 3: [1, 2]},  # triangle
            crown_graph(6),
            queen_graph(5),
            from_networkx(nx.petersen_graph()),
            from_networkx(nx.gnp_random_graph(25, 0.4, seed=3)),
        ],
    )
    def test_proper(self, heuristic, graph):
        coloring = heuristic(graph)
        assert is_proper_coloring(graph, coloring)
        assert set(coloring) == set(graph)

    @pytest.mark.parametrize("heuristic", HEURISTICS)
    def test_uses_at_most_max_degree_plus_one_colors(self, heuristic):
        graph = from_networkx(nx.gnp_random_graph(30, 0.3, seed=5))
        coloring = heuristic(graph)
        max_degree = max((len(graph[v]) for v in graph), default=0)
        assert num_colors(coloring) <= max_degree + 1


class TestKnownChromaticNumbers:
    def test_empty_graph(self):
        assert exact_chromatic_number({}) == (0, {})

    def test_single_vertex(self):
        k, coloring = exact_chromatic_number({1: []})
        assert k == 1
        assert coloring == {1: 0}

    def test_single_edge(self):
        k, _ = exact_chromatic_number({1: [2], 2: [1]})
        assert k == 2

    @pytest.mark.parametrize("n", [3, 4, 5, 8])
    def test_complete_graph_needs_n_colors(self, n):
        graph = from_networkx(nx.complete_graph(n))
        k, coloring = exact_chromatic_number(graph)
        assert k == n
        assert is_proper_coloring(graph, coloring)

    @pytest.mark.parametrize("n", [4, 6, 8, 10])
    def test_even_cycle_is_bipartite(self, n):
        graph = from_networkx(nx.cycle_graph(n))
        k, _ = exact_chromatic_number(graph)
        assert k == 2

    @pytest.mark.parametrize("n", [3, 5, 7, 9])
    def test_odd_cycle_needs_three_colors(self, n):
        graph = from_networkx(nx.cycle_graph(n))
        k, _ = exact_chromatic_number(graph)
        assert k == 3

    def test_petersen_graph_is_three_chromatic(self):
        graph = from_networkx(nx.petersen_graph())
        k, coloring = exact_chromatic_number(graph)
        assert k == 3
        assert is_proper_coloring(graph, coloring)

    def test_groetzsch_graph_is_four_chromatic(self):
        # Mycielskian of C5: the smallest triangle-free graph with chi = 4.
        groetzsch = nx.mycielskian(nx.cycle_graph(5))
        graph = from_networkx(groetzsch)
        k, coloring = exact_chromatic_number(graph)
        assert k == 4
        assert is_proper_coloring(graph, coloring)
        # Triangle-free: no 3-clique, yet chi = 4 -- greedy-family heuristics
        # can't shortcut this with a clique argument the way they can on a
        # complete graph.
        assert max(len(c) for c in nx.find_cliques(groetzsch)) == 2

    @pytest.mark.parametrize("n", [4, 5])
    def test_crown_graph_is_two_chromatic(self, n):
        graph = crown_graph(n)
        k, coloring = exact_chromatic_number(graph)
        assert k == 2
        assert is_proper_coloring(graph, coloring)


class TestCrownGraphOrderSensitivity:
    def test_natural_order_colors_optimally(self):
        graph = crown_graph(8)
        coloring = greedy_color(graph, order=sorted(graph))
        assert num_colors(coloring) == 2

    def test_interleaved_order_is_forced_far_from_optimal(self):
        n = 8
        graph = crown_graph(n)
        coloring = greedy_color(graph, order=crown_interleaved_order(n))
        assert is_proper_coloring(graph, coloring)
        # The chromatic number is 2; this asserts the empirically observed
        # failure mode is real and severe, not just "not optimal" by one.
        assert num_colors(coloring) >= n // 2

    @pytest.mark.parametrize("n", [4, 6, 8, 10])
    def test_dsatur_is_not_fooled_by_the_bad_order(self, n):
        # DSATUR recomputes which vertex to color next every step, so a fixed
        # bad input order (unlike plain greedy) can't trap it.
        graph = crown_graph(n)
        coloring = dsatur(graph)
        assert num_colors(coloring) == 2


class TestExactMatchesBruteForceOracle:
    @pytest.mark.parametrize("seed", range(10))
    def test_random_small_graphs(self, seed):
        graph = from_networkx(nx.gnp_random_graph(8, 0.5, seed=seed))
        expected = _brute_force_chromatic_number(graph)
        actual, coloring = exact_chromatic_number(graph)
        assert actual == expected
        assert is_proper_coloring(graph, coloring)
        assert num_colors(coloring) == actual


class TestQueenGraph:
    def test_queen_graph_row_col_diagonal_adjacency(self):
        graph = queen_graph(3)
        # (0,0) = 0, (0,1) = 1, (0,2) = 2, (1,0) = 3, (1,1) = 4, (1,2) = 5, ...
        assert 1 in graph[0]  # same row
        assert 3 in graph[0]  # same column
        assert 4 in graph[0]  # same diagonal
        assert 5 not in graph[0]  # not attackable from (0,0)

    def test_queen_graph_is_symmetric_and_complete_node_set(self):
        graph = queen_graph(4)
        assert set(graph) == set(range(16))
        for u, neighbors in graph.items():
            for v in neighbors:
                assert u in graph[v]


class TestCoreProperties:
    def test_num_colors_counts_distinct_colors_only(self):
        assert num_colors({1: 0, 2: 0, 3: 1}) == 2

    def test_is_proper_coloring_detects_a_conflict(self):
        graph = {1: [2], 2: [1]}
        assert not is_proper_coloring(graph, {1: 0, 2: 0})
        assert is_proper_coloring(graph, {1: 0, 2: 1})

    def test_asymmetric_input_is_normalized(self):
        # Edge listed only as 1 -> 2, not 2 -> 1.
        graph = {1: [2], 2: []}
        coloring = greedy_color(graph)
        assert is_proper_coloring(graph, coloring)
        assert coloring[1] != coloring[2]
