from __future__ import annotations

import networkx as nx
import pytest
from scc import (
    condensation,
    gabow_scc,
    kosaraju_scc,
    random_sparse_digraph,
    tarjan_scc,
    tarjan_scc_recursive,
)

ALGORITHMS = [tarjan_scc, kosaraju_scc, gabow_scc, tarjan_scc_recursive]


def _key(components):
    return frozenset(frozenset(c) for c in components)


def _nx_reference(graph: dict) -> frozenset:
    g = nx.DiGraph()
    g.add_nodes_from(graph)
    g.add_edges_from((u, v) for u, succs in graph.items() for v in succs)
    return frozenset(frozenset(c) for c in nx.strongly_connected_components(g))


@pytest.mark.parametrize("algo", ALGORITHMS)
class TestAllAlgorithmsAgree:
    def test_empty_graph(self, algo):
        assert algo({}) == []

    def test_single_node_no_edges(self, algo):
        assert _key(algo({1: []})) == frozenset({frozenset({1})})

    def test_single_node_self_loop(self, algo):
        assert _key(algo({1: [1]})) == frozenset({frozenset({1})})

    def test_two_node_cycle(self, algo):
        assert _key(algo({1: [2], 2: [1]})) == frozenset({frozenset({1, 2})})

    def test_disconnected_no_edges(self, algo):
        graph = {1: [], 2: [], 3: []}
        assert _key(algo(graph)) == frozenset(
            {frozenset({1}), frozenset({2}), frozenset({3})}
        )

    def test_dag_each_node_own_component(self, algo):
        graph = {1: [2, 3], 2: [4], 3: [4], 4: []}
        expected = frozenset({frozenset({n}) for n in graph})
        assert _key(algo(graph)) == expected

    def test_complete_bidirectional_is_one_component(self, algo):
        n = 6
        graph = {i: [j for j in range(n) if j != i] for i in range(n)}
        assert _key(algo(graph)) == frozenset({frozenset(range(n))})

    def test_clrs_textbook_graph(self, algo):
        # CLRS 3rd ed., Figure 22.9, relabeled a..h -> 1..8.
        graph = {
            1: [2],
            2: [3],
            3: [1],
            4: [2, 3, 5],
            5: [4, 6],
            6: [3, 7],
            7: [6],
            8: [5, 7, 8],
        }
        expected = frozenset(
            {frozenset({1, 2, 3}), frozenset({4, 5}), frozenset({6, 7}), frozenset({8})}
        )
        assert _key(algo(graph)) == expected

    def test_node_referenced_only_as_successor(self, algo):
        graph = {1: [2]}  # node 2 has no key of its own
        assert _key(algo(graph)) == frozenset({frozenset({1}), frozenset({2})})

    def test_chain_of_cycles(self, algo):
        # 0<->1<->2, forming a single SCC only because of the middle back-edges,
        # then 3 is a separate downstream sink.
        graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: []}
        expected = frozenset({frozenset({0, 1, 2}), frozenset({3})})
        assert _key(algo(graph)) == expected

    @pytest.mark.parametrize("seed", range(30))
    def test_matches_networkx_on_random_graphs(self, algo, seed):
        size = 3 + (seed % 25)
        graph = random_sparse_digraph(size, avg_out_degree=1.8, seed=seed)
        assert _key(algo(graph)) == _nx_reference(graph)


@pytest.mark.parametrize("a", ALGORITHMS)
@pytest.mark.parametrize("b", ALGORITHMS)
@pytest.mark.parametrize("seed", range(15))
def test_all_pairs_agree_with_each_other(a, b, seed):
    graph = random_sparse_digraph(4 + (seed % 20), avg_out_degree=2.0, seed=seed + 1000)
    assert _key(a(graph)) == _key(b(graph))


class TestRecursiveTarjanHitsLimit:
    def test_recursion_error_on_long_path(self):
        import sys

        path_len = sys.getrecursionlimit() + 200
        graph = {i: [i + 1] for i in range(path_len - 1)}
        graph[path_len - 1] = []
        with pytest.raises(RecursionError):
            tarjan_scc_recursive(graph)

    def test_iterative_tarjan_handles_the_same_path(self):
        import sys

        path_len = sys.getrecursionlimit() + 200
        graph = {i: [i + 1] for i in range(path_len - 1)}
        graph[path_len - 1] = []
        components = tarjan_scc(graph)
        assert len(components) == path_len
        assert all(len(c) == 1 for c in components)


class TestCondensation:
    def test_condensation_is_acyclic_and_correct(self):
        graph = {
            1: [2],
            2: [3],
            3: [1],
            4: [2, 3, 5],
            5: [4, 6],
            6: [3, 7],
            7: [6],
            8: [5, 7, 8],
        }
        components = tarjan_scc(graph)
        dag = condensation(graph, components)
        assert nx.is_directed_acyclic_graph(nx.DiGraph(dag))
        # 3 components (well, 4) with edges only downstream in Tarjan's
        # reverse-topological output order.
        assert len(dag) == len(components) == 4

    def test_condensation_of_single_scc_has_no_edges(self):
        graph = {1: [2], 2: [1]}
        components = tarjan_scc(graph)
        dag = condensation(graph, components)
        assert dag == {0: set()}


class TestRandomSparseDigraph:
    def test_deterministic_with_seed(self):
        g1 = random_sparse_digraph(50, 2.0, seed=42)
        g2 = random_sparse_digraph(50, 2.0, seed=42)
        assert g1 == g2

    def test_no_self_loops(self):
        graph = random_sparse_digraph(200, 3.0, seed=7)
        for node, succs in graph.items():
            assert node not in succs

    def test_all_nodes_present(self):
        graph = random_sparse_digraph(100, 1.0, seed=1)
        assert set(graph.keys()) == set(range(100))
