from __future__ import annotations

import itertools
import random

import pytest
from course_schedule import (
    PearceKellyOrder,
    can_finish,
    dfs_topological_order,
    find_cycle_groups,
    kahn_topological_order,
)


def _valid_topo_order(graph: dict, order: list) -> bool:
    if set(order) != set(graph):
        return False
    pos = {n: i for i, n in enumerate(order)}
    return all(pos[u] < pos[v] for u, succs in graph.items() for v in succs)


def _valid_cycle(graph: dict, cycle: list) -> bool:
    if not cycle or cycle[0] != cycle[-1]:
        return False
    return all(b in graph.get(a, []) for a, b in itertools.pairwise(cycle))


def _random_digraph(
    rng: random.Random, n: int, edge_prob: float
) -> dict[int, list[int]]:
    nodes = list(range(n))
    graph: dict[int, list[int]] = {node: [] for node in nodes}
    for u in nodes:
        for v in nodes:
            if u != v and rng.random() < edge_prob:
                graph[u].append(v)
    return graph


class TestKahnAndDFSAgreeOnAcyclicity:
    @pytest.mark.parametrize("seed", range(80))
    def test_random_instances(self, seed):
        rng = random.Random(seed)
        n = rng.randint(1, 12)
        graph = _random_digraph(rng, n, rng.choice([0.1, 0.2, 0.35]))

        order, blocked = kahn_topological_order(graph)
        dfs_order, cycle = dfs_topological_order(graph)

        is_dag = not blocked
        assert is_dag == (dfs_order is not None)
        if is_dag:
            assert _valid_topo_order(graph, order)
            assert _valid_topo_order(graph, dfs_order)
        else:
            assert _valid_cycle(graph, cycle)
            assert set(blocked) <= set(graph)
            assert set(cycle) <= set(blocked)


class TestKahnTopologicalOrder:
    def test_linear_chain(self):
        graph = {"a": ["b"], "b": ["c"], "c": []}
        order, blocked = kahn_topological_order(graph)
        assert order == ["a", "b", "c"]
        assert blocked == []

    def test_diamond(self):
        graph = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
        order, blocked = kahn_topological_order(graph)
        assert blocked == []
        assert _valid_topo_order(graph, order)

    def test_self_loop_is_a_cycle(self):
        graph = {"a": ["a"]}
        _, blocked = kahn_topological_order(graph)
        assert blocked == ["a"]

    def test_disconnected_components(self):
        graph = {"a": ["b"], "b": [], "x": ["y"], "y": []}
        order, blocked = kahn_topological_order(graph)
        assert blocked == []
        assert _valid_topo_order(graph, order)

    def test_three_cycle_leaves_unrelated_node_out_of_blocked(self):
        graph = {"a": ["b"], "b": ["c"], "c": ["a"], "d": []}
        order, blocked = kahn_topological_order(graph)
        assert set(blocked) == {"a", "b", "c"}
        assert order == ["d"]


class TestDFSTopologicalOrder:
    def test_linear_chain(self):
        graph = {"a": ["b"], "b": ["c"], "c": []}
        order, cycle = dfs_topological_order(graph)
        assert cycle is None
        assert order == ["a", "b", "c"]

    def test_self_loop(self):
        graph = {"a": ["a"]}
        order, cycle = dfs_topological_order(graph)
        assert order is None
        assert cycle == ["a", "a"]

    def test_three_cycle_returns_concrete_path(self):
        graph = {"a": ["b"], "b": ["c"], "c": ["a"]}
        order, cycle = dfs_topological_order(graph)
        assert order is None
        assert _valid_cycle(graph, cycle)
        assert set(cycle[:-1]) == {"a", "b", "c"}


class TestFindCycleGroups:
    def test_acyclic_graph_has_no_groups(self):
        graph = {"a": ["b"], "b": ["c"], "c": []}
        assert find_cycle_groups(graph) == []

    def test_single_three_cycle(self):
        graph = {"a": ["b"], "b": ["c"], "c": ["a"], "d": []}
        groups = find_cycle_groups(graph)
        assert len(groups) == 1
        assert set(groups[0]) == {"a", "b", "c"}

    def test_two_independent_cycles(self):
        graph = {
            "a": ["b"],
            "b": ["a"],
            "x": ["y"],
            "y": ["x"],
            "z": [],
        }
        groups = find_cycle_groups(graph)
        group_sets = {frozenset(g) for g in groups}
        assert group_sets == {frozenset({"a", "b"}), frozenset({"x", "y"})}

    def test_self_loop_is_its_own_group(self):
        graph = {"a": ["a"], "b": []}
        groups = find_cycle_groups(graph)
        assert groups == [["a"]]

    @pytest.mark.parametrize("seed", range(60))
    def test_every_group_is_mutually_reachable(self, seed):
        rng = random.Random(seed + 500)
        n = rng.randint(2, 10)
        graph = _random_digraph(rng, n, 0.3)
        for group in find_cycle_groups(graph):
            gs = set(group)
            for start in gs:
                seen = {start}
                stack = [start]
                while stack:
                    node = stack.pop()
                    for w in graph[node]:
                        if w in gs and w not in seen:
                            seen.add(w)
                            stack.append(w)
                assert gs <= seen


class TestCanFinish:
    def test_no_prerequisites(self):
        assert can_finish(3, [])

    def test_linear_prerequisites(self):
        assert can_finish(3, [(1, 0), (2, 1)])

    def test_cycle_makes_it_impossible(self):
        assert not can_finish(2, [(0, 1), (1, 0)])


class TestPearceKellyOrder:
    def test_empty(self):
        pk = PearceKellyOrder()
        assert pk.order() == []

    def test_forward_edges_need_no_reorder(self):
        pk = PearceKellyOrder(["a", "b", "c"])
        assert pk.try_add_edge("a", "b") is None
        assert pk.try_add_edge("b", "c") is None
        assert pk.order() == ["a", "b", "c"]

    def test_backward_edge_triggers_reorder(self):
        pk = PearceKellyOrder(["a", "b", "c"])
        assert pk.try_add_edge("c", "a") is None
        assert pk.position("c") < pk.position("a")

    def test_edge_that_would_close_a_cycle_is_rejected(self):
        pk = PearceKellyOrder()
        assert pk.try_add_edge("a", "b") is None
        assert pk.try_add_edge("b", "c") is None
        cycle = pk.try_add_edge("c", "a")
        assert cycle == ["c", "a", "b", "c"]
        # rejected edge must not have mutated the structure
        assert pk.order() == ["a", "b", "c"]

    def test_self_loop_rejected(self):
        pk = PearceKellyOrder(["a"])
        assert pk.try_add_edge("a", "a") == ["a", "a"]

    def test_duplicate_edge_is_a_no_op_success(self):
        pk = PearceKellyOrder()
        assert pk.try_add_edge("a", "b") is None
        before = pk.order()
        assert pk.try_add_edge("a", "b") is None
        assert pk.order() == before

    def test_auto_adds_new_vertices(self):
        pk = PearceKellyOrder()
        pk.try_add_edge("x", "y")
        assert set(pk.order()) == {"x", "y"}

    @pytest.mark.parametrize("seed", range(60))
    def test_matches_dfs_topological_order_after_every_insertion(self, seed):
        rng = random.Random(seed + 2000)
        n = rng.randint(1, 10)
        nodes = list(range(n))
        candidate_edges = [(u, v) for u in nodes for v in nodes if u != v]
        rng.shuffle(candidate_edges)
        candidate_edges = candidate_edges[: rng.randint(0, len(candidate_edges))]

        pk = PearceKellyOrder(nodes)
        applied: dict[int, list[int]] = {n: [] for n in nodes}

        for u, v in candidate_edges:
            cycle = pk.try_add_edge(u, v)
            trial_graph = {k: list(vv) for k, vv in applied.items()}
            trial_graph[u].append(v)
            _, real_cycle = dfs_topological_order(trial_graph)

            if cycle is None:
                assert real_cycle is None, (applied, u, v)
                applied[u].append(v)
                assert _valid_topo_order(applied, pk.order())
            else:
                assert real_cycle is not None, (applied, u, v)
                assert _valid_cycle(trial_graph, cycle)

        assert _valid_topo_order(applied, pk.order())
