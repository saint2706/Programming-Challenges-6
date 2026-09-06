"""Correctness tests for mst.py.

Cross-checks Kruskal, Prim, and Boruvka against each other and, for small
graphs, against a brute-force minimum spanning forest computed by enumerating
every acyclic edge subset. Also pins down the edge cases the brief and the
design conversation called out: disconnected graphs (forest, not a single
tree), negative weights, duplicate weights (tie-invariant total), self-loops,
and parallel edges.
"""

from __future__ import annotations

import itertools
import random

import pytest
from mst import DisjointSet, boruvka_steps, kruskal_steps, prim_steps, solve

ALGORITHMS = {"kruskal": kruskal_steps, "prim": prim_steps, "boruvka": boruvka_steps}


# ---------------------------------------------------------------------------
# DisjointSet
# ---------------------------------------------------------------------------


def test_disjoint_set_starts_all_singletons():
    dsu = DisjointSet(5)
    for i in range(5):
        assert dsu.find(i) == i


def test_disjoint_set_union_merges_components():
    dsu = DisjointSet(4)
    assert dsu.union(0, 1) is True
    assert dsu.find(0) == dsu.find(1)
    assert dsu.find(2) != dsu.find(0)


def test_disjoint_set_union_already_connected_returns_false():
    dsu = DisjointSet(3)
    dsu.union(0, 1)
    assert dsu.union(0, 1) is False


def test_disjoint_set_path_compression_flattens_chain():
    dsu = DisjointSet(5)
    # Build a chain 0<-1<-2<-3<-4 by unioning sequentially without letting
    # union-by-rank flatten it on its own (force root always parent[0]).
    for i in range(4):
        dsu.union(i, i + 1)
    root = dsu.find(0)
    for i in range(5):
        assert dsu.parent[i] == root


# ---------------------------------------------------------------------------
# Brute force reference
# ---------------------------------------------------------------------------


def _brute_force_min_forest_weight(
    n: int, edges: list[tuple[int, int, float]]
) -> float:
    """Minimum spanning forest weight via exhaustive search over edge subsets.

    A forest is only a *spanning* forest if it fully connects each connected
    component of the underlying (unweighted) graph -- an acyclic subset with
    fewer edges than that is not a valid answer even if it happens to weigh
    less (this matters once weights can be negative: leaving a component only
    partially connected is not "optimal", it's a different problem). The
    number of edges in any spanning forest is fixed at `n - c` where `c` is
    the underlying graph's component count, so only combinations of exactly
    that size are considered.

    Only safe for tiny graphs (len(edges) <= ~18) -- used purely as a ground
    truth oracle in tests, never in the real algorithms.
    """
    real_edges = [e for e in edges if e[0] != e[1]]
    connectivity = DisjointSet(n)
    for u, v, _ in real_edges:
        connectivity.union(u, v)
    actual_components = len({connectivity.find(i) for i in range(n)})
    target_edge_count = n - actual_components

    best = None
    m = len(real_edges)
    for combo in itertools.combinations(range(m), target_edge_count):
        dsu = DisjointSet(n)
        weight = 0.0
        ok = True
        for idx in combo:
            u, v, w = real_edges[idx]
            if not dsu.union(u, v):
                ok = False
                break
            weight += w
        if not ok:
            continue
        if best is None or weight < best:
            best = weight
    return 0.0 if best is None else best


@pytest.mark.parametrize("algo_name", ["kruskal", "prim", "boruvka"])
def test_matches_brute_force_on_random_small_graphs(algo_name):
    rng = random.Random(42)
    algo = ALGORITHMS[algo_name]
    for _ in range(30):
        n = rng.randint(1, 6)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < 0.5:
                    edges.append((u, v, rng.randint(-5, 20)))
        _mst_edges, total = solve(algo, n, edges)
        expected = _brute_force_min_forest_weight(n, edges)
        assert total == pytest.approx(expected), (n, edges, algo_name)


# ---------------------------------------------------------------------------
# Cross-algorithm agreement
# ---------------------------------------------------------------------------


def test_all_three_algorithms_agree_on_total_weight():
    rng = random.Random(7)
    for _ in range(20):
        n = rng.randint(2, 15)
        edges = [
            (u, v, rng.randint(1, 100))
            for u in range(n)
            for v in range(u + 1, n)
            if rng.random() < 0.4
        ]
        totals = {name: solve(algo, n, edges)[1] for name, algo in ALGORITHMS.items()}
        values = list(totals.values())
        assert all(v == pytest.approx(values[0]) for v in values), totals


def test_distinct_weights_give_identical_edge_sets_across_algorithms():
    # With all-distinct weights the MST is unique (cut-property exchange
    # argument), so every correct algorithm must return the same edge set,
    # not merely the same total weight.
    n = 6
    edges = [
        (0, 1, 1), (0, 2, 7), (1, 2, 3), (1, 3, 4),
        (2, 3, 2), (2, 4, 6), (3, 4, 5), (3, 5, 8), (4, 5, 9),
    ]  # fmt: skip
    edge_sets = []
    for algo in ALGORITHMS.values():
        mst_edges, _ = solve(algo, n, edges)
        normalized = frozenset(frozenset((u, v)) for u, v, _ in mst_edges)
        edge_sets.append(normalized)
    assert all(s == edge_sets[0] for s in edge_sets)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo_name", ["kruskal", "prim", "boruvka"])
def test_empty_graph(algo_name):
    mst_edges, total = solve(ALGORITHMS[algo_name], 0, [])
    assert mst_edges == ()
    assert total == 0.0


@pytest.mark.parametrize("algo_name", ["kruskal", "prim", "boruvka"])
def test_single_vertex_no_edges(algo_name):
    mst_edges, total = solve(ALGORITHMS[algo_name], 1, [])
    assert mst_edges == ()
    assert total == 0.0


@pytest.mark.parametrize("algo_name", ["kruskal", "prim", "boruvka"])
def test_disconnected_graph_yields_forest_not_single_tree(algo_name):
    # Two disjoint triangles: {0,1,2} and {3,4,5}.
    n = 6
    edges = [(0, 1, 1), (1, 2, 1), (0, 2, 1), (3, 4, 1), (4, 5, 1), (3, 5, 1)]
    mst_edges, total = solve(ALGORITHMS[algo_name], n, edges)
    # A spanning forest over 2 components on 6 vertices has n - c = 4 edges.
    assert len(mst_edges) == 4
    assert total == pytest.approx(4.0)
    dsu = DisjointSet(n)
    for u, v, _ in mst_edges:
        dsu.union(u, v)
    assert dsu.find(0) == dsu.find(1) == dsu.find(2)
    assert dsu.find(3) == dsu.find(4) == dsu.find(5)
    assert dsu.find(0) != dsu.find(3)


@pytest.mark.parametrize("algo_name", ["kruskal", "prim", "boruvka"])
def test_negative_weights_handled_correctly(algo_name):
    # MST algorithms never assume non-negative weights (unlike Dijkstra) --
    # the cut/cycle properties hold regardless of sign.
    n = 3
    edges = [(0, 1, -10), (1, 2, -5), (0, 2, 100)]
    mst_edges, total = solve(ALGORITHMS[algo_name], n, edges)
    assert total == pytest.approx(-15.0)
    normalized = frozenset(frozenset((u, v)) for u, v, _ in mst_edges)
    assert normalized == frozenset({frozenset((0, 1)), frozenset((1, 2))})


@pytest.mark.parametrize("algo_name", ["kruskal", "prim", "boruvka"])
def test_self_loops_are_ignored(algo_name):
    n = 3
    edges = [(0, 0, -999), (0, 1, 1), (1, 2, 1), (2, 2, -999), (1, 1, 5)]
    mst_edges, total = solve(ALGORITHMS[algo_name], n, edges)
    assert total == pytest.approx(2.0)
    assert all(u != v for u, v, _ in mst_edges)


@pytest.mark.parametrize("algo_name", ["kruskal", "prim", "boruvka"])
def test_parallel_edges_only_cheapest_is_ever_usable(algo_name):
    n = 2
    edges = [(0, 1, 50), (0, 1, 3), (1, 0, 40)]
    mst_edges, total = solve(ALGORITHMS[algo_name], n, edges)
    assert total == pytest.approx(3.0)
    assert len(mst_edges) == 1


def test_duplicate_weights_total_is_tie_invariant_even_if_edge_choice_differs():
    # A 4-cycle with all weights equal to 1: three different valid MSTs exist
    # (drop any one edge), but every one has total weight 3. This is the
    # cycle-property statement that tie-breaking never changes total weight.
    n = 4
    edges = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)]
    for algo in ALGORITHMS.values():
        mst_edges, total = solve(algo, n, edges)
        assert total == pytest.approx(3.0)
        assert len(mst_edges) == 3


@pytest.mark.parametrize("algo_name", ["kruskal", "prim", "boruvka"])
def test_complete_graph_stress(algo_name):
    rng = random.Random(99)
    n = 60
    edges = [(u, v, rng.randint(1, 10_000)) for u in range(n) for v in range(u + 1, n)]
    mst_edges, total = solve(ALGORITHMS[algo_name], n, edges)
    assert len(mst_edges) == n - 1
    dsu = DisjointSet(n)
    for u, v, _ in mst_edges:
        assert dsu.union(u, v)  # every accepted edge must merge two components
    assert len({dsu.find(i) for i in range(n)}) == 1
    assert total > 0


def test_step_stream_tracks_running_total_and_final_matches_solve():
    n = 4
    edges = [(0, 1, 1), (1, 2, 2), (2, 3, 3), (0, 3, 10)]
    last = None
    running = 0.0
    for step in kruskal_steps(n, edges):
        if step.accepted:
            running += step.edge[2]
        assert step.total_weight == pytest.approx(running)
        last = step
    mst_edges, total = solve(kruskal_steps, n, edges)
    assert last.total_weight == pytest.approx(total)
    assert last.mst_edges == mst_edges
