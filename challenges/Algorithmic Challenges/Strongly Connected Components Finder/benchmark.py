"""Tarjan vs Kosaraju vs Gabow vs networkx vs scipy, on growing sparse digraphs.

Four pure-Python single-DFS-pass algorithms (three of them ours, one
networkx's) against scipy.sparse.csgraph.connected_components, which is a
compiled (Cython/C) Tarjan variant -- the realistic "how fast could this go"
ceiling for the same algorithm family.

    uv run python benchmark.py
"""

from __future__ import annotations

import time
from collections.abc import Callable

import networkx as nx
import numpy as np
from scc import gabow_scc, kosaraju_scc, random_sparse_digraph, tarjan_scc
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as scipy_scc


def _time_once(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def _networkx_scc(graph: dict[int, list[int]]) -> list:
    g = nx.DiGraph()
    g.add_nodes_from(graph)
    g.add_edges_from((u, v) for u, succs in graph.items() for v in succs)
    return list(nx.strongly_connected_components(g))


def _scipy_scc(graph: dict[int, list[int]], n: int) -> int:
    rows, cols = [], []
    for u, succs in graph.items():
        for v in succs:
            rows.append(u)
            cols.append(v)
    data = np.ones(len(rows), dtype=np.int8)
    mat = csr_matrix((data, (rows, cols)), shape=(n, n))
    n_components, _ = scipy_scc(mat, directed=True, connection="strong")
    return n_components


def _bench_pure_python_vs_scipy() -> None:
    print("Pure-Python Tarjan/Kosaraju/Gabow/networkx vs scipy (compiled) SCC")
    print(
        f"{'n':>10}{'edges':>10}{'tarjan':>10}{'kosaraju':>10}"
        f"{'gabow':>10}{'networkx':>10}{'scipy':>10}{'scipy speedup':>16}"
    )
    print("-" * 96)
    for n in (5_000, 20_000, 80_000, 300_000):
        graph = random_sparse_digraph(n, avg_out_degree=2.5, seed=0)
        edges = sum(len(v) for v in graph.values())

        t_tarjan = _time_once(lambda g=graph: tarjan_scc(g))
        t_kosaraju = _time_once(lambda g=graph: kosaraju_scc(g))
        t_gabow = _time_once(lambda g=graph: gabow_scc(g))
        t_nx = _time_once(lambda g=graph: _networkx_scc(g))
        t_scipy = _time_once(lambda g=graph, n=n: _scipy_scc(g, n))

        fastest_python = min(t_tarjan, t_kosaraju, t_gabow, t_nx)
        speedup = fastest_python / t_scipy if t_scipy else float("inf")
        print(
            f"{n:>10,}{edges:>10,}{t_tarjan:9.3f}s{t_kosaraju:9.3f}s"
            f"{t_gabow:9.3f}s{t_nx:9.3f}s{t_scipy:9.4f}s{speedup:15.1f}x"
        )
    print()
    print(
        "scipy's connected_components is a compiled Tarjan variant with no "
        "Python-object or dict overhead per edge -- the gap here is the real "
        "cost of doing graph algorithms in pure Python at all, not a flaw in "
        "any of the three algorithms above it. Their relative order to each "
        "other is what actually tests this challenge's brief."
    )


def _bench_recursion_limit_cost() -> None:
    import sys

    print()
    print("Recursive Tarjan: fine until it isn't (RecursionError, not just slow)")
    limit = sys.getrecursionlimit()
    print(f"{'path length':>14}{'iterative tarjan':>20}{'recursive tarjan':>20}")
    print("-" * 54)
    from scc import tarjan_scc_recursive

    for path_len in (limit // 4, limit // 2, limit - 50):
        graph = {i: [i + 1] for i in range(path_len - 1)}
        graph[path_len - 1] = []
        t_iter = _time_once(lambda g=graph: tarjan_scc(g))
        try:
            t_rec = _time_once(lambda g=graph: tarjan_scc_recursive(g))
            rec_str = f"{t_rec:19.4f}s"
        except RecursionError:
            rec_str = "RecursionError".rjust(20)
        print(f"{path_len:>14,}{t_iter:19.4f}s{rec_str}")
    print(f"(sys.getrecursionlimit() == {limit} on this interpreter)")


def main() -> None:
    _bench_pure_python_vs_scipy()
    _bench_recursion_limit_cost()


if __name__ == "__main__":
    main()
