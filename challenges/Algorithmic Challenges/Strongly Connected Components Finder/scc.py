"""Strongly connected components: Tarjan, Kosaraju, and Gabow, iteratively.

A strongly connected component (SCC) of a directed graph is a maximal set of
vertices where every vertex can reach every other vertex along directed edges.
All three algorithms here are single-pass DFS, O(V + E) time and O(V) extra
space, and return the same partition of vertices -- they differ only in the
bookkeeping used to detect a component's root as the DFS unwinds.

Every implementation is iterative (an explicit stack standing in for the call
stack), not recursive. See the README's "why iterative" section: a recursive
DFS blows Python's ~1000-frame recursion limit on anything but a small graph
whenever the graph contains a long path, which sparse random graphs and
real dependency graphs both do.

Graphs are adjacency dicts: ``{node: [successor, ...]}``. Nodes referenced
only as a successor need not have their own key; :func:`_normalize` fills
those in with an empty successor list.
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Hashable, Iterable


def _normalize(graph: dict[Hashable, list[Hashable]]) -> dict[Hashable, list[Hashable]]:
    """Ensure every referenced node has an adjacency entry (default: no out-edges)."""
    normalized: dict[Hashable, list[Hashable]] = {
        node: list(succs) for node, succs in graph.items()
    }
    for succs in graph.values():
        for s in succs:
            normalized.setdefault(s, [])
    return normalized


def tarjan_scc(graph: dict[Hashable, list[Hashable]]) -> list[list[Hashable]]:
    """Tarjan (1972): one DFS, a node stack, and a low-link value per vertex.

    ``lowlink[v]`` is the smallest discovery index reachable from v via a path
    that uses at most one back/cross edge into the current DFS tree's stack.
    A vertex is a component root exactly when ``lowlink[v] == index[v]`` --
    nothing on the stack above it can reach back past it, so everything still
    above it on the stack is exactly its component.
    """
    graph = _normalize(graph)
    counter = itertools.count()
    index: dict[Hashable, int] = {}
    lowlink: dict[Hashable, int] = {}
    on_stack: dict[Hashable, bool] = {}
    node_stack: list[Hashable] = []
    components: list[list[Hashable]] = []

    for root in graph:
        if root in index:
            continue
        index[root] = lowlink[root] = next(counter)
        node_stack.append(root)
        on_stack[root] = True
        call_stack: list[tuple[Hashable, Iterable[Hashable]]] = [
            (root, iter(graph[root]))
        ]

        while call_stack:
            v, it = call_stack[-1]
            recursed = False
            for w in it:
                if w not in index:
                    index[w] = lowlink[w] = next(counter)
                    node_stack.append(w)
                    on_stack[w] = True
                    call_stack.append((w, iter(graph[w])))
                    recursed = True
                    break
                if on_stack.get(w, False):
                    lowlink[v] = min(lowlink[v], index[w])
            if recursed:
                continue

            call_stack.pop()
            if call_stack:
                parent = call_stack[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[v])
            if lowlink[v] == index[v]:
                component = []
                while True:
                    w = node_stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == v:
                        break
                components.append(component)

    return components


def kosaraju_scc(graph: dict[Hashable, list[Hashable]]) -> list[list[Hashable]]:
    """Kosaraju/Sharir: DFS finish order, transpose the graph, DFS again in reverse order.

    A component's finish order in the *original* graph guarantees that a DFS
    over the *transpose*, run in decreasing finish-time order, can never leak
    from one component into a component that was finished earlier -- an edge
    out of the current component in the transpose would need an edge into it
    in the original graph, which would have forced that other component to
    finish first. Two full DFS passes, no low-link bookkeeping at all.
    """
    graph = _normalize(graph)

    visited: set[Hashable] = set()
    finish_order: list[Hashable] = []
    for root in graph:
        if root in visited:
            continue
        visited.add(root)
        stack: list[tuple[Hashable, Iterable[Hashable]]] = [(root, iter(graph[root]))]
        while stack:
            v, it = stack[-1]
            recursed = False
            for w in it:
                if w not in visited:
                    visited.add(w)
                    stack.append((w, iter(graph[w])))
                    recursed = True
                    break
            if not recursed:
                stack.pop()
                finish_order.append(v)

    transpose: dict[Hashable, list[Hashable]] = {node: [] for node in graph}
    for node, succs in graph.items():
        for w in succs:
            transpose[w].append(node)

    visited.clear()
    components: list[list[Hashable]] = []
    for node in reversed(finish_order):
        if node in visited:
            continue
        visited.add(node)
        component = [node]
        stack = [node]
        while stack:
            v = stack.pop()
            for w in transpose[v]:
                if w not in visited:
                    visited.add(w)
                    component.append(w)
                    stack.append(w)
        components.append(component)

    return components


def gabow_scc(graph: dict[Hashable, list[Hashable]]) -> list[list[Hashable]]:
    """Gabow (2000): path-based, two stacks, no low-link arithmetic at all.

    ``S`` holds every visited-but-unassigned vertex, in DFS visitation order.
    ``P`` holds *candidate roots*: a shrinking subsequence of S such that
    whenever a back edge to an already-on-S vertex w is found, every
    candidate discovered after w is popped off P (they're now known to share
    w's component, since a cycle back to w merges everything discovered since
    w into one component). When a vertex is popped off the DFS stack and it
    still sits on top of P, no later edge can ever pull it into a bigger
    component -- it *is* a root, and everything above it on S is its
    component. Popping P by preorder number rather than maintaining a
    numeric low-link is the entire trick.
    """
    graph = _normalize(graph)
    counter = itertools.count()
    preorder: dict[Hashable, int] = {}
    assigned: set[Hashable] = set()
    S: list[Hashable] = []
    P: list[Hashable] = []
    components: list[list[Hashable]] = []

    for root in graph:
        if root in preorder:
            continue
        preorder[root] = next(counter)
        S.append(root)
        P.append(root)
        call_stack: list[tuple[Hashable, Iterable[Hashable]]] = [
            (root, iter(graph[root]))
        ]

        while call_stack:
            v, it = call_stack[-1]
            recursed = False
            for w in it:
                if w not in preorder:
                    preorder[w] = next(counter)
                    S.append(w)
                    P.append(w)
                    call_stack.append((w, iter(graph[w])))
                    recursed = True
                    break
                if w not in assigned:
                    while preorder[P[-1]] > preorder[w]:
                        P.pop()
            if recursed:
                continue

            call_stack.pop()
            if P[-1] is v:
                P.pop()
                component = []
                while True:
                    w = S.pop()
                    assigned.add(w)
                    component.append(w)
                    if w is v:
                        break
                components.append(component)

    return components


def tarjan_scc_recursive(graph: dict[Hashable, list[Hashable]]) -> list[list[Hashable]]:
    """Textbook recursive Tarjan -- for the README's recursion-limit demo only.

    Not exported as a "real" option: it raises ``RecursionError`` on any graph
    with a directed path longer than ``sys.getrecursionlimit()`` (~1000 by
    default), which a random sparse graph or a real build/import dependency
    graph hits at a few thousand nodes, let alone the sizes benchmarked here.
    """
    graph = _normalize(graph)
    counter = itertools.count()
    index: dict[Hashable, int] = {}
    lowlink: dict[Hashable, int] = {}
    on_stack: set[Hashable] = set()
    node_stack: list[Hashable] = []
    components: list[list[Hashable]] = []

    def strongconnect(v: Hashable) -> None:
        index[v] = lowlink[v] = next(counter)
        node_stack.append(v)
        on_stack.add(v)
        for w in graph[v]:
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            component = []
            while True:
                w = node_stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            components.append(component)

    for root in graph:
        if root not in index:
            strongconnect(root)
    return components


def condensation(
    graph: dict[Hashable, list[Hashable]], components: list[list[Hashable]]
) -> dict[int, set[int]]:
    """Collapse each SCC to a single node -- the result is always a DAG."""
    graph = _normalize(graph)
    owner: dict[Hashable, int] = {}
    for i, comp in enumerate(components):
        for node in comp:
            owner[node] = i
    dag: dict[int, set[int]] = {i: set() for i in range(len(components))}
    for node, succs in graph.items():
        for w in succs:
            if owner[node] != owner[w]:
                dag[owner[node]].add(owner[w])
    return dag


def _partition_key(components: list[list[Hashable]]) -> frozenset[frozenset[Hashable]]:
    return frozenset(frozenset(c) for c in components)


def random_sparse_digraph(
    n: int, avg_out_degree: float, seed: int | None = None
) -> dict[int, list[int]]:
    """A random directed graph with n nodes and ~n*avg_out_degree edges (numpy-backed)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    m = int(n * avg_out_degree)
    sources = rng.integers(0, n, size=m)
    targets = rng.integers(0, n, size=m)
    graph: dict[int, list[int]] = {i: [] for i in range(n)}
    for s, t in zip(sources.tolist(), targets.tolist(), strict=True):
        if s != t:
            graph[s].append(t)
    return graph


def _demo() -> None:
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
    print("Classic textbook graph (Cormen et al., CLRS 3rd ed., Fig. 22.9 relabeled):")
    for name, fn in (
        ("tarjan", tarjan_scc),
        ("kosaraju", kosaraju_scc),
        ("gabow", gabow_scc),
    ):
        comps = fn(graph)
        print(f"  {name:10s}: {comps}")
    print()
    print("Condensation DAG of the Tarjan result:")
    comps = tarjan_scc(graph)
    print(f"  components: {comps}")
    print(f"  DAG:        {condensation(graph, comps)}")


def _verify(n: int = 20_000, trials: int = 25, seed: int = 0) -> None:
    import networkx as nx

    print(
        f"Cross-verifying tarjan/kosaraju/gabow/networkx agree on {trials} random graphs..."
    )
    rng_seed = seed
    for i in range(trials):
        size = 5 + (i % 40)
        graph = random_sparse_digraph(size, avg_out_degree=1.5, seed=rng_seed + i)
        keys = {
            "tarjan": _partition_key(tarjan_scc(graph)),
            "kosaraju": _partition_key(kosaraju_scc(graph)),
            "gabow": _partition_key(gabow_scc(graph)),
            "networkx": _partition_key(
                [
                    list(c)
                    for c in nx.strongly_connected_components(nx.DiGraph(graph).copy())
                ]
            ),
        }
        # nodes with no edges at all don't appear as nx.DiGraph nodes unless added;
        # rebuild networkx graph with explicit node set to be fair.
        g = nx.DiGraph()
        g.add_nodes_from(graph)
        g.add_edges_from((u, v) for u, succs in graph.items() for v in succs)
        keys["networkx"] = _partition_key(
            [list(c) for c in nx.strongly_connected_components(g)]
        )
        reference = keys["tarjan"]
        for name, key in keys.items():
            if key != reference:
                print(f"  MISMATCH at trial {i} ({name} vs tarjan) on graph {graph}")
                return
    print(f"PASS  all {trials} trials agree across tarjan/kosaraju/gabow/networkx")

    print()
    print("Recursion-limit demo: recursive Tarjan on a long directed path")
    limit = sys.getrecursionlimit()
    path_len = limit + 500
    path_graph = {i: [i + 1] for i in range(path_len - 1)}
    path_graph[path_len - 1] = []
    try:
        tarjan_scc_recursive(path_graph)
        print(f"  recursive Tarjan survived a path of {path_len} nodes (limit={limit})")
    except RecursionError:
        print(
            f"  RecursionError on a path of {path_len} nodes (limit={limit}), as expected"
        )
    comps = tarjan_scc(path_graph)
    print(
        f"  iterative Tarjan handled the same graph fine: {len(comps)} components (all size 1)"
    )


if __name__ == "__main__":
    if "--verify" in sys.argv:
        _verify()
    else:
        _demo()
