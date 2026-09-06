"""Minimum spanning tree/forest via Kruskal, Prim, and Boruvka.

All three are the same theorem wearing different clothes: on the graphic
matroid of a graph, the greedy algorithm computes a minimum-weight basis
(Rado-Edmonds). Kruskal is that greedy algorithm run globally by weight;
Prim is the same greedy choice restricted to one growing component at a
time; Boruvka runs it in parallel across every component at once. See
README.md `## Correctness` for the cut-property / cycle-property argument
and the matroid framing.

Every algorithm below is a generator that yields a `Step` after each edge
decision, so `visualize.py` can drive a Manim scene off the exact same
function used for solving -- no second, animation-only copy of the logic.
None of them assume non-negative weights, a connected graph, no self-loops,
or no parallel edges: see the "Edge cases" section of README.md and
test_mst.py for what each of those degeneracies does and why it is still
handled correctly without special-casing.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

Edge = tuple[int, int, float]


class DisjointSet:
    """Union-find with path compression and union by rank.

    Amortized O(alpha(n)) per operation (alpha = inverse Ackermann, <= 4 for
    any n representable in this universe) -- Tarjan & van Leeuwen, 1984.
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


@dataclass
class Step:
    """One edge decision in the algorithm's execution.

    `edge` is None only for a Boruvka "round boundary" marker (not currently
    emitted, kept for forward compatibility with the visualizer). `accepted`
    is True/False for a real decision. `mst_edges`/`total_weight` are the
    cumulative forest built so far, so the last yielded Step *is* the answer
    -- `solve()` below just takes it.
    """

    edge: Edge | None
    accepted: bool | None
    mst_edges: tuple[Edge, ...]
    total_weight: float
    caption: str = ""


def solve(algorithm, n: int, edges: list[Edge]) -> tuple[tuple[Edge, ...], float]:
    """Drain a Step generator down to its final (mst_edges, total_weight)."""
    last: Step | None = None
    for step in algorithm(n, edges):
        last = step
    if last is None:
        return (), 0.0
    return last.mst_edges, last.total_weight


def kruskal_steps(n: int, edges: list[Edge]):
    """Global greedy: sort all edges, take each if it doesn't close a cycle.

    O(E log E) for the sort, O(E alpha(n)) for the union-find calls -- sort
    dominates. Self-loops are filtered before sorting since `u == v` can
    never merge two components. Parallel edges are handled for free: the
    first (cheapest, since sorted) one between a pair unions the components,
    so every later duplicate is rejected by the `find(u) != find(v)` check
    without any pre-processing. A disconnected graph is not a special case
    either -- the loop simply runs out of edges before every vertex is
    unioned, leaving a forest with `n - num_components` edges.
    """
    dsu = DisjointSet(n)
    mst: list[Edge] = []
    total = 0.0
    real_edges = [e for e in edges if e[0] != e[1]]
    ordered = sorted(enumerate(real_edges), key=lambda item: (item[1][2], item[0]))
    for _, (u, v, w) in ordered:
        if dsu.union(u, v):
            mst.append((u, v, w))
            total += w
            yield Step((u, v, w), True, tuple(mst), total, f"Accept {u}-{v} (w={w})")
        else:
            yield Step(
                (u, v, w),
                False,
                tuple(mst),
                total,
                f"Reject {u}-{v} (would close a cycle)",
            )


def prim_steps(n: int, edges: list[Edge]):
    """Grow one component at a time via a binary min-heap of frontier edges.

    O(E log V) with `heapq` (a Fibonacci heap gets O(E + V log V) --
    Fredman & Tarjan 1987 -- Python has no stdlib Fibonacci heap, and the
    constant-factor overhead of one rarely wins in practice at the sizes
    Python is used for anyway). Restarting from every unvisited vertex once
    its component is exhausted turns this into Prim-per-component, i.e. a
    minimum spanning *forest* on a disconnected graph, for free.
    """
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        if u == v:
            continue
        adj[u].append((v, w))
        adj[v].append((u, w))

    visited = [False] * n
    mst: list[Edge] = []
    total = 0.0
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        heap: list[tuple[float, int, int]] = []
        for v, w in adj[start]:
            heapq.heappush(heap, (w, start, v))
        while heap:
            w, u, v = heapq.heappop(heap)
            if visited[v]:
                yield Step(
                    (u, v, w),
                    False,
                    tuple(mst),
                    total,
                    f"Reject {u}-{v} (both endpoints in tree)",
                )
                continue
            visited[v] = True
            mst.append((u, v, w))
            total += w
            yield Step((u, v, w), True, tuple(mst), total, f"Accept {u}-{v} (w={w})")
            for v2, w2 in adj[v]:
                if not visited[v2]:
                    heapq.heappush(heap, (w2, v, v2))


def boruvka_steps(n: int, edges: list[Edge]):
    """Every component grabs its own cheapest outgoing edge, in parallel rounds.

    The oldest MST algorithm (Otakar Boruvka, 1926) and the ancestor both
    Kruskal (1956) and Prim (1957, independently Jarnik 1930) specialize:
    Kruskal is "one global sorted pass" and Prim is "one component grows",
    while Boruvka merges *every* component simultaneously each round. Each
    round at least halves the number of components (every component finds a
    distinct partner or merges into a bigger one), so it terminates in
    O(log V) rounds of O(E) work: O(E log V) total, and the round structure
    is what makes it the basis of parallel/distributed MST implementations
    (each processor can own a component and search its edges independently).
    """
    dsu = DisjointSet(n)
    mst: list[Edge] = []
    total = 0.0
    real_edges = [(idx, u, v, w) for idx, (u, v, w) in enumerate(edges) if u != v]

    while True:
        # cheapest[component_root] = (weight, tie_break_idx, u, v)
        cheapest: dict[int, tuple[float, int, int, int]] = {}
        for idx, u, v, w in real_edges:
            ru, rv = dsu.find(u), dsu.find(v)
            if ru == rv:
                continue
            candidate = (w, idx, u, v)
            for r in (ru, rv):
                if r not in cheapest or candidate < cheapest[r]:
                    cheapest[r] = candidate

        if not cheapest:
            break  # no edges leave any remaining component: done (or disconnected)

        merged_pairs: set[frozenset[int]] = set()
        progressed = False
        for w, idx, u, v in cheapest.values():
            ru, rv = dsu.find(u), dsu.find(v)
            if ru == rv:
                continue
            pair = frozenset((ru, rv))
            if pair in merged_pairs:
                yield Step(
                    (u, v, w),
                    False,
                    tuple(mst),
                    total,
                    f"Skip {u}-{v} (components already merged this round)",
                )
                continue
            merged_pairs.add(pair)
            dsu.union(u, v)
            mst.append((u, v, w))
            total += w
            progressed = True
            yield Step(
                (u, v, w),
                True,
                tuple(mst),
                total,
                f"Accept {u}-{v} (round merge, w={w})",
            )

        if not progressed:
            break


ALGORITHMS = {"kruskal": kruskal_steps, "prim": prim_steps, "boruvka": boruvka_steps}


def _random_graph(n: int, density: float, seed: int, weight_range=(1, 1000)):
    import random

    rng = random.Random(seed)
    return [
        (u, v, rng.randint(*weight_range))
        for u in range(n)
        for v in range(u + 1, n)
        if rng.random() < density
    ]


def _demo() -> None:
    n = 6
    edges = [
        (0, 1, 4), (0, 2, 4), (1, 2, 2), (1, 3, 5),
        (2, 3, 8), (2, 4, 10), (3, 4, 2), (3, 5, 6), (4, 5, 3),
    ]  # fmt: skip
    for name, algo in ALGORITHMS.items():
        mst_edges, total = solve(algo, n, edges)
        print(
            f"{name:8s} total={total:6.1f} edges={sorted((u, v) for u, v, _ in mst_edges)}"
        )


def _verify(trials: int = 200, max_n: int = 40) -> None:
    import random

    rng = random.Random(0)
    mismatches = 0
    for _ in range(trials):
        n = rng.randint(0, max_n)
        density = rng.random()
        edges = _random_graph(n, density, seed=rng.randrange(1 << 30))
        totals = {name: solve(algo, n, edges)[1] for name, algo in ALGORITHMS.items()}
        if len({round(v, 6) for v in totals.values()}) != 1:
            mismatches += 1
            print("MISMATCH", n, density, totals)
    print(f"{trials - mismatches}/{trials} trials agree across kruskal/prim/boruvka")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        _verify()
    else:
        _demo()
