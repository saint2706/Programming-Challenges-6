"""Exact and approximate TSP: Held-Karp DP, branch and bound, and four heuristics.

The traveling salesman problem: given an n x n distance matrix, find the
shortest cycle visiting every city exactly once. NP-hard; the two exact
methods below are only practical for small n, in exchange for a provable
optimum -- everything else trades that guarantee for polynomial time.

| Method              | Time             | Space   | Optimal?                      |
| -------------------- | ---------------- | ------- | ------------------------------ |
| `held_karp`          | O(2^n * n^2)     | O(2^n * n) | Always                      |
| `branch_and_bound`   | O(2^n) worst case | O(n) per branch | Always (often far faster in practice) |
| `nearest_neighbor`   | O(n^2)           | O(n)    | No guarantee                  |
| `greedy_edge`        | O(n^2 log n)     | O(n)    | No guarantee                  |
| `two_opt`            | O(n^2) per pass  | O(n)    | Local optimum only            |
| `christofides` (networkx) | O(n^3)     | O(n^2)  | <= 1.5x optimal (metric only) |

`dist` is a plain ``list[list[float]]`` distance matrix; `christofides`
additionally requires the triangle inequality to hold (a metric instance --
:func:`euclidean_instance` always satisfies this) for its 1.5x guarantee to
mean anything.
"""

from __future__ import annotations

import itertools
import math
import random


def tour_length(dist: list[list[float]], tour: list[int]) -> float:
    n = len(tour)
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def held_karp(dist: list[list[float]]) -> tuple[float, list[int]]:
    """Held & Karp (1962/1970): bitmask DP over subsets of cities, fixing city 0 as start.

    ``C[(subset, k)]`` = cheapest way to start at city 0, visit exactly the
    cities in `subset` (a bitmask over cities 1..n-1), and end at city `k` in
    `subset`. Built up by subset size so every smaller sub-solution needed by
    ``C[(subset, k)] = min over m in subset\\{k} of C[(subset\\{k}, m)] + dist[m][k]``
    is already computed. The city-0 fix and the ``2^(n-1)`` (not ``2^n``)
    subset space are both the standard trick: city 0's position is fixed by
    the problem being a *cycle* (any tour can be rotated to start there), so
    it never needs its own bit.
    """
    n = len(dist)
    if n == 0:
        return 0.0, []
    if n == 1:
        return 0.0, [0]
    if n == 2:
        return dist[0][1] + dist[1][0], [0, 1]

    # C[(bits, k)] = (min cost, predecessor city) -- bits indexes cities 1..n-1
    # via bit (city - 1), k is a city in that bitmask (excludes city 0).
    C: dict[tuple[int, int], tuple[float, int]] = {}
    for k in range(1, n):
        C[(1 << (k - 1), k)] = (dist[0][k], 0)

    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = 0
            for city in subset:
                bits |= 1 << (city - 1)
            for k in subset:
                prev_bits = bits & ~(1 << (k - 1))
                best: tuple[float, int] | None = None
                for m in subset:
                    if m == k:
                        continue
                    entry = C.get((prev_bits, m))
                    if entry is None:
                        continue
                    cost = entry[0] + dist[m][k]
                    if best is None or cost < best[0]:
                        best = (cost, m)
                if best is not None:
                    C[(bits, k)] = best

    full_bits = (1 << (n - 1)) - 1
    best_final: tuple[float, int] | None = None
    for k in range(1, n):
        entry = C.get((full_bits, k))
        if entry is None:
            continue
        cost = entry[0] + dist[k][0]
        if best_final is None or cost < best_final[0]:
            best_final = (cost, k)
    assert best_final is not None
    opt_cost, last = best_final

    tour_rev = []
    bits, k = full_bits, last
    for _ in range(n - 1):
        tour_rev.append(k)
        _, prev = C[(bits, k)]
        bits &= ~(1 << (k - 1))
        k = prev
    return opt_cost, [0, *reversed(tour_rev)]


def _mst_weight(dist: list[list[float]], nodes: frozenset[int]) -> float:
    """Prim's algorithm, O(k^2), over the induced subgraph on `nodes`."""
    if len(nodes) <= 1:
        return 0.0
    remaining = set(nodes)
    start = next(iter(remaining))
    remaining.discard(start)
    best_edge = {v: dist[start][v] for v in remaining}
    total = 0.0
    while remaining:
        v = min(remaining, key=lambda x: best_edge[x])
        total += best_edge[v]
        remaining.discard(v)
        for u in remaining:
            best_edge[u] = min(best_edge[u], dist[v][u])
    return total


def branch_and_bound(
    dist: list[list[float]], initial_tour: list[int] | None = None
) -> tuple[float, list[int]]:
    """Best-first branch and bound with an MST-based lower bound.

    For a partial tour ending at city `last` with `remaining` cities still
    unvisited, the true completion cost is at least

        MST(remaining U {last}) + (cheapest edge from any remaining city back to city 0)

    -- any way to visit every remaining city starting from `last` uses a
    connected structure spanning them all (so costs >= their MST), and the
    tour must eventually close back to city 0 through *some* remaining city
    (so costs >= the single cheapest such edge, even though the real
    closing city isn't known yet). Both pieces under-estimate the true
    remaining cost, so the sum is a valid (if not maximally tight) lower
    bound, safe to prune against without ever discarding the optimum.

    Best-first search (a min-heap keyed on lower bound) with this bound
    means the search can `break` entirely the moment a popped state's bound
    already meets or exceeds the best complete tour found so far: since the
    heap always pops its current minimum, every remaining queued state has
    an equal-or-worse bound too.
    """
    import heapq

    n = len(dist)
    if n <= 3:
        return held_karp(dist)

    if initial_tour is None:
        initial_tour = two_opt(dist, nearest_neighbor(dist))
    best_cost = tour_length(dist, initial_tour)
    best_tour = list(initial_tour)

    def lower_bound(path_cost: float, last: int, remaining: frozenset[int]) -> float:
        if not remaining:
            return path_cost + dist[last][0]
        mst = _mst_weight(dist, remaining | {last})
        closing = min(dist[r][0] for r in remaining)
        return path_cost + mst + closing

    counter = itertools.count()
    full_remaining = frozenset(range(1, n))
    start_lb = lower_bound(0.0, 0, full_remaining)
    heap = [(start_lb, next(counter), 0.0, 0, full_remaining, (0,))]

    while heap:
        lb, _, path_cost, last, remaining, path = heapq.heappop(heap)
        if lb >= best_cost:
            break  # best-first invariant: nothing left in the heap can beat this

        if not remaining:
            total = path_cost + dist[last][0]
            if total < best_cost:
                best_cost = total
                best_tour = list(path)
            continue

        for nxt in remaining:
            new_cost = path_cost + dist[last][nxt]
            new_remaining = remaining - {nxt}
            new_lb = lower_bound(new_cost, nxt, new_remaining)
            if new_lb < best_cost:
                heapq.heappush(
                    heap,
                    (new_lb, next(counter), new_cost, nxt, new_remaining, (*path, nxt)),
                )

    return best_cost, best_tour


def nearest_neighbor(dist: list[list[float]], start: int = 0) -> list[int]:
    n = len(dist)
    visited = [False] * n
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        nxt = min(
            (j for j in range(n) if not visited[j]), key=lambda j: dist[current][j]
        )
        visited[nxt] = True
        tour.append(nxt)
        current = nxt
    return tour


def greedy_edge(dist: list[list[float]]) -> list[int]:
    """Bentley's "greedy"/multi-fragment heuristic: cheapest-edge-first, no cycles until the last.

    Repeatedly add the globally cheapest remaining edge whose endpoints both
    still have degree < 2 and that doesn't close a sub-tour -- unless it's
    the n-th edge, which necessarily closes the single remaining Hamiltonian
    path into the full cycle. A union-find over components makes the
    "would this close a premature sub-tour" check O(alpha(n)).
    """
    n = len(dist)
    if n <= 2:
        return list(range(n))

    edges = sorted((dist[i][j], i, j) for i in range(n) for j in range(i + 1, n))
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    degree = [0] * n
    adjacency: list[list[int]] = [[] for _ in range(n)]
    edges_added = 0
    for _, i, j in edges:
        if edges_added == n:
            break
        if degree[i] >= 2 or degree[j] >= 2:
            continue
        ri, rj = find(i), find(j)
        if ri == rj and edges_added < n - 1:
            continue
        if ri != rj:
            parent[ri] = rj
        degree[i] += 1
        degree[j] += 1
        adjacency[i].append(j)
        adjacency[j].append(i)
        edges_added += 1

    tour = [0]
    prev, current = -1, 0
    for _ in range(n - 1):
        a, b = adjacency[current]
        nxt = a if a != prev else b
        tour.append(nxt)
        prev, current = current, nxt
    return tour


def two_opt(dist: list[list[float]], tour: list[int]) -> list[int]:
    """Repeatedly reverse a segment if it shortens the tour, until no such move exists.

    A 2-opt move removes edges (a,b) and (c,d) and reconnects as (a,c) and
    (b,d), reversing everything between b and c -- the only way to
    reconnect two removed edges into a single tour again. Full O(n^2)
    neighborhood scan per pass, repeated to convergence (a local optimum
    under this move set, not a global one).
    """
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            a, b = tour[i], tour[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # would reverse the entire tour: a no-op
                c, d = tour[j], tour[(j + 1) % n]
                if dist[a][c] + dist[b][d] < dist[a][b] + dist[c][d] - 1e-9:
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    improved = True
                    b = tour[i + 1]
    return tour


def christofides(dist: list[list[float]]) -> list[int]:
    """Christofides (1976), via networkx: MST + min-weight matching + Eulerian shortcut.

    Guaranteed <= 1.5x the optimal tour length, but *only* when `dist`
    satisfies the triangle inequality (a metric instance, e.g. Euclidean
    points) -- the proof relies on it. Not implemented from scratch here:
    the MST + minimum-weight perfect matching + Eulerian-circuit-shortcutting
    pipeline is exactly what `networkx.algorithms.approximation.christofides`
    already provides, and reimplementing a matching algorithm (Edmonds'
    blossom algorithm, for the min-weight-perfect-matching step) from
    scratch is out of scope for this challenge -- using the modern, tested
    library implementation here is the honest choice over a partial
    reimplementation of blossom matching.
    """
    import networkx as nx

    n = len(dist)
    g = nx.complete_graph(n)
    for i, j in g.edges():
        g[i][j]["weight"] = dist[i][j]
    cycle = nx.algorithms.approximation.christofides(g, weight="weight")
    if cycle[0] == cycle[-1]:
        cycle = cycle[:-1]
    return cycle


def euclidean_instance(
    n: int, seed: int | None = None
) -> tuple[list[tuple[float, float]], list[list[float]]]:
    """Random points in [0, 100]^2 and their pairwise Euclidean distance matrix (a metric instance)."""
    rng = random.Random(seed)
    points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
    dist = [[math.dist(points[i], points[j]) for j in range(n)] for i in range(n)]
    return points, dist


def _demo() -> None:
    _, dist = euclidean_instance(10, seed=0)
    opt_cost, opt_tour = held_karp(dist)
    bnb_cost, bnb_tour = branch_and_bound(dist)
    nn_tour = nearest_neighbor(dist)
    ge_tour = greedy_edge(dist)
    two_opt_tour = two_opt(dist, nn_tour)
    chr_tour = christofides(dist)

    print(f"Held-Karp (exact):      cost={opt_cost:.2f}  tour={opt_tour}")
    print(f"Branch & bound (exact): cost={bnb_cost:.2f}  tour={bnb_tour}")
    print(
        f"Nearest neighbor:       cost={tour_length(dist, nn_tour):.2f}  "
        f"ratio={tour_length(dist, nn_tour) / opt_cost:.3f}"
    )
    print(
        f"Greedy edge:            cost={tour_length(dist, ge_tour):.2f}  "
        f"ratio={tour_length(dist, ge_tour) / opt_cost:.3f}"
    )
    print(
        f"NN + 2-opt:             cost={tour_length(dist, two_opt_tour):.2f}  "
        f"ratio={tour_length(dist, two_opt_tour) / opt_cost:.3f}"
    )
    print(
        f"Christofides:           cost={tour_length(dist, chr_tour):.2f}  "
        f"ratio={tour_length(dist, chr_tour) / opt_cost:.3f}"
    )


if __name__ == "__main__":
    _demo()
