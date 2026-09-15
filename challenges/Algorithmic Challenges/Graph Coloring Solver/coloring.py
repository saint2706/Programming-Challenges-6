"""Graph coloring: four heuristics and one exact solver, cross-verified.

Vertex coloring assigns each vertex a color such that no edge joins two
same-colored vertices, using as few colors as possible. Finding the true
minimum (the chromatic number) is NP-hard; everything except
``exact_chromatic_number`` below is a fast heuristic with no optimality
guarantee.

| Method                    | Idea                                                | Guarantee                  |
| ------------------------- | ---------------------------------------------------- | --------------------------- |
| ``greedy_color``           | Color in a given/natural order, smallest free color  | <= max degree + 1 colors    |
| ``welsh_powell``           | Greedy, ordered by descending static degree          | Often better, no guarantee  |
| ``dsatur``                 | Greedy, always pick highest *saturation* degree next | Exact on bipartite/chordal  |
| ``rlf``                    | Build one color class at a time (Recursive Largest First) | No guarantee, often tight |
| ``exact_chromatic_number`` | Backtracking + clique lower bound, binary-searched k | Always optimal (NP-hard cost) |

Graphs are undirected adjacency dicts: ``{node: set-or-list-of-neighbors}``.
Edges need not be listed on both ends -- :func:`_normalize` symmetrizes them.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable


def _normalize(
    graph: dict[Hashable, Iterable[Hashable]],
) -> dict[Hashable, set[Hashable]]:
    """Symmetrize adjacency (an edge listed on only one end still counts both ways)."""
    normalized: dict[Hashable, set[Hashable]] = {
        v: set(neighbors) for v, neighbors in graph.items()
    }
    for v, neighbors in list(normalized.items()):
        for u in neighbors:
            normalized.setdefault(u, set()).add(v)
    return normalized


def is_proper_coloring(
    graph: dict[Hashable, Iterable[Hashable]], coloring: dict[Hashable, int]
) -> bool:
    graph = _normalize(graph)
    return all(
        coloring[u] != coloring[v] for u, neighbors in graph.items() for v in neighbors
    )


def num_colors(coloring: dict[Hashable, int]) -> int:
    return len(set(coloring.values()))


def greedy_color(
    graph: dict[Hashable, Iterable[Hashable]], order: list[Hashable] | None = None
) -> dict[Hashable, int]:
    """Color vertices in `order` (default: dict iteration order), smallest legal color each time.

    Guaranteed to use at most ``max_degree + 1`` colors, regardless of order --
    but the order matters enormously for *how close* to optimal that is (see
    the crown-graph example in the README: a bad order forces this same
    algorithm to use roughly n/2 colors on a graph that's 2-colorable).
    """
    graph = _normalize(graph)
    if order is None:
        order = list(graph)
    coloring: dict[Hashable, int] = {}
    for v in order:
        used = {coloring[u] for u in graph[v] if u in coloring}
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
    return coloring


def welsh_powell(graph: dict[Hashable, Iterable[Hashable]]) -> dict[Hashable, int]:
    """Welsh & Powell (1967): greedy, vertices pre-sorted by descending degree.

    High-degree vertices are the most constrained (most neighbors to avoid
    clashing with), so coloring them first -- while the most colors are still
    free -- tends to need fewer colors overall than an arbitrary order.
    """
    graph = _normalize(graph)
    order = sorted(graph, key=lambda v: -len(graph[v]))
    return greedy_color(graph, order)


def dsatur(graph: dict[Hashable, Iterable[Hashable]]) -> dict[Hashable, int]:
    """Brelaz's DSATUR (1979): always color the vertex with the highest *saturation*.

    Saturation degree = the number of *distinct* colors already used among a
    vertex's neighbors (not the count of colored neighbors -- two neighbors
    sharing a color count once). Recomputing "most constrained vertex right
    now" after every move (rather than fixing the order up front, as
    Welsh-Powell does) is provably exact on bipartite and chordal graphs, and
    is the standard heuristic of choice in practice.

    This implementation recomputes the max-saturation vertex with a linear
    scan each round: O(n) per step, O(n^2) total. A binary-heap-backed
    DSATUR reaches O((V+E) log V) by updating priorities incrementally --
    a known, real improvement, left unimplemented here for the same reason
    A-ExpJ was left out of the Reservoir Sampling challenge: a subtly-wrong
    incremental-heap DSATUR is worse than a simple, obviously-correct O(n^2)
    one, and O(n^2) never becomes the bottleneck next to the exponential
    exact solver this module also implements.
    """
    graph = _normalize(graph)
    coloring: dict[Hashable, int] = {}
    saturation: dict[Hashable, set[int]] = {v: set() for v in graph}
    degree = {v: len(graph[v]) for v in graph}
    uncolored = set(graph)
    while uncolored:
        v = max(uncolored, key=lambda x: (len(saturation[x]), degree[x]))
        used = {coloring[u] for u in graph[v] if u in coloring}
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
        uncolored.discard(v)
        for u in graph[v]:
            if u in uncolored:
                saturation[u].add(c)
    return coloring


def rlf(graph: dict[Hashable, Iterable[Hashable]]) -> dict[Hashable, int]:
    """Recursive Largest First (Leighton, 1979): build one whole color class at a time.

    Unlike the vertex-at-a-time methods above, RLF picks an entire maximal
    independent set to share color 0, then color 1, and so on. Within a
    color class: start with the vertex of highest degree among the remaining
    uncolored vertices; then repeatedly add whichever remaining candidate has
    the most neighbors *already excluded* from this class (i.e. is most
    "used up" and should be spent now rather than saved for a future class),
    breaking ties toward fewer neighbors among the *still-eligible*
    candidates. This one-class-at-a-time construction is the reason RLF
    often beats DSATUR on structured graphs at the cost of being noticeably
    slower (each class construction rescans the remaining vertex set).
    """
    graph = _normalize(graph)
    uncolored = set(graph)
    coloring: dict[Hashable, int] = {}
    color = 0
    while uncolored:
        eligible = set(uncolored)  # V1: still candidates for this color class
        excluded: set[Hashable] = set()  # V2: ruled out for this class (adjacent to it)
        this_class: list[Hashable] = []

        v = max(eligible, key=lambda x: len(graph[x] & eligible))
        this_class.append(v)
        eligible.discard(v)
        newly_excluded = graph[v] & eligible
        excluded |= newly_excluded
        eligible -= newly_excluded

        while eligible:
            v = max(
                eligible,
                key=lambda x: (len(graph[x] & excluded), -len(graph[x] & eligible)),
            )
            this_class.append(v)
            eligible.discard(v)
            newly_excluded = graph[v] & eligible
            excluded |= newly_excluded
            eligible -= newly_excluded

        for u in this_class:
            coloring[u] = color
            uncolored.discard(u)
        color += 1
    return coloring


def _max_clique_lower_bound(graph: dict[Hashable, set[Hashable]]) -> int:
    """Exact max clique size via networkx's Bron-Kerbosch (`find_cliques`).

    omega(G) <= chi(G) always: every vertex in a clique needs its own color.
    Exponential in general, but the graphs this module benchmarks are small
    enough (chromatic-number search is exponential too, and dominates).
    """
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(graph)
    g.add_edges_from((u, v) for u, neighbors in graph.items() for v in neighbors)
    return max((len(c) for c in nx.find_cliques(g)), default=0)


def _try_k_color(
    graph: dict[Hashable, set[Hashable]], order: list[Hashable], k: int
) -> tuple[bool, dict[Hashable, int] | None]:
    """Backtracking feasibility check: can `graph` be colored with exactly `k` colors?

    Symmetry breaking: a vertex is only ever allowed to introduce color
    ``m`` if colors ``0..m-1`` are already in use somewhere -- two colorings
    that only differ by permuting color *labels* are the same coloring, so
    there is no reason to explore both `{0: red, 1: blue}` and
    `{0: blue, 1: red}` as distinct branches.
    """
    coloring: dict[Hashable, int] = {}

    def backtrack(i: int, colors_used: int) -> bool:
        if i == len(order):
            return True
        v = order[i]
        forbidden = {coloring[u] for u in graph[v] if u in coloring}
        for c in range(min(k, colors_used + 1)):
            if c not in forbidden:
                coloring[v] = c
                if backtrack(i + 1, max(colors_used, c + 1)):
                    return True
                del coloring[v]
        return False

    if backtrack(0, 0):
        return True, dict(coloring)
    return False, None


def exact_chromatic_number(
    graph: dict[Hashable, Iterable[Hashable]],
) -> tuple[int, dict[Hashable, int]]:
    """The true chromatic number, via clique lower bound + backtracking search over k.

    Tries k = (max clique size) up to (best heuristic's color count), in
    increasing order, returning the first k that's feasible -- which is then
    provably minimal, since every smaller k was proven infeasible by the
    same exhaustive search. NP-hard in general: this is exponential time in
    the worst case, made practical for the sizes benchmarked here by the
    clique lower bound (often already tight -- see the README's certificate
    discussion), a good starting upper bound from the heuristics above, and
    symmetry-breaking in the backtracking search itself.
    """
    graph = _normalize(graph)
    nodes = list(graph)
    if not nodes:
        return 0, {}

    order = sorted(nodes, key=lambda v: -len(graph[v]))
    lower = max(1, _max_clique_lower_bound(graph))
    upper = min(
        num_colors(dsatur(graph)),
        num_colors(rlf(graph)),
        num_colors(welsh_powell(graph)),
    )

    for k in range(lower, upper + 1):
        feasible, coloring = _try_k_color(graph, order, k)
        if feasible:
            assert coloring is not None
            return k, coloring
    # Unreachable: `upper` colors is always achievable by construction.
    raise AssertionError("no feasible k found up to a known-achievable upper bound")


# --- graph generators used in the README / tests / benchmark -------------


def crown_graph(n: int) -> dict[int, set[int]]:
    """K(n,n) minus a perfect matching: {a_1..a_n} x {b_1..b_n}, a_i-b_j iff i != j.

    Bipartite (chromatic number 2 for n >= 2), but a "natural" interleaved
    vertex order (a_1, b_1, a_2, b_2, ...) is the textbook example of an
    ordering that forces plain greedy coloring far away from optimal --
    quantified empirically in the README rather than asserted from memory.
    """
    graph: dict[int, set[int]] = {}
    a = list(range(n))
    b = list(range(n, 2 * n))
    for i in range(n):
        graph[a[i]] = {b[j] for j in range(n) if j != i}
        graph[b[i]] = {a[j] for j in range(n) if j != i}
    return graph


def crown_interleaved_order(n: int) -> list[int]:
    order = []
    for i in range(n):
        order.append(i)
        order.append(n + i)
    return order


def queen_graph(n: int) -> dict[int, set[int]]:
    """The n x n queens graph: one vertex per square, edges between squares a queen attacks."""
    graph: dict[int, set[int]] = {i: set() for i in range(n * n)}

    def idx(r: int, c: int) -> int:
        return r * n + c

    for r1 in range(n):
        for c1 in range(n):
            u = idx(r1, c1)
            for r2 in range(n):
                for c2 in range(n):
                    if (r1, c1) == (r2, c2):
                        continue
                    if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                        graph[u].add(idx(r2, c2))
    return graph


def from_networkx(g) -> dict[Hashable, set[Hashable]]:
    return {v: set(g.neighbors(v)) for v in g.nodes}


def _demo() -> None:
    print("Crown graph (n=6): 2-colorable, but order matters for plain greedy.")
    graph = crown_graph(6)
    natural = greedy_color(graph, order=sorted(graph))  # all a's, then all b's
    bad = greedy_color(graph, order=crown_interleaved_order(6))  # a1,b1,a2,b2,...
    dsat = dsatur(graph)
    exact_k, _ = exact_chromatic_number(graph)
    print(f"  exact chromatic number:                     {exact_k}")
    print(
        f"  greedy_color (all a's, then all b's):        {num_colors(natural)} colors"
    )
    print(f"  greedy_color (a1,b1,a2,b2,... interleaved):  {num_colors(bad)} colors")
    print(f"  dsatur:                                      {num_colors(dsat)} colors")
    for name, c in (("natural", natural), ("bad", bad), ("dsatur", dsat)):
        assert is_proper_coloring(graph, c), f"{name} coloring is not proper!"


if __name__ == "__main__":
    _demo()
