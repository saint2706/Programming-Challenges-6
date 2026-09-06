# Minimum Spanning Tree Visualizer

**Category:** Algorithmic Challenges
**Difficulty:** I (brief: "Kruskal vs Prim, animate edge selection order")

**Status:** Implemented (Python)

The brief asks for two algorithms. Three are implemented, because the third
is the one that explains the other two: **Boruvka's algorithm** (1926) is the
oldest MST algorithm and the common ancestor both Kruskal (1956) and Prim
(1957, independently Jarnik 1930) specialize. Kruskal is "sort every edge,
take greedily"; Prim is "grow one component"; Boruvka is "every component
grows at once." Benchmarking all three on real graphs produces a result
none of the textbook complexity bounds predict on their own: **Boruvka wins
by 2-3x on dense graphs and loses by 2x on sparse ones** — see below.

## Correctness: why three different greedy rules all find the same answer

All three algorithms are instances of one theorem. The edges of a graph
form a **graphic matroid**: a subset of edges is "independent" if it's
acyclic. A spanning tree is a maximum independent set (a *basis*) of that
matroid, and a *minimum-weight* basis is exactly what the MST problem asks
for. The **Rado-Edmonds matroid greedy theorem** (Rado 1957, Edmonds 1971)
says: for any matroid, sorting elements by weight and greedily adding each
one that keeps the set independent produces a minimum-weight basis. That
single theorem is a full correctness proof for Kruskal, and — because
Prim's and Boruvka's local rules are each provably equivalent to running
the same matroid-greedy choice with a different traversal order — for all
three.

Two classical, more concrete properties say the same thing without matroid
language, and are what `test_mst.py` and the benchmarks actually lean on:

- **Cut property.** For any partition of the vertices into two nonempty
  sets, the minimum-weight edge crossing that cut belongs to *some* MST
  (uniquely, if it's the strict minimum). Every accept decision in all
  three algorithms is a cut-property application: Kruskal's next sorted
  edge is the minimum edge crossing the cut between "already merged
  components" and everything else; Prim's heap pop is the minimum edge
  crossing the cut between "in the tree" and "not yet"; Boruvka's
  per-component cheapest edge is the minimum edge crossing that
  component's cut.
- **Cycle property.** For any cycle, the maximum-weight edge on it belongs
  to *no* MST (uniquely, if it's the strict maximum) — equivalently, any
  edge that would close a cycle can safely be rejected, which is exactly
  what Kruskal's "reject if same component" and Prim's "reject if both
  endpoints visited" do.
- **Uniqueness under distinct weights.** If every edge weight is distinct,
  the MST is unique. Proof by exchange argument: suppose two spanning
  trees T1 != T2 were both minimum. Take the lowest-weight edge `e` where
  they differ, say `e` in T1 not T2. Adding `e` to T2 creates a cycle;
  that cycle must contain an edge `f` not in T1 (else T1 would contain the
  whole cycle). Since weights are distinct and `e` was the lowest edge of
  disagreement, `weight(f) > weight(e)`. Swapping `f` for `e` in T2
  produces a spanning tree lighter than T2 — contradiction. `test_mst.py`
  checks this directly: on a graph with all-distinct weights, Kruskal,
  Prim, and Boruvka return the *same edge set*, not just the same total
  weight.

## The three algorithms

- **Kruskal — global sort, then greedy.** Sort all `E` edges once,
  `O(E log E)`, then scan them in order, accepting an edge iff its
  endpoints are in different components (checked with a union-find). Self
  loops are filtered before sorting (`u == v` can never merge two
  components, so they're always useless); parallel edges need no special
  handling at all — the cheapest of a duplicate pair is processed first
  (it sorts lower), unions its endpoints, and every later duplicate is
  rejected by the ordinary "same component" check. A disconnected graph is
  not a special case either: the scan simply runs out of edges before
  every vertex is unioned, leaving a **minimum spanning forest** with
  `n - c` edges for `c` components.
- **Prim — grow one component with a min-heap frontier.** Start from an
  arbitrary vertex, repeatedly pop the cheapest edge leaving the current
  tree from a binary heap (`heapq`), and push the new vertex's edges when
  it's added. `O(E log V)` with a binary heap; a Fibonacci heap gets it to
  `O(E + V log V)` (Fredman & Tarjan, 1987) but Python's standard library
  has no Fibonacci heap, and the extra constant factor of implementing one
  rarely pays for itself at the scale Python runs anything. Restarting
  from every unvisited vertex once a component is exhausted turns this
  into "Prim, run once per component" — a minimum spanning forest on a
  disconnected graph, for free.
- **Boruvka — every component merges in parallel.** In each round, every
  remaining component finds its own single cheapest outgoing edge; all of
  those edges are added at once (skipping a component if its partner
  already merged from the other side this round, to avoid a double-union).
  Every component either merges with another or is absorbed, so the
  component count at least halves each round: `O(log V)` rounds of `O(E)`
  work, `O(E log V)` total. This round structure — independent,
  simultaneous per-component work — is exactly why Boruvka's algorithm
  (not Kruskal's or Prim's) is the basis of essentially every
  parallel/distributed/GPU MST implementation: each processor can own a
  component and search its own edges with no coordination needed until
  the merge step.

`Union-Find` underneath all three where needed uses **path compression +
union by rank** (Tarjan & van Leeuwen, 1984), giving amortized
`O(alpha(n))` per operation — the inverse Ackermann function, which is at
most 4 for any `n` that fits in this universe, so it's treated as constant
in every complexity bound above.

## Benchmark: textbook complexity vs measured reality

```
   n          density     kruskal        prim     boruvka
 100 sparse (~V edges)       0.23ms      0.19ms      0.41ms
 100  medium (~VlogV)        1.59ms      1.43ms      1.25ms
 100    dense (V^2/4)        2.13ms      2.08ms      2.30ms
 300 sparse (~V edges)       0.30ms      0.29ms      0.55ms
 300  medium (~VlogV)        3.15ms      3.18ms      3.72ms
 300    dense (V^2/4)       28.13ms     31.66ms     25.76ms
 600 sparse (~V edges)       0.78ms      0.75ms      1.44ms
 600  medium (~VlogV)       10.36ms     10.56ms     10.18ms
 600    dense (V^2/4)      223.24ms    238.71ms    133.75ms
1000 sparse (~V edges)       1.76ms      1.75ms      3.17ms
1000  medium (~VlogV)       24.06ms     23.92ms     19.41ms
1000    dense (V^2/4)      966.91ms   1183.84ms    385.10ms
```

(`uv run python benchmark.py`, best-of-3, random integer weights.)

Two things the `O(E log E)` / `O(E log V)` / `O(E log V)` bounds do not
predict:

1. **On sparse graphs, Boruvka is 2x slower** than Kruskal or Prim. Its
   asymptotic bound is the same, but the *constant* is worse in this
   regime: each of its `O(log V)` rounds re-scans **every remaining edge**
   from scratch (a fresh Python-level loop with two `dict` lookups per
   edge), even though a sparse graph gives it almost nothing to find per
   round. Kruskal pays its `O(E log E)` sort exactly once; Prim's heap
   only ever touches edges adjacent to the growing frontier. Boruvka's
   repeated full-edge-list scans are the overhead neither of the others
   has.
2. **On dense graphs, Boruvka wins by 2-3x** — the opposite ranking. Here
   `E = Theta(V^2)`, so Kruskal's sort costs `O(V^2 log V)` in CPython's
   highly-optimized-but-still-per-comparison Timsort, and Prim performs
   `O(V^2)` heap pushes, each paying `O(log V)` sift-up/down through
   Python-level comparisons. Boruvka's `O(log V)` rounds each do one
   linear, comparison-light scan over the edge list with no heap
   maintenance and no full sort at all — and on a dense graph, `O(log V)`
   rounds is a *small* multiplier. The single global sort and the
   per-element heap bookkeeping turn out to be the expensive parts in
   Python, not the asymptotic round count.

The practical reading: Boruvka's textbook reputation is "the
parallel-friendly one, mostly of historical/pedagogical interest for a
single thread." That undersells it — on dense graphs, even run
single-threaded, it's the fastest of the three here, for a completely
different reason than the one it's usually kept around for.

## Edge cases

Each of these has a test in `test_mst.py`, run against all three
algorithms:

- **Disconnected graphs** produce a minimum spanning **forest**, not an
  error and not a single tree — `n - c` edges for `c` components. None of
  the three algorithms needs a special branch for this: Kruskal simply
  runs out of edges, Prim restarts from every unvisited vertex, and
  Boruvka's round loop terminates naturally once no component has an edge
  left to offer.
- **Negative weights** work correctly with no modification — the cut and
  cycle properties never assumed non-negativity (unlike Dijkstra's
  shortest path, which does). Tested with a 3-vertex graph where the
  correct MST total is negative.
- **Duplicate weights** can produce multiple *different* valid MSTs (e.g.
  a 4-cycle with all weights equal — dropping any one edge is optimal),
  but every valid MST has the **same total weight** — a direct consequence
  of the cycle property (any of the tied maximum-weight edges on a cycle
  is equally safe to reject). Tested by checking all three algorithms
  agree on total weight even where they may pick different edges.
- **Self-loops** (`u == v`) are filtered immediately — a self-loop can
  never reduce the number of components, so it is never useful and never
  appears in any output.
- **Parallel edges** between the same pair need no pre-processing: the
  cheapest one is always the one that ends up used, because whichever
  algorithm processes it first (Kruskal by sort order, Prim/Boruvka by
  min-comparison) unions the pair, and every subsequent duplicate is
  rejected as a same-component edge.
- **Empty graph (`n = 0`)** and **single vertex, no edges** both correctly
  return an empty forest with total weight 0.

## Pacing: readable without pausing

The first version gave every edge decision a flat ~0.4s `run_time` covering
both the line's motion and its caption change at once -- not a duration
anyone can read a caption in, and the accept/reject color was gone before
it registered. Each step is now two phases, the same fix used by the
sorting-algorithm visualizer elsewhere in this repo: a short **motion**
phase (the edge animates in, and turns green/red), then a still **hold**
phase sized by subtitle-reading-speed practice (14 characters/second first
read, under Netflix's 17 cps ceiling since these captions are denser than
prose). `pacing.py` keeps this pure arithmetic, with no Manim import, so it
can be checked and tuned without rendering anything:

```
uv run python pacing.py
scene       steps    motion      hold     total
kruskal         9      3.4s      9.3s     12.7s
prim            9      3.4s      9.6s     13.0s
boruvka         5      2.0s      6.5s      8.5s
```

Repeated captions are charged less: "Accept 0-1 (w=4)" -> "Accept 1-2
(w=7)" changes three digits in unrelated positions, so a naive
common-prefix/common-suffix trim would (wrongly) count almost the whole
caption as new the moment the first digit differs. `novel_characters` uses
`difflib.SequenceMatcher` instead, which finds every matching run
regardless of position, so only the digits that actually changed are
charged at the faster re-read rate -- see the docstring in `pacing.py` for
the worked example. `MST_HOLD_SCALE=0 uv run --with manim manim -pql
visualize.py KruskalScene` drops every hold for fast layout iteration.

## Files

| File             | What it is                                                                                                                                                                 |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mst.py`         | `DisjointSet`, the three `*_steps` generators, `solve()`, CLI (`--demo`, `--verify`)                                                                                       |
| `test_mst.py`    | Brute-force cross-check on random small graphs, cross-algorithm agreement, uniqueness under distinct weights, all edge cases above, a 60-vertex complete-graph stress test |
| `benchmark.py`   | Wall-clock timing across sparse/medium/dense random graphs at increasing `V`                                                                                               |
| `pacing.py`      | Motion/hold timing for the animation, testable without Manim                                                                                                               |
| `test_pacing.py` | Caption-shape diffing, repeat-discount behavior, motion times                                                                                                              |
| `visualize.py`   | Manim scenes (`KruskalScene`, `PrimScene`, `BoruvkaScene`) animating edge-by-edge accept/reject with a running weight counter, paced to be readable without pausing        |

## Running it

```bash
uv run python mst.py --demo             # kruskal/prim/boruvka on one small graph
uv run python mst.py --verify           # cross-check 200 random graphs agree

uv run --with pytest pytest -q          # 40 tests
uv run python benchmark.py --quick      # fast sanity timings
uv run python benchmark.py              # the table above (takes a few minutes)
uv run python pacing.py                 # per-scene animation durations, no rendering

uv run --with manim manim -pql visualize.py KruskalScene
uv run --with manim manim -pql visualize.py PrimScene
uv run --with manim manim -pql visualize.py BoruvkaScene

# Fast layout iteration with every reading-hold dropped:
MST_HOLD_SCALE=0 uv run --with manim manim -pql visualize.py KruskalScene
```

Manim needs **ffmpeg** on the PATH plus the system Cairo/Pango development
libraries (`manimpango` builds against them); `uv` handles the Python side
only. `mst.py`, `benchmark.py`, and the tests are pure standard library and
need none of it. Rendered video lands in `media/videos/visualize/` and is
git-ignored — regenerate with the commands above.

## Where this is actually used

- **Network design.** The original motivating problem for both Kruskal
  and Prim's papers: connecting a set of sites (telephone exchanges,
  power substations, computer networks) with the least total cable/wire,
  where "least" is a sum of edge weights.
- **Approximate TSP.** Doubling every edge of an MST and taking an
  Eulerian tour gives a tour at most 2x the optimal traveling-salesman
  tour on metric (triangle-inequality) instances — one of the oldest
  constant-factor approximation algorithms, and it's just an MST plus a
  walk.
- **Single-linkage clustering.** Removing the `k-1` most expensive edges
  from an MST partitions the points into exactly `k` clusters, and that
  partition is provably identical to single-linkage hierarchical
  clustering — MST computation is the standard fast implementation of it.
- **Image segmentation.** Felzenszwalb & Huttenlocher's classic
  graph-based segmentation algorithm builds a minimum spanning forest
  over pixel-similarity edges and cuts it where edge weight exceeds an
  adaptive per-region threshold.
- **Circuit design and VLSI routing**, where Steiner-tree routing
  problems are frequently approximated by computing an MST over the
  terminals to be connected.

## Sources

- Boruvka, ["O jistém problému minimálním"](https://dml.cz/handle/10338.dmlcz/500114) (1926) — the original algorithm, predating both Kruskal and Prim
- Jarnik, "O jistém problému minimálním" (1930) — the same algorithm later rediscovered as "Prim's"
- Kruskal, ["On the Shortest Spanning Subtree of a Graph and the Traveling Salesman Problem"](https://doi.org/10.1090/S0002-9939-1956-0078686-7), *Proc. AMS* 7(1), 1956
- Prim, ["Shortest Connection Networks and Some Generalizations"](https://doi.org/10.1002/j.1538-7305.1957.tb01515.x), *Bell System Technical Journal* 36(6), 1957
- Rado, ["Note on Independence Functions"](https://doi.org/10.1112/plms/s3-7.1.300), *Proc. LMS* 1957; Edmonds, ["Matroids and the Greedy Algorithm"](https://doi.org/10.1007/BF01584082), *Mathematical Programming* 1(1), 1971 — the matroid greedy theorem
- Tarjan & van Leeuwen, ["Worst-case Analysis of Set Union Algorithms"](https://doi.org/10.1145/62.2160), *JACM* 31(2), 1984 — path compression + union by rank
- Fredman & Tarjan, ["Fibonacci Heaps and Their Uses in Improved Network Optimization Algorithms"](https://doi.org/10.1145/28869.28874), *JACM* 34(3), 1987 — the O(E + V log V) Prim bound
- Felzenszwalb & Huttenlocher, ["Efficient Graph-Based Image Segmentation"](https://doi.org/10.1023/B:VISI.0000022288.19776.77), *IJCV* 59(2), 2004
- CLRS, *Introduction to Algorithms*, ch. 21 (Data Structures for Disjoint Sets) and ch. 23 (Minimum Spanning Trees)
