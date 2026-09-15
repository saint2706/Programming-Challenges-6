# Course Schedule / Cycle Detection Toolkit

**Category:** Algorithmic Challenges
**Difficulty:** A (brief: "Kahn's vs DFS-based topological sort, detect and report cycles")

**Status:** Implemented (Python)

A course-schedule graph is a dependency DAG: edge `u -> v` means "u must be
completed before v". Deciding every course is schedulable is deciding the
graph is acyclic and producing a valid order -- topological sort. The brief
asks for Kahn's vs DFS-based topo sort, both implemented and cross-verified
below, plus reporting actual cycles rather than just "a cycle exists". This
implementation also adds the genuine state-of-the-art piece beyond either:
a real course catalog is built up one prerequisite link at a time, not
handed over as one static graph -- so it also implements the algorithm for
*maintaining* a topological order incrementally as edges are added.

| Method                   | Time                                                                                   | Finds an actual cycle path?          |
| ------------------------ | -------------------------------------------------------------------------------------- | ------------------------------------ |
| `kahn_topological_order` | O(V + E), one-shot                                                                     | No -- only the stuck vertex set      |
| `dfs_topological_order`  | O(V + E), one-shot                                                                     | Yes -- one concrete cycle            |
| `find_cycle_groups`      | O(V + E), one-shot                                                                     | Yes -- every mutually-blocking group |
| `PearceKellyOrder`       | amortized well below O(V+E) per insertion; O(m^1.5) worst case over m insertions total | Yes                                  |

## `kahn_topological_order` and `dfs_topological_order`

Kahn's (1962) repeatedly removes a zero-in-degree vertex: every vertex with
no remaining unprocessed prerequisite is immediately schedulable, so
"completing" it only lowers its successors' in-degree, uncovering the next
batch. If the queue runs dry before every vertex is output, whatever's left
has a prerequisite on itself (possibly transitively) -- that's the stuck
set, but Kahn's bookkeeping alone can't say *which path* forms the cycle.

DFS-based topo sort gets that path for free. Standard white/gray/black
coloring: a vertex is gray while it's still on the current DFS stack
(pushed, not yet finished); any edge landing back on a gray vertex is a
back edge, meaning that vertex is its own descendant -- a cycle, and the
parent-pointer chain from the current vertex back to the gray one it hit
*is* a concrete cycle. Post-order (finish-time) DFS, reversed, is a valid
topological order for a DAG the same way it is for the SCC Finder
challenge's algorithms.

Both are implemented iteratively (an explicit stack standing in for the
call stack), not recursively -- the same reason as the SCC Finder
challenge: Python's ~1000-frame recursion limit is easy to hit on a long
dependency chain, which real course catalogs and sparse random graphs both
produce.

`find_cycle_groups` goes one step further than either: real cyclic
prerequisite data can have *several* independent deadlocked clusters at
once, not just one. Each maximal set of mutually-reachable vertices (a
nontrivial strongly connected component, or a single self-looped vertex)
is one cluster -- one iterative Tarjan pass. The SCC Finder challenge
already runs a full Tarjan/Kosaraju/Gabow shootout cross-verified against
networkx and scipy; this reuses just the one algorithm needed here rather
than duplicating that comparison.

## `PearceKellyOrder`: maintaining the order incrementally

Recomputing a full topological sort after every single edge addition costs
O(V + E) *every time* -- correct, but wasteful for a catalog that's built
up one prerequisite at a time over years, which is how course-schedule
systems (and build systems, and spreadsheet dependency graphs, and package
managers) actually operate. Pearce & Kelly (2007) maintain a permutation
`pos[v]` (v's position in the current order) and, on inserting edge
`(u, v)`:

- If `pos[u] < pos[v]` already, the edge doesn't violate the order: record
  it, done -- no search needed at all.
- Otherwise, a forward DFS from `v` -- using *existing* edges only,
  restricted to the region between the two positions -- looks for `u`.
  Finding it means `v` can already reach `u`, so the new edge would close a
  cycle: reject it (graph left unchanged) and return the discovered path.
- If no cycle, a backward DFS from `u` collects everything that reaches
  `u` in that same region (`ΔB`); the forward search's visited set is
  everything reachable from `v` in that region (`ΔF`, including `v`
  itself). These two sets are provably disjoint whenever there's no cycle
  (if they overlapped at some vertex `x`, that would mean `v` reaches `x`
  reaches `u`, which combined with the new edge `u -> v` is already a
  cycle -- contradiction). `ΔB` and `ΔF` are then reassigned exactly the
  positions they collectively used to occupy: `ΔB` first (preserving its
  old relative order), then `ΔF` (preserving its old relative order). That
  places every vertex of `ΔB` (including `u`) before every vertex of `ΔF`
  (including `v`), satisfying the new edge, while every vertex *outside*
  the affected region keeps its position untouched, and no two vertices
  that were already on the same side of the affected region are ever
  reordered relative to each other.

Each insertion only touches its own affected region, not the whole graph.
Pearce & Kelly's analysis bounds the total work across `m` insertions at
`O(m^1.5)` worst case, against `O(m * (V + E))` for recomputing from
scratch after every edge -- the benchmark below measures this directly.

## Correctness

```
$ uv run --with pytest pytest -q
222 passed in 0.2s
```

`kahn_topological_order` and `dfs_topological_order` are cross-verified
against each other on 80 random digraphs (spanning both DAGs and cyclic
graphs): they must agree on whether the graph is acyclic, any produced
order must respect every edge, and any produced cycle must actually be a
closed walk using only real edges of the graph. `find_cycle_groups` is
checked, on 60 further random graphs, to only ever report groups whose
members can truly all reach each other. `PearceKellyOrder` is the one
requiring the most scrutiny -- it's checked against `dfs_topological_order`
**after every single insertion** (not just at the end) across 60 random
insertion sequences: every accepted edge must correspond to a graph
`dfs_topological_order` also finds acyclic, every rejected edge must
correspond to one `dfs_topological_order` also finds cyclic, a rejected
edge must leave the structure completely unmutated, and the maintained
order must remain a valid topological order of everything actually applied
so far, at every step -- not just once construction finishes.

## Benchmarks

```
$ uv run python benchmark.py
Kahn's vs DFS-based topological sort: both O(V + E), one-shot
       n     edges        kahn         dfs
------------------------------------------
   2,000     6,000     0.0022s     0.0013s
  10,000    30,000     0.0094s     0.0068s
  50,000   150,000     0.0776s     0.0775s
 200,000   600,000     1.2283s     1.0297s

Growing a course catalog one prerequisite edge at a time: incremental vs recompute-from-scratch
     n   edges added    incremental (PK)    recompute-every-insert   speedup
----------------------------------------------------------------------------
   100           500             0.0017s                   0.0238s     14.0x
   200         1,000             0.0064s                   0.1114s     17.5x
   400         2,000             0.0221s                   0.6321s     28.6x
   800         4,000             0.0843s                   4.4114s     52.3x
```

**Kahn's and DFS are, as predicted, close.** Both are O(V + E) with a fixed
constant amount of work per vertex/edge; the ratio between them stays near
1x across two orders of magnitude in graph size, confirming neither has a
structural speed edge over the other here -- consistent with the README's
framing above that DFS's advantage on this challenge is the free cycle
path, not raw throughput.

**The incremental-vs-recompute comparison is the headline result, and the
gap widens as the catalog grows.** At `n=100` the incremental structure is
already 14x faster; by `n=800` that's grown to over 52x, because
recompute-from-scratch's per-insertion cost scales with the *entire*
current graph (`O(V+E)`, climbing every single edge) while
`PearceKellyOrder`'s per-insertion cost scales with the *affected region*
of that one edge, which stays comparatively small. This is exactly the
same "same guaranteed-correct answer, different growth rate" story as
Held-Karp vs branch-and-bound and MLCS's brute-force vs dominant-point
methods elsewhere in this repo -- here the axis is *incremental
maintenance* rather than *state-space pruning*, but the shape of the win
is the same idea: don't redo work that a smarter bookkeeping scheme can
avoid.

## Run it

```bash
cd "challenges/Algorithmic Challenges/Course Schedule - Cycle Detection Toolkit"

uv run python course_schedule.py     # Kahn's, DFS, cycle groups, and PearceKellyOrder demos
uv run python benchmark.py           # Kahn vs DFS, and incremental vs recompute-from-scratch

uv run --with pytest pytest -q          # 222 tests
```

## Where this is used

**Build systems and task runners.** Make, Bazel, and CI pipeline DAGs are
literally this problem: a target depends on its inputs being built first,
a cycle means a broken build graph, and tools that rebuild the dependency
graph incrementally as `BUILD` files change (rather than reparsing the
entire repository on every edit) face exactly the incremental-maintenance
problem `PearceKellyOrder` solves.

**Package managers.** Resolving install order from a dependency
specification is a topological sort; a version-conflict cycle (A requires
B requires A, transitively) is detected the same way `find_cycle_groups`
finds a mutual-prerequisite deadlock.

**Spreadsheet and reactive-programming dependency graphs.** A cell's
formula depending on other cells (or a reactive framework's derived-value
graph) needs recomputation in dependency order, and adding a new formula
is exactly one incremental edge insertion -- Pearce & Kelly's own paper
cites exactly this kind of "maintain order under a stream of small
changes" application as the motivation.

**Static analysis and compilers.** Points-to/pointer analysis and other
dataflow-graph maintenance during incremental compilation is cited
directly in later work building on Pearce & Kelly's algorithm (Bender,
Fineman, Gilbert & Kelly's 2015 follow-up tightens the same problem's
bounds further, motivated by exactly this use case).

## Sources

- [Kahn, A.B., "Topological Sorting of Large Networks," *Communications of the ACM* 5(11):558-562, 1962](https://doi.org/10.1145/368996.369025) -- the in-degree-removal algorithm.
- Cormen, T.H., Leiserson, C.E., Rivest, R.L. & Stein, C., *Introduction to Algorithms*, 3rd ed., Ch. 22 -- DFS-based topological sort and the white/gray/black coloring scheme this implementation follows.
- [Pearce, D.J. & Kelly, P.H.J., "A Dynamic Topological Sort Algorithm for Directed Acyclic Graphs," *ACM Journal of Experimental Algorithmics* 11, 2007](https://doi.org/10.1145/1187436.1210590) -- the incremental order-maintenance algorithm `PearceKellyOrder` implements.
- [Bender, M.A., Fineman, J.T., Gilbert, S. & Kelly, P.H.J., "A New Approach to Incremental Cycle Detection and Related Problems," *ACM Transactions on Algorithms* 12(2), 2015](https://doi.org/10.1145/2756553) -- a later, further-improved-bound approach to the same incremental problem, noted here as the current state of the art beyond this implementation's scope.
- [Tarjan, R.E., "Depth-First Search and Linear Graph Algorithms," *SIAM Journal on Computing* 1(2):146-160, 1972](https://doi.org/10.1137/0201010) -- the SCC algorithm `find_cycle_groups` reuses (also cited by the SCC Finder challenge, which implements it alongside Kosaraju and Gabow).
