# Held-Karp Exact TSP Solver

**Category:** Algorithmic Challenges
**Difficulty:** A (brief: "Bitmask DP, exact solution for small n; compare vs heuristics")

**Status:** Implemented (Python)

The traveling salesman problem: given pairwise distances between n cities,
find the shortest cycle visiting every city exactly once. NP-hard -- the
two exact methods below only stay practical for small n, in exchange for a
provable optimum; everything else trades that guarantee for polynomial time.

| Method                     | Time               | Space      | Optimal?                       |
| --------------------------- | ------------------ | ---------- | -------------------------------- |
| `held_karp`                 | O(2^n * n^2)       | O(2^n * n) | Always                          |
| `branch_and_bound`          | O(2^n) worst case  | O(n)/branch | Always, often far faster        |
| `nearest_neighbor`          | O(n^2)             | O(n)       | No guarantee                    |
| `greedy_edge`               | O(n^2 log n)       | O(n)       | No guarantee                    |
| `two_opt`                   | O(n^2) per pass    | O(n)       | Local optimum only              |
| `christofides` (via networkx) | O(n^3)           | O(n^2)     | <= 1.5x optimal, metric instances only |

The brief names Held-Karp and "vs heuristics." Branch and bound is added as
the second exact method because it's the other half of Held & Karp's own
1970 paper (the DP and a minimum-1-tree-based branch-and-bound lower bound
appear together there) and gives a genuinely useful comparison: **same
guaranteed-optimal answer, radically different practical cost** -- see the
benchmark below. Nearest-neighbor, greedy-edge, 2-opt, and Christofides
round out "vs heuristics" with the four most commonly cited approaches, one
of them (Christofides) carrying an actual worst-case approximation
guarantee rather than just "usually works well."

## Held-Karp: the bitmask DP

`C[(subset, k)]` = the cheapest way to start at city 0, visit exactly the
cities in `subset` (a bitmask over cities `1..n-1`), and end at city `k`.
Built up by subset size, smallest first, so every sub-solution the
recurrence needs is already available:

```
C[(subset, k)] = min over m in subset \ {k} of  C[(subset \ {k}, m)] + dist[m][k]
```

Fixing city 0 as the start (never given its own bit) is the standard trick:
any optimal tour can be rotated to start wherever we like without changing
its cost, so the search only needs `2^(n-1)` subsets, not `2^n`. The final
answer closes the loop back to city 0 from whichever `k` minimizes
`C[(full_set, k)] + dist[k][0]`, and the path is reconstructed by walking
the predecessor pointers stored alongside each cost.

This implementation stores `C` in a plain dict keyed by `(bitmask, city)`
rather than a dense array indexed by `bitmask * n + city`; a numpy-backed
dense array, processing every bitmask `1..2^(n-1)-1` directly (by
popcount-independent iteration order, not `itertools.combinations` grouped
by size) with the inner `m` loop vectorized, is a known, real speedup —
left unimplemented here because the benchmark below shows the DP hits its
practical wall from raw state-space size (`2^n`) well before dict overhead
is the bottleneck: n=20 already takes 15+ seconds with either
representation's fundamental `2^19 * 19` states.

## Branch and bound: an MST-based lower bound

Held & Karp's 1970 paper also describes a branch-and-bound lower bound
built from **minimum 1-trees** with iteratively adjusted Lagrangian
multipliers -- full Lagrangian ascent is out of scope here (an iterative
optimization procedure in its own right). What's implemented instead is the
simpler piece that still gives a *valid* (if less tight) bound: for a
partial tour ending at city `last` with `remaining` cities still unvisited,

```
lower_bound = path_cost_so_far + MST(remaining U {last}) + min_edge(remaining -> city 0)
```

Any way to visit every remaining city from `last` uses some connected
structure spanning them, which can't cost less than their MST; the tour
must also eventually close back to city 0 through *some* remaining city,
which can't cost less than the single cheapest such edge (even though which
city actually closes the loop isn't known yet). Both terms under-estimate,
so the sum never prunes away the true optimum. Search is **best-first**
(a min-heap keyed on lower bound): whenever a popped state's bound already
meets the best complete tour found so far, the search can stop entirely --
since the heap always pops its current minimum, every other queued state is
provably at least as bad.

## Correctness

```
$ uv run --with pytest --with networkx pytest -q
113 passed in 0.55s
```

`held_karp` is checked against a fully independent brute-force oracle
(every permutation, for n up to 7) across 5 random instances per size.
`branch_and_bound` is checked against `held_karp` itself (not the brute
force oracle again) across n up to 10 -- two structurally different exact
methods agreeing is strong evidence neither has a shared blind spot.
`two_opt` is checked to never make a tour worse than its input; every
heuristic is checked to never beat the true optimum (a heuristic "beating"
`held_karp` would mean one of them has a bug); `christofides` is checked to
stay within its proven 1.5x bound on genuinely metric (Euclidean) instances,
and `euclidean_instance` itself is checked to actually satisfy the triangle
inequality it's relied on for.

## Benchmarks

```
$ uv run --with networkx python benchmark.py
Held-Karp vs branch and bound: same exact answer, very different cost
   n    held-karp time   branch&bound time    optimal cost   agree?
-------------------------------------------------------------------
   8           0.0007s             0.0026s          318.61     True
  10           0.0020s             0.0035s          344.77     True
  12           0.0086s             0.0032s          362.67     True
  14           0.0606s             0.0234s          380.07     True
  16           0.3397s             0.0389s          385.29     True
  18           2.0404s             0.0922s          401.97     True
  20          15.6345s             0.0885s          413.06     True

Heuristic solution quality: ratio to optimal, averaged over 40 random n=10 instances
            method  mean ratio  max ratio   mean time
-----------------------------------------------------
  nearest_neighbor      1.0965     1.4100     0.012ms
       greedy_edge      1.0793     1.2129     0.019ms
      nn + two_opt      1.0062     1.0879     0.023ms
      christofides      1.0612     1.1551     5.622ms
```

**Held-Karp vs branch and bound is the headline result.** Both solve every
instance to the *identical* optimal cost -- the "agree?" column is a
correctness check on two structurally unrelated algorithms, not a
coincidence -- yet by n=20, Held-Karp takes 15.6 seconds while branch and
bound takes 0.09: nearly 180x faster for the same guaranteed answer, on
these instances. That gap is **not free lunch, it's the lower bound doing
its job on well-behaved (Euclidean) inputs**: separately, `branch_and_bound`
did not finish a random Euclidean n=30 instance within a 45-second manual
check either -- past some point, the MST bound stops being tight enough to
prune the exponential search space fast enough, and both exact methods are
exponential in the worst case regardless of which one wins in practice at a
given n. Held-Karp's own timings roughly track its `2^n` prediction (each
+2 cities is close to a 4x-8x slowdown, matching `2^n * n^2`'s dependence).

**Heuristic quality tells a sharper story than "which one is best."**
2-opt applied on top of nearest-neighbor cuts the average gap from ~9.65%
down to ~0.62% -- *cheaper* than Christofides and with *no formal
guarantee at all*, which is exactly the point: a proven worst-case bound
(Christofides' 1.5x) and typical-case performance on well-behaved instances
are different claims, and this benchmark's instances are the easy
(Euclidean, no adversarial construction) kind where local search does very
well. Christofides' own ratios stay comfortably under its 1.5x guarantee
here (max observed 1.155), but at roughly 300-500x the wall-clock cost of
every other method -- the minimum-weight perfect matching step (Edmonds'
blossom algorithm, via networkx, not reimplemented here) is genuinely
expensive, and its value is the guarantee it carries into adversarial
instances where 2-opt has no such promise, not typical-case speed or
quality on instances like these.

## Run it

```bash
cd "challenges/Algorithmic Challenges/Held-Karp Exact TSP Solver"

uv run --with networkx python tsp.py         # exact + heuristics on one instance
uv run --with networkx python benchmark.py   # timing and quality tables

uv run --with pytest --with networkx pytest -q   # 113 tests
```

## Where this is used

**Vehicle routing and logistics.** Delivery-route optimization (last-mile
delivery, school bus routing) is TSP with extra constraints (time windows,
capacity); the exact/heuristic tradeoff here is exactly the one routing
software makes -- exact solvers for small depot clusters, 2-opt/Lin-
Kernighan-family local search for anything at real-world scale.

**Circuit board drilling and manufacturing.** Positioning a drill head to
visit every hole on a PCB in the shortest total travel distance is a
literal TSP instance -- this is one of the applications that motivated
early exact TSP research, including Held & Karp's own work.

**DNA sequencing (shortest superstring / fragment assembly).** Certain
genome assembly formulations reduce to a TSP-like shortest-path-visiting-
all-fragments problem, using a "distance" defined by sequence overlap
rather than physical space -- the reason `dist` here is left as an
abstract matrix rather than assumed to come from real coordinates.

**Why Christofides mattered historically, and what's changed since.**
Christofides' 1.5x bound stood as the best known approximation guarantee
for metric TSP from 1976 until Karlin, Klein & Oveis Gharan's 2021 result
(*STOC* 2021, and refined further since) proved a bound *strictly* below
1.5x for the first time -- a genuine, recent state-of-the-art advance, and
one whose algorithm is far more involved than the MST-plus-matching
pipeline implemented here. Noted as a pointer for context, not implemented:
reproducing that result is a research-paper-scale undertaking on its own,
well beyond this challenge's scope.

## Sources

- [Held, M. & Karp, R.M., "A Dynamic Programming Approach to Sequencing Problems," *Journal of the Society for Industrial and Applied Mathematics* 10(1):196-210, 1962](https://doi.org/10.1137/0110015) -- the bitmask DP.
- [Held, M. & Karp, R.M., "The Traveling-Salesman Problem and Minimum Spanning Trees," *Operations Research* 18(6):1138-1162, 1970](https://doi.org/10.1287/opre.18.6.1138) -- the 1-tree lower bound and Lagrangian relaxation this challenge's branch-and-bound bound simplifies from.
- [Christofides, N., "Worst-Case Analysis of a New Heuristic for the Travelling Salesman Problem," Technical Report, Carnegie Mellon University, 1976](https://apps.dtic.mil/sti/citations/ADA025602) -- the 1.5x-approximation algorithm (used here via `networkx.algorithms.approximation.christofides`).
- [Karlin, A.R., Klein, N. & Oveis Gharan, S., "A (Slightly) Improved Approximation Algorithm for Metric TSP," *STOC* 2021](https://doi.org/10.1145/3406325.3451009) -- the first approximation ratio proven strictly below 1.5x, mentioned above as the actual current state of the art.
- [Edmonds, J., "Paths, Trees, and Flowers," *Canadian Journal of Mathematics* 17:449-467, 1965](https://doi.org/10.4153/CJM-1965-045-4) -- the blossom algorithm for minimum-weight perfect matching that Christofides' method (and networkx's implementation of it) depends on.
