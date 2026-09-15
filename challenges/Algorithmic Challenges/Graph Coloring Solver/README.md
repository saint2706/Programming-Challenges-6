# Graph Coloring Solver

**Category:** Algorithmic Challenges
**Difficulty:** A (brief: "Greedy, backtracking, and Welsh-Powell, compare chromatic counts")

**Status:** Implemented (Python)

Vertex coloring assigns every vertex a color so that no edge joins two
same-colored vertices, using as few colors as possible. The minimum possible
count is the graph's **chromatic number**, and deciding it is NP-hard --
so everything here except the exact solver is a heuristic with no
optimality guarantee, however good it looks in practice:

| Method                   | Idea                                                 | Guarantee                     |
| ------------------------ | ---------------------------------------------------- | ----------------------------- |
| `greedy_color`           | Color in a given/natural order, smallest free color  | <= max degree + 1 colors      |
| `welsh_powell`           | Greedy, pre-sorted by descending *static* degree     | Often better, no guarantee    |
| `dsatur`                 | Greedy, always recompute the most-constrained vertex | Exact on bipartite/chordal    |
| `rlf`                    | Build one whole color class at a time                | No guarantee, often tight     |
| `exact_chromatic_number` | Clique lower bound + backtracking, binary-searched k | Always optimal (NP-hard cost) |

The brief names greedy, backtracking, and Welsh-Powell. DSATUR (Brelaz,
1979) and RLF (Leighton, 1979) are added because they're the two heuristics
most commonly cited as *actual* improvements over Welsh-Powell in the
graph-coloring literature, and comparing four heuristics against a real
exact solver (rather than against each other alone) is the only way to know
which of "close to optimal" and "optimal" any of them actually achieve.

## Why order is everything for greedy coloring

`greedy_color` never uses more than `max_degree + 1` colors, for *any*
order -- that bound is easy to prove and doesn't depend on picking a good
order at all. What the bound doesn't promise is that greedy's actual output
gets anywhere near the chromatic number. The **crown graph** `S_n^0`
(complete bipartite `K(n,n)` minus a perfect matching -- vertex `a_i`
adjacent to every `b_j` except `b_i`) is the standard demonstration: it's
bipartite, so its chromatic number is 2 for any n, but its *default*
insertion order (`a_1, b_1, a_2, b_2, ...`) is exactly the pathological
order that forces greedy far away from that:

```
$ uv run --with networkx python coloring.py
Crown graph (n=6): 2-colorable, but order matters for plain greedy.
  exact chromatic number:                     2
  greedy_color (all a's, then all b's):        2 colors
  greedy_color (a1,b1,a2,b2,... interleaved):  6 colors
  dsatur:                                      2 colors
```

Six colors for a 2-chromatic graph, from the same algorithm that gets it
exactly right when handed the vertices in a different order. **Welsh-Powell
doesn't rescue this case either**, and the reason is specific to this
graph's structure: `S_n^0` is regular (every vertex has the same degree,
`n - 1`), so sorting by descending degree has nothing to distinguish
vertices by, and Python's *stable* sort leaves the original (bad) order
untouched -- confirmed in the benchmark table below, where Welsh-Powell
matches plain greedy's 10 colors on `crown n=10` exactly. **DSATUR is
immune** to this specific trap because it never commits to a fixed order at
all: saturation degree (colors already forced onto a vertex's neighbors)
changes as coloring proceeds, so DSATUR keeps re-asking "who's most
constrained *right now*" instead of trusting a snapshot taken before any
color was placed.

## DSATUR: saturation degree, not just degree

Static degree (how many neighbors a vertex has) is available before
coloring starts. **Saturation degree** -- the number of *distinct* colors
already used among a vertex's neighbors -- can only be computed as coloring
proceeds, and it's the sharper signal: a vertex touching three neighbors
that all happen to share one color is *less* constrained right now than a
vertex touching two neighbors with two different colors. DSATUR always
colors whichever uncolored vertex currently has the highest saturation
degree (ties broken by static degree), which makes it provably exact on
bipartite and chordal graphs and, in practice, one of the strongest
heuristics that isn't outright exponential.

This implementation recomputes the max-saturation vertex with a linear scan
every round: O(n) per step, O(n^2) total. A binary-heap-backed DSATUR
reaches O((V+E) log V) with incremental priority updates -- a known, real
improvement, deliberately left unimplemented here for the same reason
A-ExpJ was left out of the Reservoir Sampling challenge in this repo: an
incremental-heap DSATUR that's subtly wrong about *when* to re-key a vertex
is worse than an obviously-correct O(n^2) version, and O(n^2) is nowhere
near the actual bottleneck once `exact_chromatic_number`'s exponential
backtracking is in the same file.

## RLF: one color class at a time

Every method above colors one *vertex* at a time. Recursive Largest First
(Leighton, 1979) instead builds one whole *color class* at a time: start
the class with the highest-degree remaining vertex, then repeatedly add
whichever remaining candidate has the most neighbors already excluded from
this class (spend the "most used up" vertices now, save the more flexible
ones for later classes), until no candidate is left that doesn't conflict
with the class so far. This is the heuristic that comes out ahead most
often in the benchmark below -- narrowly beating DSATUR on both queen
graphs -- at the cost of being the most expensive heuristic here (each
color class construction rescans the shrinking candidate set).

## Exact solving: clique lower bound + backtracking

Every clique of size `omega` in the graph needs `omega` distinct colors, so
`omega(G) <= chi(G)` always -- computing the exact maximum clique size
(via networkx's Bron-Kerbosch `find_cliques`) gives a real starting lower
bound for free. The best of the four heuristics above gives an upper bound.
`exact_chromatic_number` then tries `k` from that lower bound upward,
running a backtracking search for each `k` with one symmetry-breaking rule
(a vertex may only introduce color `m` if colors `0..m-1` are already used
somewhere -- coloring is only defined up to relabeling colors, so there's no
reason to explore permutations of the same partition twice). The first
feasible `k` is returned, which is provably minimal because every smaller
`k` was exhaustively proven infeasible along the way.

**A free optimality certificate.** Whenever the returned `k` equals the
clique lower bound exactly, `chi(G) = omega(G)` is *proven* on the spot --
no external reference chromatic number is needed to trust the answer. That
happens for every "easy" family in the benchmark below (cycles, complete
graphs, Petersen, crown, and every random graph tested). It does *not*
happen for the queen graphs (`chi > omega` there, confirmed by the
backtracking search actually ruling out every `k` below the answer it
found) or for the Groetzsch graph (triangle-free, `omega = 2`, yet
`chi = 4` -- the textbook example of a graph where the clique bound is
nowhere close to tight). Separately, for every graph with <= 8 vertices,
`test_coloring.py` checks `exact_chromatic_number`'s answer against a fully
independent brute-force oracle (try every `k` from 1 up, test literally
every `k^n` coloring) -- so the exact solver's correctness doesn't rest on
trusting its own clique-bound reasoning alone.

## Correctness

```
$ uv run --with pytest --with networkx pytest -q
76 passed in 0.35s
```

Covers: every heuristic produces a proper coloring and respects the
`max_degree + 1` bound on random graphs; known chromatic numbers (empty
graph, single vertex/edge, `K_n` needs exactly `n` colors, even cycles are
bipartite, odd cycles need 3, Petersen is 3-chromatic, the Groetzsch graph
is 4-chromatic and triangle-free, crown graphs are 2-chromatic for any n);
the crown-graph order-sensitivity result stated above, both directions
(natural order optimal, interleaved order forced to `>= n/2` colors, DSATUR
immune to both); `exact_chromatic_number` against the brute-force oracle on
10 random 8-vertex graphs; and structural checks on the `queen_graph`
generator (row/column/diagonal adjacency, symmetry).

## Benchmarks

```
$ uv run --with networkx python benchmark.py
Chromatic count: heuristics vs the true chromatic number
graph                         n  greedy  welsh-p  dsatur   rlf  EXACT
---------------------------------------------------------------------
cycle C7 (odd)                7       3        3       3     3      3
cycle C8 (even)               8       2        2       2     2      2
complete K8                   8       8        8       8     8      8
petersen                     10       3        3       3     3      3
groetzsch (mycielski C5)     11       4        4       4     4      4
crown n=10 (2-chromatic)     20      10       10       2     2      2
queen4_4                     16       6        6       5     5      5
queen5_5                     25       8        7       5     5      5
queen6_6                     36      11        9       9     8      7
queen7_7                     49      10       12      11     9      7
random G(20, 0.5)            20       8        6       6     6      6
random G(40, 0.5)            40      11        9       8     9      8

exact_chromatic_number runtime as the (queen graph) instance grows
graph           n  chromatic number        time
-----------------------------------------------
queen4_4        16                 5      0.000s
queen5_5        25                 5      0.001s
queen6_6        36                 7      0.011s
queen7_7        49                 7      0.379s

exact_chromatic_number runtime on random G(n, 0.5) graphs
     n  chromatic number      time
----------------------------------
    30                 7    0.001s
    35                 7    0.003s
    40                 8    0.071s
    45                 9    0.399s
    50                 9    2.098s
    55                 9    0.741s
```

`queen8_8` (64 vertices) is deliberately not in the automated benchmark: it
did not finish within a 45-second budget checked manually. That's the real
behavior of an NP-hard exact solver meeting a genuinely hard instance, not
something to paper over with a bigger timeout -- queen graphs are dense,
highly symmetric, and a well-known stress case for exactly this reason
(they show up as `queen*_*` instances in the DIMACS graph-coloring
benchmark suite). Contrast with the random-graph table, which reaches
`n = 55` in well under a second: **how big a graph the exact solver can
handle depends on structure, not vertex count** -- `queen7_7` (49 vertices)
takes noticeably longer than `random G(55, 0.5)` (55 vertices), the
opposite of what raw size would predict. The chromatic-count table adds two
more specific, sometimes counterintuitive results: **RLF wins outright** on
both queen graphs (8 and 9 colors respectively, the closest of the four
heuristics to EXACT's 7 on each), and on `queen7_7`, **Welsh-Powell (12) is
worse than plain default-order greedy (10)** -- a concrete demonstration
that a "smarter-looking" static ordering rule is not a strict improvement
over no ordering rule at all, unless (like DSATUR) it's recomputed
dynamically as coloring proceeds.

## Run it

```bash
cd "challenges/Algorithmic Challenges/Graph Coloring Solver"

uv run --with networkx python coloring.py      # crown-graph order demo
uv run --with networkx python benchmark.py     # chromatic-count + runtime tables

uv run --with pytest --with networkx pytest -q   # 76 tests
```

## Where this is used

**Register allocation in compilers.** The classic Chaitin-Briggs approach
models variables as vertices, "live at the same time" as edges, and finds a
coloring with `k` colors = `k` physical registers; when no such coloring
exists, some variables get "spilled" to memory instead. This is the
application that made graph coloring a core compilers topic, not just a
graph-theory curiosity.

**Exam / meeting scheduling.** Vertices are exams (or meetings); an edge
means "share a student (or participant) and can't happen at the same time."
A coloring with `k` colors is a valid schedule using `k` time slots -- exact
solving matters here specifically because minimizing slots is the actual
business goal, not just "a working schedule."

**Wireless frequency assignment.** Transmitters that could interfere (are
within range of each other) get an edge; colors are frequency channels. The
same DSATUR/RLF-family heuristics used here are standard in that literature
because exact solving is infeasible at real network sizes but "provably
close to optimal in practice" heuristics matter for spectrum, which is
expensive.

**Sudoku, and constraint satisfaction generally.** A Sudoku puzzle is
literally graph coloring: 81 vertices (cells), edges between cells sharing a
row/column/3x3 box, 9 colors (digits), with some vertices pre-colored as
givens. The backtracking-with-forward-pruning structure in
`exact_chromatic_number` is the same family of technique (just without the
pre-colored constraints) used by real constraint-satisfaction solvers.

## Sources

- Welsh, D.J.A. & Powell, M.B., "An upper bound for the chromatic number of a graph and its application to timetabling problems," *The Computer Journal* 10(1):85-86, 1967.
- [Brelaz, D., "New methods to color the vertices of a graph," *Communications of the ACM* 22(4):251-256, 1979](https://doi.org/10.1145/359094.359101) -- DSATUR.
- Leighton, F.T., "A Graph Coloring Algorithm for Large Scheduling Problems," *Journal of Research of the National Bureau of Standards* 84(6):489-506, 1979 -- Recursive Largest First.
- Garey, M.R. & Johnson, D.S., *Computers and Intractability*, 1979 -- graph k-colorability's NP-completeness (for k >= 3).
- [Bron, C. & Kerbosch, J., "Algorithm 457: finding all cliques of an undirected graph," *Communications of the ACM* 16(9):575-577, 1973](https://doi.org/10.1145/362342.362367) -- the max-clique algorithm behind the lower bound (via `networkx.find_cliques`).
- [DIMACS Graph Coloring Instances](https://mat.tepper.cmu.edu/COLOR/instances.html) -- the standard hard-instance benchmark suite that queen graphs (`queen5_5`, `queen6_6`, ...) belong to.
