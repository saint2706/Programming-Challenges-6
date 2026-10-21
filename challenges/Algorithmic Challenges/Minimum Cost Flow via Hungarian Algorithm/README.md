# Minimum Cost Flow via Hungarian Algorithm

**Category:** Algorithmic Challenges
**Difficulty:** X (brief: "Solve the assignment problem optimally on a weighted bipartite graph.")

**Status:** Implemented (Rust)

The brief asks for the Hungarian algorithm. This crate builds it, then asks what a person who needs an
optimal assignment today should run, and answers by building the alternatives and racing them: eight
assignment solvers, three of which are general min-cost-flow solvers, every answer proved optimal by
a check that does not trust the solver, benchmarked against SciPy's algorithm and two Rust crates.

| Solver            | What it is                                                      | Input                                                     | Notes                                         |
| ----------------- | --------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------- |
| `hungarian`       | Kuhn-Munkres, potentials form (Edmonds-Karp, Tomizawa)          | dense, any shape, forbidden edges                         | O(n²m); the tie-break rule decides its speed  |
| `crouse`          | Jonker-Volgenant shortest augmenting path, Crouse's form        | dense, any shape, forbidden edges                         | what SciPy's `linear_sum_assignment` runs     |
| `jv`              | full LAPJV: column reduction, reduction transfer, row reduction | dense, square (rectangular is padded), no forbidden edges | 65x faster than `crouse` on one family        |
| `lapmod`          | sparse shortest augmenting paths over CSR                       | sparse, any shape                                         | in the spirit of LAPMOD, not a port           |
| `auction`         | Bertsekas' auction with epsilon scaling                         | sparse or dense, square (rectangular is padded)           | exact on integer costs; flat across families  |
| `ssp`             | successive shortest paths with potentials                       | any network                                               | the Hungarian algorithm, seen as a flow       |
| `cost-scaling`    | Goldberg-Tarjan push-relabel with global price updates          | any network                                               | what OR-Tools and LEMON use for general flows |
| `network-simplex` | primal simplex on a strongly feasible spanning tree             | any network                                               | block-search pivoting                         |

All eight also solve assignments: the last three through `flow::assignment_network`, which turns a
cost matrix into a transportation network. `#![forbid(unsafe_code)]` in the library; edition 2024
and current crates (`clap` 4.6, `criterion` 0.8, `proptest` 1.12, `pathfinding` 4.16, `lapjv` 0.3).
`ndarray` is pinned to 0.16 because the `lapjv` crate's matrix type is that version's `Array2`; it
is a dev-dependency only.

```bash
cargo test --release                                         # 50 tests (also run in debug for overflow checks)
uv run --script scripts/make_fixtures.py                     # rebuild tests/fixtures from SciPy and NetworkX
cargo run --release --example scale                          # every table below (about 10 minutes)
cargo run --release --example scale -- dense                 # or sparse | flow | tuning
cargo bench --bench solvers                                  # criterion intervals, one size per family
cargo run --release -- gen assignment uniform --n 500 --out a.csv
cargo run --release -- compare a.csv                         # all eight solvers, one table
cargo run --release -- verify a.csv --algo auction           # solve, then prove it optimal two ways
cargo run --release -- convert a.csv --to min > a.min        # the assignment as a DIMACS min-cost flow
cargo run --release -- solve a.min --algo network-simplex
```

## What I looked for first

I searched for what has replaced the Hungarian algorithm and found no new algorithm that clearly
has, only a division of labour:

- **Dense assignment on one CPU core**: Jonker-Volgenant. SciPy's documentation describes
  `linear_sum_assignment` as "a modified Jonker-Volgenant algorithm with no initialization",
  from Crouse, *IEEE TAES* 52(4), 2016. This is `crouse` here; `jv` adds the initialisation
  phases that Crouse leaves out.
- **Assignment at scale**: Goldberg and Kennedy's cost-scaling algorithm (1995), which is what
  OR-Tools' `LinearSumAssignment` implements, and Guthe and Thuerck's *Algorithm 1015* (ACM TOMS,
  2021): successive shortest paths whose duals are first approximated with the auction algorithm's
  epsilon scaling, built for parallel hardware. I did not build either (see Limitations).
- **General min-cost flow**: cost-scaling push-relabel (OR-Tools' `SimpleMinCostFlow` documents
  exactly that; LEMON's `CostScaling`) and network simplex (LEMON's `NetworkSimplex`, POT's
  `ot.emd`). LEMON's own documentation names those two as its fastest implementations.
- **Almost-linear time**: Chen, Kyng, Liu, Peng, Probst Gutenberg and Sachdeva (FOCS 2022,
  arXiv 2203.00671) compute min-cost flow in `m^(1+o(1))` time. The abstract describes no implementation
  or experiment, so it is not here.

So the question this crate can answer is the practical one: among the methods people ship, which wins
on which instance, and by how much.

## Design notes

### Exact integers, with limits stated up front (`src/matrix.rs`, `src/flow.rs`)

Costs are `i64`. Floating-point duals are the classic way to get an assignment solver stuck: a
reduced cost that should be exactly 0 comes out as -1e-16, a Dijkstra step goes the wrong way, and the
solver loops or returns a slightly wrong matching. Exactness costs a range limit instead, checked on
construction: `|cost| <= 2^32`, dimensions up to `2^24`, flow capacities and supplies up to `2^30`.
Inside those limits every potential and every sum fits an `i64`. The two solvers that scale costs
(auction, cost scaling) check their own tighter bound and return `OutOfRange`, never a wrapped value.

### Shapes and forbidden edges

`rows <= cols` matches every row and leaves columns free; `rows > cols` matches every column and
leaves rows free (SciPy's convention). Forbidden edges are `FORBIDDEN` in a dense matrix and absent
in a sparse one; when no full-size matching avoids them the answer is `Error::Infeasible`, from every
solver. Dense solvers that need a square matrix pad with zero-cost dummy rows, which leaves the
optimum unchanged.

### Proofs, not trust (`src/verify.rs`)

Every optimal answer can be checked two independent ways:

- `check_duals`: the solver's dual variables satisfy complementary slackness (`c_ij >= u_i + v_j`,
  equality on matched edges, `<= 0` and zero-when-free on the larger side), so `sum(u) + sum(v)` is a
  lower bound that the assignment meets. O(nnz).
- `certify`: no negative cycle exists in the residual graph (Bellman-Ford, a zero-cost hub joins free
  and matched vertices so that swapping a column also shows up as a cycle). It uses no dual at all, so
  it works for the auction and the flow solvers, which hand back none.

The tests run both on every solver, against a bitmask-DP brute force on small instances, against SciPy
and NetworkX on 102 fixtures (including which ones are infeasible), against `lapjv` and `pathfinding`,
and against Machol-Wien's closed-form optimum at several sizes. The checks themselves were tested by
planting bugs: a flipped sign in the simplex flow update, a halved potential step in SSP, a Hungarian
dual off by one. Each is caught. One planted bug was not: relabelling cost scaling by `min + 2 eps`
instead of `min + eps` still returned optimal flows on every test instance, because the weaker
invariant only changes an answer when two solutions differ by one cost unit along a long cycle. So
`cost_scaling::solve_with_stats` reports the worst reduced cost left after every phase, and a test
asserts it never exceeds epsilon; that bug now fails.

### The Hungarian algorithm and its tie-break (`src/hungarian.rs`)

The potentials form adds rows one at a time with a Dijkstra-like search over tight edges. One line
decides how it behaves on matrices with many equal costs: when several columns tie for the smallest
slack, take the first, or take one that is still free? The usual code takes the first, which can walk
through every tied *matched* column before meeting a free one. Taking a free one ends the search at
once. On costs `0..10` at n = 3000 that is the difference between 23.4 s and 28 ms
([measured below](#tuning)); `pathfinding::kuhn_munkres` takes 20.5 s on the same matrix. `TieBreak`
exposes both and the solvers use `PreferFree`.

### Crouse and LAPJV (`src/jv.rs`)

`crouse` is the rectangular shortest-augmenting-path search with no initialisation, and it already
prefers a free column on ties. `jv` is the original LAPJV: column reduction, reduction transfer, two
rounds of augmenting row reduction, then shortest paths for whatever is still free. The initialisation
is worth nothing on a hard instance and nearly everything on a structured one.

### Sparse: LAPMOD-style (`src/lapmod.rs`)

A CSR matrix, every row starting on its cheapest edge, then one binary-heap Dijkstra per still-free row
that stops at the first free column. Dual updates are Crouse's, so the same certificate applies. It is
a sparse JV-style solver, not LAPMOD: it has neither LAPMOD's core-oriented refinement nor its
column-reduction initialisation.

### Auction (`src/auction.rs`)

Persons bid for objects, the winner pays its margin over its runner-up plus epsilon, and a displaced
person bids again. With costs multiplied by `n + 1`, a final `epsilon = 1` leaves under one unit of
total slack, which on integer costs means optimal. Epsilon falls by a constant factor per phase while
prices carry over. A Hopcroft-Karp pass decides feasibility first, because an infeasible auction does
not stop; it just keeps raising prices.

### Min-cost flow (`src/ssp.rs`, `src/cost_scaling.rs`, `src/simplex.rs`)

`assignment_network` is the transportation network: each row supplies one unit, each column demands
one, each allowed edge is a unit-capacity arc, and a dummy node absorbs the surplus when the sides
differ. **SSP on that network is the Hungarian algorithm**, with the potentials as its dual variables.
It matters which network, though: a first version used a source feeding every row, and every Dijkstra
started by flooding all free rows (886 ms against 25 ms on a sparse n = 2000 instance), because the search must begin at *one* row to be the Hungarian step.

- **SSP** saturates negative-cost arcs up front so Dijkstra is valid, stops at the first deficit node,
  and moves only the settled nodes' potentials (by `dist - D`, which changes no reduced cost), so an
  iteration costs what it explored.
- **Cost scaling** runs push-relabel refine phases at falling epsilon, with Goldberg's global price
  update (a backwards shortest-path labelling from the deficit nodes) triggered by the arc-scan work of
  relabels. Without it, dense assignment is slow; see [Tuning](#tuning).
- **Network simplex** keeps a strongly feasible tree (Cunningham's rule: the blocking arc met last on
  the cycle leaves), block-search pivoting, and re-hangs only the moved subtree through child lists.

## Results

Windows 11, Intel Core Ultra 7 255H, 31 GB RAM, rustc 1.99.1, single-threaded solvers, seed 1. Each
cell runs in its own process with a 90 s limit, builds its instance first, and times `build + solve`
from the matrix (for the flow solvers that includes building the network; for the rival crates, their
matrix type). It reports the median of up to 7 runs while a cell is short, and the peak heap above the
instance. `skip` means the input would not fit: flow solvers above 2^21 arcs, dense solvers on a sparse
instance with more than 2^25 cells. Every table's costs agree across all solvers (the harness prints a
warning when they do not). `rival-lapjv` is the `lapjv` crate, `rival-pathfinding` is
`pathfinding::kuhn_munkres_min`.

### Dense assignment, uniform costs 0..1000

| solver              | n=100          | n=300          | n=1000          | n=3000          |
| ------------------- | -------------- | -------------- | --------------- | --------------- |
| `hungarian`         | 0.23 ms / 1 MB | 2.72 ms / 1 MB | 41 ms / 1 MB    | 1.58 s / 1 MB   |
| `crouse`            | 0.18 ms / 1 MB | 1.81 ms / 1 MB | 18 ms / 1 MB    | 713 ms / 1 MB   |
| `jv`                | 0.03 ms / 1 MB | 0.49 ms / 1 MB | 7.17 ms / 1 MB  | 236 ms / 1 MB   |
| `lapmod`            | 0.35 ms / 1 MB | 3.58 ms / 2 MB | 46 ms / 12 MB   | 2.93 s / 193 MB |
| `auction`           | 0.32 ms / 1 MB | 2.50 ms / 2 MB | 40 ms / 12 MB   | 969 ms / 192 MB |
| `ssp`               | 0.91 ms / 1 MB | 12 ms / 9 MB   | 261 ms / 90 MB  | skip            |
| `cost-scaling`      | 3.00 ms / 2 MB | 38 ms / 13 MB  | 1.89 s / 128 MB | skip            |
| `network-simplex`   | 0.81 ms / 1 MB | 14 ms / 8 MB   | 146 ms / 75 MB  | skip            |
| `rival-lapjv`       | 0.03 ms / 1 MB | 0.41 ms / 1 MB | 10 ms / 8 MB    | 210 ms / 69 MB  |
| `rival-pathfinding` | 0.32 ms / 1 MB | 4.15 ms / 2 MB | 52 ms / 23 MB   | 4.15 s / 231 MB |

### Dense assignment, costs 0..10 (many ties)

| solver              | n=100          | n=300          | n=1000          | n=3000           |
| ------------------- | -------------- | -------------- | --------------- | ---------------- |
| `hungarian`         | 0.09 ms / 1 MB | 0.35 ms / 1 MB | 3.25 ms / 1 MB  | 31 ms / 1 MB     |
| `crouse`            | 0.04 ms / 1 MB | 0.33 ms / 1 MB | 2.83 ms / 1 MB  | 33 ms / 1 MB     |
| `jv`                | 0.02 ms / 1 MB | 0.26 ms / 1 MB | 3.18 ms / 1 MB  | 82 ms / 1 MB     |
| `lapmod`            | 0.26 ms / 1 MB | 1.77 ms / 2 MB | 16 ms / 12 MB   | 233 ms / 192 MB  |
| `auction`           | 0.27 ms / 1 MB | 2.76 ms / 2 MB | 41 ms / 12 MB   | 704 ms / 192 MB  |
| `ssp`               | 2.34 ms / 1 MB | 46 ms / 9 MB   | 3.12 s / 90 MB  | skip             |
| `cost-scaling`      | 2.66 ms / 2 MB | 32 ms / 13 MB  | 892 ms / 128 MB | skip             |
| `network-simplex`   | 0.59 ms / 1 MB | 7.39 ms / 8 MB | 58 ms / 75 MB   | skip             |
| `rival-lapjv`       | 0.04 ms / 1 MB | 0.42 ms / 1 MB | 8.53 ms / 8 MB  | 114 ms / 69 MB   |
| `rival-pathfinding` | 0.99 ms / 1 MB | 16 ms / 2 MB   | 471 ms / 23 MB  | 20.53 s / 231 MB |

### Dense assignment, Euclidean distances

| solver              | n=100          | n=300          | n=1000          | n=3000          |
| ------------------- | -------------- | -------------- | --------------- | --------------- |
| `hungarian`         | 0.28 ms / 1 MB | 3.90 ms / 1 MB | 109 ms / 1 MB   | 1.45 s / 1 MB   |
| `crouse`            | 0.16 ms / 1 MB | 2.07 ms / 1 MB | 44 ms / 1 MB    | 975 ms / 1 MB   |
| `jv`                | 0.13 ms / 1 MB | 1.80 ms / 1 MB | 41 ms / 1 MB    | 943 ms / 1 MB   |
| `lapmod`            | 0.47 ms / 1 MB | 5.98 ms / 2 MB | 108 ms / 13 MB  | 1.41 s / 196 MB |
| `auction`           | 0.28 ms / 1 MB | 3.19 ms / 2 MB | 54 ms / 12 MB   | 816 ms / 192 MB |
| `ssp`               | 1.12 ms / 1 MB | 14 ms / 10 MB  | 587 ms / 90 MB  | skip            |
| `cost-scaling`      | 3.66 ms / 2 MB | 41 ms / 13 MB  | 2.43 s / 128 MB | skip            |
| `network-simplex`   | 0.84 ms / 1 MB | 7.09 ms / 8 MB | 92 ms / 75 MB   | skip            |
| `rival-lapjv`       | 0.16 ms / 1 MB | 1.92 ms / 1 MB | 40 ms / 8 MB    | 806 ms / 69 MB  |
| `rival-pathfinding` | 0.37 ms / 1 MB | 4.71 ms / 2 MB | 98 ms / 23 MB   | 1.52 s / 231 MB |

### Dense assignment, separable `a_i + b_j + noise`

| solver              | n=100          | n=300          | n=1000          | n=3000           |
| ------------------- | -------------- | -------------- | --------------- | ---------------- |
| `hungarian`         | 1.29 ms / 1 MB | 21 ms / 1 MB   | 526 ms / 1 MB   | 11.18 s / 1 MB   |
| `crouse`            | 0.80 ms / 1 MB | 12 ms / 1 MB   | 327 ms / 1 MB   | 9.03 s / 1 MB    |
| `jv`                | 0.07 ms / 1 MB | 1.20 ms / 1 MB | 10 ms / 1 MB    | 139 ms / 1 MB    |
| `lapmod`            | 1.22 ms / 1 MB | 16 ms / 2 MB   | 384 ms / 12 MB  | 16.38 s / 192 MB |
| `auction`           | 0.34 ms / 1 MB | 2.90 ms / 2 MB | 34 ms / 12 MB   | 553 ms / 192 MB  |
| `ssp`               | 2.44 ms / 1 MB | 47 ms / 9 MB   | 3.50 s / 90 MB  | skip             |
| `cost-scaling`      | 4.30 ms / 2 MB | 37 ms / 13 MB  | 1.22 s / 128 MB | skip             |
| `network-simplex`   | 0.69 ms / 1 MB | 5.97 ms / 8 MB | 68 ms / 75 MB   | skip             |
| `rival-lapjv`       | 0.09 ms / 1 MB | 1.37 ms / 1 MB | 9.38 ms / 8 MB  | 145 ms / 69 MB   |
| `rival-pathfinding` | 1.01 ms / 1 MB | 18 ms / 2 MB   | 486 ms / 23 MB  | 18.85 s / 231 MB |

### Dense assignment, Machol-Wien `c_ij = i * j`

| solver              | n=100          | n=300          | n=1000          | n=3000           |
| ------------------- | -------------- | -------------- | --------------- | ---------------- |
| `hungarian`         | 0.76 ms / 1 MB | 19 ms / 1 MB   | 706 ms / 1 MB   | 23.79 s / 1 MB   |
| `crouse`            | 0.81 ms / 1 MB | 7.70 ms / 1 MB | 287 ms / 1 MB   | 14.92 s / 1 MB   |
| `jv`                | 0.65 ms / 1 MB | 13 ms / 1 MB   | 423 ms / 1 MB   | 16.58 s / 1 MB   |
| `lapmod`            | 2.25 ms / 1 MB | 59 ms / 2 MB   | 2.22 s / 20 MB  | 84.28 s / 320 MB |
| `auction`           | 0.32 ms / 1 MB | 4.20 ms / 2 MB | 66 ms / 12 MB   | 2.43 s / 192 MB  |
| `ssp`               | 3.71 ms / 1 MB | 81 ms / 10 MB  | 4.83 s / 98 MB  | skip             |
| `cost-scaling`      | 5.88 ms / 2 MB | 92 ms / 13 MB  | 2.73 s / 128 MB | skip             |
| `network-simplex`   | 1.58 ms / 1 MB | 28 ms / 8 MB   | 1.17 s / 75 MB  | skip             |
| `rival-lapjv`       | 0.51 ms / 1 MB | 12 ms / 1 MB   | 423 ms / 8 MB   | 16.37 s / 69 MB  |
| `rival-pathfinding` | 0.90 ms / 1 MB | 21 ms / 2 MB   | 681 ms / 23 MB  | 27.81 s / 231 MB |

### Sparse assignment (DIMACS "low-cost" shape: degree `2 * ceil(log2 n)`, costs 0..100)

| solver            | n=2048         | n=8192         | n=32768         |
| ----------------- | -------------- | -------------- | --------------- |
| `hungarian`       | 241 ms / 32 MB | skip           | skip            |
| `crouse`          | 116 ms / 32 MB | skip           | skip            |
| `lapmod`          | 12 ms / 1 MB   | 106 ms / 1 MB  | 1.10 s / 4 MB   |
| `auction`         | 1.20 ms / 1 MB | 8.07 ms / 1 MB | 63 ms / 1 MB    |
| `ssp`             | 20 ms / 4 MB   | 208 ms / 19 MB | 4.28 s / 83 MB  |
| `cost-scaling`    | 38 ms / 6 MB   | 298 ms / 27 MB | 5.73 s / 119 MB |
| `network-simplex` | 13 ms / 4 MB   | 115 ms / 16 MB | 1.28 s / 70 MB  |

### General min-cost flow (random feasible networks, 5 arcs per node, capacities 1..100)

Non-negative costs 0..1000:

| solver            | n=1000                         | n=5000                          | n=20000                             | n=50000                             |
| ----------------- | ------------------------------ | ------------------------------- | ----------------------------------- | ----------------------------------- |
| `ssp`             | 9.37 ms / 1 MB                 | 196 ms / 2 MB                   | 1.34 s / 7 MB                       | 8.96 s / 17 MB                      |
| `cost-scaling`    | 13 ms / 1 MB (35,977 relabels) | 92 ms / 2 MB (231,927 relabels) | 513 ms / 10 MB (1,073,509 relabels) | 2.03 s / 24 MB (2,749,928 relabels) |
| `network-simplex` | 3.25 ms / 1 MB (2,827 pivots)  | 33 ms / 2 MB (13,552 pivots)    | 223 ms / 7 MB (47,743 pivots)       | 1.06 s / 18 MB (123,705 pivots)     |

Costs -250..1000 (negative cycles present):

| solver            | n=1000                         | n=5000                           | n=20000                             | n=50000                             |
| ----------------- | ------------------------------ | -------------------------------- | ----------------------------------- | ----------------------------------- |
| `ssp`             | 24 ms / 1 MB                   | 494 ms / 2 MB                    | 5.81 s / 7 MB                       | 36.85 s / 18 MB                     |
| `cost-scaling`    | 17 ms / 1 MB (48,637 relabels) | 151 ms / 2 MB (383,635 relabels) | 887 ms / 10 MB (1,819,915 relabels) | 3.48 s / 24 MB (5,773,852 relabels) |
| `network-simplex` | 8.81 ms / 1 MB (4,243 pivots)  | 228 ms / 2 MB (30,311 pivots)    | 5.59 s / 7 MB (169,100 pivots)      | 36.34 s / 18 MB (518,405 pivots)    |

### Tuning

Cost scaling on uniform dense assignment, with the global price update off (`cs-plain`) and on:

| solver         | n=100                           | n=300                           | n=1000                             |
| -------------- | ------------------------------- | ------------------------------- | ---------------------------------- |
| `cs-plain`     | 3.15 ms / 2 MB (6,599 relabels) | 51 ms / 13 MB (46,206 relabels) | 2.39 s / 128 MB (294,127 relabels) |
| `cost-scaling` | 2.83 ms / 2 MB                  | 32 ms / 13 MB                   | 1.80 s / 128 MB                    |

The trigger for an update is the number of arcs relabels have scanned, as a multiple of the residual
arc count. I swept 0.05 to 2 on three dense families: small multiples are much worse (2.95 s at 0.05
against 1.53 s at 1 on uniform n = 1000), and 1 and 2 are close (1.53 and 1.69 s on uniform, 0.91 and
0.77 s on costs 0..10, 1.29 and 1.04 s on Euclidean n = 800). I kept 1.

Hungarian tie-break on costs 0..10:

| solver            | n=300          | n=1000         | n=3000         |
| ----------------- | -------------- | -------------- | -------------- |
| `hungarian-first` | 22 ms / 1 MB   | 693 ms / 1 MB  | 23.42 s / 1 MB |
| `hungarian`       | 0.34 ms / 1 MB | 3.06 ms / 1 MB | 28 ms / 1 MB   |

Auction, epsilon-scaling factor (bids and time; the default is 7):

| factor | uniform 1000 bids | time  | sparse 8192 bids | time    |
| -----: | ----------------: | ----- | ---------------: | ------- |
|      2 |            43,995 | 35 ms |          348,154 | 12 ms   |
|      4 |            31,159 | 24 ms |          226,882 | 8.51 ms |
|      7 |            34,755 | 27 ms |          214,236 | 8.23 ms |
|     10 |            34,009 | 26 ms |          208,475 | 8.08 ms |
|     16 |            37,394 | 29 ms |          215,016 | 8.23 ms |
|     32 |            52,073 | 41 ms |          245,016 | 9.41 ms |

### What the numbers say

- **One CPU core, dense, typical costs: use LAPJV.** From n = 1000 up, `jv` is within 20% of the
  `lapjv` crate on every family and faster on some (82 against 114 ms on costs 0..10 at n = 3000,
  but 943 against 806 ms on Euclidean). On uniform n = 3000 it takes 236 ms, 6.7x faster than the
  Hungarian algorithm and 18x faster than `pathfinding::kuhn_munkres`.
- **Initialisation is worth 65x or nothing.** On separable costs `jv` takes 139 ms at n = 3000,
  while `crouse` (the same search, no initialisation) takes 9.0 s and `hungarian` 11.2 s. On
  Machol-Wien, built to defeat the Hungarian method, `jv` is *slower* than `crouse` (16.6 s against
  14.9 s): the initialisation finds nothing to exploit.
- **The tie-break line is worth up to 840x.** Costs 0..10 at n = 3000: 23.4 s with "first minimal
  column", 28 ms with "a free one"; `pathfinding::kuhn_munkres` takes 20.5 s on that matrix.
- **The auction is the only solver whose time barely depends on the instance.** At n = 3000 it takes
  553 ms to 2.43 s across the five families, where the others range from 31 ms to 84 s. It is 4x
  slower than `jv` on uniform costs and 6x faster than the best other solver on Machol-Wien (2.43 s
  against 14.9 s). That fits the motivation of Guthe and Thuerck's Algorithm 1015, whose abstract claims
  a runtime nearly independent of instance difficulty; this is the one-core, forward-auction version of
  the idea, not their algorithm.
- **Sparse inputs reward leaving the matrix alone.** At n = 32,768 with about 30 edges per row the
  auction takes 63 ms against 1.10 s for `lapmod`, 1.28 s for network simplex, 4.3 s for SSP and 5.7 s
  for cost scaling. A dense solver would need a matrix of 10^9 cells; at n = 2048, where it fits,
  `crouse` takes 116 ms against `lapmod`'s 12 ms.
- **General flow has no single winner.** On non-negative costs network simplex is fastest (1.06 s at
  50,000 nodes, against 2.03 s for cost scaling and 8.96 s for SSP). With negative costs the order
  flips: cost scaling takes 3.5 s against 36 s for both network simplex and SSP. Network simplex's
  cost per pivot jumps from 8.6 us to 70 us there (518,405 pivots). I did not isolate why. One
  candidate is the ring my generator closes through every node, which may make the spanning tree
  deep so that re-hanging a subtree walks a long way; I have not tested that.
- **Cost scaling is the slowest dense solver on uniform and Euclidean costs** (1.89 s at n = 1000
  against `jv`'s 7 ms), even with price updates, which cut it from 2.39 s to 1.80 s in the tuning
  run. SSP is the slowest on the other three families (3.12 s against cost scaling's 0.89 s on
  costs 0..10). Cost scaling here is the general-network algorithm; the assignment-specific one
  (Goldberg and Kennedy's, see Limitations) is what OR-Tools ships for this problem.
- **Memory.** The in-place dense solvers need nothing beyond the matrix. The flow solvers build
  about 90 bytes per arc (75 to 128 MB at n = 1000, one million arcs), which is why they are skipped
  at n = 3000.

## Limitations

- **Not built: Goldberg and Kennedy's assignment cost scaling** (double push, the algorithm in
  OR-Tools' `LinearSumAssignment`), **Guthe and Thuerck's Algorithm 1015**, anything parallel or on a
  GPU, and the almost-linear-time algorithm. The cost-scaling numbers above are for the general
  network version only.
- `lapmod` is a sparse JV-style solver, not LAPMOD itself. The auction is forward only, square only,
  and reports no duals (the verifier derives its certificate from the assignment instead).
- Costs are bounded by `2^32` in magnitude, flow capacities and supplies by `2^30`; self-loops and
  arc lower bounds other than 0 are rejected.
- One machine, one seed per cell, the median of at most 7 runs, no confidence intervals in the
  tables (`cargo bench --bench solvers` gives them for one size per family). Nothing was measured on
  another CPU or operating system.
- The flow-network results depend on my generator, which closes a ring through every node so that
  every instance is feasible. A different generator could move the comparison between network
  simplex and cost scaling, especially with negative costs.
- The DIMACS "low-cost" family is reproduced from its description (degree `2 log2 n`, costs up to
  100), with a planted perfect matching so every instance is feasible, not from the original
  generator. The original families are described as easy for modern solvers, so they are one data
  point here, not the whole picture.

## Where this is actually used

SciPy's `linear_sum_assignment` is the workhorse: its documentation describes it as a modified
Jonker-Volgenant algorithm with no initialization (Crouse 2016), the routine `crouse` reimplements.
Facebook's DETR object detector calls it in its `HungarianMatcher` to pair predicted boxes with
ground-truth boxes while training. Multi-object trackers use the same family: SORT's
`linear_assignment` calls `lap.lapjv` (falling back to SciPy), and ByteTrack calls `lap.lapjv` to
match detections with tracks. For general flows, OR-Tools documents its min-cost-flow solver as a
cost-scaling push-relabel implementation and its `LinearSumAssignment` as Goldberg and Kennedy's
cost-scaling algorithm; LEMON's `NetworkSimplex` and `CostScaling` are the two implementations its
documentation calls fastest; and POT's `ot.emd`, which computes exact earth mover's distances, is a
network simplex solver (Bonneel et al. 2011).

## Layout

```
src/
  lib.rs            module map and re-exports
  matrix.rs         Dense and Sparse (CSR) cost matrices, the exact-arithmetic limits
  assign.rs         Algorithm enum, solve_dense / solve_sparse, transposition and padding
  hungarian.rs      Kuhn-Munkres potentials form, TieBreak
  jv.rs             Crouse's shortest augmenting path and full LAPJV
  lapmod.rs         sparse shortest augmenting paths
  auction.rs        epsilon-scaling auction
  matching.rs       Hopcroft-Karp (feasibility for the auction)
  flow.rs           Network, FlowAlgorithm, assignment_network, the residual graph
  ssp.rs            successive shortest paths
  cost_scaling.rs   push-relabel cost scaling, global price update, per-phase statistics
  simplex.rs        network simplex, pivot statistics
  maxflow.rs        Dinic (feasibility for cost scaling, min-cut tests)
  verify.rs         dual certificates and negative-cycle search for assignments and flows
  instances.rs      SplitMix64 and the instance families (uniform, Machol-Wien, geometric, ...)
  dimacs.rs         DIMACS min / asn readers and writers, the matrix format
  bin/mcflow.rs     CLI: solve, compare, verify, gen, convert
tests/              assignment, flow, fixtures (SciPy / NetworkX), rivals, formats, cli
benches/solvers.rs  criterion, one size per family
examples/scale.rs   the tables above, one child process per cell
scripts/make_fixtures.py  reference answers from SciPy and NetworkX
```
