# Link-Cut Tree for Dynamic Tree Connectivity

**Category:** Algorithmic Challenges
**Difficulty:** X (brief: "Support link/cut/connectivity queries in amortized O(log n).")

**Status:** Implemented (Rust)

A forest starts as `n` isolated vertices and changes by `link(u, v)` and `cut(u, v)`; the question is
whether `u` and `v` are in one tree. The brief's answer is the link-cut tree, and this crate builds
it with its usual extensions (evert, lazy path add, subtree sums). It then asks the more interesting
question, **is the link-cut tree still the right answer in 2026?**, by building the two designs that
claim to beat it and racing all of them on the same operation streams.

| Structure       | link / cut / connected | path aggregate + add | subtree aggregate           |
| --------------- | ---------------------- | -------------------- | --------------------------- |
| `NaiveForest`   | O(n)                   | O(n)                 | O(n); the test oracle       |
| `PathLct`       | O(log n) amortized     | O(log n) amortized   | no                          |
| `SubtreeLct`    | O(log n) amortized     | O(log n) amortized   | **sum and count**, O(log n) |
| `EulerTourTree` | O(log n) expected      | no                   | sum, count, **min, max**    |
| `UfoTree`       | O(min(log n, D))       | no                   | no                          |

On top of the link-cut tree sits **offline fully dynamic connectivity** on a general graph: edges
arrive and disappear, cycles are allowed, and every `conn u v` is answered in O(log n) with no
replacement-edge search ([`connectivity`](src/connectivity.rs)).

`#![forbid(unsafe_code)]` in the library. Edition 2024, current crates (`criterion` 0.8,
`proptest` 1.12, `clap` 4.6, `foldhash` 0.2). Every structure is checked against a walk-the-graph
oracle, and the Euler tour tree and UFO tree also re-verify their own structural invariants after
every operation.

```bash
cargo test --release                       # 21 integration tests (also run in debug for overflow checks)
cargo bench --bench forest                 # criterion, n = 100,000, four tree shapes
cargo run --release --example scale        # the markdown tables below, n = 1,000,000 (about 12 minutes)
cargo run --release -- gen --shape random --n 1000 --ops 2000 > forest.txt
cargo run --release -- run forest.txt --structure subtree-lct
cargo run --release -- gen-events --n 1000 --edges 2000 --events 10000 > events.txt
cargo run --release -- connect events.txt --verify
```

## What I looked for first

The brief says "amortized O(log n)", which Sleator and Tarjan's splay-based tree has met since 1983.
Before writing anything I searched for what has replaced it:

- **UFO trees** (De Man et al., [arXiv 2601.10706](https://arxiv.org/html/2601.10706v1), 2026) are
  the newest sequential contender: a tree-contraction hierarchy with unbounded fan-out and
  O(min(log n, D)) updates, where D is the diameter. The same paper benchmarks them against ten
  other implementations and reports that the link-cut tree is "still the fastest sequential data
  structure", with UFO trees 3.23x slower on average. Its case for UFO is parallel batch updates and
  low-diameter inputs, not raw sequential speed.
- **Batch-parallel link-cut trees** (De Man and Dhulipala,
  [arXiv 2609.14746](https://arxiv.org/abs/2609.14746), September 2026) beat every other parallel
  batch-dynamic tree on path queries, but the headline is parallel throughput. I did not build a
  parallel structure.
- **Euler tour trees** are what dynamic connectivity is actually built on (Holm, de Lichtenberg and
  Thorup keep one per level), though an experimental comparison of tree structures for connectivity
  on fully dynamic graphs ([arXiv 2501.02278](https://arxiv.org/pdf/2501.02278)) found the
  Euler-tour-based ones never the best on any dimension. They are the only one of the three with
  subtree min/max.

So the plan was: build all three, make them agree with an oracle, and measure. None of the code is
copied; the UFO tree in particular is written from the paper's contraction rules (below) and is
validated by its own invariant checker rather than by the paper's delete-ancestors pseudocode.

## Design notes

### Link-cut tree (`src/lct.rs`)

A tree is cut into vertex-disjoint **preferred paths**; each path is a splay tree ordered by depth,
and the root of a splay tree points at the parent of its path's top vertex. Everything is built on
`access(v)`, which makes the root-to-`v` path preferred by splaying up through path-parent pointers.

- **Arena of 64-byte nodes, node 0 a null sentinel.** A missing child is index 0, so rotations and
  tag pushes need no `Option` branches, and the whole forest is one `Vec`.
- **One `expose(u, v)` serves connectivity, path queries and link.** It runs `make_root(u)`,
  `access(v)`, `splay(u)`. If `u` and `v` are in one tree, the splay gives `v` a parent; if not, `v`
  stays the root of its own tree. So connectivity falls out of a splay that was needed anyway, and
  afterwards the splay tree holds exactly the path `u..=v` with its aggregate at the root.
- **Evert is a lazy reversal flag.** The aggregates are commutative (`count`, `sum`, `min`, `max`),
  so reversing a path only swaps children and never recomputes anything.
- **Subtree sums** (`SubtreeLct`, selected by a `const SUB: bool`) keep, per vertex, the total of its
  *virtual* children, the subtrees hanging off it that are not on its preferred path. `access` moves
  one subtree into that total and one out, which needs subtraction. That is why the LCT answers
  subtree **sum and count only**: min and max have no inverse, and keeping them would need a
  multiset per vertex. The extra bookkeeping lives in a separate array, so `PathLct` keeps 64-byte
  nodes and pays nothing for it.
- **`subtree_query(p, v)`** is `make_root(p)`, `access(v)`, then `v`'s own value plus its virtual
  total: everything on `v`'s side of the edge.

### Euler tour tree (`src/ett.rs`)

Each tree is the closed walk that crosses every edge once in each direction, stored as a cyclic
sequence in a treap with parent pointers: one node per vertex (holding the value) and two neutral
**arc** nodes per edge.

- `reroot` rotates the cycle (one split, one merge); `link` is reroot, arc, reroot, arc; `cut`
  removes the two arcs and what lies between them is the far side.
- **The subtree of `v` across the edge `p - v` is exactly the cyclic stretch between the arc
  `p > v` and the arc `v > p`.** Sums are two prefix reads that never mutate the tour; min and max
  split the tour, read, and join it back.
- Connectivity is "same treap root", found by walking parent pointers.

### UFO tree (`src/ufo.rs`)

The forest is contracted level by level. Level 0 is the vertices; one round merges clusters by:

1. **Stars.** A cluster of degree 3 or more (a *centre*) merges with **all** of its degree-1
   neighbours, however many, into one parent. A star of a million leaves has height 2.
2. **Pairs.** The remaining degree-1 and degree-2 clusters are matched along a *maximal* matching.
3. A cluster of degree 0 is the top of its tree and has no parent.

An edge change is applied at level 0 and projected up through the parents; every cluster whose
neighbourhood changed is queued, and the levels are repaired bottom-up, each queued cluster
re-checking the three rules for itself and queueing whatever it displaces. Attaching or detaching a
leaf of a big star touches only that leaf, because the star's parent is kept; that is what makes
a hub cheap.

It supports connectivity only (compare the roots of the two cluster trees). The paper's path and
subtree queries are not ported.

`UfoTree::check_invariants` re-derives every rule from scratch (symmetric neighbours on one level,
parent/child agreement, neighbours equal to the projection of the children's neighbours, every
cluster connected, stars and pairs well-formed, maximality), and the tests call it after each
update. Testing paid off early: the first version let a star outlive its centre, leaving a
cluster disconnected, and a debug assertion plus a shrinking search pinned it to a 7-operation
sequence.

### Offline dynamic connectivity (`src/connectivity.rs`)

A link-cut tree only holds a forest, so a graph with cycles needs a policy. Knowing every deletion
time in advance, keep a **maximum spanning forest keyed by deletion time**. Each edge becomes its own
tree node carrying `deletion_time << 32 | id`, so the minimum over a tree path names the edge that
dies soonest:

- Inserting `(u, v)` between different trees links through the edge node. Inside one tree, compare
  with the soonest-dying edge on the path: if it dies first, swap; otherwise the new edge is dropped
  for good, since it will be dead before any edge it could replace.
- Deleting a tree edge never searches for a replacement, because the deleted edge has the earliest
  deletion time of every live edge, so no live non-tree edge can cover it.

I did not find a published source for this exact formulation, so the correctness argument above is
my own; the tests check it against a union-find rebuilt for every query. The price is that the
event list must be known up front. The online version needs Holm, de Lichtenberg and Thorup's level
hierarchy of Euler tour forests and O(log² n) amortized, which is out of scope here.

## Results

Operation streams are valid by construction: each step cuts a random tree edge `(u, v)` and
relinks `u` to a random vertex of the other side, so the forest keeps changing shape. A stream is
followed by its exact inverse so it can loop on a built forest. Times are nanoseconds per
operation. One machine (Core Ultra 7 255H), one thread, one seed; individual cells moved by up to
40% between two runs (the random-tree 90%-queries LCT cell read 3351 and 2408), so read the
orderings, not the third digit. Heap is bytes per vertex after the replays (steady) and at the
high-water mark (peak, which also counts the temporary values array of the weighted structures).

**n = 1,000,000**, from `cargo run --release --example scale`:

| shape  | structure    | build ns/edge | churn, 10% queries | 90% queries | 90% queries on 64 hot vertices | heap B/vertex |
| ------ | ------------ | ------------- | ------------------ | ----------- | ------------------------------ | ------------- |
| random | `PathLct`    | 2469          | **706**            | 2408        | **378**                        | 64            |
|        | `SubtreeLct` | 2147          | 840                | 2962        | 434                            | 96            |
|        | `EulerTour`  | 5039          | 4888               | 2974        | 925                            | 228           |
|        | `UfoTree`    | 3863          | 3591               | **1811**    | 688                            | 268           |
| path   | `PathLct`    | 354           | **1849**           | **2906**    | **552**                        | 66            |
|        | `SubtreeLct` | 316           | 2130               | 3671        | 579                            | 98            |
|        | `EulerTour`  | 964           | 8488               | 5161        | 1641                           | 228           |
|        | `UfoTree`    | 621           | 16889              | 6887        | 4084                           | 258           |
| star   | `PathLct`    | 353           | **195**            | 481         | 106                            | 64            |
|        | `SubtreeLct` | 420           | 236                | 735         | 137                            | 96            |
|        | `EulerTour`  | 742           | 1465               | 3862        | 300                            | 228           |
|        | `UfoTree`    | 620           | 454                | **101**     | **36**                         | 149           |
| binary | `PathLct`    | 649           | **1175**           | 3792        | 563                            | 64            |
|        | `SubtreeLct` | 744           | 1268               | 4536        | 626                            | 96            |
|        | `EulerTour`  | 1053          | 6266               | 3675        | 943                            | 228           |
|        | `UfoTree`    | 1018          | 2572               | **1768**    | **484**                        | 256           |

UFO tree heights at n = 10⁶: 24 (random), 25 (path), 21 (binary), 2 (star); log₂ n is 19.9.

Aggregate queries, criterion, n = 100,000, per query (a batch of 1,024 random pairs, or random tree
edges for the subtree queries):

| query                            | random tree | star    |
| -------------------------------- | ----------- | ------- |
| path aggregate, `PathLct`        | 830 ns      | 93 ns   |
| path aggregate, `SubtreeLct`     | 911 ns      | 116 ns  |
| subtree sum, `SubtreeLct`        | 391 ns      | 59 ns   |
| subtree sum, `EulerTourTree`     | 677 ns      | 509 ns  |
| subtree min/max, `EulerTourTree` | 1452 ns     | 1222 ns |

Offline connectivity, 20,000 events (30% queries) on 2,000 vertices holding about 4,000 edges: the
link-cut-tree solver takes **11.0 ms**; rebuilding a union-find for every query takes **181 ms**.

### What the numbers say

- **For updates the splay link-cut tree wins everywhere**, as the paper found: 4.6 to 7.5x faster
  than the Euler tour tree and 2.2 to 9.1x faster than the UFO tree on churn. It also has the smallest
  footprint (64 bytes per vertex, 96 with subtree sums).
- **UFO trees win connectivity queries when the diameter is small**, for the reason the paper gives:
  a UFO query only reads parent pointers, while an LCT `connected` restructures the tree three
  times. At 90% queries the UFO tree is 4.8x faster on the star, 2.1x on the binary tree and 1.3x
  on the random tree. On a path, with diameter n, it is 2.4x *slower*, and its updates are the
  slowest of all (16.9 µs per operation on a hierarchy 25 levels tall).
- **A self-adjusting tree rewards locality.** On 64 hot vertices the LCT gets 6.4x faster than on
  uniform queries (2408 to 378 ns on random trees), the Euler tour tree 3.2x and the UFO tree
  2.6x. Part of that is plain cache locality for all three, so the gap is an upper bound on what
  adaptivity buys, not a measurement of it.
- **The Euler tour tree is dominated on speed and size** but is the only structure that answers
  subtree *min and max*. Even for subtree sums the augmented LCT is 1.7x faster on random trees and
  8.6x on a star.
- **Subtree bookkeeping costs the LCT about 10% on path queries, 8 to 21% on churn and 20 to 50%
  on connectivity queries**, plus 32 bytes per vertex. Use `PathLct` unless you need subtree sums.

## Limitations

- **The UFO tree answers connectivity only.** The paper's path, subtree and LCA queries (its
  appendix) are not ported, so it is not a drop-in replacement for the LCT here.
- **No parallelism.** The batch-parallel structures (UFO batches, MOJOS) are not built; the
  comparison is sequential, which is where the link-cut tree is strongest.
- **`SubtreeLct` has no subtree min/max** (no inverse); use `EulerTourTree` for those.
- **Offline connectivity only.** The online problem is Holm, de Lichtenberg and Thorup's.
- **Sums are `i64` and unchecked in release.** Keep `|sum| < 2⁶³`; debug builds panic on overflow.
- **Vertex counts are `u32` internally.** More than about 4 billion vertices overflows.
- **Treap depth is random.** The Euler tour tree uses a fixed seed, so runs are repeatable, but its
  O(log n) is expected rather than amortized.

## Where this is actually used

I did not find a production system that ships a link-cut tree; the evidence is from the literature.

**Link-cut trees.** Sleator and Tarjan introduced them (Journal of Computer and System Sciences,
1983) for network flow: nearest common ancestors, blocking and maximum flows in O(mn log n) on
sparse networks, constrained minimum spanning trees, and the network simplex method for min-cost
flow. Goldberg and Tarjan (1988) put the same structure inside push-relabel for O(nm log(n²/m)), and
Goldberg, Grigoriadis and Tarjan (1991) used it to bring a network simplex maximum-flow method to
O(nm log n). Those are asymptotic results; the sources I found do not say whether the trees paid off
in practice.

**Euler tour trees.** JGraphT ships `TreeDynamicConnectivity`, a dynamic forest "based on the Euler
tour technique" of Henzinger and King (STOC 1995) with `link`, `cut` and `connected` in O(log n).
The Python package `hdtgraph` implements Holm, de Lichtenberg and Thorup's fully dynamic
connectivity with an Euler tour tree for each spanning forest.

**UFO trees** and the batch-parallel structures are 2026 research results; I found no deployment.

## Layout

```
src/lib.rs           traits: Forest, Weighted, PathOps, SubtreeOps
src/lct.rs           splay link-cut tree (PathLct, SubtreeLct)
src/ett.rs           Euler tour tree on a treap
src/ufo.rs           UFO tree and its invariant checker
src/connectivity.rs  offline dynamic connectivity
src/naive.rs         the oracle
src/script.rs        the command language, runner and generators
src/shapes.rs        tree shapes, operation streams, SplitMix64
src/bin/lct.rs       CLI: run, gen, connect, gen-events
tests/               equivalence (proptest), connectivity, ufo, script_cli
benches/forest.rs    criterion; examples/scale.rs prints the tables above
```
