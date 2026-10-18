# Heavy-Light Decomposition for Tree Path Queries

**Category:** Algorithmic Challenges
**Difficulty:** X (brief: "O(log² n) path queries/updates on a static tree.")

**Status:** Implemented (Rust)

The brief's O(log² n) is the textbook bound: heavy-light decomposition (HLD) cuts any root path into
O(log n) chains, and a segment tree answers each chain in O(log n). This crate builds that baseline
and then the faster designs that beat it, so the interesting question is where, and by how much.

A static tree, a value on every vertex, and these operations: sum/min/max over a **path** or a
**subtree**, **add** to every vertex of a path (lazy), **assign** one vertex, **LCA** and **k-th
ancestor**. There is also an **edge-weighted** variant and a **CLI** over a text file.

| Structure     | path query / add | point set | subtree query | Idea                                                     |
| ------------- | ---------------- | --------- | ------------- | -------------------------------------------------------- |
| `NaiveTree`   | O(depth)         | O(1)      | O(n)          | walk parent pointers; the test oracle                    |
| `ClassicHld`  | O(log² n)        | O(log n)  | O(log n)      | one lazy segment tree over the heavy-first preorder      |
| `BalancedHld` | **O(log n)**     | O(log n)  | not supported | each chain is a static weighted-centroid tree            |
| `TopTree`     | **O(log n)**     | O(log n)  | **O(log n)**  | `BalancedHld` plus balanced rake trees of light subtrees |

`#![forbid(unsafe_code)]`. Edition 2024, current crates (`criterion` 0.8, `proptest` 1.12, `clap`
4.6). All passes are iterative, so a million-vertex path cannot overflow the stack.

```bash
cargo test --release                    # 19 tests (also run in debug so overflow checks fire)
cargo bench --bench queries             # n = 1,000,000, four tree shapes
cargo run --release --example depth     # measured depth of the global trees
cargo run --release -- gen --shape adversarial --n 100000 --queries 5000 > t.txt
cargo run --release -- run t.txt --structure toptree
cargo run --release -- run edges.txt --edges        # weights on edges
```

## Where the designs come from

I looked for existing implementations first. No Rust implementation of either idea turned up (the
one HLD crate found was a brand-new classic one), so these are Rust ports written against the
following sources. None of their code is copied.

- **Classic HLD**: [cp-algorithms](https://cp-algorithms.com/graph/hld.html) (one global segment tree,
  heavy-first positions, the path loop that picks the endpoint with the deeper chain head) and
  Al.Cash's Codeforces template.
- **Globally balanced HLD** ("global biased tree"): [myee's write-up](https://www.cnblogs.com/myee/p/global-biased-tree.html).
  Each chain becomes a static binary tree whose root is the *weighted centroid*, where vertex `v`
  weighs `size[v] - size[heavy[v]]` (itself plus its light subtrees). A light edge is a path-parent
  pointer as in a link-cut tree, so a root walk crosses O(log n) nodes in total: stepping over a
  light edge halves the subtree, stepping down a chain tree halves the weight.
- **Static top tree**: the [AtCoder ABC351-G editorial](https://atcoder.jp/contests/abc351/editorial/9899)
  and a [scrapbox walk-through](https://scrapbox.io/Atcoder-myReflection/StaticTopTree). Light
  children are combined by a *rake* tree balanced by size, so changing one light child costs
  O(log(size(v) / size(child))), which telescopes along a root path to O(log n). A heap or multiset
  per vertex (what the global-biased-tree post uses) is O(log degree) per light edge and O(log² n) in
  the worst case.

The two newest ideas are really one data structure: `BalancedHld` is the chain-tree half, and
`TopTree` is that plus the rake half. They share one engine (`GlobalTree<const SUB: bool>`), so the
benchmark difference between them is exactly the cost of subtree aggregates.

## Design notes

**One aggregate for everything.** `Stats { count, sum, min, max }` is commutative and has no
inverse, which is why min/max need a real structure and not prefix sums. The `count` is what makes a
*neutral* vertex possible: `shift(delta)` multiplies by `count` and leaves an empty aggregate alone,
so the `i64::MAX`/`MIN` sentinels never overflow.

**Edge weights by subdividing every edge.** Edge `e = (a, b)` becomes a vertex `n + e` between `a`
and `b`. Edge vertices carry the weight, original vertices are neutral. A path then covers exactly
the edges on it, a subtree covers the edges strictly below `v`, and every structure above works
unchanged. `kth_ancestor(v, k)` becomes `2k` in the subdivided tree. Nothing needs an inverse, so
min/max work on edges too.

**Chain trees, lazily.** Chain-tree node ids are vertex ids. A range add tags a whole subtree of a
chain tree; a node's own aggregate is updated immediately and the tag waits for its children, so
`push` runs top-down along the path before any recompute. A prefix of a chain (the pieces of a path
query) costs the depth of its endpoint, and the sum over the chains of a path is the O(log n).

**Subtree aggregates need two numbers per node.** A chain-tree node keeps `own` (its chain vertices
only, which a lazy add can shift) and `light_sub` (the light subtrees hanging off those vertices,
which a path add never touches). The whole-subtree aggregate is their combine, recomputed on demand,
because `min(own + delta, light)` is not `min(own, light) + delta`.

**A refresh climb that stays O(log n).** After a path add, every touched chain must refresh its rake
node and its parent vertex, then keep climbing. Naively each touched chain climbs to the root, which
is O(log n) chains times O(log n) steps. Each climb instead stops at the next touched chain, which
refreshes itself (an epoch stamp marks them), so one add is one climb. I caught this reading the code
before benchmarking and did not measure the naive version, so the saving is argued, not timed.

## Measured

**Depth.** The deepest vertex of the combined structure (chain-tree edges plus rake edges), from
`cargo run --release --example depth`, for the shapes in `src/shapes.rs`:

| n         | log₂ n | worst `BalancedHld` | worst `TopTree` | worst light edges on a root path |
| --------- | -----: | ------------------: | --------------: | -------------------------------: |
| 1,000     |   10.0 |                  13 |              13 |                                8 |
| 100,000   |   16.6 |                  17 |              17 |                               15 |
| 1,000,000 |   19.9 |                  22 |              22 |                               18 |

The worst case over every shape tracks `log₂ n + 2`, and a test pins it to `ceil(log₂ n) + 4`. A
single long chain, the case where binary-search-tree HLD usually degrades, comes out at 19 for a
million vertices.

**Time per operation**, n = 1,000,000, one thread, criterion medians over batches of 1,024 operations
(`path_add` counts the +1 and the −1 separately), Core Ultra 7 255H, rustc 1.98.1:

| Operation     | Shape (pairs)         | `ClassicHld` | `BalancedHld` |    `TopTree` |
| ------------- | --------------------- | -----------: | ------------: | -----------: |
| path query    | random tree, random   |     3,192 ns |  **1,090 ns** | **1,074 ns** |
| path query    | binary tree, random   |     5,024 ns |  **2,333 ns** | **2,196 ns** |
| path query    | adversarial, far pair |     1,408 ns |    **196 ns** |   **212 ns** |
| path query    | adversarial, random   |   **744 ns** |      2,036 ns |     1,865 ns |
| path query    | one path, random      |   **564 ns** |      1,966 ns |     1,941 ns |
| path add      | random tree, random   |     3,275 ns |  **1,241 ns** |     3,530 ns |
| path add      | binary tree, random   |     6,166 ns |  **2,254 ns** |     6,240 ns |
| path add      | adversarial, far pair |     1,711 ns |    **301 ns** |       548 ns |
| path add      | one path, random      |   **515 ns** |      2,200 ns |     4,018 ns |
| point set     | random tree           |       178 ns |     **42 ns** |       876 ns |
| point set     | one path              |   **181 ns** |        625 ns |     1,017 ns |
| subtree query | random tree           |       280 ns |           n/a |    **55 ns** |
| subtree query | one path              |   **284 ns** |           n/a |       978 ns |
| build         | random, 1M            |   **139 ms** |        401 ms |       936 ms |

(The `build` row includes cloning the layout but not the 335 ms `Layout::new` pass every structure needs.)

What the numbers say, honestly:

- **The log² gap is real, and the adversarial far pair shows it.** On the tree built so a root path
  crosses ~11 long chains, balancing is 7x faster for queries and 5.7x faster for path adds.
- **On a plain tree it is a 2-3x win, not an asymptotic one.** Random and binary trees have short
  chains, so the segment tree's extra log is small.
- **On a long single chain, classic HLD wins by 3.5-4x**: one contiguous segment-tree range,
  predictable memory, against a ~19-deep pointer descent. The same source notes the global balanced
  tree is often no faster than HLD on random data and sometimes slower; that holds here.
- **Updates are where the top tree pays.** A point set climbs through the rake trees too, about
  0.9-1.5 µs against 178 ns for the segment tree; subtree queries are the reverse on bushy trees
  (55 ns against 280 ns) and 3.5x worse on a path.
- **`BalancedHld` is the better default for path-only work on branching trees; `ClassicHld` wins on
  chain-like trees and for point updates; `TopTree` is what to reach for when the workload mixes
  path adds with subtree aggregates.**

## Testing

19 tests, all also run in a debug build so integer-overflow checks are live.

- **Equivalence with the oracle (proptest)**: random trees of up to 70 vertices, a random root, 20%
  neutral vertices, up to 120 mixed operations, all four structures, every answer compared to
  `NaiveTree`; plus deep narrow trees and every shape up to 2,500 vertices. I broke the code three
  ways (skip the light refresh in `set`, skip it in `path_add`, flip the chain-head comparison in
  classic HLD) and each break made the suite fail.
- **Layout against brute force**: sizes, depths, subtree contiguity, heavy children, chain
  partition, light-edge counts, LCA and k-th ancestor, recomputed with plain loops.
- **Depth bound** for both global trees on every shape, up to 200,000 vertices.
- **Edge-weighted variant** against an independent edge-walking oracle, on all four structures.
- **Input validation**: wrong edge count, out-of-range vertex, self loop, cycle, bad root.
- **CLI and script parser**: hand-computed answers, identical output across structures, line-numbered
  errors, exit code 2 on bad input.
- A million-vertex path builds and answers without overflowing the stack.

Testing caught one real bug: `EdgeTree` accepted an *edge vertex* as the root, because the
subdivided tree has more vertices than the original.

## Limitations

- **No subtree aggregates in `BalancedHld`**, by design: without the rake trees a chain tree has
  nowhere to keep them. `SubtreeOps` is implemented only where it is supported.
- **No subtree *add*.** Path add is lazy everywhere; adding to a whole subtree would need a second
  tag over the rake trees.
- **Static topology.** No link/cut; that is the next challenge (a link-cut tree).
- **Sums are `i64` and unchecked in release.** Keep `|sum| < 2⁶³`; debug builds panic on overflow.
- **LCA and k-th ancestor** use the shared chain-hopping layout, O(log n), not the balanced trees.
- **One machine, one thread, one seed.** Absolute numbers will move; the ordering between shapes is
  the finding. The adversarial tree is a worst case for chain-by-chain HLD, not typical input.
- **Memory is not measured.** `TopTree` keeps four 32-byte `Stats` per vertex plus child, parent and
  range pointers; `ClassicHld` allocates one `Stats` and one tag for each of 4n segment-tree slots.

## Where this is actually used

HLD is mainly a competitive-programming technique. I did not find a verified production system that
runs it, so this section names none. The documented story is what came after it:

**Top trees.** Alstrup, Holm, de Lichtenberg and Thorup (ACM Transactions on Algorithms, 2005) give
top trees as a general interface for maintaining information over a fully dynamic forest. Their
examples include the diameter, center and median of each tree, with O(log n) updates. The `TopTree`
in this crate is the static, fixed-topology version of that idea, so it is the closest relative.

**Link-cut trees.** Sleator and Tarjan's 1983 dynamic trees handle link, cut and path queries in
O(log n) per operation. They were used for maximum flow, giving an O(mn log n) algorithm for sparse
networks. HLD cannot do any of the link or cut work, because it assumes the tree never changes shape.

**Where the static assumption breaks.** Any system that adds or removes edges while answering path
queries needs a dynamic structure. The `Limitations` section above lists this crate's boundary.
