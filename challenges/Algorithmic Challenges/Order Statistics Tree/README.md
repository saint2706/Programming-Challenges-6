# Order Statistics Tree

**Category:** Algorithmic Challenges
**Difficulty:** A (brief: "Augmented BST for O(log n) rank/select queries.")

**Status:** Implemented (Rust)

An order-statistic structure is an ordered multiset that, besides
insert/remove, answers two questions in O(log n):

- `rank(x)`: how many elements are strictly smaller than `x`?
- `select(i)`: what is the `i`-th smallest element (0-based)?

The textbook answer (CLRS ch. 14) is a balanced binary search tree with a
`size` field in every node. That is implemented here as three variants (AVL,
treap, red-black) and is the correctness baseline. The **headline
implementation** is the structure that is faster in practice: an
arena-allocated **B-tree with subtree counts**, following the cache-layout
argument of Khuong & Morin's *Cache-Friendly Search Trees* (arXiv 1907.01631).
Wide nodes make a descent touch a handful of cache lines instead of ~1.4·log₂ n
scattered binary-tree nodes.

| Structure              | rank/select | insert/remove | Notes                                               |
| ---------------------- | ----------- | ------------- | --------------------------------------------------- |
| `OrderStatBTree<K, T>` | O(log n)    | O(log n)      | **headline**; const-generic min degree `T`          |
| `AvlTree`              | O(log n)    | O(log n)      | CLRS size augmentation, arena                       |
| `Treap`                | O(log n) ev | O(log n) ev   | split/merge, SplitMix64 priorities                  |
| `RedBlackTree`         | O(log n)    | O(log n)      | CLRS parent-pointer + NIL sentinel, augmented sizes |
| `SortedVec`            | O(log n)    | O(n)          | memmove insert                                      |
| `Fenwick`              | O(log U)    | O(log U)      | fixed integer universe only                         |
| `NaiveBTreeMap` (std)  | O(n)        | O(log n)      | shows why `BTreeMap` can't do this                  |

All of them implement one trait, `OrderStatistics<K>`, so tests and benchmarks
treat them uniformly. The crate is `#![forbid(unsafe_code)]`: every tree is a
`Vec` of nodes addressed by index, with a free list for recycled slots.

## Design of `OrderStatBTree`

- **Classic B-tree, keys in every node**, min degree `T` (non-root nodes hold
  `T-1 ..= 2T-1` keys). Each internal node stores `counts[i]`, the number of
  keys in the subtree under `children[i]`.
- **Duplicates** are a first-class multiset. Inserts land *after* equal keys
  (upper bound); `rank` is a lower bound (count of keys `< x`); `rank_upper`
  counts `<= x`; `count(x) = rank_upper - rank`. Because keys live in internal
  nodes too, there is one in-order sequence and no separator-with-duplicates
  corner cases (which are what make duplicate-tolerant B+-trees painful).
- **`rank`**: descend, at each node `partition_point` the keys and add
  `i + sum(counts[..i])`. **`select`**: walk `counts` left to right,
  subtracting, until the index lands on a key or inside a child.
- **Removal is by position** (`remove_at(idx)`), and `remove(&key)` is
  `rank` + `remove_at`. A key in an internal node is replaced by its in-order
  predecessor; underflow is repaired on the way back up (borrow from a
  sibling, else merge). All count bookkeeping happens in `borrow_*`/`merge`.
- **Derived queries** all fall out of `rank`/`select`: `predecessor`,
  `successor`, `count_range`, `min`/`max`, `range(..)` iterator (start located
  by rank, so it is O(log n + k)).
- **Bulk load** `from_sorted(Vec<K>)` builds a perfectly packed tree in O(n):
  pick the minimal height that fits `n`, then split keys evenly among the
  fewest legal children per level.
- `check_invariants()` verifies sortedness, fill bounds, counts, uniform leaf
  depth and that no arena slot leaked; the tests call it after every step.

Not implemented: `split_off_at_rank`/join. A correct O(log n) B-tree split
needs per-level node splitting and rebalancing on both edges, and an O(n)
implementation (drain + `from_sorted`) would add nothing over what bulk
loading already shows.

## Usage

```bash
cargo test --release                 # 26 tests (12 unit-style API tests + 14 property tests)
cargo nextest run --release          # same, via cargo-nextest
cargo clippy --all-targets -- -D warnings
cargo run --release --bin demo       # leaderboard, sliding-window median, vs std BTreeMap
cargo bench --bench structures       # all structures, n = 1e3..1e6
cargo bench --bench btree_tuning     # node-size sweep + bulk-load vs incremental
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --test btree_api -- miri_smoke
```

```rust
use ost::OrderStatBTree;

let mut t: OrderStatBTree<u32> = OrderStatBTree::new();
for score in [50, 20, 90, 20, 70] { t.insert(score); }
assert_eq!(t.rank(&70), 3);              // 20, 20, 50 are below 70
assert_eq!(t.select(0), Some(&20));      // minimum
assert_eq!(t.count_range(20..=70), 4);   // counted, not iterated
assert_eq!(t.predecessor(&70), Some(&50));
```

Real output of `cargo run --release --bin demo`:

```
== leaderboard ==
score 9000: 9783 players are strictly better; percentile 90.2
p50 = 4993
p90 = 8978
p99 = 9895
players scoring 4000..=4100: 1040 (counted, not iterated)
height 4 for 100000 keys

== rank on 1M keys: order-stat B-tree vs std BTreeMap ==
rank(750000) = 750000: 3.2µs vs 1.4458ms (452x)
```

## Verification

- **Property tests** (proptest): each of the seven structures is driven by
  random insert/remove/rank/select sequences and compared with a sorted-`Vec`
  oracle after every operation. Small key domains force heavy duplication and
  rebalancing; the B-tree is tested at `T = 2, 3, 4, 16` (`T = 2` is a 2-3-4
  tree, the most rebalancing-dense shape). Invariants are validated after every
  step.
- **Edge cases**: empty tree, single element, remove-to-empty, 50 duplicates,
  strict predecessor/successor, every range-bound combination, non-`Copy` keys
  (`String`), ascending/descending insert-then-drain (the classic rebalancing
  killer), arena slot recycling, height bound at 100k keys.
- **Bulk load**: every `n` in `0..1500` for `T = 2, 3, 4`, checked for
  invariants and content, plus further inserts/removals on a bulk-loaded tree.
- **Miri** (nightly, `miri_smoke` and five API tests) passes. The crate has no
  `unsafe`, so this mostly confirms the standard library usage is clean.
- **cargo-fuzz** was tried and does not work on this Windows/MSVC machine (the
  ASan runtime DLL is missing, and `-s none` fails to link `__sancov_pcs`), so
  no fuzz target is shipped rather than an unverified one.

Testing found one real bug during development: `iter_from_rank(len)` (an
empty tail, e.g. `range(a..)` with `a` above every key) fell off the end of an
internal node because the last child must absorb `idx == len`. It is fixed and
covered by `ranges_and_iteration_match_a_sorted_vec`.

## Benchmarks

Single run of `cargo bench` (criterion, 10 samples, ~1 s each) on a Windows 11
laptop, keys uniform in `0..2^22`, median time **per operation** in ns. Treat
differences under ~30% as noise; the ratios that matter are much larger.

| ns/op, n = 1,000,000 | insert | rank | select | mixed (25% each) |
| -------------------- | -----: | ---: | -----: | ---------------: |
| **B-tree (T=16)**    |    274 |  158 |     93 |          **472** |
| AVL                  |    767 |  258 |    307 |            2,293 |
| Red-black            |    708 |  360 |    381 |            2,719 |
| Treap                |  1,476 |  699 |    754 |            2,467 |
| Fenwick (U=2^22)     |     79 |   19 |     97 |              127 |

| ns/op, n = 100,000       | insert |   rank |  select |   mixed |
| ------------------------ | -----: | -----: | ------: | ------: |
| B-tree (T=16)            |     91 |     90 |      28 |     153 |
| AVL                      |    257 |     86 |     106 |     425 |
| Red-black                |    315 |     93 |      78 |     200 |
| Treap                    |    733 |    189 |     219 |     652 |
| Sorted `Vec`             |  3,799 |     17 |     0.4 |   2,844 |
| `BTreeMap` + linear walk |     79 | 98,507 | 102,870 | 112,359 |

Findings:

1. **The B-tree wins where the tree no longer fits in cache.** At 10⁶ keys it
   inserts 2.6-2.8x faster than AVL/red-black and its mixed workload is ~5x
   faster. At 10³-10⁵ keys, where everything is cache-resident, the gap
   shrinks to 1-2x and AVL is level with it on rank at 100k (86 vs 90 ns, within noise).
   This is the Khuong-Morin result reproduced: layout, not big-O, is the lever.
2. **A Fenwick tree still beats everything** on rank (19 ns), but only because
   it is a flat 16 MB array over a fixed integer universe. It cannot hold
   arbitrary keys; that is exactly the constraint a tree removes. (Its 1k/10k
   "insert" cells are dominated by allocating the 16 MB array per iteration
   and are not meaningful.)
3. **`std::collections::BTreeMap` is not an alternative.** It has no rank, so
   the best you can do is walk: ~98 µs per rank at n = 100k vs ~90 ns, over
   1,000x. The demo shows 452x for one query on 10⁶ keys.
4. **Sorted `Vec` is a trap that hides for a while.** Rank/select are the
   fastest of all (binary search, 0.4 ns per select at 100k), but insertion is
   O(n): 3.8 µs vs 91 ns at 100k, and at 1M it would be roughly 10x worse per
   insert (extrapolated, not measured), so it is skipped there.
5. **The treap is slowest** despite the same asymptotics: split/merge touches more
   nodes per insert than a rotation-based tree, and its random priorities
   give a less tightly balanced tree than AVL (an explanation, not a
   measurement).

### Node-size tuning (n = 1,000,000, ns/op)

| T   | keys/node (max) | insert | rank | select | mixed |
| --- | --------------: | -----: | ---: | -----: | ----: |
| 2   |               3 |  1,994 |  521 |    312 | 2,905 |
| 4   |               7 |    554 |  298 |    168 | 1,378 |
| 8   |              15 |    282 |  137 |     82 |   600 |
| 16  |              31 |    228 |  145 |     85 |   417 |
| 32  |              63 |    200 |  102 |     62 |   229 |
| 64  |             127 |    123 |   60 |     81 |   220 |
| 128 |             255 |    203 |   77 |     92 |   326 |

`T = 2` (a 2-3-4 tree, the "textbook" small B-tree) is ~9x worse than the
best setting: the tree is deeper and every level is a cache miss. Performance
flattens for `T` between 32 and 64 and starts to degrade at 128, where
`Vec::insert` shifts and the per-node `sum(counts[..i])` grow. The library
default `T = 16` is deliberately conservative; `T = 32` or `64` is measurably
better on large `u64`-sized keys. The sweet spot depends on key size (larger
keys favour smaller `T`).

### Bulk load

Building from 10⁶ pre-sorted keys: `from_sorted` **4.4 ns/key** vs
incremental insert **59.9 ns/key** (13.6x). Both produce valid trees; the
bulk-loaded one is perfectly packed (no half-full nodes from splits).

## What did not go the way the brief suggests

- The brief says "augmented BST". At scale, the BST variants are the
  slowest options here; the best in-memory answer is not a binary tree at
  all.
- The size-augmented red-black tree is the most code by far (CLRS delete
  fix-up with a shared sentinel and size bookkeeping across `transplant`),
  yet it is not faster than the simpler AVL tree in these runs.
- Per-node `Vec`s are not the tightest possible layout (fixed-size inline
  arrays would be better still, but need `unsafe` or a fixed-capacity array
  dependency). The arena keeps indices, not pointers, and the tree is
  still ~5x faster on mixed workloads.

## Sources

- Khuong & Morin, *Cache-Friendly Search Trees; or, In Which Everything Beats
  std::set* — <https://arxiv.org/abs/1907.01631>
- Cormen, Leiserson, Rivest & Stein, *Introduction to Algorithms*, ch. 14
  (augmenting data structures: order-statistic trees) and ch. 13 (red-black
  trees), ch. 18 (B-trees)
- Aragon & Seidel, *Randomized Search Trees* (treaps), 1989
- Fenwick, *A new data structure for cumulative frequency tables*, 1994
