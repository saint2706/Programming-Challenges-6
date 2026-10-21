# Persistent Segment Tree for Historical Range Queries

**Category:** Algorithmic Challenges
**Difficulty:** X (brief: "Query any past version of the structure in O(log n).")

**Status:** Implemented (Rust)

The brief asks for a segment tree whose every past version stays queryable in O(log n). This crate
builds it by path copying, then asks what the extras cost and whether it is the right tool: range
updates with permanent marks against push-down, version garbage collection by reference counting and
by mark-and-compact, fat nodes, persistent vectors (`imbl`, `rpds`), and a static wavelet matrix for
the question persistent trees are most often used to answer, the k-th smallest in a subarray.

| Structure       | What it is                                                    | Persistence                                           | Notes                                                                 |
| --------------- | ------------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------- |
| `Persistent<M>` | path-copying tree over a sum, min or max monoid               | full: any live version can be the parent of a new one | `ceil(log2 n) + 1` nodes per update, sparse start, retire and compact |
| `KthIndex`      | one count tree per prefix of an array                         | static array, every prefix                            | k-th smallest and rank of any subarray in O(log n)                    |
| `DistinctIndex` | marks each value's last occurrence, one version per prefix    | static array, every prefix                            | online distinct count of any subarray                                 |
| `LazyAdd`       | range add, sum / min / max, **tags never pushed down**        | full                                                  | an update copies only the nodes it visits                             |
| `PushDown`      | range add and range **assign**, sum, tags pushed as it copies | full                                                  | affine tags `v -> a*v + b`; the general mechanism                     |
| `FatTree`       | fat-node method: a version history per node                   | partial: one timeline                                 | binary search per visited node                                        |
| `WaveletMatrix` | static; k-th smallest, rank and range frequency               | none                                                  | about 3 bytes per element where the persistent index takes 230 to 280 |
| `History<M>`    | timestamped table with retention                              | full                                                  | `version_at(time)`, `forget_before(time)`                             |

`#![forbid(unsafe_code)]` in the library; edition 2024. `imbl` 7.0 and `rpds` 1.2 are dev-dependencies
used only as benchmark baselines.

```bash
cargo test --release                                           # 36 tests (also run in debug for overflow checks)
cargo run --release --example scale                            # every table below (about 3 minutes)
cargo run --release --example scale -- updates                 # or kth | lazy | gc
cargo bench --bench queries                                    # criterion intervals at n = 65,536
cargo run --release -- kth data.txt 100 5000 7 --engine wavelet     # 7th smallest of numbers 100..5000
cargo run --release -- distinct data.txt 100 5000
cargo run --release -- sizes data.txt                          # bytes per engine on your data
cargo run --release -- history ledger.txt                      # replay a time-travel script
```

A time-travel script is a list of timestamped writes and as-of queries:

```
size 6
set 10 0 100
set 20 0 70
sum 15 0 6        # as of time 15: 100
sum 25 0 6        # as of time 25: 70
forget 20         # drop history older than the version in effect at 20
sum 15 0 6        # (history forgotten)
```

## What I looked for first

I searched for what has replaced path copying and found no new result; the picture is the classical
one with a few practical corrections.

- **Path copying** (Sarnak and Tarjan, planar point location with persistent search trees, CACM 29,
  1986) is the standard way to get full persistence from a tree: copy the root-to-leaf path, share
  the rest, keep the old root. It is what this crate is built on.
- **Fat nodes and node copying** (Driscoll, Sarnak, Sleator and Tarjan, "Making data structures
  persistent", JCSS 38(1), 1989) cost O(1) per modified *field*. A segment-tree update modifies a field
  (the aggregate) in every ancestor of the leaf, so no persistence technique can beat Theta(log n)
  new field versions per update here. That is an argument, not a measurement; `FatTree` measures what
  fat nodes cost in that setting, and it is not less.
- **Wavelet matrix** (Claude, Navarro and Ordonez, *Information Systems* 47, 2015) and the range
  quantile query on wavelet trees (Gagie, Puglisi and Turpin, arXiv 0903.4726): the static answer to
  "k-th smallest in a subarray".
- **Persistent vectors** in functional languages: Clojure's documentation says its collections are
  "immutable and persistent" with structural sharing, and vectors take "log32N hops" to index.
  `imbl` and `rpds` are the Rust equivalents. They share structure but keep no range aggregates, so
  a historical range sum is a scan.
- **Memory reduction for persistent segment trees:** I found no 2023 to 2026 result, and I found no
  published measurement of bytes per update for any of these structures in the sources I searched, so
  the figures below are my own.
- **"Mark permanence"**, leaving an add-tag where an update put it instead of pushing it down, is
  folklore in competitive-programming write-ups. I found no primary source, so the term here names the
  idea and the behaviour is measured, not cited.

## Design notes

### The arena and the sparse tree (`src/arena.rs`)

Every tree is a DAG of immutable nodes in one `Vec`, children named by `u32` index. A node is 12
bytes for a `u32` count, 16 for an `i64` aggregate, 24 for the lazy-add payload and 32 for the
push-down payload. Index 0 is `NIL`, the shared empty subtree, so a tree over a million positions
with nothing set is one node and each update materialises one path. Node ids are 32-bit; going past
2^32 nodes panics with a message rather than wrapping.

### Path copying (`src/persistent.rs`)

`modify(from, pos, f)` copies the path from the root to position `pos`: exactly `depth + 1` nodes on
a power-of-two tree (a test asserts that for `n = 2^0 .. 2^10`), at most `ceil(log2 n) + 1` in
general. Any live version can be `from`, so versions branch. A query is the usual range descent over
whichever root it is given. The multiset difference of two count-tree versions answers "k-th smallest
between two prefixes" in one simultaneous descent.

### Range queries over prefixes (`src/kth.rs`)

*k-th smallest*: compress the values to ranks; version `i` is a count tree holding `a[0..i]`; the
multiset of `a[l..r]` is version `r` minus version `l`. *Distinct count*: version `i` marks, over
positions, the last occurrence of each value in `a[0..i]`; the distinct values of `a[l..r]` are the
marks at positions `>= l` in version `r`. Adding `a[i]` unmarks the previous occurrence and marks `i`,
two updates, and the version between them is retired at once so its nodes are recycled.

### Range updates: permanent marks or push-down (`src/lazy.rs`)

In an ordinary lazy segment tree a later visit pushes a pending tag from a node into its children. In
a persistent tree pushing is costly because a node cannot be modified: pushing means copying both
children. Two designs avoid or limit that:

- **`LazyAdd`** never pushes. An add-tag stays on the node where the update put it and that node's
  aggregate includes it; a query adds up the tags of the nodes it passes through. An update copies
  exactly the nodes it visits. It works because adds commute, and it works for min and max too (adding
  `d` to every element adds `d` to the minimum).
- **`PushDown`** pushes, folding the push into the copy of each visited node, and uses affine tags
  (`v -> a*v + b`: add is `(1, d)`, assign is `(0, x)`) so it can do range **assign**. Assign and add
  do not commute, so a tag left in an ancestor could overwrite a newer update below it; the order has
  to be fixed by pushing. An update copies the visited nodes plus, wherever a tag is pending, the
  sibling it must pass the tag to.

Queries copy nothing in either: they carry the tags they have passed instead of pushing them. The
tests include a worked assign/add sequence in both orders, and a check that push-down never allocates
fewer nodes than permanent marks for the same update (both visit the same nodes).

### Garbage collection (`src/arena.rs`)

Old versions pin memory. Two ways to let it go, on the same arena:

- **Reference counts** (`counting = true`): a `u32` per node counts the parent pointers and version
  roots that reach it. `retire(v)` releases the root, and every node whose count reaches zero goes on a
  free list for the next allocation. Memory comes back immediately; the price is 4 bytes per node and
  two increments per allocation.
- **Mark and compact** (`compact()`): trace everything the live roots reach and copy it, children
  before parents, into a fresh `Vec`. No per-node cost while running; a pause proportional to the live
  data when it runs; and the only way the arena ever shrinks (the free list recycles slots but the
  `Vec` keeps its length).

With counting off, `retire` only forgets the root, and the memory stays until `compact()`.

### Fat nodes, the wavelet matrix and the history table

`FatTree` keeps `(version, aggregate)` pairs per node, appended on change, and binary-searches them on
query. `WaveletMatrix` stores the array as `ceil(log2(max + 1))` levels of one bit per element,
stably reordering by each bit, so a range maps to the next level by two rank operations. `History`
wraps the persistent tree with timestamps (`version_at` is a binary search) and retention
(`forget_before` retires the older versions).

## Results

Windows 11, Intel Core Ultra 7 255H, 31 GB RAM, rustc 1.99.1, single thread. One process, a counting
allocator, so "bytes" is heap the structure keeps alive, measured as the change in live bytes while it
exists. Workloads are seeded; while a table is built the structures are cross-checked and an answer
that differs aborts the run. Path-copying arenas are sized exactly (`reserve_updates`) so the figure is
the nodes and not a `Vec`'s doubling slack, which can add up to 2x (the first version of this harness
counted it; fat nodes cannot pre-reserve and keep theirs). Short timings vary from run to run by up
to about 1.5x on this laptop; I ran the suite several times and the orderings below held in each.

### Point updates and historical sums

A linear history: every update is applied to the previous version. "Sum of 64" is a random range of 64
positions in a random version; "sum of n/2" a random range of half the array.

n = 65,536 positions, 65,536 updates:

| structure                       | update    | bytes / update | sum of 64 | sum of n/2 |
| ------------------------------- | --------- | -------------- | --------- | ---------- |
| path copying                    | 264 ns    | 288            | 1.45 us   | 2.61 us    |
| path copying + reference counts | 266 ns    | 356            | 1.37 us   | 2.29 us    |
| fat nodes                       | 1.37 us   | 425            | 1.01 us   | 4.95 us    |
| `imbl::Vector` (scan)           | 2.58 us   | 3,774          | 3.19 us   | 360.67 us  |
| `rpds::Vector` (scan)           | 869 ns    | 1,024          | 2.88 us   | 853.32 us  |
| full copy per version           | 164.66 us | 524,319        | -         | -          |

n = 1,048,576 positions, 262,144 updates:

| structure                       | update  | bytes / update | sum of 64 | sum of n/2 |
| ------------------------------- | ------- | -------------- | --------- | ---------- |
| path copying                    | 899 ns  | 352            | 3.10 us   | 4.31 us    |
| path copying + reference counts | 1.16 us | 436            | 2.45 us   | 3.60 us    |
| fat nodes                       | 2.88 us | 550            | 1.55 us   | 9.55 us    |
| `imbl::Vector` (scan)           | 4.29 us | 5,351          | 3.11 us   | 8.91 ms    |
| `rpds::Vector` (scan)           | 2.25 us | 1,264          | 6.89 us   | 22.23 ms   |
| full copy per version           | 3.19 ms | 8,388,639      | -         | -          |

The scans index each element (`imbl` and `rpds` have no range aggregate). Their bytes per update
include the `Vec` of retained versions.

### Range k-th smallest

Ranges of 1,000 and of n/2, 20,000 random queries each (the sort baseline runs fewer, 200 and 20).

| engine                                     | build   | bytes    | bytes / number | k-th, range 1000 | k-th, range n/2 |
| ------------------------------------------ | ------- | -------- | -------------- | ---------------- | --------------- |
| *n = 65,536, 65,536 possible values*       |         |          |                |                  |                 |
| persistent count tree                      | 18 ms   | 14.5 MB  | 232.0          | 334 ns           | 442 ns          |
| wavelet matrix                             | 11 ms   | 0.2 MB   | 3.02           | 171 ns           | 171 ns          |
| sort the range                             | -       | 0.5 MB   | 8.0            | 7.35 us          | 317.14 us       |
| *n = 1,048,576, 1,048,576 possible values* |         |          |                |                  |                 |
| persistent count tree                      | 1489 ms | 280.0 MB | 280.0          | 2.49 us          | 3.89 us         |
| wavelet matrix                             | 319 ms  | 3.8 MB   | 3.75           | 900 ns           | 719 ns          |
| sort the range                             | -       | 8.0 MB   | 8.0            | 8.68 us          | 8.92 ms         |
| *n = 1,048,576, 256 possible values*       |         |          |                |                  |                 |
| persistent count tree                      | 104 ms  | 136.0 MB | 136.0          | 929 ns           | 973 ns          |
| wavelet matrix                             | 123 ms  | 1.5 MB   | 1.50           | 377 ns           | 274 ns          |
| sort the range                             | -       | 8.0 MB   | 8.0            | 19.28 us         | 5.44 ms         |

Distinct values in a range of 1,000, the online persistent index against hashing the range (200
queries):

| n         | values    | build   | bytes     | nodes      | query   | hashing  |
| --------- | --------- | ------- | --------- | ---------- | ------- | -------- |
| 65,536    | 65,536    | 24 ms   | 26.8 MB   | 1,471,311  | 616 ns  | 9.68 us  |
| 1,048,576 | 1,048,576 | 1599 ms | 524.0 MB  | 29,262,167 | 1.90 us | 8.00 us  |
| 1,048,576 | 1,000     | 640 ms  | 1028.0 MB | 33,617,095 | 3.31 us | 38.80 us |

### Range updates

Random ranges of any length up to n, then sum queries on random versions. `LazyAdd` has a 24-byte
node, `PushDown` a 32-byte one. Bytes include `Vec` growth slack here (no pre-reservation), so compare
the nodes column.

| variant                          | n         | update  | nodes / update | bytes / update | sum query |
| -------------------------------- | --------- | ------- | -------------- | -------------- | --------- |
| add, permanent marks             | 65,536    | 1.64 us | 42.0           | 1,504          | 3.10 us   |
| add, push-down                   | 65,536    | 2.11 us | 53.5           | 2,000          | 4.18 us   |
| half add, half assign, push-down | 65,536    | 2.26 us | 53.5           | 2,000          | 4.42 us   |
| add, permanent marks             | 1,048,576 | 3.25 us | 54.0           | 2,704          | 6.91 us   |
| add, push-down                   | 1,048,576 | 4.69 us | 69.5           | 3,600          | 7.99 us   |
| half add, half assign, push-down | 1,048,576 | 3.92 us | 69.6           | 3,600          | 9.36 us   |

### Garbage collection

n = 65,536, 262,144 point updates from a full tree, keeping a window of the most recent versions:

| retention                                                                      | update | live nodes | arena slots | arena bytes |
| ------------------------------------------------------------------------------ | ------ | ---------- | ----------- | ----------- |
| keep every version                                                             | 288 ns | 4,587,519  | 4,587,519   | 74.0 MB     |
| keep every version, reference counts on                                        | 364 ns | 4,587,519  | 4,587,519   | 91.5 MB     |
| last 1, reference counts                                                       | 403 ns | 131,071    | 131,088     | 9.0 MB      |
| last 16, reference counts                                                      | 441 ns | 131,326    | 131,343     | 9.0 MB      |
| last 256, reference counts                                                     | 388 ns | 135,406    | 135,423     | 9.0 MB      |
| last 4096, reference counts                                                    | 521 ns | 200,686    | 200,703     | 9.0 MB      |
| last 16, retire in bulk, then compact once (20 ms, 4,587,519 to 131,326 slots) | -      | 131,326    | 131,326     | 6.0 MB      |

### What the numbers say

- **A point update allocates the path and nothing else.** 17 nodes at n = 65,536 and 21 at
  n = 2^20, 288 and 352 bytes with the version table's share. That is 13x to 15x less than
  `imbl::Vector` (3,774 and 5,351 bytes), 3.6x less than `rpds::Vector` at n = 65,536, and about
  1,800x and 24,000x less than copying the array.
- **Aggregates are what the persistent vectors lack.** A sum of half the array takes 4.3 us in the
  path-copying tree and 8.9 ms in `imbl` (2,000x) and 22 ms in `rpds` (5,000x) at n = 2^20. For a
  range of 64 the scan is competitive (3 to 7 us against 1.5 to 3 us), so the gap is entirely about
  long ranges.
- **Fat nodes lose, as the argument predicted, except on short queries.** They update 3x to 5x slower
  (1.37 us against 264 ns), keep more memory per update (425 and 550 bytes against 288 and 352, and
  that is with `Vec` slack the arena avoids), and answer half-array sums 1.9x to 2.2x slower. They
  answer short sums faster (1.0 us against 1.45 us; 1.55 us against 3.10 us), because their tree is a
  contiguous array and a short query touches few nodes.
- **The wavelet matrix is the better answer for a static array.** It takes 3.0 to 3.75 bytes per
  number where the persistent index takes 232 to 280 (75x to 91x less), builds 4.7x faster at n = 2^20, and answers
  2x to 5x faster (900 ns against 2.49 us on 1,000-element ranges at n = 2^20; 719 ns against
  3.89 us on half the array). The persistent tree is only 3.5x faster than sorting a 1,000-element
  range at n = 2^20 (2.49 us against 8.68 us), and 2,300x faster on half the array. Use it when the
  array changes or when you need the history, not for a fixed array.
- **Permanent marks save about a fifth of the nodes and 20% to 30% of the time on adds**: 42.0
  against 53.5 nodes per update (54.0 against 69.5 at n = 2^20), 1.64 against 2.11 us (3.25 against
  4.69 us). The nodes are also smaller (24 against 32 bytes), so the memory saving is about 40%. The
  price is generality: marks cannot express assign, and push-down can, at the same node cost as
  push-down add.
- **Reference counts cost 25%** (364 against 288 ns per update, 91.5 against 74.0 MB for 4.6 million
  nodes) and buy a flat footprint: a window of 16 versions keeps 131,326 live nodes (the current tree's
  131,071 plus 15 paths of 17) and a 9 MB arena however many updates have happened, where keeping
  everything grows to 4.6 million. Updates under a window run 390 to 520 ns, slower than keeping
  everything (288 ns); presumably that is the work of retiring, which walks the freed nodes, but I did
  not isolate it.
- **Compaction is the cheap bulk option.** Retiring all but 16 of 262,144 versions without counts and
  compacting once took 20 ms and brought 4,587,519 slots down to 131,326, with the arena at 6.0 MB.
- **The online distinct index is expensive.** It holds 29 to 34 million nodes at n = 2^20 (524 MB to
  1 GB) because each element makes two updates, and it answers 4x to 16x faster than hashing a range of
  1,000. It earns its memory only with many queries over long ranges. (The 1 GB figure is partly the
  arena doubling once: my pre-sizing estimate of 1.5 updates per element was just under the 1.6
  this input needs.)

## Limitations

- One machine, one seed per workload, short timings that vary by up to 1.5x between runs. The
  orderings held; the ratios are indicative. Nothing was measured on another CPU or operating system.
- `imbl` and `rpds` are timed with an indexing scan, because they offer no range aggregate. A persistent
  vector with a cached aggregate per node would be a different structure, and a closer competitor.
- Single-threaded: every mutating call takes `&mut self`. There is no concurrent reader support, which
  is the usual reason to want persistence; a published root is immutable, but no `Arc`-based sharing
  is provided.
- A retired version panics if queried; `is_live` is the check, and there is no `Result`-returning
  query.
- `DistinctIndex` does two separate updates per element rather than one combined pass over the union
  of the two paths, and its arena pre-sizing is a guess that can be low.
- `KthIndex` rebuilds from a whole array; there is no incremental append, and ranks are over the values
  present at build time.
- The persistent tree's aggregates are fixed-size `Copy` values. Structures with variable-size nodes
  (a persistent trie, a persistent order-statistic tree) would need a different arena.
- Not implemented: node copying (DSST's bounded-degree method) and confluent persistence. The
  argument above is why node copying would not help here, not a measurement of it.

## Where this is actually used

The idea behind path copying, a write copies the path from the leaf to the root and a reader keeps the
old root, is also the idea behind LMDB. Its Wikipedia article says new data is written without
overwriting or moving existing data, that it uses multiversion concurrency control, and that write
transactions do not block readers nor readers writers; the source comment in `mdb.c` says readers need
no locking for that reason. (The correspondence to path copying is my reading, not a statement from
either source.) Clojure's collections are documented as immutable and persistent with structural
sharing, and its vectors as indexed "in log32N hops"; `imbl` and `rpds` bring the same structures to
Rust, which is why they are the baselines here. The k-th smallest in a sublist is the range quantile
query that Gagie, Puglisi and Turpin answer with wavelet trees, and wavelet-matrix libraries exist for
Rust (`wavelet-matrix`) and Go (`go-watrix`) with quantile and rank queries. The persistent segment
tree itself I did not find documented in a named production system; its home is competitive
programming, where "k-th in a range" and "distinct in a range" are staple problems.

## Layout

```
src/
  lib.rs          module map and re-exports
  arena.rs        node arena, reference counts, mark-and-compact, the version Store
  persistent.rs   Persistent<M>: path copying over sum / min / max, kth_between, count_below_between
  kth.rs          KthIndex and DistinctIndex
  lazy.rs         LazyAdd (permanent marks) and PushDown (affine tags, add and assign)
  fat.rs          FatTree
  wavelet.rs      WaveletMatrix
  history.rs      History: timestamps, version_at, forget_before
  oracle.rs       brute-force reference used by the tests and the CLI
  bin/pst.rs      CLI: kth, distinct, history, sizes
tests/            persistent (model + memory), queries, lazy, cli
benches/queries.rs  criterion, n = 65,536
examples/scale.rs   the tables above
```
