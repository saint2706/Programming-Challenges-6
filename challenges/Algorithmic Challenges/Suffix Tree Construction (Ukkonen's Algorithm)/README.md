# Suffix Tree Construction (Ukkonen's Algorithm)

**Category:** Algorithmic Challenges
**Difficulty:** X (brief: "Online O(n) construction; compare against suffix array/LCP.")

**Status:** Implemented (Rust)

The brief asks for Ukkonen's online linear-time construction and a comparison with the suffix array.
This crate builds Ukkonen's algorithm and then asks whether anyone should still use it, by building
the two things that replaced it in practice and racing all of them on real text.

| Builder                       | Time                                   | Memory (peak, per input byte) | Notes                                       |
| ----------------------------- | -------------------------------------- | ----------------------------- | ------------------------------------------- |
| `ukkonen` (sibling lists)     | O(n)                                   | 100 B                         | online: query the tree after every byte     |
| `ukkonen` (hash map)          | O(n)                                   | 107 B                         | same algorithm, different branch lookup     |
| SA-IS + Φ-LCP + LCP intervals | O(n)                                   | 53 B                          | three flat array passes; no pointers chased |
| PWOTD (partition, top-down)   | **O(n²) worst case**, fast in practice | 40 B                          | the 2004 VLDB cache-efficient builder       |

All four emit the *same* flat preorder tree (about 28 bytes per input byte with its text), checked
node for node. On top of it: substring search, longest repeated substring, a generalized
tree over several strings with the **longest common substring**, **shortest unique substring** and
**matching statistics**, a suffix array + LCP index to compare against, and a CLI.

`#![forbid(unsafe_code)]` in the library. Edition 2024, current crates (`criterion` 0.8,
`proptest` 1.12, `clap` 4.6, `foldhash` 0.2). `libsais` 0.2 (the C library, without OpenMP, so the
comparison is single-threaded) is a dev-dependency used only by the benchmarks and one cross-check
test; the library itself is pure Rust.

```bash
cargo test --release                                    # 33 tests (also run in debug for overflow checks)
uv run --no-project python scripts/fetch_corpus.py      # four real texts, 4 x 50 MB, into data/
cargo bench --bench construction                        # criterion at 1,000,000 bytes
cargo run --release --example scale 1000000 4000000 16000000    # the tables below (about 12 minutes)
cargo run --release -- build data/dna --algo sais       # algos: ukkonen, ukkonen-hash, sais, pwotd, wotd
cargo run --release -- find data/english "the king" "queen" --show 5
cargo run --release -- lrs data/dna
cargo run --release -- lcs a.txt b.txt c.txt            # add --array for the suffix-array version
cargo run --release -- ms data/english query.txt        # matching statistics of query against the text
cargo run --release -- online data/english "whale" --every 1000000
```

## What I looked for first

I searched for what has replaced Ukkonen's algorithm and found no new construction algorithm that
does, only different trade-offs:

- **Suffix arrays by induced sorting** are the practical default. `libsais`
  ([IlyaGrebnov/libsais](https://github.com/IlyaGrebnov/libsais)) is the fast implementation, with a
  [Rust wrapper](https://docs.rs/libsais). MUMmer, which is built around suffix trees, moved from a
  32-bit suffix tree to a 48-bit suffix array in MUMmer4 to lift its genome-size limit
  ([Marçais et al., PLoS Comput Biol 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5802927/)).
- **Enhanced suffix arrays** (Abouelhoda, Kurtz and Ohlebusch,
  [Journal of Discrete Algorithms 2, 2004](https://hollywood.zbh.uni-hamburg.de/uploads/pubs/pdf/AboKurOhl2004.pdf))
  argue that an algorithm built on a suffix tree can be replaced by one on a suffix array plus an
  LCP array with the same time complexity. The tree here is built from exactly that, in linear time.
- **Cache-efficient top-down construction.** Tata, Hankins and Patel ("Practical Suffix Tree
  Construction", [VLDB 2004](https://www.khoury.northeastern.edu/home/pandey/courses/cs7280/spring25/papers/suffixtreeconst.pdf))
  showed that an O(n²) algorithm beat Ukkonen's O(n) even entirely in memory, because Ukkonen
  jumps along suffix links with poor locality. That is PWOTD below.
- **Making Ukkonen itself fast.** Larsson, Fuglsang and Karlsson
  ([arXiv 1403.0457](https://arxiv.org/abs/1403.0457)) identify the *branch lookup*, finding a
  node's child for a symbol, as the bottleneck and report roughly halving construction time with
  edge-oriented suffix links. I did not port that; instead the branch lookup is a pluggable
  backend, so the two simplest choices can be compared.

So the plan was: Ukkonen as asked (with the lookup swappable), the suffix-array route, and PWOTD,
all producing one tree type, tested against each other and against brute force.

## Design notes

### The text and the terminator (`src/text.rs`)

A suffix tree needs a terminator that occurs nowhere else. Reserving a byte value would make a text
containing that byte unrepresentable, so symbols are `u32` and the terminator is a symbol *below*
every byte: all 256 byte values work, including `0`. For a generalized tree, string `j` gets its own
separator, symbol `j`, so every separator is unique and the last one is the terminator. Because the
terminator is the smallest symbol, "a prefix sorts before its extensions", the same convention as
`libsais`.

### The tree (`src/tree.rs`)

One array of `{ start, depth, end }` per node, in preorder with children in symbol order. The first
child of `v` is `v + 1`, the next sibling of a child `c` is `end[c]`, and a leaf for suffix `i` has
depth `len - i`, so it needs no suffix field. There are no child pointers. Reading the leaves in
order is the suffix array, and `lcp_array()` is read off the structure too. `check_invariants`
re-derives every rule: preorder and `end` consistency, branching, sibling order, one leaf per
suffix, and that each edge label spells the same string as the leaf below it.

### Ukkonen (`src/ukkonen.rs`)

The textbook active-point algorithm ([Ukkonen, Algorithmica 14, 1995](https://doi.org/10.1007/BF01206331)):
open-ended leaves, the (active node, edge, length) triple,
`remainder`, and suffix links. The branch lookup is the [`ChildMap`] trait with two backends:

- `ListChildren`: first-child / next-sibling lists with the edge's first symbol cached per node.
- `HashChildren`: one `foldhash` map keyed by `(parent, symbol)`.

`OnlineSuffixTree` keeps the live structure. A query on the *implicit* tree (before the terminator)
finds every suffix that already has a leaf below the match, and checks directly the few suffixes
that do not (at most `remainder` of them; their positions are exactly `leaf_count..len`). The tests
compare it with brute force after **every single appended byte**.

### Suffix array route (`src/sais.rs`, `src/lcp.rs`, `SuffixTree::from_sa_lcp`)

- **SA-IS** (Nong, Zhang and Chan,
  [DCC 2009](https://doi.org/10.1109/DCC.2009.42)), generic over `u8` and `u32` symbols. It is the
  plain version, with the reduced string in its own vector, and is about 3 to 5 times slower than
  `libsais`.
- **Φ-algorithm LCP** (Kärkkäinen, Manzini and Puglisi,
  [CPM 2009](https://doi.org/10.1007/978-3-642-02441-2_17)); Kasai's algorithm is kept as a
  cross-check.
- **Tree from SA + LCP in O(n), directly in preorder.** The usual bottom-up LCP-interval traversal
  emits nodes in post-order. Running it over the suffix array *backwards* emits post-order with
  children right to left, and reversing that list is exactly left-to-right preorder, with subtree
  sizes known as it goes. No pointers, no sort, no recursion.

### PWOTD (`src/pwotd.rs`)

Phase 1 scans the text once and buckets every suffix by its first `p` symbols. Phase 2 expands
groups depth first: find the longest prefix `L` the whole group shares (the node's depth), counting-
sort the group by the symbol at depth `L`, emit the node, recurse in symbol order.

What the partition buys, exactly: a range sorted on its first `s` symbols has its shared prefix
(within `s`) equal to the LCP of its *first and last* suffix, and its sub-groups are found by binary
search, so the first `p` levels cost O(children · log n) text reads instead of O(group size).
Past depth `p` it is the unsorted scan. `p = 0` is plain write-only top-down (WOTD).

### Applications (`src/apps.rs`, `src/array_index.rs`)

- **Longest common substring of k strings**: a bitset of strings per node, OR-ed bottom-up in
  reverse preorder; the deepest full node wins. The array version slides a window over the
  generalized suffix array, keeping the window minimum LCP in a monotone deque.
- **Shortest unique substring**: one symbol past a leaf's parent depth, unless that symbol is the
  terminator.
- **Matching statistics** (Chang and Lawler, Algorithmica 1994): for each query position, how much
  of it the text contains, in O(|query|) by walking suffix links, which are built in O(n) with
  skip/count.

## Results

Four real texts from the [Pizza&Chili corpus](http://pizzachili.dcc.uchile.cl/texts.html) (the
50 MB prefixes: DNA, English, program sources, protein sequences), plus uniform random text over 4
letters and the Fibonacci word. They are downloaded, SHA-256-checked and kept out of git; the corpus
page states no licence. One machine (Core Ultra 7 255H), one thread, one run per cell; the
repeat-heavy cells vary the most. Seconds to build the flat tree, including the text it owns.

**16,000,000 bytes**, from `cargo run --release --example scale`:

| text      | SA-IS (own) | libsais | Ukkonen, lists | Ukkonen, hash | SA+LCP+tree | PWOTD     | WOTD      | nodes/byte |
| --------- | ----------- | ------- | -------------- | ------------- | ----------- | --------- | --------- | ---------- |
| dna       | 2.05        | 0.42    | 24.69          | 25.88         | 3.12        | **1.97**  | 2.90      | 1.67       |
| english   | 2.03        | 0.45    | 31.08          | 21.09         | **3.02**    | 41.78     | 39.97     | 1.67       |
| sources   | 1.51        | 0.40    | 19.40          | 15.23         | **2.42**    | 2.86      | 2.46      | 1.63       |
| proteins  | 2.44        | 0.53    | 43.14          | 21.91         | **3.52**    | 4.36      | 4.67      | 1.66       |
| uniform4  | 2.41        | 0.48    | 26.28          | 26.79         | 3.14        | **2.04**  | 2.36      | 1.62       |
| fibonacci | 1.25        | 0.35    | 8.88           | 19.17         | **2.12**    | quadratic | quadratic | 2.00       |

At 1,000,000 bytes (criterion, medians) on DNA: Ukkonen 528 ms (lists) and 631 ms (hash), SA-IS +
LCP + tree 106 ms, PWOTD 84 ms, WOTD 92 ms; the suffix array alone 52 ms with this crate's SA-IS and
18 ms with `libsais`.

**Peak heap, bytes per input byte, identical on every text** (the text the tree owns is counted):

| structure                                         | B/byte |
| ------------------------------------------------- | ------ |
| Ukkonen tree while it is growing                  | 49     |
| Ukkonen, peak including the freeze to flat arrays | 100    |
| Ukkonen with the hash map, peak                   | 107    |
| SA-IS + LCP + tree, peak                          | 53     |
| PWOTD, peak                                       | 40     |
| the finished tree plus its text                   | 28     |
| this crate's SA-IS + LCP array                    | 12     |
| `libsais`, suffix array only (no LCP)             | 4      |

**Searching** 200,000 patterns of 16 bytes that occur in the text (nanoseconds per query):

| text     | tree, 1 MB | array, 1 MB | tree, 16 MB | array, 16 MB |
| -------- | ---------- | ----------- | ----------- | ------------ |
| dna      | 623        | 771         | 2697        | 4181         |
| english  | 1351       | 767         | 6311        | 4433         |
| sources  | 857        | 655         | 5798        | 2932         |
| proteins | 708        | 685         | 4238        | 4163         |
| uniform4 | 619        | 743         | 2708        | 4766         |

**Applications**, criterion, 1 MB of DNA (longest repeat) and 3 strings of 100 kB (common substring):

| task                                     | suffix tree | suffix array |
| ---------------------------------------- | ----------- | ------------ |
| longest repeated substring               | 5.63 ms     | **0.84 ms**  |
| longest common substring, build included | 37.6 ms     | **24.5 ms**  |

**Top-down construction is quadratic on repetitive text** (seconds):

| text              | n      | Ukkonen | SA+LCP+tree | PWOTD  |
| ----------------- | ------ | ------- | ----------- | ------ |
| Fibonacci word    | 8,000  | 0.0005  | 0.0004      | 0.0084 |
|                   | 16,000 | 0.0011  | 0.0011      | 0.0309 |
|                   | 32,000 | 0.0049  | 0.0031      | 0.1318 |
| run of one symbol | 8,000  | 0.0013  | 0.0007      | 0.1647 |
|                   | 16,000 | 0.0011  | 0.0008      | 0.6474 |
|                   | 32,000 | 0.0041  | 0.0018      | 2.4488 |

### What the numbers say

- **Ukkonen's O(n) is the slowest way to get a tree.** The suffix-array route is 5x faster at 1 MB
  and 8 to 12x faster at 16 MB on real text, and the gap grows because Ukkonen's time per byte is
  not flat: in one run DNA takes 0.82 s at 1 MB, 4.74 s at 4 MB and 24.7 s at 16 MB, which is 30x
  the time for 16x the data. Its accesses follow suffix links across the whole tree, which is the
  cache argument of the VLDB paper made concrete.
- **The branch lookup matters, and neither backend wins.** At 16 MB the hash map is 2x faster on
  proteins and 1.5x on English, level on DNA, and 2x *slower* on the Fibonacci word. Large
  alphabets favour the hash; tiny ones favour the lists.
- **PWOTD is the fastest builder until it is catastrophic.** It beats the suffix-array route on DNA
  and random text (1.97 s against 3.12 s at 16 MB) and uses the least memory, but on English it takes
  41.8 s, 14x *slower* than the SA route. The English file's long repeats make the group-scan
  quadratic: it went from 0.08 s at 1 MB to 5.3 s at 4 MB. The synthetic table shows the mechanism
  with a clean fourfold increase for every doubling. This is the O(n²) the VLDB paper warned about,
  and it bites real text, not just contrived strings.
- **The partition phase is a modest win.** PWOTD against plain WOTD is 32% faster on DNA, 14% on
  random text, 7% on proteins, and 16% *slower* on program sources.
- **Memory favours arrays.** Ukkonen peaks at 100 bytes per input byte (49 live, the rest the
  transient flat copy, which the paper's O(n) hides); the SA route at 53; the suffix array and LCP
  alone at 12, and `libsais`'s suffix array alone at 4. The finished flat tree plus its text is 28
  bytes per input byte, about 20 of them node records.
- **`libsais` is 3 to 5x faster than a clean SA-IS** (2.9x at 1 MB, 3.8 to 4.9x at 16 MB). The own
  implementation is linear and correct but allocates its reduced string separately and does not
  tune its bucket scans.
- **Tree search against binary search is a wash with a twist.** The tree wins by up to 1.8x on
  small alphabets (DNA, random), where a descent is a few steps. On English it loses, because the
  root and top nodes have over a hundred children and a lookup scans them linearly: this tree has
  no child index, trading speed on wide nodes for 24 bytes per symbol.
- **For single-string questions the arrays win.** The longest repeat is 6.7x faster as a scan of
  the LCP array than as a scan of the tree, and the common substring is 1.5x faster with the array
  window even before counting the tree's memory.

## Limitations

- **Texts up to about 4 billion symbols** (`u32` positions). Searching is O(m · σ) at wide nodes.
- **Ukkonen's online property is kept only by `OnlineSuffixTree`.** It uses sibling lists; the hash
  backend is construction-only because a map cannot list one node's children.
- **Edge-oriented suffix links** (Larsson et al.), which that paper reports roughly halve Ukkonen's
  time, are not implemented. The comparison therefore uses the textbook algorithm; even halving
  Ukkonen's time would leave it 4 to 6x behind the suffix-array route at 16 MB.
- **SA-IS is the clear version, not the tuned one.** The `libsais` column shows what tuning buys.
- **No parallelism, no compressed trees.** `libsais` is run single-threaded to match.
- **`check_invariants` is O(total edge length)**, quadratic on repeated text, so tests call it on
  small inputs only.

## Where this is actually used

**Suffix trees in bioinformatics.** The VLDB paper this crate reproduces names MUMmer, REPuter and
OASIS as applications that rely on them for sequence queries. MUMmer is the useful story: the tool
was built on a 32-bit suffix tree, and MUMmer4 (Marçais et al., 2018) replaced that core structure
with a 48-bit suffix array to lift its genome-size limit; MUMmer3 handled references of about 500
Mb. The memory table above measures the same trade: 4 to 12 bytes per input byte for a suffix array
and its LCP, 28 for the finished flat tree, and 100 at Ukkonen's peak.

**Suffix trees in language-model serving.** SuffixDecoding
([arXiv 2411.04975](https://arxiv.org/html/2411.04975v3)) builds suffix trees over previous model
outputs and the current prompt, updates the per-request tree as tokens are generated, and matches the
latest tokens against it to propose continuations for speculative decoding. The abstract reports up
to 5.3x speedup over vanilla decoding. A tree that grows token by token is the use case for an
*online* construction like Ukkonen's.

**Suffix array libraries.** `libsais` is a C library for linear-time suffix array, LCP array and
Burrows-Wheeler transform construction; this crate benchmarks against it through its Rust wrapper.

## Layout

```
src/text.rs         terminator-aware symbols: Bytes, Words (generalized)
src/sais.rs         SA-IS suffix array;  src/lcp.rs  Phi and Kasai LCP
src/tree.rs         the flat preorder SuffixTree, queries, suffix links, invariants
src/satree.rs       SA + LCP -> tree
src/ukkonen.rs      Ukkonen with two branch backends, and OnlineSuffixTree
src/pwotd.rs        partitioned top-down construction
src/array_index.rs  suffix array + LCP index, window-based common substring
src/apps.rs         common substring, shortest unique substring, matching statistics
src/texts.rs        synthetic texts;  src/bin/sufftree.rs  the CLI
scripts/fetch_corpus.py + corpus.json    pinned download of the Pizza&Chili texts
tests/              arrays, tree, ukkonen (+ pwotd), apps, cli
benches/construction.rs   criterion;  examples/scale.rs   prints the tables above
```
