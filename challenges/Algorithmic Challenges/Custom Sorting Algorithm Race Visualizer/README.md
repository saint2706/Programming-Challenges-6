# Custom Sorting Algorithm Race Visualizer

**Category:** Algorithmic Challenges
**Difficulty:** B (scope expanded well past the original brief -- see below)

**Status:** Implemented (Python)

Manim animations for 14 sorting algorithms, spanning comparison-based, non-comparison-based, and hybrid strategies, all racing the same input array so their behavior can be compared directly.

## The array

```python
BASE_ARRAY = [23, 8, 12, 29, 12, 4, 8, 19, 23, 15]
```

Chosen deliberately:
- **10 elements** -- small enough that every bar and every compare/swap is individually readable, large enough to show real algorithmic behavior (multiple merge levels, a real heap, three Radix Sort passes).
- **Two-digit range (4-29)** -- gives Radix Sort two genuinely different digit passes (units, then tens) instead of a trivial single pass.
- **Three duplicate pairs** (`8`, `12`, `23`) -- makes stability visible on screen: a stable algorithm keeps each pair's original left-to-right order; an unstable one may not.

## Which algorithms visualize well

All 14 are viable. One is weaker than the rest:

- **Pigeonhole Sort** is mechanically almost identical to **Counting Sort** (both place elements directly by key into indexed slots). It's included for completeness, but its clip leans on the one real difference it has -- pigeonholes hold the literal duplicate elements as small stacks, while Counting Sort's story is about building a running-count / prefix-sum table -- rather than offering much new information.

Standouts:
- **Radix Sort** -- watching the array settle further with each digit pass is genuinely satisfying to watch.
- **TimSort** -- identifies runs, insertion-sorts each one, then merges them; it calls back to both Insertion Sort and Merge Sort's visual grammar in one clip.
- **IntroSort** -- the whole point is the *mode switch* (Quick -> Heap -> Insertion under recursion-depth/size pressure), and a color-coded "Mode: X" banner makes that switch unambiguous on screen.
- **Heap Sort** -- reuses the same bars as the heap's array representation and overlays parent/child edges on top, so you see the heap *and* the array at once without a separate structure.

## Design

- `sorting_algorithms.py` -- 14 algorithms, each an instrumented generator that mutates the array in place and `yield`s a `Step` (compare / swap / write / sorted indices / an optional algorithm-specific `aux` payload) after every meaningful operation. Pure algorithm logic, no Manim import. Run directly (`python sorting_algorithms.py`) to self-check that every algorithm produces a correctly sorted array.
- `visualize.py` -- one generic `SortRaceScene` driver that reacts only to the `Step` fields above; it has no per-algorithm animation code. Bars stay at a fixed x-position per array index -- a swap is shown as two bars simultaneously changing height and flashing red, not as bars physically sliding past each other. This is what makes 14 correct, watchable animations tractable from one small driver instead of 14 bespoke ones. Each of the 14 `Scene` subclasses is just a title + subtitle + a lookup key into `ALGORITHMS`.
- Auxiliary panels (rendered only for the algorithms that need them): a `[l, r)` bracket for merge segments, a pivot marker for Quick Sort, heap parent/child edges, a cycle-start marker, bucket/digit/hole columns for the non-comparison sorts, a colored mode banner for IntroSort, and colored run-bands for TimSort.

## The fourteen algorithms

All measurements below are from this repository's own code, on `BASE_ARRAY`:
step counts from `python sorting_algorithms.py`, stability from a search over
40 000 random arrays for a counterexample rather than from a textbook.

| # | Algorithm | Family | Time (avg / worst) | Extra space | Stable? | Steps |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Selection Sort | comparison | Θ(n²) / Θ(n²) | O(1) | **no** | 63 |
| 2 | Bubble Sort | comparison | O(n²) / O(n²) | O(1) | yes | 66 |
| 3 | Insertion Sort | comparison | O(n²) / O(n²) | O(1) | yes | 57 |
| 4 | Merge Sort | comparison | Θ(n log n) | O(n) | yes | 66 |
| 5 | Quick Sort | comparison | O(n log n) / O(n²) | O(log n) | **no** | 35 |
| 6 | Heap Sort | comparison | Θ(n log n) | O(1) | **no** | 69 |
| 7 | Cycle Sort | comparison | Θ(n²) / Θ(n²) | O(1) | **no** | 107 |
| 8 | 3-Way Merge Sort | comparison | Θ(n log₃ n) | O(n) | yes | 50 |
| 9 | Counting Sort | non-comparison | Θ(n + k) | O(n + k) | yes | 46 |
| 10 | Radix Sort | non-comparison | Θ(d·(n + b)) | O(n + b) | yes | 23 |
| 11 | Bucket Sort | non-comparison | Θ(n + k) avg / O(n²) | O(n + k) | yes | 16 |
| 12 | Pigeonhole Sort | non-comparison | Θ(n + k) | O(n + k) | yes | 11 |
| 13 | IntroSort | hybrid | Θ(n log n) guaranteed | O(log n) | **no** | 34 |
| 14 | TimSort | hybrid | O(n) best / Θ(n log n) | O(n) | yes | 56 |

`k` is the value range, `d` the number of digits, `b` the radix (10 here).
"Steps" is yielded animation frames, not operations — it is a rough proxy for
clip length, and the ordering is *not* the ordering of asymptotic cost, because
a step is emitted per meaningful event and the algorithms differ in how much
each event accomplishes.

### Stability, measured rather than assumed

The three duplicate pairs in `BASE_ARRAY` exist so this is visible on screen. A
search for counterexamples over 40 000 random arrays (n ≤ 9, values 0–3) found
one for exactly five of the ten comparison sorts:

```
Selection Sort  UNSTABLE  [3,0,2,3,3,2,3,2] -> 3e before 3a
Quick Sort      UNSTABLE  [3,0,1,0,3,0,3,3,0] -> 3a moved to the end
Heap Sort       UNSTABLE  [2,2] -> 2b before 2a
Cycle Sort      UNSTABLE  [2,3,1,2,3,2,0,1,3] -> 1h before 1c
IntroSort       UNSTABLE  [3,0,0,2,3,2,1] -> 3a after 3e
Bubble / Insertion / Merge / 3-Way Merge / TimSort — no counterexample found
```

Heap Sort's minimal counterexample is `[2, 2]` — two equal elements is all it
takes, because the first thing the sort does is swap `arr[0]` with `arr[end]`.
Note also that Quick Sort and IntroSort are stable *on `BASE_ARRAY`*: running
one array and concluding "stable" is exactly the mistake this search avoids.

---

### 1–3. The quadratic three

**Selection Sort** scans the unsorted suffix for its minimum, then swaps it into
place. Its distinguishing property is that it does **exactly n−1 swaps** no
matter what the input is — the fewest writes of any algorithm here. That is why
it survives in settings where writes are expensive (EEPROM, flash) even though
it always does Θ(n²) comparisons, including on already-sorted input. The swap
is also what breaks stability: moving a distant minimum into position `i`
leapfrogs it over equal elements in between.

**Bubble Sort** compares adjacent pairs and swaps them, bubbling the largest
element to the end each pass. The implementation keeps the `swapped` flag, so on
already-sorted input it exits after one pass — Ω(n) best case. It is the only
algorithm here whose animation reads as a physical process, which is the entire
reason it is still taught. It is also the slowest useful one, and the clip's job
is to make that visible rather than to argue for it.

**Insertion Sort** holds a key and shifts larger elements right until the key's
slot opens. It is the one to actually reach for at n ≤ 10: near-linear on nearly
sorted input, in-place, stable, and with a tiny constant. That is not a
consolation prize — it is why it appears twice more in this list, as the base
case of both hybrid sorts. Watch the shift-then-insert rhythm here, then watch
IntroSort and TimSort fall back to the same motion.

### 4–8. Divide, conquer, and one oddity

**Merge Sort** splits to singletons and merges back. It is the reference for
Θ(n log n) worst-case behaviour with guaranteed stability (the merge takes from
the left run on `<=`, which is the whole reason it is stable), at the cost of an
O(n) buffer. The clip draws the `[l, r)` bracket for the segment being merged,
so the recursion tree is legible as the bracket narrows and widens.

**Quick Sort** partitions around a pivot (Lomuto, last element) and recurses. It
wins on step count here — 35, the second fewest of the comparison sorts —
because partitioning does a lot per step. Its O(n²) worst case on
already-sorted input under a fixed pivot choice is not academic; it is the
reason IntroSort exists. The pivot marker in the clip is the thing to watch: the
whole algorithm is "where does the pivot end up".

**Heap Sort** builds a max-heap in place, then repeatedly swaps the root to the
end and sifts down. It is the only Θ(n log n) sort here needing **O(1)** extra
space, which is its reason for existing and why IntroSort falls back to it
rather than to merge sort. The clip overlays parent/child edges on the same bars
that represent the array, so the heap and its array encoding are visible
simultaneously — the point being that they are the same thing.

**Cycle Sort** is the outlier and earns its slot for one property: it performs
the **theoretical minimum number of writes**, because every element is written
directly to its final position and never moved again. It pays for this with
Θ(n²) comparisons and, at 107 steps, the longest clip in the set — it counts
how many elements are smaller, places the item, then repeats with whatever it
displaced. That trade (most comparisons, fewest writes) is the exact mirror of
Selection Sort, and putting them side by side is the reason both are here.

**3-Way Merge Sort** splits into thirds instead of halves. It shows that the
base of the logarithm is a constant factor, not a complexity change: log₃ n
levels instead of log₂ n, but each merge compares three run heads instead of
two, and the two effects very nearly cancel. Its 50 steps against Merge Sort's
66 is the visible version of "the constant moved, the class did not". It stays
stable because `min()` over the candidate list returns the first minimum and the
runs are offered in left-to-right order.

### 9–12. Not comparing at all

These four beat the Ω(n log n) comparison lower bound by not comparing —
they use values as array indices, which requires knowing the value range.

**Counting Sort** tallies occurrences, builds a prefix-sum table that says where
each value's block starts, then places elements. The clip's story is that
prefix-sum table. Placement walks the input **in reverse** while decrementing
the counter, which is what makes it stable — the standard formulation, and the
reason Radix Sort can use it as a subroutine at all.

**Radix Sort** sorts by each digit position, least significant first. `BASE_ARRAY`
was chosen with a two-digit range specifically so this runs two genuinely
different passes rather than one trivial one, and watching the array become
more ordered after each pass is the most satisfying clip in the set. Its
stability is not a nice-to-have but a **correctness precondition**: if the
per-digit pass reordered equal digits, the work of the previous pass would be
destroyed and the output would simply be wrong.

**Bucket Sort** scatters values into range-partitioned buckets, insertion-sorts
each, and concatenates. It is Θ(n) on average *for uniformly distributed input*
and O(n²) when everything lands in one bucket — the only algorithm here whose
performance depends on the input's distribution rather than its order. Fewest
steps but one, at 16.

**Pigeonhole Sort** gives every distinct value its own hole, drops each element
in, and reads the holes back. At 11 steps it is the shortest clip. It is
mechanically almost identical to Counting Sort, and the README is honest that
it is the weakest of the fourteen: its clip leans on the one real difference,
that pigeonholes hold the literal duplicate elements as small stacks while
Counting Sort's story is a running-count table.

### 13–14. What production actually runs

**IntroSort** is `std::sort` in libstdc++. It runs Quick Sort, but switches to
Heap Sort when recursion depth exceeds 2·log₂(n), and to Insertion Sort on small
ranges. The depth limit is a **security control**, not a tuning knob: Quick
Sort's O(n²) worst case is reachable by anyone who can influence input order, and
the fallback caps it at Θ(n log n) guaranteed. The clip's colour-coded
`Mode: Quick / Heap / Insertion` banner exists to make the switch unambiguous,
and `size_threshold` is set to 4 — deliberately small — so that a ten-element
demo actually exercises the fallback instead of finishing before it triggers.

**TimSort** is `list.sort` in Python and `Arrays.sort` for objects in Java. It
finds already-ascending runs, insertion-sorts each up to a minimum length, then
merges pairs of runs bottom-up. It exists because real data is rarely random:
log lines arrive in time order, database rows come back sorted by another key,
appended records are already ordered. On fully sorted input the run scan finds
one run and the whole sort is O(n). Its clip is the only one that calls back to
two others — coloured run-bands during the insertion phase, then Merge Sort's
segment bracket during merging — which is exactly what the algorithm is.

*(The real CPython implementation adds galloping merges, a computed `minrun`,
and a merge-invariant stack that this teaching version leaves out; what is here
is the run-detect-then-merge skeleton the animation needs.)*


## Where this is actually used

The animations are a teaching artifact. The two things worth taking from this
directory are the *pattern* and the *conclusions*.

**Algorithms as instrumented generators.** `sorting_algorithms.py` yields a
`Step` after every compare/swap/write and knows nothing about rendering. That
separation is how you build a debugger's step-through, a deterministic replay
log, or an operation *counter* — and counting operations is the honest metric
whenever a comparison is expensive: comparing database rows, ICU collation of
Unicode strings, or keys that have to be fetched over a network. Wall time
measures your machine; comparison counts measure the algorithm.

**Why the hybrids exist.** Two of the fourteen are what production actually
runs, and the clips show exactly why:

- **TimSort** is `list.sort` in Python and `Arrays.sort` for objects in Java.
  It detects existing runs first, because real data is rarely random: log lines
  arrive in time order, database rows come back sorted by some other key,
  appended records are already ordered. The run-detection phase is the entire
  reason it beats a textbook merge sort on real input.
- **IntroSort** is `std::sort` in libstdc++. The mode switch that the banner
  makes visible is a defence, not an optimization — Quicksort's O(n²) worst
  case is reachable by anyone who can influence input order, so it bails out to
  Heapsort on recursion depth.

**Stability is not an aesthetic property.** The three duplicate pairs in the
array exist to make it visible on screen, and the reason it matters is
multi-key sorting: "sort by date, then stably by name" only produces the
intended order if the second sort is stable. Every spreadsheet sort, every
`ORDER BY a, b` plan that sorts in two passes, and every UI table with
clickable column headers depends on it.

Counting and radix sorts, meanwhile, are what GPU sorting libraries and
columnar sort-merge joins use on fixed-width keys. The O(n log n) comparison
lower bound only binds if you insist on comparing.

### Where each one actually ships

Sorting is not one algorithm with a winner; it is a family, and which member
you want is decided by the constraint you are actually under — writes, memory,
worst-case bounds, stability, key width, or whether the data even fits in RAM.

| Algorithm | Where it genuinely runs, and the constraint that puts it there |
| --- | --- |
| **Insertion Sort** | Sweep-and-prune broad-phase collision detection in physics engines (Bullet, Box2D): object order barely changes between frames, so a nearly-sorted list re-sorts in ~O(n). Also the small-range base case inside `std::sort` and CPython's sort. |
| **Merge Sort** | **External sorting** — the merge is the only phase that works on data you cannot hold in memory, so it is what `sort(1)`, database `ORDER BY`, sort-merge joins, and Hadoop/Spark shuffle all run. Also linked lists, where there is no random access to partition around: the Linux kernel's `list_sort()` is a merge sort. |
| **k-way merge** (3-Way's generalisation) | **Log-structured merge trees** — RocksDB, LevelDB and Cassandra compact by k-way merging sorted runs. Merging k runs at once instead of pairwise is what cuts the number of passes over disk. Also the merge phase of any external sort, including [Anagram Grouping](../Anagram%20Grouping%20at%20Scale/)'s out-of-core path. |
| **Quick Sort** | The default for arrays of primitives, because it is in-place with excellent cache locality. Java's `Arrays.sort` on primitives is a dual-pivot Quicksort. Stability is irrelevant when the elements *are* the keys, which is exactly when it is chosen. |
| **Heap Sort** | O(1) space and a hard Θ(n log n) ceiling: real-time and embedded code that cannot allocate and cannot risk a worst case. Its sift-down is also the priority queue behind OS schedulers, Dijkstra, and top-k selection. |
| **Selection / Cycle Sort** | Chosen only when **writes** are the scarce resource — EEPROM and flash have limited erase cycles. Selection does exactly n−1 swaps; Cycle Sort does the theoretical minimum number of writes. Both pay Θ(n²) comparisons for it. |
| **Counting Sort** | Histogram operations in image processing, and the per-digit subroutine inside Radix Sort. Needs a bounded, small key range. |
| **Radix Sort** | **The fastest sort on GPUs** — CUB and Thrust's radix sort is what large-scale GPU sorting actually uses. Also fixed-width database keys, IP routing table construction, and suffix array construction (SA-IS). |
| **Bucket Sort** | Sample sort, its distributed cousin, is how parallel and cluster sorts partition work — Spark's range partitioner samples the data to pick bucket boundaries for exactly this reason. The catch is in the table above: it is the one algorithm whose cost depends on the input *distribution*. |
| **IntroSort** | `std::sort` in libstdc++ and MSVC. |
| **TimSort** | `list.sort` in Python, `Arrays.sort` for objects in Java and Android, and `Array.prototype.sort` in V8 — so every sort in Chrome and Node. Rust's stable sort is from the same lineage. |
| **Bubble / Pigeonhole Sort** | Nowhere, honestly. Bubble Sort is a teaching device; Pigeonhole is Counting Sort with worse memory behaviour. They are in this set to be compared against, not adopted. |

### Why sorting is worth this much attention

Almost nothing here is sorted for the sake of being sorted. Sorting is the
*enabling* step:

- **Binary search, and every index built on it** — B-trees, sorted string
  tables, and search-engine posting lists.
- **Deduplication, `GROUP BY`, and joins.** A sort-merge join is two sorts and
  one linear pass; grouping is a sort and a scan. Sorted order is what turns an
  O(n²) pairwise problem into an O(n log n) one.
- **Set intersection.** Two sorted posting lists intersect in a linear merge —
  which is how a search engine answers a two-term query.
- **Order statistics.** Median, percentiles and top-k all fall out of order,
  and the partial versions (`nth_element`, heapselect) are sorting algorithms
  stopped early.
- **Compression.** The Burrows–Wheeler transform is a sort of rotations; it is
  why `bzip2` works.

That is also why the stability question in this README is not pedantry. The
moment you sort *records* by a key rather than sorting the keys themselves —
which is what every one of the uses above does — stability decides whether a
second sort preserves the first one's work.


## Reproduce

```bash
pip install manim   # needs ffmpeg; no LaTeX required -- this uses plain Text throughout, no MathTex
cd "challenges/Algorithmic Challenges/Custom Sorting Algorithm Race Visualizer"
python sorting_algorithms.py                       # self-check: all 14 sort correctly

# One algorithm, fast iteration:
manim -pql visualize.py BubbleSortScene

# Every algorithm, final quality:
manim -qm visualize.py SelectionSortScene BubbleSortScene InsertionSortScene \
    MergeSortScene QuickSortScene HeapSortScene CycleSortScene \
    ThreeWayMergeSortScene CountingSortScene RadixSortScene BucketSortScene \
    PigeonholeSortScene IntroSortScene TimSortScene
```

Rendered output lands in `media/videos/visualize/<quality>/<Scene>.mp4` and is not committed to the repo (regenerate with the command above -- `media/` is git-ignored).
