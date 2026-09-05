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
