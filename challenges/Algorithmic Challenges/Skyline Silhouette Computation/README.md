# Skyline Silhouette Computation

**Category:** Algorithmic Challenges
**Difficulty:** I (brief: "divide-and-conquer or sweep-line over building outlines")

**Status:** Implemented (Python)

Given a set of buildings, each a rectangle `(left, right, height)` on the
x-axis, trace the outline of their combined silhouette: the minimal list of
"key points" `(x, height)` such that reading them in order — flat at that
height until the next x — reconstructs the skyline exactly. This is
[LeetCode 218](https://leetcode.com/problems/the-skyline-problem/), treated
with more depth than a single accepted solution: four independent methods,
cross-checked against each other, with the tie-heavy edge cases that a quick
solution tends to get subtly wrong.

| Method               | Time                | Space | Role                                                                           |
| -------------------- | ------------------- | ----- | ------------------------------------------------------------------------------ |
| `brute_force`        | O(n · gaps) ≈ O(n²) | O(n)  | The definition; the oracle everything else is checked against                  |
| `sweep_line`         | O(n log n)          | O(n)  | Max-heap over active buildings, lazy deletion                                  |
| `sweep_line_bst`     | O(n log n)          | O(n)  | Same sweep, `sortedcontainers.SortedList` over active buildings, true deletion |
| `divide_and_conquer` | O(n log n)          | O(n)  | Merge-sort-style: split, recurse, merge two skylines                           |

`skyline(buildings, method="auto")` dispatches to `sweep_line` — see
[Benchmarks](#benchmarks) for why. `sweep_line_bst` is the only method with a
third-party dependency (`sortedcontainers`); every other method, and the CLI's
default paths, need nothing beyond the standard library.

No known algorithm beats O(n log n) for this problem in the comparison
model: the standard argument (believed to follow from the same
sorting-reduction used for convex hull — sorting n reals reduces to computing
a skyline of n unit-width, uniquely-ordered-height buildings, and reading the
key points back off in order sorts them) puts a lower bound of Ω(n log n) on
any comparison-based skyline algorithm, so none of the three real methods
here is leaving asymptotic performance on the table. It's worth noting that
convex hull has *output-sensitive* algorithms (Kirkpatrick–Seidel,
Chan's algorithm) that beat O(n log n) when the hull itself has few vertices
— no such bound is established for skyline specifically, and even if one
were found, it wouldn't help in the worst case here: a skyline's output size
is itself Θ(n) for dense, non-overlapping buildings (see the `sparse`
benchmark below, where `points` tracks `n` almost 1:1), so an
output-sensitive bound degrades to the same O(n log n) exactly when it would
matter most.

## The trap: minimal means minimal, not just correct

The actual spec of "skyline" isn't just "describe the right silhouette" — a
description with redundant points describes the same silhouette but is
*wrong output*. Two adjacent buildings of equal height sharing an edge:

```python
buildings = [(0, 5, 10), (5, 10, 10)]
```

A naive sweep emits a point whenever *something happens* — a building starts,
a building ends — and gets:

```
[(0, 10), (5, 10), (10, 0)]     # wrong: (5, 10) is redundant
```

`(5, 10)` says "the height changes to 10 at x=5", but it was already 10. The
correct output has no key point there at all:

```
[(0, 10), (10, 0)]              # right
```

Every method in this file avoids the trap the same way: compare the
*current* height against the *last height emitted*, and emit only on a real
change — never react to an event just because one occurred.

```python
if height != prev_height:
    points.append((x, height))
    prev_height = height
```

That one `if` is the entire fix, and it is why `sweep_line` tracks "current
max height" rather than "current active building", and why
`divide_and_conquer`'s merge tracks "current *merged* height" rather than
"an event from either side".

## The divide-and-conquer merge, precisely

`divide_and_conquer` splits the building list in half **by index**, not by
position — the two halves usually overlap in x-range, and that's fine. Each
half's skyline is still correct on its own (recursion bottoms out at a single
building: `[(left, height), (right, 0)]`, or `[]` for a height-0 building,
which would otherwise leave two redundant zero-height points behind). The
work is entirely in merging two already-correct skylines into one.

The merge walks both outlines left to right in lockstep, like a merge-sort
merge step, except each side also carries a **current height** — the height
its own skyline is holding at the x under consideration, which is just the
height of the last point consumed from that side:

```python
i = j = 0
h1 = h2 = 0  # what each side is doing "right now"
while i < len(a) and j < len(b):
    if a[i][0] < b[j][0]:
        x, h1 = a[i]
        i += 1
    elif b[j][0] < a[i][0]:
        x, h2 = b[j]
        j += 1
    else:  # shared x: both sides step together
        x, h1 = a[i]
        _, h2 = b[j]
        i += 1
        j += 1
    emit(x, max(h1, h2))  # emit() only appends if the height changed
```

At every x visited from either side, the merged silhouette's height is
`max(h1, h2)` — the taller of "what the left half is doing" and "what the
right half is doing" — and `emit` only appends when that merged height
differs from the last one appended. That second part is what makes the merge
correct rather than merely a plausible-looking loop: it would be easy to emit
whenever *either side* has a point, which over-emits exactly when one side's
skyline changes but the combined max doesn't move — e.g. side A drops from 10
to 7 while side B is holding steady at 12 the whole time. The visible skyline
doesn't change at all; only `h1` did.

Worked example — merging skyline `A = [(0,10),(5,0)]` with
`B = [(2,15),(8,0)]`:

```
x:      0        2        5        8
A says: 10   ->   10   ->   0   ->   0     (h1)
B says:  0   ->   15   ->  15   ->   0     (h2)
max:    10        15        15        0
emit?    yes       yes       no        yes
```

`x=5` is skipped even though `A` has a point there, because
`max(h1=0, h2=15) = 15`, unchanged from the previous emission at `x=2`. Result:
`[(0,10), (2,15), (8,0)]` — merging two 4-point-and-2-point outlines into
three points, not five.

## Correctness / verification

`brute_force` is the oracle: collect every candidate x (every `left` and
`right`), and for each gap between consecutive candidates, scan every
building for the max height that covers the whole gap. No heap, no
recursion — just the definition, so it's the thing to trust.

```
$ uv run --with sortedcontainers python skyline.py --verify
verify: 410 building sets x 3 methods vs brute_force -- OK
```

That's 10 hand-picked adversarial cases (shared edges on both sides, full
containment, identical footprints, a zero-width building sandwiched between
two real ones, all-zero heights, dense ties with 20 buildings sharing one
span, negative coordinates) plus 400 randomized trials that vary count (0–40
buildings), overlap density (`span` in `{5, 15, 60}` — narrow spans force
heavy overlap), and height distribution (`max_h` in `{1, 3, 20}` — small
`max_h` forces lots of tied heights, which is exactly what stresses the
equal-height-merge rule), each checked against all three non-oracle methods
(`sweep_line`, `sweep_line_bst`, `divide_and_conquer`). If
`sortedcontainers` isn't installed, `verify()` skips `sweep_line_bst` with a
note instead of failing outright, and the tally reads "x 2 methods".
`test_skyline.py` runs the same check as a property-based-style test
(`test_verify_many_random_trials`, marked `slow`, 4000 trials) plus a 4×4
grid over span × max_h explicitly (`test_cross_verification_grid`), and
asserts a structural invariant directly — no two consecutive emitted points
ever share a height (`test_output_has_no_consecutive_duplicate_heights`) —
rather than only checking equality against the oracle.

## Edge cases, and how each method handles them

| Case                                                                 | Handling                                                                                                                                                                                                                                                 |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Buildings sharing a left or right edge                               | `sweep_line` pushes every building starting at x, *then* pops expired ones, *then* reads the height — never reads an intermediate state mid-x. `divide_and_conquer`'s merge treats equal x's from both sides as one lockstep step.                       |
| Adjacent equal-height buildings                                      | Merged into one flat segment with no point at the shared edge — the "compare to last emitted height" rule above, tested explicitly (`test_adjacent_equal_height_merges_no_extra_point`, `test_three_adjacent_equal_height_buildings`).                   |
| Zero-width (`left == right`) or malformed (`left > right`) buildings | Filtered out before any method runs — a footprint with no width casts no shadow, and "malformed" degrades to "zero-width" rather than raising.                                                                                                           |
| Fully-contained buildings                                            | Present in the bookkeeping (pushed to the heap; one leaf of the recursion) but never surface a key point, because they never become the current maximum.                                                                                                 |
| Dense ties (many buildings start/end at the same x)                  | Handled identically to the shared-edge case: consume every event at that x before asking "what's the height now".                                                                                                                                        |
| Height-zero buildings                                                | Contribute nothing (`max(h, anything) `never changes because of them) — `divide_and_conquer`'s single-building base case explicitly returns `[]` rather than `[(l,0),(r,0)]`, which would otherwise leave two redundant background-height points behind. |
| End of the skyline                                                   | Falls out of the same rule with no special case: the height genuinely changes to 0 once the last building's right edge passes, so a point is emitted for it.                                                                                             |

## Watching it happen

```
$ uv run python skyline.py --demo
buildings: [(2, 9, 10), (3, 7, 15), (5, 12, 12), (15, 20, 10), (19, 24, 8)]

  15 | ####
  14 | ####
  13 | ####
  12 | #########
  11 | #########
  10 |##########   #####
   9 |##########   #####
   8 |##########   #########
   7 |##########   #########
   6 |##########   #########
   5 |##########   #########
   4 |##########   #########
   3 |##########   #########
   2 |##########   #########
   1 |##########   #########
     +----------------------
     2345678901234567890123

key points: (2, 10), (3, 15), (7, 12), (12, 0), (15, 10), (20, 8), (24, 0)

 brute: [(2, 10), (3, 15), (7, 12), (12, 0), (15, 10), (20, 8), (24, 0)]
 sweep: [(2, 10), (3, 15), (7, 12), (12, 0), (15, 10), (20, 8), (24, 0)]
    dc: [(2, 10), (3, 15), (7, 12), (12, 0), (15, 10), (20, 8), (24, 0)]
```

Each filled column is the max of every building covering it — the block
rendering *is* the traced outline, since the skyline is already the step
function that decides where each `#` stops.

## Benchmarks

Both `sweep_line` and `divide_and_conquer` are O(n log n), so raw
size-scaling doesn't separate them much — the real question is whether
*structure* (how much buildings overlap) favors one over the other. It does,
and the direction is not the obvious one.

**Throughput vs n**, two densities (`uv run python benchmark.py`, `sparse` =
non-overlapping buildings back to back, `downtown` = buildings packed into a
narrow span with heavy overlap):

```
== throughput, sparse buildings, seconds (lower is better) ==
         n   points       brute       sweep          dc
-------------------------------------------------------
      1000      978      0.0215      0.0004      0.0020
     10000     9810          --      0.0045      0.0260
    100000    97977          --      0.0512      0.3306
    500000   490028          --      0.2976      2.9423

== throughput, downtown buildings, seconds (lower is better) ==
         n   points       brute       sweep          dc
-------------------------------------------------------
      1000        9      0.0017      0.0007      0.0021
     10000       11          --      0.0072      0.0330
    100000       10          --      0.2147      0.3396
    500000        9          --      3.2119      1.5758
```

Two things jump out. First, `sweep_line` wins by a wide and fairly stable
margin (~9–11x) on **sparse** input at every size — no crossover there.
Second, and this is the actual finding: on **downtown** (heavily overlapping)
input, the ranking *flips* somewhere between n = 200,000 and n = 300,000:

```
== sweep_line vs divide_and_conquer, downtown density, by n ==
        n   points      sweep         dc     winner    ratio
--------------------------------------------------------------
   100000        9     0.1236     0.1812      sweep    1.47x
   150000       10     0.4584     0.5559      sweep    1.21x
   200000        9     0.7143     0.7515      sweep    1.05x
   300000        7     1.3981     1.0640         dc    1.31x
   500000        6     3.0441     1.7060         dc    1.78x
```

`divide_and_conquer` overtakes `sweep_line` once overlap is heavy enough and
n is large enough — and the gap *widens* in `dc`'s favor as n grows further,
the opposite of the sparse case.

**Why**, once you look at what each method actually keeps live: `sweep_line`'s
heap holds one entry per building that is *currently active* — started but
not yet ended — regardless of whether that building ever becomes tall enough
to matter for the visible skyline. Under heavy overlap, many buildings are
simultaneously active even though only a handful ever reach the top, so the
heap stays large (many O(log heap-size) pushes and lazy-pops) for the whole
sweep. `divide_and_conquer` never builds anything proportional to "how many
buildings overlap in x" — it only ever holds two already-*collapsed*
sub-skylines at a time, and under heavy overlap those sub-skylines shrink to
a handful of points almost immediately (most of a downtown block's buildings
get dominated within the first few merge levels), so the merges near the top
of the recursion stay cheap. Sparse input has the opposite shape: nothing
ever collapses, sub-skylines stay large all the way up, and `dc`'s per-level
list-slicing and recursive call overhead (which `sweep_line`'s flat loop over
a pre-sorted array never pays) dominates instead.

**Overlap sweep at fixed n = 40,000**, `span` controlling how much buildings
overlap (small span = dense; large span = sparse) — confirms the shape rather
than the flip, since 40,000 buildings never reaches the crossover size seen
above:

```
      span   points      sweep         dc     winner
----------------------------------------------------
        50        2     0.0656     0.1282      sweep
       200        3     0.0529     0.1342      sweep
      1000        8     0.0611     0.1511      sweep
      5000       13     0.0646     0.1267      sweep
     20000       16     0.0603     0.1169      sweep
    100000        7     0.1438     0.1795      sweep
```

`sweep_line` is faster everywhere here because 40,000 buildings simply isn't
enough for the heap-size effect above to outweigh `dc`'s recursion/slicing
overhead — the crossover in the n-sweep is a size effect as much as a density
one. Run `uv run --with sortedcontainers python benchmark.py` (or with
`--sizes` for custom points) to reproduce; numbers above are from an unloaded
machine and will vary run to run, though the *direction* of the flip
reproduces consistently.

### The other sweep: heap vs balanced-BST active set

`sweep_line_bst` runs the *identical* sweep as `sweep_line` — same events,
same "push starts, then pop/remove ends, then read the max" order at each x —
with one structural change: the active set is a
[`sortedcontainers.SortedList`](https://grantjenks.com/docs/sortedcontainers/sortedlist.html)
instead of a `heapq` max-heap. That isolates the trade-off to one thing: how
each structure deletes an ended building.

`heapq` has no arbitrary-element removal, so `sweep_line` never removes an
ended building when it ends — it's left in the heap and discarded lazily,
the next time it happens to surface at the top. Under heavy overlap with
many buildings ending around the same x, that means real wasted work: pops
that return a stale entry, thrown away, before the true current max is
found. `SortedList.remove` does true `O(log n)` deletion the instant a
building ends, so `sweep_line_bst` never carries a single byte of that
garbage — `len(active)` is always exactly the count of buildings genuinely
live at the current x.

That sounds like it should make `sweep_line_bst` win under heavy overlap.
Measured, it doesn't, anywhere:

```
== throughput, sparse buildings, seconds (lower is better) ==
         n   points       brute       sweep   sweep_bst          dc
-------------------------------------------------------------------
      1000      978      0.0215      0.0004      0.0009      0.0020
     10000     9810          --      0.0046      0.0093      0.0262
    100000    97977          --      0.0520      0.0975      0.3213
    500000   490028          --      0.2774      0.5151      3.5588

== throughput, downtown buildings, seconds (lower is better) ==
         n   points       brute       sweep   sweep_bst          dc
-------------------------------------------------------------------
      1000        9      0.0020      0.0014      0.0011      0.0034
     10000       11          --      0.0119      0.0218      0.0322
    100000       10          --      0.2349      0.3511      0.2879
    500000        9          --      3.0498      3.0702      1.7398
```

`sweep_line_bst` is consistently ~1.5–2x slower than `sweep_line` on
**sparse** input at every size, and on **downtown** it's slower or, at
best (n = 500,000, where both are swamped by `dc`), a statistical tie —
never a clear win. That holds even in a scenario built specifically to
maximize the heap's lazy-garbage cost — 300,000 buildings all starting at
x = 0 with scattered end times and descending heights, so most are buried
under a taller one and sit as pure lazy-deletion garbage until they
eventually surface and get popped:

```
adversarial same-start scattered-end: sweep=0.2472 sweep_bst=0.6008 winner=sweep
```

Even there, `sweep_line` wins by ~2.4x. So unlike `sweep_line` vs
`divide_and_conquer`, which really does flip depending on density and size,
**this comparison has no crossover in anything measured here** — the
heap's smaller constant factor (plain array pushes/pops on a flat Python
list) outweighs the BST's saved wasted-pop work at every density and size
tried, including the one built to be worst-case for the heap. `SortedList`'s
true `O(log n)` deletion is doing algorithmically less work per removal, but
that work still costs more per operation in practice than a heap pop
in CPython — a balanced BST (or `SortedList`'s list-of-sublists
approximation of one) has a larger constant factor than a binary heap in
most implementations, and here that constant dominates over the range of
sizes and densities this benchmark reaches. `sweep_line` is the better
default for exactly the reason `skyline()`'s `"auto"` already picks it; the
BST sweep exists to make that measured, not assumed. Reproduce with
`uv run --with sortedcontainers python benchmark.py`; numbers above are from
an unloaded machine and will vary run to run.

## Run it

```bash
cd "challenges/Algorithmic Challenges/Skyline Silhouette Computation"

uv run python skyline.py 2,9,10 3,7,15 5,12,12 15,20,10 19,24,8
uv run python skyline.py --demo
uv run python skyline.py --verify
uv run python skyline.py --method dc 0,5,10 5,10,10   # force one implementation

# sweep_line_bst needs sortedcontainers; every other path above works without it
uv run --with sortedcontainers python skyline.py --method sweep_bst 0,5,10 5,10,10
uv run --with sortedcontainers python skyline.py --demo     # includes sweep_bst in the comparison
uv run --with sortedcontainers python skyline.py --verify   # "x 3 methods" instead of "x 2"

uv run --with sortedcontainers python benchmark.py            # full run
uv run --with sortedcontainers python benchmark.py --quick    # ~10s

uv run --with pytest --with sortedcontainers pytest -q                # 43 tests
uv run --with pytest --with sortedcontainers pytest -q -m "not slow"  # skip the 4000-trial + grid tests
```

`brute_force`, `sweep_line`, and `divide_and_conquer` are standard library
only (`heapq`, `argparse`, `random`). `sweep_line_bst` additionally needs the
third-party `sortedcontainers` package; every command above that doesn't name
`sweep_bst` still runs with nothing beyond the standard library, and
`--verify` / `verify()` degrade gracefully (skip that one method, with a
note) rather than fail if it isn't installed.

## Where this is used

The skyline problem is a stand-in for a family of "interval max" questions
that show up wherever overlapping rectangles need to be reduced to their
combined outline:

- **Overlapping UI panels / z-order compositing** — the visible top edge of a
  stack of windows or cards at each x is exactly this problem, one dimension
  at a time.
- **City and game-world skyline rendering** — literal building silhouettes
  for a backdrop, procedurally generated or loaded from data, without
  rasterizing every building's full footprint.
- **Load-shedding / "max concurrent" problems** — "what's the peak number of
  simultaneously active X at any instant" (connections, jobs, reservations)
  is the same sweep with `height=1` per interval and `sum` instead of `max`;
  the lazy-deletion heap pattern here transfers directly.
- **Interval-max problems generally** — histogram-of-intervals compression,
  timeline/Gantt-chart rendering, and any "reduce many overlapping ranges to
  their envelope" task reach for the same sweep-line-with-a-heap or
  divide-and-conquer-merge shape.

## Sources

- [The Skyline Problem — LeetCode 218](https://leetcode.com/problems/the-skyline-problem/)
- [Skyline problem — GeeksforGeeks](https://www.geeksforgeeks.org/skyline-problem-using-divide-and-conquer-algorithm/) (the divide-and-conquer merge)
- [CLRS, *Introduction to Algorithms*](https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/) — merge-sort-style divide and conquer, and sweep-line techniques generally
- [sortedcontainers — `SortedList` documentation](https://grantjenks.com/docs/sortedcontainers/sortedlist.html) — the balanced-BST-like sorted sequence backing `sweep_line_bst`
- [Kirkpatrick–Seidel algorithm](https://en.wikipedia.org/wiki/Kirkpatrick%E2%80%93Seidel_algorithm) and [Chan's algorithm](https://en.wikipedia.org/wiki/Chan%27s_algorithm) — output-sensitive convex hull algorithms, cited for contrast in the lower-bound discussion above (no equivalent output-sensitive bound is established for skyline)
