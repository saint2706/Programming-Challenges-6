"""Instrumented sorting algorithms.

Every algorithm below is a generator that mutates ``arr`` in place and
``yield``s a :class:`Step` after each meaningful operation. The step stream is
the contract between algorithm logic and the Manim visualization layer in
``visualize.py`` — the animation driver never contains algorithm-specific
code, it just reacts to ``compare`` / ``swap`` / ``write`` / ``aux`` fields.

Run this file directly to self-check that every algorithm produces a sorted
array from ``BASE_ARRAY`` and that the step stream is well-formed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# 10 elements, two-digit range (drives 2 Radix Sort passes), three duplicate
# pairs (8, 12, 23) so stable vs. unstable behavior is visible on screen.
BASE_ARRAY = [23, 8, 12, 29, 12, 4, 8, 19, 23, 15]


@dataclass
class Step:
    array: list
    compare: tuple | None = None
    swap: tuple | None = None
    write: tuple | None = None
    sorted_idx: frozenset = field(default_factory=frozenset)
    aux: dict | None = None
    caption: str = ""


# ---------------------------------------------------------------------------
# Comparison-based
# ---------------------------------------------------------------------------


def selection_sort_steps(arr):
    n = len(arr)
    sorted_idx = set()
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            yield Step(
                list(arr),
                compare=(min_idx, j),
                sorted_idx=frozenset(sorted_idx),
                caption=f"Scan for minimum: compare {min_idx} and {j}",
            )
            if arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            yield Step(
                list(arr),
                swap=(i, min_idx),
                sorted_idx=frozenset(sorted_idx),
                caption=f"Swap minimum into place: {i} <-> {min_idx}",
            )
        sorted_idx.add(i)
        yield Step(
            list(arr), sorted_idx=frozenset(sorted_idx), caption=f"Index {i} finalized"
        )


def bubble_sort_steps(arr):
    n = len(arr)
    sorted_idx = set()
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            yield Step(
                list(arr),
                compare=(j, j + 1),
                sorted_idx=frozenset(sorted_idx),
                caption=f"Compare adjacent pair {j}, {j + 1}",
            )
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
                yield Step(
                    list(arr),
                    swap=(j, j + 1),
                    sorted_idx=frozenset(sorted_idx),
                    caption=f"Swap {j}, {j + 1}",
                )
        sorted_idx.add(n - i - 1)
        yield Step(
            list(arr),
            sorted_idx=frozenset(sorted_idx),
            caption=f"Index {n - i - 1} finalized",
        )
        if not swapped:
            break
    sorted_idx.update(range(n))
    yield Step(list(arr), sorted_idx=frozenset(sorted_idx), caption="Sorted")


def insertion_sort_steps(arr):
    n = len(arr)
    yield Step(
        list(arr), sorted_idx=frozenset({0}), caption="Index 0 is trivially sorted"
    )
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        yield Step(list(arr), compare=(j, i), caption=f"Hold key={key} from index {i}")
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            yield Step(list(arr), write=(j + 1,), caption=f"Shift {j} -> {j + 1}")
            j -= 1
            if j >= 0:
                yield Step(
                    list(arr), compare=(j, i), caption="Compare key against next"
                )
        arr[j + 1] = key
        yield Step(
            list(arr),
            write=(j + 1,),
            sorted_idx=frozenset(range(i + 1)),
            caption=f"Insert key at {j + 1}",
        )


def merge_sort_steps(arr):
    def merge_sort(l, r):
        if r - l <= 1:
            return
        m = (l + r) // 2
        yield from merge_sort(l, m)
        yield from merge_sort(m, r)
        yield Step(
            list(arr),
            aux={"type": "segment", "range": (l, r)},
            caption=f"Merging [{l}, {r})",
        )
        left, right = arr[l:m], arr[m:r]
        i = j = 0
        k = l
        while i < len(left) and j < len(right):
            yield Step(
                list(arr),
                compare=(l + i, m + j),
                aux={"type": "segment", "range": (l, r)},
                caption="Compare run heads",
            )
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            yield Step(
                list(arr),
                write=(k,),
                aux={"type": "segment", "range": (l, r)},
                caption=f"Write index {k}",
            )
            k += 1
        while i < len(left):
            arr[k] = left[i]
            i += 1
            yield Step(
                list(arr),
                write=(k,),
                aux={"type": "segment", "range": (l, r)},
                caption="Flush left run",
            )
            k += 1
        while j < len(right):
            arr[k] = right[j]
            j += 1
            yield Step(
                list(arr),
                write=(k,),
                aux={"type": "segment", "range": (l, r)},
                caption="Flush right run",
            )
            k += 1

    yield from merge_sort(0, len(arr))
    yield Step(list(arr), sorted_idx=frozenset(range(len(arr))), caption="Sorted")


def quick_sort_steps(arr):
    def qsort(lo, hi, sorted_idx):
        if lo > hi:
            return
        if lo == hi:
            sorted_idx.add(lo)
            return
        pivot = arr[hi]
        i = lo - 1
        for j in range(lo, hi):
            yield Step(
                list(arr),
                compare=(j, hi),
                aux={"type": "pivot", "idx": hi},
                sorted_idx=frozenset(sorted_idx),
                caption=f"Compare {j} against pivot",
            )
            if arr[j] <= pivot:
                i += 1
                if i != j:
                    arr[i], arr[j] = arr[j], arr[i]
                    yield Step(
                        list(arr),
                        swap=(i, j),
                        aux={"type": "pivot", "idx": hi},
                        sorted_idx=frozenset(sorted_idx),
                        caption=f"Swap {i}, {j}",
                    )
        arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
        yield Step(
            list(arr),
            swap=(i + 1, hi),
            sorted_idx=frozenset(sorted_idx),
            caption=f"Pivot placed at {i + 1}",
        )
        sorted_idx.add(i + 1)
        yield from qsort(lo, i, sorted_idx)
        yield from qsort(i + 2, hi, sorted_idx)

    sorted_idx = set()
    yield from qsort(0, len(arr) - 1, sorted_idx)
    yield Step(list(arr), sorted_idx=frozenset(range(len(arr))), caption="Sorted")


def heap_sort_steps(arr):
    n = len(arr)

    def sift_down(start, end):
        root = start
        while True:
            child = 2 * root + 1
            if child > end:
                break
            yield Step(
                list(arr),
                compare=(root, child),
                aux={"type": "heap", "size": end + 1},
                caption="Compare parent with left child",
            )
            if child + 1 <= end:
                yield Step(
                    list(arr),
                    compare=(child, child + 1),
                    aux={"type": "heap", "size": end + 1},
                    caption="Compare children",
                )
                if arr[child + 1] > arr[child]:
                    child += 1
            if arr[root] < arr[child]:
                arr[root], arr[child] = arr[child], arr[root]
                yield Step(
                    list(arr),
                    swap=(root, child),
                    aux={"type": "heap", "size": end + 1},
                    caption=f"Sift down: swap {root}, {child}",
                )
                root = child
            else:
                break

    for start in range(n // 2 - 1, -1, -1):
        yield from sift_down(start, n - 1)
    sorted_idx = set()
    for end in range(n - 1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]
        sorted_idx.add(end)
        yield Step(
            list(arr),
            swap=(0, end),
            aux={"type": "heap", "size": end},
            sorted_idx=frozenset(sorted_idx),
            caption=f"Move max to {end}",
        )
        yield from sift_down(0, end - 1)
    sorted_idx.add(0)
    yield Step(list(arr), sorted_idx=frozenset(range(n)), caption="Sorted")


def cycle_sort_steps(arr):
    n = len(arr)
    sorted_idx = set()
    for cycle_start in range(n - 1):
        item = arr[cycle_start]
        pos = cycle_start
        for i in range(cycle_start + 1, n):
            yield Step(
                list(arr),
                compare=(cycle_start, i),
                aux={"type": "cycle", "start": cycle_start},
                caption="Count elements smaller than item",
            )
            if arr[i] < item:
                pos += 1
        if pos != cycle_start:
            while item == arr[pos]:
                pos += 1
            arr[pos], item = item, arr[pos]
            yield Step(
                list(arr),
                write=(pos,),
                aux={"type": "cycle", "start": cycle_start},
                caption=f"Place item directly at {pos}",
            )
            while pos != cycle_start:
                pos = cycle_start
                for i in range(cycle_start + 1, n):
                    yield Step(
                        list(arr),
                        compare=(cycle_start, i),
                        aux={"type": "cycle", "start": cycle_start},
                        caption="Count elements smaller than item",
                    )
                    if arr[i] < item:
                        pos += 1
                while item == arr[pos]:
                    pos += 1
                arr[pos], item = item, arr[pos]
                yield Step(
                    list(arr),
                    write=(pos,),
                    aux={"type": "cycle", "start": cycle_start},
                    caption=f"Continue cycle: place item at {pos}",
                )
        sorted_idx.add(cycle_start)
    yield Step(list(arr), sorted_idx=frozenset(range(n)), caption="Sorted")


def three_way_merge_sort_steps(arr):
    def merge3(l, r):
        if r - l <= 1:
            return
        third = max(1, (r - l) // 3)
        m1 = min(l + third, r - 1)
        m2 = min(m1 + third, r)
        if m2 <= m1:
            m2 = min(m1 + 1, r)
        yield from merge3(l, m1)
        yield from merge3(m1, m2)
        yield from merge3(m2, r)
        yield Step(
            list(arr),
            aux={"type": "segment", "range": (l, r)},
            caption=f"3-way merging [{l}, {r})",
        )
        a, b, c = arr[l:m1], arr[m1:m2], arr[m2:r]
        i = j = k = 0
        idx = l
        while i < len(a) or j < len(b) or k < len(c):
            yield Step(
                list(arr),
                aux={"type": "segment", "range": (l, r)},
                caption="Compare heads of 3 runs",
            )
            candidates = []
            if i < len(a):
                candidates.append(("a", a[i]))
            if j < len(b):
                candidates.append(("b", b[j]))
            if k < len(c):
                candidates.append(("c", c[k]))
            which, _ = min(candidates, key=lambda t: t[1])
            if which == "a":
                arr[idx] = a[i]
                i += 1
            elif which == "b":
                arr[idx] = b[j]
                j += 1
            else:
                arr[idx] = c[k]
                k += 1
            yield Step(
                list(arr),
                write=(idx,),
                aux={"type": "segment", "range": (l, r)},
                caption=f"Write index {idx}",
            )
            idx += 1

    yield from merge3(0, len(arr))
    yield Step(list(arr), sorted_idx=frozenset(range(len(arr))), caption="Sorted")


# ---------------------------------------------------------------------------
# Non-comparison-based
# ---------------------------------------------------------------------------


def counting_sort_steps(arr):
    n = len(arr)
    lo, hi = min(arr), max(arr)
    count = [0] * (hi - lo + 1)
    for v in arr:
        count[v - lo] += 1
        yield Step(
            list(arr),
            aux={"type": "count", "table": list(count), "lo": lo},
            caption=f"Tally value {v}",
        )
    for i in range(1, len(count)):
        count[i] += count[i - 1]
        yield Step(
            list(arr),
            aux={"type": "count", "table": list(count), "lo": lo, "prefix": True},
            caption="Build prefix-sum (placement) table",
        )
    output = [None] * n
    for v in reversed(arr):
        count[v - lo] -= 1
        output[count[v - lo]] = v
        yield Step(
            list(output),
            aux={"type": "count", "table": list(count), "lo": lo},
            caption=f"Place {v} at output index {count[v - lo]}",
        )
    arr[:] = output
    yield Step(list(arr), sorted_idx=frozenset(range(n)), caption="Sorted")


def radix_sort_steps(arr):
    n = len(arr)
    max_val = max(arr)
    exp = 1
    pass_num = 0
    while max_val // exp > 0:
        pass_num += 1
        buckets = [[] for _ in range(10)]
        for v in arr:
            d = (v // exp) % 10
            buckets[d].append(v)
            yield Step(
                list(arr),
                aux={
                    "type": "radix",
                    "buckets": [list(b) for b in buckets],
                    "digit": d,
                },
                caption=f"Pass {pass_num}: digit {d} of {v} (place value {exp})",
            )
        arr[:] = [v for bucket in buckets for v in bucket]
        yield Step(
            list(arr),
            aux={"type": "radix", "buckets": [list(b) for b in buckets]},
            caption=f"Pass {pass_num} complete: array re-collected from buckets",
        )
        exp *= 10
    yield Step(list(arr), sorted_idx=frozenset(range(n)), caption="Sorted")


def bucket_sort_steps(arr):
    n = len(arr)
    lo, hi = min(arr), max(arr)
    num_buckets = min(n, 5) or 1
    width = (hi - lo) / num_buckets if hi > lo else 1
    buckets = [[] for _ in range(num_buckets)]
    for v in arr:
        idx = min(int((v - lo) / width), num_buckets - 1) if width else 0
        buckets[idx].append(v)
        yield Step(
            list(arr),
            aux={"type": "buckets", "buckets": [list(b) for b in buckets]},
            caption=f"Scatter {v} into bucket {idx}",
        )
    for b in buckets:
        for i in range(1, len(b)):
            key = b[i]
            j = i - 1
            while j >= 0 and b[j] > key:
                b[j + 1] = b[j]
                j -= 1
            b[j + 1] = key
        yield Step(
            list(arr),
            aux={"type": "buckets", "buckets": [list(x) for x in buckets]},
            caption="Insertion-sort within each bucket",
        )
    arr[:] = [v for bucket in buckets for v in bucket]
    yield Step(
        list(arr),
        sorted_idx=frozenset(range(n)),
        aux={"type": "buckets", "buckets": [list(x) for x in buckets]},
        caption="Concatenate buckets in order",
    )


def pigeonhole_sort_steps(arr):
    n = len(arr)
    lo, hi = min(arr), max(arr)
    holes = [[] for _ in range(hi - lo + 1)]
    for v in arr:
        holes[v - lo].append(v)
        yield Step(
            list(arr),
            aux={"type": "holes", "holes": [list(h) for h in holes], "lo": lo},
            caption=f"Drop {v} directly into hole {v}",
        )
    arr[:] = [v for hole in holes for v in hole]
    yield Step(
        list(arr),
        sorted_idx=frozenset(range(n)),
        aux={"type": "holes", "holes": [list(h) for h in holes], "lo": lo},
        caption="Read holes back in order",
    )


# ---------------------------------------------------------------------------
# Hybrid
# ---------------------------------------------------------------------------


def introsort_steps(arr):
    n = len(arr)
    max_depth = max(1, 2 * int(math.log2(n))) if n > 1 else 1
    size_threshold = 4  # small on purpose so this 10-element demo actually falls back

    def ins_sort(lo, hi):
        yield Step(
            list(arr),
            aux={"type": "mode", "mode": "insertion"},
            caption=f"Range too small [{lo},{hi}] -> Insertion Sort",
        )
        for i in range(lo + 1, hi + 1):
            key = arr[i]
            j = i - 1
            while j >= lo and arr[j] > key:
                arr[j + 1] = arr[j]
                yield Step(
                    list(arr),
                    write=(j + 1,),
                    aux={"type": "mode", "mode": "insertion"},
                    caption="Shift",
                )
                j -= 1
            arr[j + 1] = key
            yield Step(
                list(arr),
                write=(j + 1,),
                aux={"type": "mode", "mode": "insertion"},
                caption="Insert",
            )

    def heap_range(lo, hi):
        yield Step(
            list(arr),
            aux={"type": "mode", "mode": "heap"},
            caption=f"Depth limit hit on [{lo},{hi}] -> Heap Sort",
        )
        sub = arr[lo : hi + 1]

        def remap(idxs):
            return tuple(x + lo for x in idxs) if idxs else None

        for step in heap_sort_steps(sub):
            arr[lo : hi + 1] = sub
            yield Step(
                list(arr),
                compare=remap(step.compare),
                swap=remap(step.swap),
                write=remap(step.write),
                aux={"type": "mode", "mode": "heap"},
                caption=step.caption,
            )

    def qsort(lo, hi, depth):
        while hi - lo + 1 > size_threshold:
            if depth <= 0:
                yield from heap_range(lo, hi)
                return
            depth -= 1
            yield Step(
                list(arr),
                aux={"type": "mode", "mode": "quick"},
                caption=f"Quicksort partition [{lo},{hi}]",
            )
            pivot = arr[hi]
            i = lo - 1
            for j in range(lo, hi):
                yield Step(
                    list(arr),
                    compare=(j, hi),
                    aux={"type": "mode", "mode": "quick"},
                    caption="Compare to pivot",
                )
                if arr[j] <= pivot:
                    i += 1
                    if i != j:
                        arr[i], arr[j] = arr[j], arr[i]
                        yield Step(
                            list(arr),
                            swap=(i, j),
                            aux={"type": "mode", "mode": "quick"},
                            caption="Swap",
                        )
            arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
            yield Step(
                list(arr),
                swap=(i + 1, hi),
                aux={"type": "mode", "mode": "quick"},
                caption="Pivot placed",
            )
            p = i + 1
            if p - lo < hi - p:
                yield from qsort(lo, p - 1, depth)
                lo = p + 1
            else:
                yield from qsort(p + 1, hi, depth)
                hi = p - 1
        yield from ins_sort(lo, hi)

    yield from qsort(0, n - 1, max_depth)
    yield Step(list(arr), sorted_idx=frozenset(range(n)), caption="Sorted")


def timsort_steps(arr):
    n = len(arr)
    min_run = 3
    runs = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and arr[j] >= arr[j - 1]:
            j += 1
        run_end = min(max(j, i + min_run), n)
        yield Step(
            list(arr),
            aux={"type": "runs", "runs": runs + [(i, run_end)]},
            caption=f"Insertion-sort run [{i}, {run_end})",
        )
        for k in range(i + 1, run_end):
            key = arr[k]
            p = k - 1
            while p >= i and arr[p] > key:
                arr[p + 1] = arr[p]
                yield Step(
                    list(arr),
                    write=(p + 1,),
                    aux={"type": "runs", "runs": runs + [(i, run_end)]},
                    caption="Shift within run",
                )
                p -= 1
            arr[p + 1] = key
            yield Step(
                list(arr),
                write=(p + 1,),
                aux={"type": "runs", "runs": runs + [(i, run_end)]},
                caption="Insert within run",
            )
        runs.append((i, run_end))
        i = run_end
    yield Step(
        list(arr),
        aux={"type": "runs", "runs": list(runs)},
        caption=f"Identified {len(runs)} natural runs",
    )

    while len(runs) > 1:
        new_runs = []
        it = iter(runs)
        for a in it:
            b = next(it, None)
            if b is None:
                new_runs.append(a)
                continue
            l, m, r = a[0], a[1], b[1]
            yield Step(
                list(arr),
                aux={"type": "segment", "range": (l, r)},
                caption=f"Merging runs [{l},{m}) + [{m},{r})",
            )
            left, right = arr[l:m], arr[m:r]
            x = y = 0
            k = l
            while x < len(left) and y < len(right):
                yield Step(
                    list(arr),
                    compare=(l + x, m + y),
                    aux={"type": "segment", "range": (l, r)},
                    caption="Compare run heads",
                )
                if left[x] <= right[y]:
                    arr[k] = left[x]
                    x += 1
                else:
                    arr[k] = right[y]
                    y += 1
                yield Step(
                    list(arr),
                    write=(k,),
                    aux={"type": "segment", "range": (l, r)},
                    caption=f"Write {k}",
                )
                k += 1
            while x < len(left):
                arr[k] = left[x]
                x += 1
                yield Step(
                    list(arr),
                    write=(k,),
                    aux={"type": "segment", "range": (l, r)},
                    caption="Flush left",
                )
                k += 1
            while y < len(right):
                arr[k] = right[y]
                y += 1
                yield Step(
                    list(arr),
                    write=(k,),
                    aux={"type": "segment", "range": (l, r)},
                    caption="Flush right",
                )
                k += 1
            new_runs.append((l, r))
        runs = new_runs
    yield Step(list(arr), sorted_idx=frozenset(range(n)), caption="Sorted")


ALGORITHMS = {
    "Selection Sort": selection_sort_steps,
    "Bubble Sort": bubble_sort_steps,
    "Insertion Sort": insertion_sort_steps,
    "Merge Sort": merge_sort_steps,
    "Quick Sort": quick_sort_steps,
    "Heap Sort": heap_sort_steps,
    "Cycle Sort": cycle_sort_steps,
    "3-Way Merge Sort": three_way_merge_sort_steps,
    "Counting Sort": counting_sort_steps,
    "Radix Sort": radix_sort_steps,
    "Bucket Sort": bucket_sort_steps,
    "Pigeonhole Sort": pigeonhole_sort_steps,
    "IntroSort": introsort_steps,
    "TimSort": timsort_steps,
}

# The one-line characterisation shown under each title. It lives here rather
# than on the Manim scene classes because it is a fact about the algorithm,
# not about the animation -- and because `pacing.py` has to budget reading
# time for it without importing Manim.
SUBTITLES = {
    "Selection Sort": "Unstable | O(n^2) always | minimal writes",
    "Bubble Sort": "Stable | O(n^2) | early-exit when no swaps happen",
    "Insertion Sort": "Stable | O(n^2) worst | O(n) best on nearly-sorted input",
    "Merge Sort": "Stable | O(n log n) always | needs O(n) extra space",
    "Quick Sort": "Unstable | O(n log n) average | O(n^2) worst case",
    "Heap Sort": "Unstable | O(n log n) always | O(1) extra space",
    "Cycle Sort": "Stable-ish | O(n^2) | minimizes total number of writes",
    "3-Way Merge Sort": "Stable | O(n log3 n) comparisons | fewer merge levels than 2-way",
    "Counting Sort": "Stable | O(n + k), k = value range | non-comparison",
    "Radix Sort": "Stable | O(d * (n + b)), d = digits | non-comparison",
    "Bucket Sort": "Stable | O(n + k) average | needs a roughly uniform distribution",
    "Pigeonhole Sort": "Stable | O(n + range) | mechanically close to Counting Sort",
    "IntroSort": "Unstable | O(n log n) worst-case guaranteed | Quick -> Heap -> Insertion",
    "TimSort": "Stable | O(n log n) worst, O(n) best | runs + merge, powers Python's sort",
}


def _self_check():
    expected = sorted(BASE_ARRAY)
    for name, fn in ALGORITHMS.items():
        arr = list(BASE_ARRAY)
        last = None
        step_count = 0
        for step in fn(arr):
            last = step
            step_count += 1
        assert last is not None, f"{name}: produced no steps"
        assert list(last.array) == expected, (
            f"{name}: final array {last.array} != {expected}"
        )
        assert arr == expected, f"{name}: mutated arr {arr} != {expected}"
        print(f"OK  {name:<20} {step_count:>4} steps")


if __name__ == "__main__":
    _self_check()
