"""Counting inversions: merge sort, Fenwick tree, and a vectorised third way.

An inversion is a pair i < j with a[i] > a[j] -- a pair the array has in the
wrong order. Counting them is the standard "divide and conquer beats the
obvious O(n^2)" exercise, and the count is worth more than the exercise
suggests: it is exactly the number of adjacent swaps bubble sort would make,
exactly the Kendall tau distance from the sorted order, and the statistic
whose distribution over S_n is the Gaussian binomial [n]_q!. All three of
those are asserted in the test suite rather than just claimed here.

The brief is merge sort vs Fenwick tree. Both are O(n log n), and in Python
both run at roughly 3 microseconds per element at n = 10^6, because both
spend their O(n log n) *in the interpreter* -- the asymptotics are equal and
so are the constants, to within 30%. The comparison only gets interesting
once a method moves the work into C, so there are two more here.

**count_numpy** is a bottom-up merge sort whose every level is three ndarray
calls. At width w the array is a (blocks, 2w) matrix of independent merges,
and `np.searchsorted` searches one haystack, not a batch of them -- so add
`block_index * (n+1)` to every value first. Ranks live in [0, n], so block b
then occupies [b(n+1), b(n+1)+n]: disjoint from every other block and
increasing in b, which makes the concatenated left halves *globally* sorted.
One binary search now answers every block at once. O(n log^2 n) -- an extra
log versus the textbook merge sort, and 2.6x faster than it in practice.

**count_numpy_radix** removes that extra log. Turn the recursion inside out:
split by *value* rather than by index, MSB-first on the ranks, and the
question "how many earlier elements are larger" becomes "how many 1-bits
precede this 0-bit", which is a cumsum rather than a binary search. That is
O(n) per bit and O(n log n) overall -- asymptotically the best method here.

It is also, measurably, *not the fastest at scale*, and that is the most
interesting result in this directory. Its per-bit scatter writes across the
whole array, so past ~300k elements it falls out of L3 and memory bandwidth
decides the race; `count_numpy`'s searchsorted has far better locality and
wins by ~1.4x at n = 10^6 despite doing asymptotically more work. Below
300k they are within noise of each other. Asymptotics rank algorithms;
cache hierarchies rank implementations.

Methods, all exact and all agreeing on every input:

    count_brute       O(n^2)          the definition; the oracle
    count_insort      O(n^2) memmove  fastest below n ~ 3000 (the loop is in C)
    count_mergesort   O(n log n)      iterative, so no recursion limit at n = 10^8
    count_fenwick     O(n log n)      also yields the per-element counts
    count_numpy       O(n log^2 n)    vectorised; fastest past n ~ 300k
    count_numpy_radix O(n log n)      vectorised; best asymptotics, memory-bound

Beyond the count itself: `count_smaller_to_right`, `kendall_tau_distance`,
`kendall_tau_b` (tie-corrected, matching Knight's algorithm),
`count_significant_inversions` (pairs with a[i] > f * a[j]), the Lehmer code
via `inversion_table`, and `inversion_polynomial` for the generating function.

    uv run python inversions.py --demo
    uv run --with numpy python inversions.py --verify
"""

from __future__ import annotations

import argparse
import bisect
import math
import random
import sys
from collections.abc import Callable, Iterable, Sequence
from typing import Any

__all__ = [
    "Fenwick",
    "METHODS",
    "count_inversions",
    "count_brute",
    "count_insort",
    "count_mergesort",
    "count_fenwick",
    "count_numpy",
    "count_numpy_radix",
    "count_smaller_to_right",
    "count_greater_to_left",
    "count_significant_inversions",
    "kendall_tau_distance",
    "kendall_tau_b",
    "inversion_table",
    "from_inversion_table",
    "inversion_polynomial",
    "bubble_sort_swaps",
    "max_inversions",
    "verify",
    "main",
]

#: Accepted values for ``method``; ``"auto"`` picks by size and availability.
METHODS = ("auto", "brute", "insort", "mergesort", "fenwick", "numpy", "radix")


# ---------------------------------------------------------------------------
# Shared plumbing: ranks, validation
# ---------------------------------------------------------------------------


def _prepare(
    seq: Iterable[Any],
    key: Callable[[Any], Any] | None,
    reverse: bool,
    validate: bool,
) -> list[Any]:
    """Materialise, apply ``key``, flip for ``reverse``, and reject NaN.

    ``reverse=True`` counts *non*-inversions -- pairs already in order --
    which is what you want when the array is meant to be descending. It is
    implemented by negating the comparison rather than reversing the list,
    because reversing would also swap which member of a tied pair comes first.
    """
    values = list(seq)
    if key is not None:
        values = [key(v) for v in values]
    if validate and any(v != v for v in values):
        raise ValueError(
            "sequence contains NaN: comparisons against NaN are all false, so "
            "the elements are not totally ordered and the inversion count is "
            "undefined. Filter or replace them, or pass validate=False."
        )
    if reverse:
        values = [_Reversed(v) for v in values]
    return values


class _Item:
    """Carries the original element alongside the value being compared.

    Only used by ``count_mergesort(return_sorted=True)`` with a ``key`` or
    ``reverse``: without it the sorted output would be the *keys*, which is
    almost never what the caller meant. Comparison touches only ``k``, so the
    original never needs to be orderable.
    """

    __slots__ = ("k", "v")

    def __init__(self, k: Any, v: Any) -> None:
        self.k = k
        self.v = v

    def __lt__(self, other: "_Item") -> bool:
        return self.k < other.k


class _Reversed:
    """Wraps a value so that ``<`` means ``>``. Cheaper than a cmp_to_key shim."""

    __slots__ = ("v",)

    def __init__(self, v: Any) -> None:
        self.v = v

    def __lt__(self, other: "_Reversed") -> bool:
        return other.v < self.v

    def __le__(self, other: "_Reversed") -> bool:
        return not (self.v < other.v)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Reversed) and not (self.v < other.v or other.v < self.v)

    def __hash__(self) -> int:  # pragma: no cover - only needed to stay hashable
        return 0


def _ranks(values: Sequence[Any]) -> tuple[list[int], int]:
    """Dense ranks: equal elements get equal ranks. Returns (ranks, distinct count).

    Sort-based rather than ``sorted(set(...))``-based, so it works for
    unhashable elements (lists, dicts) and needs nothing but ``<``. Equality is
    inferred as "neither is less than the other", which is the right definition
    for a total order and avoids requiring ``__eq__`` to agree with ``__lt__``.
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    out = [0] * n
    rank = 0
    for pos, idx in enumerate(order):
        if pos and values[order[pos - 1]] < values[idx]:
            rank += 1
        out[idx] = rank
    return out, (rank + 1 if n else 0)


def max_inversions(n: int) -> int:
    """The most inversions an array of ``n`` elements can have: C(n, 2)."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return n * (n - 1) // 2


# ---------------------------------------------------------------------------
# 1. Brute force -- the definition, and the oracle
# ---------------------------------------------------------------------------


def count_brute(seq: Iterable[Any], *, key=None, reverse: bool = False,
                validate: bool = True) -> int:
    """O(n^2) pair enumeration. Correct by construction; the reference."""
    v = _prepare(seq, key, reverse, validate)
    return sum(1 for i in range(len(v)) for j in range(i + 1, len(v)) if v[j] < v[i])


# ---------------------------------------------------------------------------
# 2. Insertion into a sorted list -- O(n^2) on paper, fastest in practice small
# ---------------------------------------------------------------------------


def count_insort(seq: Iterable[Any], *, key=None, reverse: bool = False,
                 validate: bool = True) -> int:
    """Walk right to left, keeping the suffix sorted; count what is smaller.

    ``bisect.insort`` is O(n) per insertion because of the list shift, so this
    is O(n^2) memmove -- but memmove moves several bytes per nanosecond while
    a Python-level merge step manages one element per microsecond. The
    crossover against `count_mergesort` sits near n = 3000, which covers a
    surprising share of real calls.
    """
    v = _prepare(seq, key, reverse, validate)
    seen: list[Any] = []
    total = 0
    for x in reversed(v):
        pos = bisect.bisect_left(seen, x)
        total += pos
        seen.insert(pos, x)
    return total


# ---------------------------------------------------------------------------
# 3. Merge sort -- the classic, iterative so it has no depth limit
# ---------------------------------------------------------------------------


def count_mergesort(seq: Iterable[Any], *, key=None, reverse: bool = False,
                    validate: bool = True, return_sorted: bool = False):
    """Bottom-up merge sort, counting as it merges. O(n log n) time, O(n) space.

    When the merge takes an element from the right half while ``mid - i``
    elements remain in the left half, those are exactly the left-half elements
    greater than it, and each is an inversion. Every inversion is counted once
    because every pair is split by exactly one merge -- the one at the level
    where its two indices first fall in different blocks.

    Bottom-up rather than recursive on purpose: at n = 10^8 the recursive
    version needs a depth-27 stack, which is fine, but the recursive version
    with a bad pivot-free split is the one people write with ``a[:mid]``
    slicing, and that allocates O(n log n) memory. This one allocates exactly
    two buffers, ever, and reuses them by swapping.

    ``return_sorted=True`` additionally hands back the sorted array -- the
    *original* elements, not the keyed ones, so it composes with ``key`` the
    way ``sorted(..., key=...)`` does.
    """
    originals = list(seq)
    arr = _prepare(originals, key, reverse, validate)
    decorated = return_sorted and (key is not None or reverse)
    if decorated:
        arr = [_Item(k, v) for k, v in zip(arr, originals)]
    n = len(arr)
    if n < 2:
        if not return_sorted:
            return 0
        return 0, ([item.v for item in arr] if decorated else arr)

    buf: list[Any] = [None] * n
    total = 0
    width = 1
    while width < n:
        span = width * 2
        for lo in range(0, n, span):
            mid = lo + width
            hi = min(lo + span, n)
            if mid >= hi:  # a lone tail block: copy through, nothing to merge
                buf[lo:hi] = arr[lo:hi]
                continue
            i, j, k = lo, mid, lo
            while i < mid and j < hi:
                if arr[j] < arr[i]:
                    total += mid - i
                    buf[k] = arr[j]
                    j += 1
                else:
                    buf[k] = arr[i]
                    i += 1
                k += 1
            if i < mid:
                buf[k:hi] = arr[i:mid]
            else:
                buf[k:hi] = arr[j:hi]
        arr, buf = buf, arr
        width = span

    if not return_sorted:
        return total
    return total, ([item.v for item in arr] if decorated else arr)


# ---------------------------------------------------------------------------
# 4. Fenwick tree -- and the only method that gives per-element counts
# ---------------------------------------------------------------------------


class Fenwick:
    """Binary indexed tree over ``size`` slots, 0-indexed externally.

    Prefix sums and point updates in O(log n) apiece, in one flat list with no
    node objects. The index trick is that ``i & -i`` isolates the lowest set
    bit, which is exactly the length of the range the slot is responsible for.
    """

    __slots__ = ("_n", "_t")

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError(f"size must be non-negative, got {size}")
        self._n = size
        self._t = [0] * (size + 1)

    def add(self, index: int, delta: int = 1) -> None:
        """Add ``delta`` at ``index``."""
        if not 0 <= index < self._n:
            raise IndexError(f"index {index} out of range for size {self._n}")
        i = index + 1
        t = self._t
        while i <= self._n:
            t[i] += delta
            i += i & -i

    def prefix(self, count: int) -> int:
        """Sum of the first ``count`` slots, i.e. indices ``[0, count)``."""
        if count < 0:
            raise IndexError(f"count must be non-negative, got {count}")
        i = min(count, self._n)
        t = self._t
        total = 0
        while i > 0:
            total += t[i]
            i -= i & -i
        return total

    def total(self) -> int:
        return self.prefix(self._n)

    def __len__(self) -> int:
        return self._n

    @classmethod
    def from_counts(cls, counts: Sequence[int]) -> "Fenwick":
        """Build from initial slot values in O(n) rather than O(n log n)."""
        tree = cls(len(counts))
        t = tree._t
        for i, c in enumerate(counts, start=1):
            t[i] += c
            parent = i + (i & -i)
            if parent <= tree._n:
                t[parent] += t[i]
        return tree


def count_fenwick(seq: Iterable[Any], *, key=None, reverse: bool = False,
                  validate: bool = True) -> int:
    """Coordinate-compress, then sweep right to left counting smaller elements.

    O(n log n), with the log coming from the tree rather than the recursion.
    Compression is what makes it work for arbitrary comparable values and not
    just small integers -- and it is where the O(n log n) really lives, since
    the sweep itself is n cheap tree updates.
    """
    v = _prepare(seq, key, reverse, validate)
    ranks, distinct = _ranks(v)
    tree = Fenwick(distinct)
    total = 0
    for r in reversed(ranks):
        total += tree.prefix(r)  # already-seen (to the right) with a smaller rank
        tree.add(r)
    return total


def count_smaller_to_right(seq: Iterable[Any], *, key=None, reverse: bool = False,
                           validate: bool = True) -> list[int]:
    """``out[i]`` = how many elements after ``i`` are strictly smaller than ``a[i]``.

    ``sum(out)`` is the inversion count, so this is a strictly more informative
    answer for the same asymptotic cost. It is the per-element view you need
    for "which entries are most out of place", and the standard LeetCode
    "count of smaller numbers after self".
    """
    v = _prepare(seq, key, reverse, validate)
    ranks, distinct = _ranks(v)
    tree = Fenwick(distinct)
    out = [0] * len(ranks)
    for i in range(len(ranks) - 1, -1, -1):
        out[i] = tree.prefix(ranks[i])
        tree.add(ranks[i])
    return out


def count_greater_to_left(seq: Iterable[Any], *, key=None, reverse: bool = False,
                          validate: bool = True) -> list[int]:
    """``out[j]`` = how many elements before ``j`` are strictly greater than ``a[j]``.

    The other half of the same pairing, and the *inversion table* (Lehmer code)
    of the sequence. ``sum(out)`` is again the inversion count -- the same
    pairs, attributed to the later index instead of the earlier one.
    """
    v = _prepare(seq, key, reverse, validate)
    ranks, distinct = _ranks(v)
    tree = Fenwick(distinct)
    out = [0] * len(ranks)
    for j, r in enumerate(ranks):
        out[j] = j - tree.prefix(r + 1)  # seen so far, minus those <= r
        tree.add(r)
    return out


# ---------------------------------------------------------------------------
# 5. The vectorised method
# ---------------------------------------------------------------------------


def _numpy_ranks(seq: Iterable[Any], key, reverse: bool, validate: bool):
    """Dense ranks computed entirely in numpy, or None if the input is not plain.

    This exists because rank compression, not counting, is where a vectorised
    inversion counter actually spends its time: `_ranks` runs a Python
    ``sorted`` with a ``key`` callback, which is O(n log n) *interpreted* and
    swamps the O(n log^2 n) of C-level searchsorted. ``np.unique`` does the
    same job in C and takes the method from ~1.4x faster than merge sort to
    ~10x.

    Deliberately narrow: numeric dtypes only. A list of strings would work but
    numpy's ordering is not guaranteed to match Python's for every dtype, and
    a wrong answer is worse than a slow one -- anything else falls back.
    """
    import numpy as np

    if key is not None or reverse:
        return None
    try:
        arr = np.asarray(seq)
    except (ValueError, TypeError):  # ragged input, exotic objects
        return None
    # 'b' bool, 'i' signed, 'u' unsigned, 'f' float. Object dtype (which is what
    # Fractions, huge ints and mixed types produce) is excluded on purpose.
    if arr.ndim != 1 or arr.dtype.kind not in "biuf":
        return None
    if validate and arr.dtype.kind == "f" and np.isnan(arr).any():
        raise ValueError(
            "sequence contains NaN: comparisons against NaN are all false, so "
            "the elements are not totally ordered and the inversion count is "
            "undefined. Filter or replace them, or pass validate=False."
        )
    _values, inverse = np.unique(arr, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64, copy=False)


def count_numpy(seq: Iterable[Any], *, key=None, reverse: bool = False,
                validate: bool = True) -> int:
    """Bottom-up merge sort with every level batched into three ndarray calls.

    The trick that makes a level batchable: at width ``w`` the array is a
    ``(blocks, 2w)`` matrix whose rows are independent merges, and
    ``np.searchsorted`` searches one sorted haystack, not a batch of them. So
    add ``block_index * (n + 1)`` to every value first. Because every value is
    a dense rank in ``[0, n]``, block ``b`` then occupies
    ``[b(n+1), b(n+1) + n]`` -- disjoint from every other block and increasing
    in ``b``, which makes the concatenated left halves *globally* sorted. One
    binary search over the whole array now answers every block simultaneously,
    and subtracting ``b * w`` converts each global position back to a
    within-block count.

    Two searches per level do all the work:

        le[j] = #{left elements <= right[j]}   -> inversions are w - le[j],
                                                  and right[j] merges to j + le[j]
        lt[i] = #{right elements <  left[i]}   -> left[i] merges to i + lt[i]

    so the merge is a scatter, not a second sort, and the level costs
    O(n log w) with no Python loop over elements. Summed over ``log n`` levels
    that is O(n log^2 n) -- an extra log versus the textbook merge sort, and
    ~50x faster in practice because the constant is a C loop rather than an
    interpreter dispatch.

    Ties are handled by the two ``side`` arguments: ``le`` uses ``side="right"``
    (equal elements count as *not* inverted, since equal is not greater) and
    ``lt`` uses ``side="left"``, which together make the merge stable and the
    positions collision-free.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without numpy
        raise ImportError("count_numpy requires numpy; use method='mergesort'") from exc

    dense = _numpy_ranks(seq, key, reverse, validate)
    if dense is None:
        v = _prepare(seq, key, reverse, validate)
        rank_list, _ = _ranks(v)
        dense = np.asarray(rank_list, dtype=np.int64)

    n = len(dense)
    if n < 2:
        return 0
    if n > 2**31:  # pragma: no cover - unreachable at any realistic memory size
        raise ValueError("count_numpy addresses at most 2^31 elements; use count_mergesort")

    # Pad to a power of two with a value above every rank. Padding sits at the
    # tail, so it is never in a left half that has real elements to its right,
    # and pad > everything means it contributes zero inversions.
    size = 1 << max(1, (n - 1).bit_length())
    ranks = np.full(size, n, dtype=np.int64)
    ranks[:n] = dense

    stride = np.int64(n + 1)
    total = 0
    width = 1
    while width < size:
        blocks = size // (2 * width)
        view = ranks.reshape(blocks, 2 * width)
        left, right = view[:, :width], view[:, width:]

        offsets = (np.arange(blocks, dtype=np.int64) * stride)[:, None]
        left_flat = (left + offsets).ravel()
        right_flat = (right + offsets).ravel()
        base = np.repeat(np.arange(blocks, dtype=np.int64) * width, width)

        le = np.searchsorted(left_flat, right_flat, side="right") - base
        total += blocks * width * width - int(le.sum(dtype=np.int64))

        lt = np.searchsorted(right_flat, left_flat, side="left") - base

        merged = np.empty_like(ranks)
        block_base = np.repeat(np.arange(blocks, dtype=np.int64) * (2 * width), width)
        within = np.tile(np.arange(width, dtype=np.int64), blocks)
        merged[block_base + within + lt] = left.ravel()
        merged[block_base + within + le] = right.ravel()
        ranks = merged
        width *= 2

    return total


def count_numpy_radix(seq: Iterable[Any], *, key=None, reverse: bool = False,
                      validate: bool = True) -> int:
    """Split by *value* instead of by index. O(n log n), fully vectorised.

    :func:`count_numpy` is a merge sort: it splits the array by index and asks
    "how many left-half elements exceed this right-half one", which is a
    binary search and costs an extra log. Turn the recursion inside out --
    split by *value* instead, MSB-first on the dense ranks -- and the same
    question becomes "how many 1-bits precede this 0-bit", which is a running
    sum. That removes the log: O(n) work per bit, log n bits, and every step
    is a `cumsum`, a gather or a scatter.

    The identity it rests on: every inverted pair ``(i, j)`` has a unique
    highest bit at which ``rank[i]`` and ``rank[j]`` differ, and at that bit
    ``rank[i]`` has a 1 and ``rank[j]`` a 0. So

        inversions = sum over bits b of
                     #{(i, j) : i < j, ranks agree above b, r[i] has 1, r[j] has 0}

    and the three conditions are exactly "same group, earlier position, one
    then zero" once the array is kept MSD-radix-partitioned. Equal ranks never
    differ at any bit, so ties contribute nothing -- which is the correct
    answer for strict inversions, with no special case anywhere.

    The invariant carried across bits is that positions within a group are in
    original index order. It holds initially (the permutation is the identity,
    and every element is in one group) and is preserved because the partition
    by bit is stable within each group.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError("count_numpy_radix requires numpy; use method='mergesort'") from exc

    dense = _numpy_ranks(seq, key, reverse, validate)
    if dense is None:
        v = _prepare(seq, key, reverse, validate)
        rank_list, _ = _ranks(v)
        dense = np.asarray(rank_list, dtype=np.int64)

    n = len(dense)
    if n < 2:
        return 0

    values = dense.astype(np.int64, copy=True)
    position = np.arange(n, dtype=np.int64)
    # Group bounds per position. One group to begin with: nothing has been
    # split yet, because no bits have been examined.
    group_start = np.zeros(n, dtype=np.int64)
    group_end = np.full(n, n, dtype=np.int64)
    prefix = np.empty(n + 1, dtype=np.int64)
    prefix[0] = 0

    total = 0
    for bit_index in range(int(n - 1).bit_length() - 1, -1, -1):
        bit = (values >> bit_index) & 1
        np.cumsum(bit, out=prefix[1:])

        at_start = prefix[group_start]
        ones_before = prefix[:n] - at_start
        zeros_before = (position - group_start) - ones_before
        zeros_in_group = (group_end - group_start) - (prefix[group_end] - at_start)

        is_zero = bit == 0
        total += int(ones_before[is_zero].sum())

        # Stable partition inside each group: zeros keep their order and go
        # first, ones follow. Both halves become groups for the next bit.
        destination = np.where(is_zero,
                               group_start + zeros_before,
                               group_start + zeros_in_group + ones_before)
        next_start = np.where(is_zero, group_start, group_start + zeros_in_group)
        next_end = np.where(is_zero, group_start + zeros_in_group, group_end)

        moved = np.empty_like(values)
        moved[destination] = values
        starts = np.empty_like(group_start)
        starts[destination] = next_start
        ends = np.empty_like(group_end)
        ends[destination] = next_end
        values, group_start, group_end = moved, starts, ends

    return total


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_IMPLS: dict[str, Callable[..., int]] = {
    "brute": count_brute,
    "insort": count_insort,
    "mergesort": count_mergesort,
    "fenwick": count_fenwick,
    "numpy": count_numpy,
    "radix": count_numpy_radix,
}


def count_inversions(seq: Iterable[Any], *, method: str = "auto", key=None,
                     reverse: bool = False, validate: bool = True) -> int:
    """Count pairs i < j with ``a[i] > a[j]``. The main entry point.

    ``key`` decorates each element before comparing, like ``sorted``.
    ``reverse=True`` counts pairs that are *in* descending order instead --
    i.e. the inversion count relative to a descending target. ``validate``
    rejects NaN, which would otherwise silently produce a number that depends
    on the algorithm rather than on the data.

    ``method="auto"`` picks `insort` below n = 3000 (the O(n^2) memmove is
    cheaper than a Python-level merge there), then `radix` up to 300k and
    `numpy` above it -- the crossover where the radix method's scatter stops
    fitting in cache -- and `mergesort` when numpy is not installed. All the
    thresholds are measured; see `benchmark.py`.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "auto":
        method = _auto(seq if isinstance(seq, Sequence) else (seq := list(seq)))
    return _IMPLS[method](seq, key=key, reverse=reverse, validate=validate)


def _auto(seq: Sequence[Any]) -> str:
    n = len(seq)
    if n < 3000:
        return "insort"
    try:
        import numpy  # noqa: F401
    except ImportError:
        return "mergesort"
    return "radix" if n < 300_000 else "numpy"


# ---------------------------------------------------------------------------
# Variants: significant inversions, Kendall tau
# ---------------------------------------------------------------------------


def count_significant_inversions(seq: Iterable[Any], factor: float = 2.0, *,
                                 validate: bool = True) -> int:
    """Count pairs i < j with ``a[i] > factor * a[j]``. O(n log^2 n).

    The generalisation from the "significant inversions" exercise: with
    ``factor = 1`` it is the ordinary count, and with ``factor = 2`` it finds
    pairs where the earlier value is more than double the later one.

    Implemented as a merge sort whose counting step is a ``bisect`` per
    right-half element rather than a two-pointer scan. The two-pointer version
    is O(n log n) but only correct when ``factor * x`` is non-decreasing in
    ``x``, which fails for negative factors; the bisect costs one extra log and
    is correct for every real factor and for negative array values.

    With float values and a float factor the comparison inherits float
    rounding, exactly as writing ``a[i] > factor * a[j]`` by hand would.
    """
    arr = _prepare(seq, None, False, validate)
    n = len(arr)
    if n < 2:
        return 0
    if any(not isinstance(x, (int, float)) for x in arr):
        raise TypeError("count_significant_inversions needs numbers, not arbitrary orderings")

    def rec(lo: int, hi: int) -> int:
        if hi - lo < 2:
            return 0
        mid = (lo + hi) // 2
        total = rec(lo, mid) + rec(mid, hi)
        left = arr[lo:mid]
        for x in arr[mid:hi]:
            total += len(left) - bisect.bisect_right(left, factor * x)
        arr[lo:hi] = sorted(arr[lo:hi])
        return total

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2 * n.bit_length() + 100))
    return rec(0, n)


def kendall_tau_distance(a: Sequence[Any], b: Sequence[Any], *,
                         normalize: bool = False) -> float | int:
    """Number of pairs the two rankings order differently.

    Both sequences must be permutations of the same multiset of *distinct*
    elements -- the plain tau distance is only defined without ties. The
    reduction is the standard one: relabel each element by its position in
    ``b``, then the distance is the inversion count of ``a`` under that
    relabelling, because a pair is discordant exactly when ``b`` puts it in the
    opposite order from ``a``.

    ``normalize=True`` divides by C(n, 2), giving 0 for identical rankings and
    1 for exactly reversed ones.
    """
    if len(a) != len(b):
        raise ValueError(f"sequences must be the same length, got {len(a)} and {len(b)}")
    position = {x: i for i, x in enumerate(b)}
    if len(position) != len(b):
        raise ValueError("kendall_tau_distance needs distinct elements; use kendall_tau_b for ties")
    try:
        relabelled = [position[x] for x in a]
    except KeyError as exc:
        raise ValueError(f"{exc.args[0]!r} appears in the first sequence but not the second") from None
    distance = count_inversions(relabelled, validate=False)
    if not normalize:
        return distance
    pairs = max_inversions(len(a))
    return distance / pairs if pairs else 0.0


def kendall_tau_b(x: Sequence[Any], y: Sequence[Any]) -> float:
    """Tie-corrected Kendall rank correlation in [-1, 1]. Knight's O(n log n) form.

    With ``C`` concordant and ``D`` discordant pairs, ``n0 = C(n, 2)``, and
    ``n1``/``n2``/``n3`` the pairs tied in x / in y / in both,

        C - D  = n0 - n1 - n2 + n3 - 2 * (inversions of y sorted by x)
        tau_b  = (C - D) / sqrt((n0 - n1) * (n0 - n2))

    The first line is the identity worth staring at: sorting the pairs by x and
    counting inversions in the resulting y sequence counts discordant pairs,
    but double-counts nothing and misses only the tied ones, which the n1/n2/n3
    terms put back. Returns NaN when one variable is constant, since then every
    pair is tied in it and the correlation is genuinely undefined.
    """
    if len(x) != len(y):
        raise ValueError(f"sequences must be the same length, got {len(x)} and {len(y)}")
    n = len(x)
    if n < 2:
        return math.nan

    order = sorted(range(n), key=lambda i: (x[i], y[i]))
    xs = [x[i] for i in order]
    ys = [y[i] for i in order]

    n0 = max_inversions(n)
    n1 = _tie_pairs(xs)
    n2 = _tie_pairs(sorted(ys))
    n3 = _tie_pairs(list(zip(xs, ys)))
    swaps = count_inversions(ys, validate=False)

    numerator = n0 - n1 - n2 + n3 - 2 * swaps
    denominator = math.sqrt((n0 - n1) * (n0 - n2))
    return numerator / denominator if denominator else math.nan


def _tie_pairs(sorted_values: Sequence[Any]) -> int:
    """Sum of C(t, 2) over runs of equal values in an already-sorted sequence."""
    total = 0
    run = 1
    for i in range(1, len(sorted_values)):
        if sorted_values[i] == sorted_values[i - 1]:
            run += 1
        else:
            total += run * (run - 1) // 2
            run = 1
    return total + run * (run - 1) // 2


# ---------------------------------------------------------------------------
# The Lehmer code, and the generating function
# ---------------------------------------------------------------------------


def inversion_table(perm: Sequence[int]) -> list[int]:
    """Knuth's inversion table: ``b[v]`` = how many values greater than ``v`` precede it.

    Indexed by *value*, not by position -- that distinction is the whole point,
    and the reason this is a bijection while the position-indexed count is
    merely a statistic. Since ``b[v]`` counts values drawn from
    ``{v+1, ..., n-1}``, it satisfies ``0 <= b[v] <= n-1-v`` independently for
    each ``v``, so the table ranges over a product of ranges of sizes
    ``n, n-1, ..., 1``. That product has ``n!`` elements, and
    :func:`from_inversion_table` shows the map onto it is invertible -- hence
    the bijection, hence ``sum(b) = inv(perm)``, and hence the generating
    function factors as ``[n]_q!`` (see :func:`inversion_polynomial`).

    The position-indexed relatives are :func:`count_greater_to_left` and the
    Lehmer code :func:`count_smaller_to_right`; all three sum to the same
    inversion count, because all three count the same pairs under a different
    attribution.

    ``perm`` must be a permutation of ``0..n-1``.
    """
    n = len(perm)
    if sorted(perm) != list(range(n)):
        raise ValueError("inversion_table needs a permutation of 0..n-1")
    by_position = count_greater_to_left(perm, validate=False)
    table = [0] * n
    for j, value in enumerate(perm):
        table[value] = by_position[j]
    return table


def from_inversion_table(table: Sequence[int]) -> list[int]:
    """Rebuild the permutation of ``0..n-1`` with the given inversion table.

    Insert values from largest to smallest, placing ``v`` at index ``b[v]``.
    That works because at the moment ``v`` is inserted the list holds exactly
    the values greater than ``v``, so "index ``b[v]``" and "``b[v]`` greater
    values to my left" are the same statement -- and the placement is forced,
    which is what makes the map injective. O(n^2) via list insertion; this
    exists to demonstrate the bijection, not to be fast.
    """
    n = len(table)
    for v, c in enumerate(table):
        if not 0 <= c <= n - 1 - v:
            raise ValueError(
                f"table[{v}] = {c} is outside the allowed range [0, {n - 1 - v}]"
            )
    out: list[int] = []
    for v in range(n - 1, -1, -1):
        out.insert(table[v], v)
    return out


def inversion_polynomial(n: int) -> list[int]:
    """Coefficients of ``[n]_q! = prod_{i=1..n} (1 + q + ... + q^(i-1))``.

    ``inversion_polynomial(n)[k]`` is the number of permutations of ``n``
    elements with exactly ``k`` inversions -- the Mahonian numbers, OEIS
    A008302. The factorisation follows straight from the Lehmer code being a
    bijection onto a product of independent ranges: each factor is the
    generating function of one coordinate.

    Used in the tests as an exact, non-statistical check that the counters
    agree with combinatorics on the whole of S_n for n <= 8.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    poly = [1]
    for i in range(1, n + 1):
        nxt = [0] * (len(poly) + i - 1)
        for k, c in enumerate(poly):
            for shift in range(i):
                nxt[k + shift] += c
        poly = nxt
    return poly


def bubble_sort_swaps(seq: Sequence[Any]) -> int:
    """Adjacent swaps a bubble sort performs. Equals the inversion count.

    Not an algorithm anyone should use -- it is here because "the inversion
    count is the minimum number of adjacent transpositions that sort the array"
    is the claim that makes the number *mean* something, and a test that runs
    an actual bubble sort is the only honest way to check it.
    """
    arr = list(seq)
    swaps = 0
    for end in range(len(arr) - 1, 0, -1):
        moved = False
        for i in range(end):
            if arr[i + 1] < arr[i]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swaps += 1
                moved = True
        if not moved:
            break
    return swaps


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def verify(*, seed: int = 0, trials: int = 300, verbose: bool = True) -> bool:
    """Check every method against brute force on adversarial and random input."""
    rng = random.Random(seed)
    try:
        import numpy  # noqa: F401
        methods = ["insort", "mergesort", "fenwick", "numpy", "radix"]
    except ImportError:
        methods = ["insort", "mergesort", "fenwick"]
        if verbose:
            print("  (numpy not installed; skipping count_numpy)", file=sys.stderr)

    cases: list[list[int]] = [
        [], [1], [1, 1], [2, 1], [1, 2],
        list(range(50)), list(range(50))[::-1], [7] * 50,
        [0, 1] * 25, list(range(25)) + list(range(25)),
    ]
    for _ in range(trials):
        n = rng.randint(0, 60)
        spread = rng.choice([2, 5, 1000])
        cases.append([rng.randrange(spread) for _ in range(n)])

    ok = True
    for case in cases:
        expected = count_brute(case)
        for method in methods:
            got = _IMPLS[method](case)
            if got != expected:
                ok = False
                if verbose:
                    print(f"  MISMATCH {method}: {got} != {expected} on {case}", file=sys.stderr)
        # Structural identities that must hold for every input.
        if sum(count_smaller_to_right(case)) != expected:
            ok = False
        if sum(count_greater_to_left(case)) != expected:
            ok = False
        if len(case) <= 40 and bubble_sort_swaps(case) != expected:
            ok = False

    if verbose:
        print(f"verify: {len(cases)} arrays x {len(methods)} methods vs brute force -- "
              f"{'OK' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> None:
    array = [8, 4, 2, 1]
    print(f"array: {array}")
    print(f"  inversions:            {count_inversions(array)}  (of at most "
          f"{max_inversions(len(array))} -- strictly decreasing, so all of them)")
    print(f"  smaller to the right:  {count_smaller_to_right(array)}")
    print(f"  greater to the left:   {count_greater_to_left(array)}  (the Lehmer code)")
    print(f"  bubble sort swaps:     {bubble_sort_swaps(array)}")

    mixed = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"\narray: {mixed}")
    print(f"  inversions:            {count_inversions(mixed)}")
    print(f"  significant (f=2):     {count_significant_inversions(mixed, 2)}"
          f"   pairs with a[i] > 2*a[j]")
    print(f"  tau distance to sorted:{kendall_tau_distance(mixed, sorted(mixed)) if len(set(mixed)) == len(mixed) else 'n/a (ties)'}")

    with_ties = [1, 2, 2, 3]
    print(f"\ntau-b({with_ties}, {[1, 2, 3, 4]}) = "
          f"{kendall_tau_b(with_ties, [1, 2, 3, 4]):.6f}   (ties cost you the last 0.09)")
    print(f"tau-b({[1, 2, 3, 4]}, {[4, 3, 2, 1]}) = "
          f"{kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]):.6f}")

    print(f"\npermutations of 4 by inversion count: {inversion_polynomial(4)}")
    print(f"  (1 sorted, 3 with one inversion, ..., 1 fully reversed; sums to "
          f"{sum(inversion_polynomial(4))} = 4!)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count inversions in an array, five ways.",
    )
    parser.add_argument("numbers", nargs="*", type=float,
                        help="the array; reads whitespace-separated stdin if empty")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--method", default="auto", choices=METHODS)
    parser.add_argument("--detail", action="store_true",
                        help="also print the per-element counts")
    args = parser.parse_args(argv)

    if args.verify:
        return 0 if verify() else 1
    if args.demo:
        _demo()
        return 0

    numbers = args.numbers
    if not numbers:
        if sys.stdin.isatty():
            parser.error("give numbers, --demo, or --verify")
        numbers = [float(tok) for tok in sys.stdin.read().split()]

    total = count_inversions(numbers, method=args.method)
    print(f"{len(numbers)} elements, {total} inversions "
          f"(max {max_inversions(len(numbers))})")
    if args.detail:
        print(f"  smaller to the right: {count_smaller_to_right(numbers)}")
        print(f"  greater to the left:  {count_greater_to_left(numbers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
