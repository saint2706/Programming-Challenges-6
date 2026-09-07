"""The skyline problem, four ways: brute force, two sweeps, and divide and conquer.

A building is a rectangle ``(left, right, height)``: base on the x-axis from
``left`` to ``right``, top at ``height``. Given many such rectangles, possibly
overlapping, trace the outline of their silhouette -- the classic LeetCode 218
problem. The output is the minimal list of "key points" ``(x, height)``: the
x-coordinate where the skyline's height changes, and what it changes to.
Reading consecutive key points as (x, height) -> flat segment at that height
until the next x reconstructs the whole outline.

    brute_force        O(n * range) / O(n^2)   the definition; the oracle
    sweep_line         O(n log n)               max-heap, lazy deletion
    sweep_line_bst      O(n log n)               SortedList, true deletion
    divide_and_conquer O(n log n)               merge-sort-style skyline merge

All four must produce *exactly* the same list, not merely an equivalent step
function -- getting that exact match right is most of the difficulty here,
and it is where a naive implementation usually leaks redundant points.

**The equal-height-merge trap.** Two adjacent buildings of the same height,
sharing an edge, must merge into one flat segment with no key point at the
shared edge -- e.g. buildings (0, 5, 10) and (5, 10, 10) must skyline to
``[(0, 10), (10, 0)]``, not ``[(0, 10), (5, 10), (10, 0)]``. The second form is
not wrong in the sense of describing a different silhouette (10 is still 10 at
every x), but it is not *minimal*, and "minimal key points" is the actual
spec of the problem. Every method below avoids it the same way: compare the
*current* height against the *last emitted* height, and only emit when they
differ -- never emit just because "an event happened here". A naive
sweep that emits whenever the active building changes (rather than when the
current max height changes) fails exactly this case, because the building
changes at x=5 while the height does not.

**Other edge cases, and how each method handles them:**

* Buildings sharing the exact same left/right edge -- both a start and an end
  event land on the same x. ``sweep_line`` pushes every building starting at x
  before checking which have expired, so a same-x handoff never dips to a
  wrong intermediate height. ``divide_and_conquer``'s merge treats "a on the
  left of b" (``a[i][0] < b[j][0]``) as strict, so equal x's are consumed from
  both sides in the same step.
* Zero-width or malformed buildings (``left >= right``) contribute no area and
  are silently dropped before any method runs -- a footprint with no width
  casts no silhouette, and "malformed" degrades to "zero-width" rather than
  crashing.
* A building fully contained in a taller one is present in every method's
  bookkeeping (it is pushed onto the heap, it is one of the ``n`` leaves of
  the recursion) but never emits a key point, because it never becomes the
  current maximum.
* Dense ties -- many buildings starting or ending at the same x -- are handled
  by processing *all* of them before asking "what's the current height now?",
  in both methods; asking after each one individually is what causes spurious
  intermediate points.
* The skyline always ends at height 0 once the last building's right edge
  passes, which fits the same "emit on change" rule with no special case: the
  height genuinely changes to 0, so a point is emitted for it.

    uv run python skyline.py --demo
    uv run python skyline.py --verify
"""

from __future__ import annotations

import argparse
import heapq
import random
import sys
from collections.abc import Sequence

__all__ = [
    "Building",
    "Point",
    "brute_force",
    "divide_and_conquer",
    "main",
    "render_ascii",
    "skyline",
    "sweep_line",
    "sweep_line_bst",
    "verify",
]

Number = int | float
Building = tuple[Number, Number, Number]
Point = tuple[Number, Number]

METHODS = ("auto", "brute", "sweep", "sweep_bst", "dc")


def _valid(buildings: Sequence[Building]) -> list[Building]:
    """Drop zero-width or malformed (``left >= right``) buildings.

    A rectangle with no width casts no shadow, so this is not a special case
    for the algorithms below to worry about -- it is a filter applied once,
    up front, and every method sees the same reduced input.
    """
    for _, _, h in buildings:
        if h < 0:
            raise ValueError(f"height must be non-negative, got {h}")
    return [b for b in buildings if b[0] < b[1]]


# ---------------------------------------------------------------------------
# 1. Brute force -- the definition, and the oracle
# ---------------------------------------------------------------------------


def brute_force(buildings: Sequence[Building]) -> list[Point]:
    """Critical x-coordinates, then max height covering each gap. O(n^2).

    Every ``left`` and ``right`` is a candidate x where the skyline *could*
    change; between two consecutive critical x's, no building starts or ends,
    so the covering set -- and hence the max height -- is constant. That
    reduces "trace a continuous outline" to "compute one number per gap",
    which is what makes this small enough to trust as an oracle: no heap, no
    recursion, just "for each interval, scan every building".
    """
    bs = _valid(buildings)
    if not bs:
        return []
    xs = sorted({x for l, r, _ in bs for x in (l, r)})

    points: list[Point] = []
    prev_height: Number = 0
    for i in range(len(xs) - 1):
        lo, hi = xs[i], xs[i + 1]
        height = max((h for l, r, h in bs if l <= lo and r >= hi), default=0)
        if height != prev_height:
            points.append((lo, height))
            prev_height = height
    if prev_height != 0:
        points.append((xs[-1], 0))
    return points


# ---------------------------------------------------------------------------
# 2. Sweep line -- O(n log n), max-heap with lazy deletion
# ---------------------------------------------------------------------------


def sweep_line(buildings: Sequence[Building]) -> list[Point]:
    """Sweep left to right; a max-heap tracks the tallest active building.

    ``heapq`` has no decrease-key or arbitrary removal, so a building that has
    ended is not pulled out of the heap -- it is left there and skipped
    ("lazy deletion") the next time it would matter, which is only when it
    surfaces at the top. That keeps every heap operation O(log n) instead of
    O(n): eager removal would need a linear scan (or a second index structure)
    to find the entry.

    At each distinct x, *all* buildings starting there are pushed before any
    expired ones are popped, and the current height is read only after both
    steps -- so a building ending exactly where another begins never produces
    a spurious dip to a lower intermediate height, and dense ties at one x
    are resolved as a single point rather than one per event.
    """
    bs = sorted(_valid(buildings), key=lambda b: b[0])
    if not bs:
        return []
    xs = sorted({x for l, r, _ in bs for x in (l, r)})

    heap: list[tuple[Number, Number]] = []  # (-height, right), lazily expired
    points: list[Point] = []
    prev_height: Number = 0
    i, n = 0, len(bs)
    for x in xs:
        while i < n and bs[i][0] == x:
            _, r, h = bs[i]
            heapq.heappush(heap, (-h, r))
            i += 1
        while heap and heap[0][1] <= x:
            heapq.heappop(heap)
        height = -heap[0][0] if heap else 0
        if height != prev_height:
            points.append((x, height))
            prev_height = height
    return points


# ---------------------------------------------------------------------------
# 2b. Sweep line -- O(n log n), balanced-BST (SortedList) variant
# ---------------------------------------------------------------------------


def sweep_line_bst(buildings: Sequence[Building]) -> list[Point]:
    """Same sweep as :func:`sweep_line`, but a sorted multiset instead of a heap.

    Structurally the same algorithm -- sweep left to right, track the tallest
    *active* building -- but the data structure backing "tallest active
    building" is different. ``sweep_line``'s heap never removes an ended
    building eagerly (``heapq`` has no arbitrary-element removal), so stale
    entries pile up and are discarded lazily, only when they would otherwise
    surface at the top. This version uses ``sortedcontainers.SortedList``, a
    pure-Python balanced-BST-like sorted sequence: on a building's end event
    its height is removed with true ``O(log n)`` deletion, immediately, so
    the structure never carries garbage at all -- ``len(active)`` is always
    exactly the count of buildings genuinely active at the current x, not
    "active or merely not-yet-popped".

    That is the entire trade-off, and it is real, not hypothetical: a heap's
    ``push``/``pop`` are simple array operations with a tiny constant factor,
    while a balanced BST's insert/delete do more bookkeeping (rebalancing,
    or here, a shifted-list-of-sublists insert/delete) for the same
    asymptotic cost. Whether true deletion's saved wasted-pop work outweighs
    the BST's larger constant is an empirical question, not a proof -- see
    the README's benchmark section for what actually gets measured across
    building density, the same way ``sweep_line`` vs ``divide_and_conquer``
    turned out to have a real, density-dependent crossover rather than a
    universal winner.

    Requires the third-party ``sortedcontainers`` package (imported lazily,
    here, so every other function and the CLI's default paths stay
    dependency-free): ``uv run --with sortedcontainers python skyline.py ...``.
    """
    try:
        from sortedcontainers import SortedList
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "sweep_line_bst requires the 'sortedcontainers' package: "
            "uv run --with sortedcontainers python skyline.py ..."
        ) from exc

    bs = _valid(buildings)
    if not bs:
        return []
    starts = sorted(bs, key=lambda b: b[0])
    ends = sorted(bs, key=lambda b: b[1])
    xs = sorted({x for l, r, _ in bs for x in (l, r)})

    active: SortedList = SortedList()  # active heights, ascending; max is active[-1]
    points: list[Point] = []
    prev_height: Number = 0
    i = j = 0
    n = len(bs)
    for x in xs:
        # As in sweep_line: every building starting at x is added before any
        # ending at x is removed, so a same-x handoff never dips through a
        # spurious intermediate height.
        while i < n and starts[i][0] == x:
            active.add(starts[i][2])
            i += 1
        while j < n and ends[j][1] == x:
            active.remove(ends[j][2])  # true O(log n) removal, not lazy
            j += 1
        height = active[-1] if active else 0
        if height != prev_height:
            points.append((x, height))
            prev_height = height
    return points


# ---------------------------------------------------------------------------
# 3. Divide and conquer -- O(n log n), merge two skylines like merge sort
# ---------------------------------------------------------------------------


def _merge_skylines(a: list[Point], b: list[Point]) -> list[Point]:
    """Merge two already-computed skylines the way merge sort merges runs.

    Walk both outlines left to right in lockstep. Each side contributes a
    "current height" -- the height of the last key point consumed from that
    side, i.e. what that side's skyline is doing *right now* at the x under
    consideration. At every x visited (from either side, or both at once when
    they coincide), the merged height is ``max(h1, h2)``, and a point is
    emitted only when that merged height differs from the last one emitted.

    That last rule is the whole trick and the fix for the equal-height-merge
    trap: it would be easy, and wrong, to emit a point whenever *either* side
    has an event, which over-emits every time one skyline changes but the
    combined max does not (one side stepping down from 10 to 7 while the
    other is holding steady at 12, say). Comparing against the *merged*
    previous height instead of reacting to raw events is what keeps the
    result minimal.
    """
    i = j = 0
    h1: Number = 0
    h2: Number = 0
    merged: list[Point] = []

    def emit(x: Number, height: Number) -> None:
        if not merged or merged[-1][1] != height:
            merged.append((x, height))

    while i < len(a) and j < len(b):
        if a[i][0] < b[j][0]:
            x, h1 = a[i]
            i += 1
        elif b[j][0] < a[i][0]:
            x, h2 = b[j]
            j += 1
        else:
            x, h1 = a[i]
            _, h2 = b[j]
            i += 1
            j += 1
        emit(x, max(h1, h2))

    while i < len(a):
        x, h1 = a[i]
        emit(x, max(h1, h2))
        i += 1
    while j < len(b):
        x, h2 = b[j]
        emit(x, max(h1, h2))
        j += 1
    return merged


def divide_and_conquer(buildings: Sequence[Building]) -> list[Point]:
    """Split the buildings in half, recurse, merge. O(n log n) like merge sort.

    The split is by index, not by position -- the two halves can (and, for a
    random split, usually do) overlap in x-range. That is fine: each half's
    ``divide_and_conquer`` call still returns a *correct* skyline for just
    that half's buildings, because the recursion bottoms out at single
    buildings, and :func:`_merge_skylines` makes no assumption that its two
    inputs occupy disjoint ranges. Recursion depth is log2(n), so this never
    approaches Python's recursion limit at any size worth running.
    """
    bs = _valid(buildings)

    def rec(items: list[Building]) -> list[Point]:
        if not items:
            return []
        if len(items) == 1:
            l, r, h = items[0]
            # A height-0 building contributes nothing -- returning [(l, 0), (r, 0)]
            # here would be two key points at the same (background) height,
            # which fails the "no redundant equal-height point" invariant every
            # other method upholds for the same input.
            return [(l, h), (r, 0)] if h else []
        mid = len(items) // 2
        return _merge_skylines(rec(items[:mid]), rec(items[mid:]))

    return rec(bs)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_IMPLS = {
    "brute": brute_force,
    "sweep": sweep_line,
    "sweep_bst": sweep_line_bst,
    "dc": divide_and_conquer,
}


def skyline(buildings: Sequence[Building], *, method: str = "auto") -> list[Point]:
    """Compute the skyline. ``method="auto"`` picks ``sweep`` for large input.

    Brute force is O(n^2)-ish (a linear scan per gap, with up to 2n gaps) and
    only worth it as an oracle, so ``auto`` never chooses it; among the three
    O(n log n) methods it defaults to ``sweep`` (the heap sweep), which the
    benchmark in this directory shows is the steadier of the bunch across
    densities and needs no third-party dependency. Pass ``method="dc"`` or
    ``method="sweep_bst"`` explicitly to compare -- the latter needs
    ``sortedcontainers`` installed.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "auto":
        method = "sweep"
    return _IMPLS[method](buildings)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def _random_buildings(
    rng: random.Random, n: int, *, span: int, max_h: int
) -> list[Building]:
    out = []
    for _ in range(n):
        a = rng.randint(0, span)
        b = rng.randint(0, span)
        l, r = (a, b) if a <= b else (b, a)
        out.append((l, r, rng.randint(0, max_h)))
    return out


def verify(*, seed: int = 0, trials: int = 400, verbose: bool = True) -> bool:
    """Check ``sweep_line``, ``sweep_line_bst``, and ``divide_and_conquer`` vs ``brute_force``.

    Random trials vary count, overlap density (via how wide ``span`` is
    relative to ``n`` -- narrow spans force heavy overlap), and height
    distribution (including many zero heights and repeated heights, which is
    exactly what stresses the equal-height-merge rule). A handful of
    hand-picked adversarial cases are checked first because random trials
    rarely happen to hit them: shared edges, full containment, and a
    zero-width building mixed in with real ones.

    ``sweep_line_bst`` needs the third-party ``sortedcontainers`` package; if
    it is not installed, this skips just that method (with a note, when
    ``verbose``) rather than failing the whole check -- the other two are
    standard-library-only and always run.
    """
    names = ["sweep", "dc"]
    try:
        import sortedcontainers  # noqa: F401

        names.append("sweep_bst")
    except ImportError:
        if verbose:
            print(
                "  (sortedcontainers not installed -- skipping sweep_bst in verify)",
                file=sys.stderr,
            )

    rng = random.Random(seed)
    cases: list[list[Building]] = [
        [],
        [(0, 5, 10)],
        [
            (2, 9, 10),
            (3, 7, 15),
            (5, 12, 12),
            (15, 20, 10),
            (19, 24, 8),
        ],  # LeetCode 218
        [(0, 2, 3), (2, 5, 3)],  # equal-height merge across a shared edge
        [(0, 5, 10), (1, 4, 5)],  # fully contained
        [(0, 5, 10), (0, 5, 10)],  # identical footprints
        [(0, 5, 5), (5, 5, 999), (5, 10, 5)],  # zero-width building sandwiched in
        [(0, 3, 0), (3, 6, 0)],  # all heights zero
        [(0, 10, 5), (0, 10, 5), (0, 10, 5)],  # dense ties, identical everything
        [(-5, 0, 4), (-3, 2, 6)],  # negative coordinates
    ]
    for _ in range(trials):
        n = rng.randint(0, 40)
        span = rng.choice([5, 15, 60])  # small span -> dense overlap; large -> sparse
        max_h = rng.choice([1, 3, 20])  # small max_h -> lots of tied heights
        cases.append(_random_buildings(rng, n, span=span, max_h=max_h))

    ok = True
    for case in cases:
        expected = brute_force(case)
        for name in names:
            got = _IMPLS[name](case)
            if got != expected:
                ok = False
                if verbose:
                    print(
                        f"  MISMATCH {name}: {got} != {expected} on {case}",
                        file=sys.stderr,
                    )
    if verbose:
        print(
            f"verify: {len(cases)} building sets x {len(names)} methods vs brute_force -- "
            f"{'OK' if ok else 'FAILED'}"
        )
    return ok


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------


def render_ascii(buildings: Sequence[Building]) -> list[str]:
    """Draw the buildings as blocks and the traced silhouette, on one grid.

    The step function defined by the key points *is* the silhouette, so
    filling each integer column up to its skyline height already draws the
    outline -- there is no separate "outline pass" needed once the key points
    are correct.
    """
    bs = _valid(buildings)
    if not bs:
        return ["(no buildings)"]
    points = sweep_line(bs)
    min_x = int(min(b[0] for b in bs))
    max_x = int(max(b[1] for b in bs))
    max_h = int(max(h for _, h in points)) if points else 0

    def height_at(x: Number) -> Number:
        h: Number = 0
        for kx, kh in points:
            if kx <= x:
                h = kh
            else:
                break
        return h

    xs = list(range(min_x, max_x))
    heights = [height_at(x) for x in xs]

    lines = []
    for level in range(max_h, 0, -1):
        row = "".join("#" if h >= level else " " for h in heights)
        lines.append(f"{level:>4} |{row}")
    lines.append(f"{'':>4} +" + "-" * len(xs))
    axis = "".join(str(x)[-1] for x in xs)
    lines.append(f"{'':>5}{axis}")
    lines.append("")
    lines.append("key points: " + ", ".join(f"({x}, {h})" for x, h in points))
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_building(spec: str) -> Building:
    parts = spec.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'left,right,height', got {spec!r}")
    l, r, h = (float(p) for p in parts)
    l = int(l) if l.is_integer() else l
    r = int(r) if r.is_integer() else r
    h = int(h) if h.is_integer() else h
    return (l, r, h)


def _demo() -> None:
    buildings: list[Building] = [
        (2, 9, 10),
        (3, 7, 15),
        (5, 12, 12),
        (15, 20, 10),
        (19, 24, 8),
    ]
    print(f"buildings: {buildings}")
    print()
    for line in render_ascii(buildings):
        print(line)
    print()
    for name in ("brute", "sweep", "sweep_bst", "dc"):
        try:
            print(f"{name:>9}: {_IMPLS[name](buildings)}")
        except ImportError as exc:
            print(f"{name:>9}: skipped -- {exc}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute the skyline silhouette of a set of buildings."
    )
    parser.add_argument(
        "buildings",
        nargs="*",
        type=_parse_building,
        help="each as 'left,right,height'; reads whitespace-separated stdin triples if empty",
    )
    parser.add_argument(
        "--demo", action="store_true", help="ASCII render of a worked example"
    )
    parser.add_argument(
        "--verify", action="store_true", help="cross-check every method"
    )
    parser.add_argument(
        "--trials", type=int, default=400, help="random trials for --verify"
    )
    parser.add_argument("--method", default="auto", choices=METHODS)
    args = parser.parse_args(argv)

    if args.demo:
        _demo()
        return 0
    if args.verify:
        return 0 if verify(trials=args.trials) else 1

    buildings = args.buildings
    if not buildings:
        if sys.stdin.isatty():
            parser.error("give buildings as 'L,R,H', or use --demo / --verify")
        tokens = sys.stdin.read().split()
        buildings = [
            _parse_building(",".join(tokens[i : i + 3]))
            for i in range(0, len(tokens), 3)
        ]

    points = skyline(buildings, method=args.method)
    print(f"{len(buildings)} buildings -> {len(points)} key points")
    print(points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
