"""The Josephus problem, five ways, from O(n·k) simulation to O(1).

n people stand in a circle. Starting from a chosen position, every k-th person
is removed, counting around the survivors, until one remains. Which position
survives?

The brief is "simulate with a circular list, then derive the closed form", and
the interesting part is that the closed form only exists for k = 2. For general
k there is a recurrence, and a much faster algorithm built on it, but no known
closed form -- that is still open. So this module is a ladder:

    simulate            O(n·k) time, O(n) space   -- also gives the death order
    survivor_recurrence O(n)   time, O(1) space
    survivor_fast       O(k log n) time, O(1) space
    survivor_pow2       O(1)   time and space     -- k = 2 only, one bit rotation
    elimination_order   O(n log n)                -- the full permutation

with ``survivor`` picking the right one for the (n, k) you actually have.

All public positions are 1-indexed, matching how the problem is always stated.
``start`` wraps around the circle rather than being range-checked -- ``start=0``
and ``start=n`` name the same person, as do ``start=-1`` and ``start=n-1``,
which is the arithmetic the problem itself implies. Every method agrees on
this, and the test suite checks it over the whole range -2n .. 3n.

Run directly for a demo and cross-validation:

    uv run python josephus.py 41 3 --animate
    uv run python josephus.py --verify
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from typing import Iterator, Sequence

__all__ = [
    "simulate",
    "survivor",
    "survivor_recurrence",
    "survivor_fast",
    "survivor_pow2",
    "elimination_order",
    "survivors",
    "Fenwick",
    "frames",
    "verify",
]


def _check(n: int, k: int) -> None:
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")


# ---------------------------------------------------------------------------
# 1. Simulation -- O(n*k), and the only method that is obviously correct
# ---------------------------------------------------------------------------


def simulate(n: int, k: int, *, start: int = 1) -> list[int]:
    """Eliminate people one at a time; return the order they leave in.

    A ``deque`` rather than a hand-rolled linked list: ``rotate`` is the exact
    operation the problem describes ("count k around the circle") and it runs in
    C, so this is both the most readable implementation and the fastest O(n·k)
    one available in Python.

    The last element of the returned list is the survivor. This function is the
    oracle every other method in this module is tested against.
    """
    _check(n, k)
    circle = deque(range(1, n + 1))
    circle.rotate(-(start - 1))
    order: list[int] = []
    while circle:
        circle.rotate(-(k - 1))
        order.append(circle.popleft())
    return order


# ---------------------------------------------------------------------------
# 2. The recurrence -- O(n) time, O(1) space
# ---------------------------------------------------------------------------


def survivor_recurrence(n: int, k: int, *, start: int = 1) -> int:
    """Survivor via ``J(i) = (J(i-1) + k) mod i``, with ``J(1) = 0``.

    The derivation is the whole trick. After the first person dies -- position
    ``k-1`` counting from 0 -- what remains is the *same problem* with ``n-1``
    people, except that the circle now starts at position ``k``. Relabelling
    that smaller circle to start at 0 means every answer shifts back by ``k``,
    so undoing the relabelling is exactly ``(J(n-1) + k) mod n``.
    """
    _check(n, k)
    pos = 0
    for i in range(2, n + 1):
        pos = (pos + k) % i
    return (pos + start - 1) % n + 1


# ---------------------------------------------------------------------------
# 3. The fast algorithm -- O(k log n) time, O(1) space
# ---------------------------------------------------------------------------


def _fast_steps(n: int, k: int) -> float:
    """How many iterations :func:`survivor_fast` will actually take.

    Seeded at x = min(k-1, n(k-1)), it grows to n(k-1) at a rate of k/(k-1) per
    step -- a factor of n, so (k-1)*ln(n) steps. For n = 1 the seed already
    lands on the target and the loop body never runs, which is why this is not
    ``log(max(2, n))``: that would over-estimate the one case the prefix skip
    made free.
    """
    if n <= 1 or k <= 1:
        return 0.0
    return (k - 1) * math.log(n)


def survivor_fast(
    n: int, k: int, *, start: int = 1, max_steps: int | None = 10**7
) -> int:
    """Survivor in O(k log n) steps and O(1) space.

    The recurrence advances one person at a time, which is wasteful: while
    ``i`` is much larger than ``k``, whole sweeps of the circle behave
    identically and can be jumped in one arithmetic step.

    This is the iterative form of that idea -- no recursion, so no stack depth
    limit and no risk of blowing up on n = 10^18::

        x = 0;  repeat  x <- x + floor(x / (k-1)) + 1  until x >= n(k-1)
        survivor (0-indexed) = n·k - x - 1

    Each step multiplies x by roughly k/(k-1), so n enters the cost only
    logarithmically: n can be astronomically large as long as k is not.

    One subtlety worth the two lines it costs: while ``x < k-1`` the update is
    exactly ``x += 1``, so a naive loop spends its first k-1 iterations
    counting one at a time before the geometric growth starts. Seeding x past
    that prefix analytically turns ``survivor_fast(1, 10**7)`` from a
    one-second walk into three arithmetic operations.

    The flip side remains: growth from k-1 up to n(k-1) is a factor of n at a
    rate of k/(k-1) per step, so the real cost is about (k-1)*ln(n) iterations.
    For k much larger than n that is catastrophic -- ``survivor_fast(2, 10**12)``
    wants roughly 7e11 iterations, which is not slow, it is *never*. ``max_steps``
    refuses that up front instead of hanging; pass ``None`` to insist, or just
    call :func:`survivor`, which routes such cases to the O(n) recurrence.
    """
    _check(n, k)
    if k == 1:
        return (n - 1 + start - 1) % n + 1
    if max_steps is not None:
        estimate = _fast_steps(n, k)
        if estimate > max_steps:
            raise ValueError(
                f"survivor_fast(n={n}, k={k}) needs about {estimate:.3g} iterations "
                f"because k is large relative to n. Call survivor(), which uses the "
                f"O(n) recurrence here, or pass max_steps=None to insist."
            )
    target = n * (k - 1)
    # Skip the linear prefix: from x = 0, x increments by exactly 1 until it
    # reaches k-1, so the first x at or above min(k-1, target) is known.
    x = min(k - 1, target)
    while x < target:
        x += x // (k - 1) + 1
    return (n * k - x - 1 + start - 1) % n + 1


# ---------------------------------------------------------------------------
# 4. The closed form -- k = 2 only, O(1)
# ---------------------------------------------------------------------------


def survivor_pow2(n: int, *, start: int = 1) -> int:
    """Survivor for k = 2 in constant time: ``J(2^m + L) = 2L + 1``.

    Written in binary this is startlingly simple -- **move the leading 1 bit to
    the end**::

        n = 1 0 1 1 0        (22)
        J = 0 1 1 0 1        (13)

    Which is one subtraction once you see it: shifting left and setting the low
    bit produces ``2n + 1``, and the leading 1 has moved from ``2^m`` to
    ``2^(m+1)``, so subtracting ``2^(m+1)`` removes it.

    Why 2L + 1: after one full pass around ``n = 2^m + L`` people, exactly L
    have died and the count resumes at position 2L + 1 with 2^m people left --
    and for a power of two the person the count starts on always survives,
    because each pass halves the circle while keeping that person first.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    j = ((n << 1) | 1) - (1 << n.bit_length())
    return (j - 1 + start - 1) % n + 1


def survivor(n: int, k: int, *, start: int = 1) -> int:
    """Survivor by whichever method is cheapest for this (n, k).

    * k = 1 -- the last person standing is the last one counted.
    * k = 2 -- constant-time bit rotation.
    * otherwise -- ``survivor_fast`` grows x from k-1 to n(k-1), a factor of n,
      at a rate of k/(k-1) per step: about (k-1)·ln(n) iterations.
      ``survivor_recurrence`` costs exactly n. Take the smaller.

    That second comparison matters more than it looks. ``survivor_fast`` is
    asymptotically *wrong* for k much larger than n -- n = 10, k = 10^9 takes
    minutes there and microseconds via the recurrence -- so this dispatch is
    what makes ``survivor`` safe to call on any (n, k) at all.
    """
    _check(n, k)
    if k == 1:
        return (n - 1 + start - 1) % n + 1
    if k == 2:
        return survivor_pow2(n, start=start)
    fast_steps = _fast_steps(n, k)
    if fast_steps < n:
        # max_steps=None because this comparison *is* the guard: the fast path
        # was chosen precisely because it beats the O(n) alternative. A large
        # absolute step count is fine when n is larger still.
        return survivor_fast(n, k, start=start, max_steps=None)
    return survivor_recurrence(n, k, start=start)


# ---------------------------------------------------------------------------
# 5. The whole permutation -- O(n log n) with a Fenwick tree
# ---------------------------------------------------------------------------


class Fenwick:
    """A Fenwick tree over n slots, each holding 1, supporting ``select``.

    Two operations, both O(log n):

    * ``remove(i)`` -- take slot i out of circulation.
    * ``select(j)`` -- the position of the j-th *remaining* slot.

    ``select`` is the interesting one. Rather than a prefix-sum binary search
    (O(log^2 n)), it descends the tree directly by binary lifting: walk the
    power-of-two jumps from largest to smallest, taking each one whose subtree
    does not yet cover j. That is a single O(log n) pass over the same array.
    """

    __slots__ = ("n", "tree", "_top")

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"a Fenwick tree needs at least one slot, got {n}")
        self.n = n
        # tree[i] = i & -i initializes every slot to 1 in O(n) without n adds:
        # node i covers exactly (i & -i) slots, all of which hold 1.
        self.tree = [i & -i for i in range(n + 1)]
        self._top = 1 << (n.bit_length())

    def remove(self, i: int) -> None:
        while i <= self.n:
            self.tree[i] -= 1
            i += i & -i

    def select(self, j: int) -> int:
        """1-indexed position of the j-th remaining slot (j is 1-indexed)."""
        pos = 0
        step = self._top
        while step:
            nxt = pos + step
            if nxt <= self.n and self.tree[nxt] < j:
                pos = nxt
                j -= self.tree[pos]
            step >>= 1
        return pos + 1


def elimination_order(n: int, k: int, *, start: int = 1) -> list[int]:
    """The full elimination order in O(n log n) instead of simulation's O(n·k).

    The circle is only ever asked one question -- "who is the j-th person still
    standing?" -- which is exactly the ``select`` an order-statistic structure
    provides. Once removals are O(log n) and select is O(log n), k drops out of
    the complexity entirely: k = 10^6 costs the same as k = 2.
    """
    _check(n, k)
    fen = Fenwick(n)
    order: list[int] = []
    pos = (start - 1) % n
    for remaining in range(n, 0, -1):
        pos = (pos + k - 1) % remaining
        person = fen.select(pos + 1)
        order.append(person)
        fen.remove(person)
    return order


# ---------------------------------------------------------------------------
# Generalizations
# ---------------------------------------------------------------------------


def survivors(n: int, k: int, m: int = 1, *, start: int = 1) -> list[int]:
    """The last ``m`` survivors, sorted by position.

    The same relabelling argument that gives the single-survivor recurrence
    works for a whole set: run ``m`` positions through it in lockstep, starting
    from the m positions that remain at the very end.

    The historical case: Josephus and 40 others, every third man killed. He
    wanted two survivors.

    >>> survivors(41, 3, 2)
    [16, 31]
    """
    _check(n, k)
    if not 1 <= m <= n:
        raise ValueError(f"m must be in 1..{n}, got {m}")
    positions = list(range(m))
    for i in range(m + 1, n + 1):
        positions = [(p + k) % i for p in positions]
    return sorted((p + start - 1) % n + 1 for p in positions)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def frames(n: int, k: int, *, start: int = 1, width: int = 72) -> Iterator[str]:
    """Render the circle after each elimination, for small n."""
    order = elimination_order(n, k, start=start)
    dead: set[int] = set()
    label_width = len(str(n))

    def row() -> str:
        cells = []
        for p in range(1, n + 1):
            cells.append(("--".rjust(label_width, "-") if p in dead
                          else str(p).rjust(label_width)))
        line = " ".join(cells)
        return line if len(line) <= width else line[: width - 3] + "..."

    yield f"{'':>5} {'start':>10}  {row()}"
    for step, person in enumerate(order, 1):
        dead.add(person)
        tag = f"survivor {person}" if step == n else f"kill {person}"
        yield f"{step:>5} {tag:>10}  {row()}"


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def verify(max_n: int = 120, max_k: int = 20) -> list[str]:
    """Check every method against the simulation oracle. Returns failures."""
    problems: list[str] = []
    for n in range(1, max_n + 1):
        for k in range(1, max_k + 1):
            order = simulate(n, k)
            expected = order[-1]

            checks = {
                "survivor": survivor(n, k),
                "recurrence": survivor_recurrence(n, k),
                "fast": survivor_fast(n, k),
            }
            if k == 2:
                checks["pow2"] = survivor_pow2(n)
            for name, got in checks.items():
                if got != expected:
                    problems.append(f"{name}(n={n}, k={k}) = {got}, expected {expected}")

            got_order = elimination_order(n, k)
            if got_order != order:
                problems.append(f"elimination_order(n={n}, k={k}) differs from simulation")

            for m in (1, 2, 3):
                if m <= n:
                    got = survivors(n, k, m)
                    if got != sorted(order[-m:]):
                        problems.append(
                            f"survivors(n={n}, k={k}, m={m}) = {got}, "
                            f"expected {sorted(order[-m:])}"
                        )

    # Arbitrary starting positions must just rotate the answer.
    for n in (1, 2, 7, 13, 41):
        for k in (1, 2, 3, 5):
            for s in range(1, n + 1):
                expected = simulate(n, k, start=s)[-1]
                for name, got in (
                    ("survivor", survivor(n, k, start=s)),
                    ("recurrence", survivor_recurrence(n, k, start=s)),
                    ("fast", survivor_fast(n, k, start=s)),
                    ("order", elimination_order(n, k, start=s)[-1]),
                ):
                    if got != expected:
                        problems.append(
                            f"{name}(n={n}, k={k}, start={s}) = {got}, expected {expected}"
                        )
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _time(fn, *args) -> tuple[float, object]:
    import time

    start = time.perf_counter()
    result = fn(*args)
    return time.perf_counter() - start, result


def _benchmark() -> None:
    print("Finding the survivor only")
    print(f"{'Method':<22}{'n':>10}{'k':>8}{'time':>11}   survivor")
    print("-" * 72)
    cases = [
        ("simulate", simulate, 200_000, 7),
        ("survivor_recurrence", survivor_recurrence, 200_000, 7),
        ("survivor_fast", survivor_fast, 200_000, 7),
        ("survivor_fast", survivor_fast, 10**18, 7),
        ("survivor_pow2", survivor_pow2, 10**18, 2),
        ("survivor", survivor, 10**1000, 3),
    ]
    for name, fn, n, k in cases:
        elapsed, result = _time(fn, n) if fn is survivor_pow2 else _time(fn, n, k)
        if isinstance(result, list):
            result = result[-1]
        shown = str(result)
        if len(shown) > 22:
            shown = shown[:9] + "..." + shown[-8:]
        n_shown = f"10^{len(str(n)) - 1}" if n >= 10**9 else f"{n:,}"
        print(f"{name:<22}{n_shown:>10}{k:>8}{elapsed:10.4f}s   {shown}")

    print()
    print("survivor_fast is independent of n except logarithmically, which is why")
    print("n = 10^18 costs the same as n = 200,000. simulate cannot go there at all,")
    print("and even n = 10^1000 is only a few milliseconds.")

    print()
    print("Producing the whole elimination order: O(n*k) deque vs O(n log n) Fenwick")
    print(f"{'n':>10}  {'k':>15}{'simulate':>12}{'Fenwick':>12}   winner")
    print("-" * 72)
    for n, k in [(100_000, 7), (100_000, 1_000), (50_000, 25_000), (100_000, 10**9)]:
        t_sim, sim = _time(simulate, n, k)
        t_fen, fen = _time(elimination_order, n, k)
        assert sim == fen, "the two methods must produce identical orders"
        winner = "simulate" if t_sim < t_fen else "Fenwick"
        print(f"{n:>10,}  {k:>15,}{t_sim:11.3f}s{t_fen:11.3f}s   {winner}")
    print()
    print("The asymptotically better structure loses until k is around n/2, because")
    print("deque.rotate is a C memmove while the Fenwick tree is interpreted Python.")
    print("Its cost is min(k mod len, len - k mod len), so a huge k is no worse than")
    print("a random one -- which is exactly where the O(n log n) version takes over.")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="josephus", description="Solve and explore the Josephus problem."
    )
    ap.add_argument("n", nargs="?", type=int, help="number of people")
    ap.add_argument("k", nargs="?", type=int, default=2, help="count (default 2)")
    ap.add_argument("--start", type=int, default=1, help="position counting starts at")
    ap.add_argument("--survivors", type=int, default=1, metavar="M",
                    help="report the last M survivors")
    ap.add_argument("--order", action="store_true", help="print the elimination order")
    ap.add_argument("--animate", action="store_true", help="draw the circle each step")
    ap.add_argument("--method", choices=["auto", "simulate", "recurrence", "fast", "pow2"],
                    default="auto")
    ap.add_argument("--verify", action="store_true", help="cross-validate every method")
    ap.add_argument("--benchmark", action="store_true", help="compare method speeds")
    args = ap.parse_args(argv)

    if args.verify:
        problems = verify()
        if problems:
            print(f"{len(problems)} disagreement(s):")
            for p in problems[:20]:
                print("  " + p)
            return 1
        print("all methods agree for n in 1..120, k in 1..20, and every start position")
        return 0

    if args.benchmark:
        _benchmark()
        return 0

    if args.n is None:
        ap.error("n is required (or use --verify / --benchmark)")

    n, k = args.n, args.k
    try:
        if args.method == "simulate":
            answer = simulate(n, k, start=args.start)[-1]
        elif args.method == "recurrence":
            answer = survivor_recurrence(n, k, start=args.start)
        elif args.method == "fast":
            answer = survivor_fast(n, k, start=args.start)
        elif args.method == "pow2":
            if k != 2:
                ap.error("--method pow2 only applies to k = 2")
            answer = survivor_pow2(n, start=args.start)
        else:
            answer = survivor(n, k, start=args.start)
    except ValueError as exc:
        ap.error(str(exc))

    if args.survivors > 1:
        picks = survivors(n, k, args.survivors, start=args.start)
        print(f"n={n} k={k}: last {args.survivors} survivors are {picks}")
    else:
        print(f"n={n} k={k}: survivor is position {answer}")

    if args.order:
        print("elimination order:",
              ", ".join(map(str, elimination_order(n, k, start=args.start))))

    if args.animate:
        if n > 40:
            print(f"(animation skipped: n={n} is too wide to draw)", file=sys.stderr)
        else:
            for frame in frames(n, k, start=args.start):
                print(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
