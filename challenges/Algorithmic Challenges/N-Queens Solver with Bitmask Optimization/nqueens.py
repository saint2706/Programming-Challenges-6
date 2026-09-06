"""N-Queens: bitmask backtracking, mirror-symmetry counting, and the 8-fold
dihedral symmetry classification of solutions.

A solution is represented as a tuple `cols` of length `n` where `cols[r]` is
the column of the queen in row `r` (0-indexed) -- i.e. a permutation of
`range(n)` with no two entries on the same diagonal.

Two solvers are provided for two different jobs:

- `count_solutions` / `_count_from` -- the fast counter, using the classic
  "only search the left half of row 0, double the result" mirror-symmetry
  optimization. This is what the benchmark and any real solution count use.
- `solve_bitmask_steps` -- a plain (non-halved) generator that yields a
  `Step` per placement/backtrack/solution, for the visualizer and for
  `all_solutions`. It is intentionally the simple version: halving the
  search *during* animation would mean half the found solutions are never
  actually searched for on screen, only mirrored after the fact, which
  would be a worse visualization even though it is a faster algorithm.

See README.md for why full 8-fold symmetry pruning is not attempted during
the search itself (2-fold captures nearly all the achievable speedup for
much less complexity), and for the Burnside's-lemma verification of the
fundamental-solution count below.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache


@dataclass
class Step:
    row: int
    col: int
    action: str  # "place", "backtrack", "solution"
    board: tuple[int, ...]
    caption: str = ""


def _count_from(n: int, row: int, cols: int, diag1: int, diag2: int, mask: int) -> int:
    if row == n:
        return 1
    available = mask & ~(cols | diag1 | diag2)
    count = 0
    while available:
        bit = available & (-available)
        available ^= bit
        count += _count_from(
            n, row + 1, cols | bit, (diag1 | bit) << 1 & mask, (diag2 | bit) >> 1, mask
        )
    return count


@cache
def count_solutions(n: int) -> int:
    """Total number of n-queens solutions, via 2-fold mirror-symmetry halving.

    Only the left half of row 0's columns is searched; each solution found
    there has a distinct mirror-image solution with the queen in the
    mirrored column of row 0, so it counts double. For odd `n` there is also
    a middle column with no distinct partner under this specific mirror (its
    own subtree already contains complete mirror pairs, since reflecting the
    whole board maps a middle-column-row-0 solution to another
    middle-column-row-0 solution), so that subtree is searched in full and
    counted once, not doubled.
    """
    if n == 0:
        return 1
    if n == 1:
        return 1
    mask = (1 << n) - 1
    half = n // 2
    total = 0
    for col in range(half):
        bit = 1 << col
        total += 2 * _count_from(n, 1, bit, (bit << 1) & mask, bit >> 1, mask)
    if n % 2 == 1:
        mid_bit = 1 << (n // 2)
        total += _count_from(n, 1, mid_bit, (mid_bit << 1) & mask, mid_bit >> 1, mask)
    return total


def solve_bitmask_steps(n: int, limit: int | None = None):
    """Yield a Step per queen placement/backtrack/solution (full search).

    This is the un-halved search: every row-0 column is tried in order, so
    every solution is actually discovered by the animation rather than half
    of them being produced by mirroring after the fact. `limit` stops the
    whole search after that many solutions have been yielded -- used by the
    visualizer to animate a *sample* of solutions for larger `n` rather than
    all of them, since the count explodes quickly (n=12 already has 14,200).
    """
    if n == 0:
        yield Step(0, -1, "solution", (), "n=0: the vacuous placement")
        return

    mask = (1 << n) - 1
    board: list[int] = []
    found = 0

    def backtrack(row: int, cols: int, diag1: int, diag2: int):
        nonlocal found
        if limit is not None and found >= limit:
            return
        if row == n:
            found += 1
            yield Step(row, -1, "solution", tuple(board), f"Solution #{found}")
            return
        available = mask & ~(cols | diag1 | diag2)
        while available:
            if limit is not None and found >= limit:
                return
            bit = available & (-available)
            available ^= bit
            col = bit.bit_length() - 1
            board.append(col)
            yield Step(
                row, col, "place", tuple(board), f"Place row {row} at column {col}"
            )
            yield from backtrack(
                row + 1, cols | bit, (diag1 | bit) << 1 & mask, (diag2 | bit) >> 1
            )
            board.pop()
            yield Step(
                row,
                col,
                "backtrack",
                tuple(board),
                f"Backtrack from row {row}, column {col}",
            )

    yield from backtrack(0, 0, 0, 0)


@cache
def all_solutions(n: int) -> list[tuple[int, ...]]:
    return [step.board for step in solve_bitmask_steps(n) if step.action == "solution"]


def is_valid_solution(cols: tuple[int, ...]) -> bool:
    n = len(cols)
    if len(set(cols)) != n:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if abs(cols[i] - cols[j]) == j - i:
                return False
    return True


# ---------------------------------------------------------------------------
# The 8-fold dihedral group D4 acting on solutions
# ---------------------------------------------------------------------------


def _inverse(cols: tuple[int, ...]) -> list[int]:
    n = len(cols)
    inv = [0] * n
    for r, c in enumerate(cols):
        inv[c] = r
    return inv


def _identity(cols: tuple[int, ...], n: int) -> tuple[int, ...]:
    return cols


def _rot90(cols: tuple[int, ...], n: int) -> tuple[int, ...]:
    # (r, c) -> (c, n-1-r): newcols[c] = n-1-r for each r.
    new = [0] * n
    for r, c in enumerate(cols):
        new[c] = n - 1 - r
    return tuple(new)


def _rot180(cols: tuple[int, ...], n: int) -> tuple[int, ...]:
    return tuple(n - 1 - cols[n - 1 - i] for i in range(n))


def _rot270(cols: tuple[int, ...], n: int) -> tuple[int, ...]:
    inv = _inverse(cols)
    return tuple(inv[n - 1 - i] for i in range(n))


def _flip_h(cols: tuple[int, ...], n: int) -> tuple[int, ...]:
    return tuple(n - 1 - c for c in cols)


def _flip_v(cols: tuple[int, ...], n: int) -> tuple[int, ...]:
    return tuple(cols[n - 1 - i] for i in range(n))


def _flip_diag(cols: tuple[int, ...], n: int) -> tuple[int, ...]:
    return tuple(_inverse(cols))


def _flip_antidiag(cols: tuple[int, ...], n: int) -> tuple[int, ...]:
    inv = _inverse(cols)
    return tuple(n - 1 - inv[n - 1 - i] for i in range(n))


D4_TRANSFORMS = {
    "identity": _identity,
    "rot90": _rot90,
    "rot180": _rot180,
    "rot270": _rot270,
    "flip_h": _flip_h,
    "flip_v": _flip_v,
    "flip_diag": _flip_diag,
    "flip_antidiag": _flip_antidiag,
}


def canonical_form(cols: tuple[int, ...], n: int) -> tuple[int, ...]:
    return min(fn(cols, n) for fn in D4_TRANSFORMS.values())


@cache
def count_fundamental_solutions(n: int) -> int:
    """Number of solutions up to the full 8-fold symmetry of the square.

    Ground truth by direct canonicalization + dedup, cross-checked in
    test_nqueens.py against the Burnside's-lemma computation in
    `fixed_point_counts` -- see README.md for why this counting-based
    approach was chosen over pruning the search itself by all 8 symmetries.
    """
    solutions = all_solutions(n)
    return len({canonical_form(s, n) for s in solutions})


def fixed_point_counts(n: int) -> dict[str, int]:
    """|Fix(g)| for each of the 8 symmetries, computed directly (not assumed).

    Feeds the Burnside's-lemma check: #orbits = (1/8) * sum(|Fix(g)|).
    """
    solutions = all_solutions(n)
    return {
        name: sum(1 for s in solutions if fn(s, n) == s)
        for name, fn in D4_TRANSFORMS.items()
    }


def _demo(n: int = 8) -> None:
    total = count_solutions(n)
    fundamental = count_fundamental_solutions(n)
    print(f"n={n}: {total} total solutions, {fundamental} fundamental (up to symmetry)")
    fixed = fixed_point_counts(n)
    burnside = sum(fixed.values()) / 8
    print(f"Burnside check: sum(|Fix(g)|)/8 = {burnside} (fixed counts: {fixed})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=8)
    parser.add_argument("--verify", action="store_true", help="run counts for n=0..12")
    args = parser.parse_args()

    if args.verify:
        for n in range(13):
            _demo(n)
    else:
        _demo(args.n)
