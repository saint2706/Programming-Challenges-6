"""Benchmark: mirror-halved counting vs full search, and the cost of going
from "how many solutions" to "how many *fundamentally different* solutions".

    uv run python benchmark.py
    uv run python benchmark.py --quick
"""

from __future__ import annotations

import argparse
import time

from nqueens import _count_from, count_fundamental_solutions, count_solutions


def _full_count(n: int) -> int:
    """The same bitmask search with no mirror-symmetry halving at all."""
    if n == 0:
        return 1
    mask = (1 << n) - 1
    return _count_from(n, 0, 0, 0, 0, mask)


def time_it(fn, *args) -> tuple[float, object]:
    start = time.perf_counter()
    result = fn(*args)
    return time.perf_counter() - start, result


def run(sizes: list[int]) -> None:
    print(
        f"{'n':>3} {'full (s)':>12} {'halved (s)':>12} {'speedup':>9} {'solutions':>10}"
    )
    for n in sizes:
        count_solutions.cache_clear()
        t_full, full = time_it(_full_count, n)
        count_solutions.cache_clear()
        t_half, half = time_it(count_solutions, n)
        assert full == half, (n, full, half)
        speedup = t_full / t_half if t_half > 0 else float("inf")
        print(f"{n:3d} {t_full:12.4f} {t_half:12.4f} {speedup:8.2f}x {full:10d}")


def run_fundamental(sizes: list[int]) -> None:
    print(
        f"\n{'n':>3} {'count_solutions (s)':>20} {'count_fundamental (s)':>22} {'ratio':>8}"
    )
    for n in sizes:
        count_solutions.cache_clear()
        t_total, _ = time_it(count_solutions, n)
        count_fundamental_solutions.cache_clear()
        t_fund, _ = time_it(count_fundamental_solutions, n)
        ratio = t_fund / t_total if t_total > 0 else float("inf")
        print(f"{n:3d} {t_total:20.4f} {t_fund:22.4f} {ratio:7.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        run(list(range(8, 13)))
        run_fundamental(list(range(8, 11)))
    else:
        run(list(range(8, 16)))
        run_fundamental(list(range(8, 13)))
