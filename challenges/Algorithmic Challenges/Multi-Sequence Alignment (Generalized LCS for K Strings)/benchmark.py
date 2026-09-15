"""Brute-force K-dim DP vs the dominant-point method: state-space size and wall-clock time.

    uv run python benchmark.py

Two comparisons:

1. How many states each exact method actually visits. Brute force visits
   exactly `prod(n_i)` DP cells by construction; the dominant-point method
   visits only the dominant points it generates level by level -- the
   benchmark counts both directly.
2. How that translates to wall-clock time as sequence length grows, until
   brute force's `prod(n_i)` growth makes it impractical.
"""

from __future__ import annotations

import math
import random
import time

from mlcs import (
    brute_force_mlcs,
    dominant_point_levels,
    dominant_point_mlcs,
    progressive_mlcs,
)


def _random_sequences(k: int, n: int, alphabet: str, seed: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(alphabet) for _ in range(n)) for _ in range(k)]


def _bench_state_space(k: int = 4, alphabet: str = "ACGT") -> None:
    print(
        f"State space size: brute force (prod n_i) vs dominant points actually generated (K={k}, alphabet={alphabet!r})"
    )
    print(f"{'n':>4}{'grid cells':>14}{'dominant points':>18}{'reduction':>12}")
    print("-" * 48)
    for n in (4, 6, 8, 10, 12, 14, 16):
        seqs = _random_sequences(k, n, alphabet, seed=n)
        grid_cells = math.prod(len(s) + 1 for s in seqs)
        levels, _ = dominant_point_levels(seqs)
        dominant_points = sum(len(level) for level in levels)
        reduction = grid_cells / dominant_points if dominant_points else float("inf")
        print(f"{n:>4}{grid_cells:>14,}{dominant_points:>18,}{reduction:>11.1f}x")
    print()
    print(
        "'grid cells' is every state brute_force_mlcs's memoized recursion could "
        "visit -- prod(n_i + 1) by construction. 'dominant points' is the actual "
        "total size of every level the dominant-point method generated. The gap "
        "widens with n because the grid grows as n^K while the number of points "
        "that can survive the Pareto/dominance filter is bounded by how many "
        "distinct, mutually-non-dominated ways there are to have matched a "
        "length-l common subsequence so far -- a much smaller quantity on "
        "typical (non-adversarial) sequences."
    )


def _bench_wall_clock(k: int = 4, alphabet: str = "ACGT") -> None:
    print()
    print(
        f"Wall-clock time: brute_force_mlcs vs dominant_point_mlcs (K={k}, alphabet={alphabet!r})"
    )
    print(f"{'n':>4}{'brute force':>16}{'dominant point':>18}{'speedup':>10}")
    print("-" * 48)
    for n in (16, 20, 24, 28, 32, 36):
        seqs = _random_sequences(k, n, alphabet, seed=n + 1)

        start = time.perf_counter()
        exact = brute_force_mlcs(seqs)
        bf_time = time.perf_counter() - start

        start = time.perf_counter()
        fast = dominant_point_mlcs(seqs)
        dp_time = time.perf_counter() - start

        assert len(exact) == len(fast), (n, exact, fast)
        speedup = bf_time / dp_time if dp_time > 0 else float("inf")
        print(f"{n:>4}{bf_time:15.5f}s{dp_time:17.5f}s{speedup:9.1f}x")
    print()
    print(
        "Both columns agree on length by construction -- the point is the "
        "growth rate. brute_force_mlcs is O(n^K): each +4 to n multiplies its "
        "grid by roughly (1 + 4/n)^K, compounding fast. dominant_point_mlcs's "
        "cost tracks how many dominant points actually exist, which grows far "
        "more slowly on these instances -- the same trade the SCC challenge's "
        "Tarjan/Kosaraju/Gabow shootout makes for compiled-vs-interpreted, and "
        "TSP's Held-Karp/branch-and-bound comparison makes for exact algorithms "
        "generally: same guaranteed-correct answer, different state-space size."
    )


def _bench_progressive_quality(trials: int = 300) -> None:
    print()
    print(
        f"progressive_mlcs solution quality vs the true optimum, {trials} random K=3 instances"
    )
    rng = random.Random(7)
    total_gap = 0
    instances_with_gap = 0
    for _ in range(trials):
        seqs = [
            "".join(rng.choice("AB") for _ in range(rng.randint(2, 6)))
            for _ in range(3)
        ]
        exact_len = len(brute_force_mlcs(seqs))
        prog_len = len(progressive_mlcs(seqs))
        gap = exact_len - prog_len
        total_gap += gap
        if gap > 0:
            instances_with_gap += 1
    print(
        f"{instances_with_gap}/{trials} instances ({100 * instances_with_gap / trials:.1f}%) "
        f"had progressive_mlcs strictly below the true MLCS length; mean gap "
        f"{total_gap / trials:.3f} characters. progressive_mlcs is O(K) pairwise "
        "LCS computations -- cheap -- but every one of these gaps is an early, "
        "locally-optimal tie-break that turned out to be the wrong one once the "
        "next sequence was considered, with no way to backtrack and fix it."
    )


def main() -> None:
    _bench_state_space()
    _bench_wall_clock()
    _bench_progressive_quality()


if __name__ == "__main__":
    main()
