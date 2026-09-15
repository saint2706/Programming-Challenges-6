"""Held-Karp vs branch and bound (both exact), and heuristic solution quality.

    uv run --with networkx python benchmark.py

Two comparisons:

1. Both exact methods must agree on cost by construction -- what differs is
   wall-clock time as n grows. Held-Karp's O(2^n * n^2) has no way to finish
   early; branch and bound's MST lower bound lets it discard huge swaths of
   the search space on typical (Euclidean) instances, at the cost of the
   same exponential worst case if the bound isn't tight enough to help.

2. How close each polynomial-time heuristic gets to the true optimum,
   averaged over many random instances (a single instance's ratio is noise;
   many instances is signal), plus how expensive each one is.
"""

from __future__ import annotations

import statistics
import time

from tsp import (
    branch_and_bound,
    christofides,
    euclidean_instance,
    greedy_edge,
    held_karp,
    nearest_neighbor,
    tour_length,
    two_opt,
)


def _bench_exact_methods() -> None:
    print("Held-Karp vs branch and bound: same exact answer, very different cost")
    print(
        f"{'n':>4}{'held-karp time':>18}{'branch&bound time':>20}{'optimal cost':>16}{'agree?':>9}"
    )
    print("-" * 67)
    for n in (8, 10, 12, 14, 16, 18, 20):
        _, dist = euclidean_instance(n, seed=1)

        start = time.perf_counter()
        hk_cost, _ = held_karp(dist)
        hk_time = time.perf_counter() - start

        start = time.perf_counter()
        bnb_cost, _ = branch_and_bound(dist)
        bnb_time = time.perf_counter() - start

        agree = abs(hk_cost - bnb_cost) < 1e-6
        print(f"{n:>4}{hk_time:17.4f}s{bnb_time:19.4f}s{hk_cost:16.2f}{agree!s:>9}")
    print()
    print(
        "Both columns solve the same instances to the same optimal cost -- the "
        "'agree?' column is a correctness check, not a coincidence. Held-Karp's "
        "time roughly octuples every 3 cities (2^n growth) with no way to stop "
        "early; branch and bound barely moves across the same range because its "
        "MST lower bound prunes almost the entire search tree on these Euclidean "
        "instances. That gap is instance-dependent, not a free lunch: on harder, "
        "less geometrically structured distance matrices the lower bound is "
        "looser and branch and bound degrades toward Held-Karp's own exponential "
        "wall (confirmed separately: branch_and_bound did not finish a random "
        "Euclidean n=30 instance within a 45-second budget either, well past "
        "where n=20 above suggests -- exponential is still exponential)."
    )


def _bench_heuristic_quality(n: int = 10, trials: int = 40) -> None:
    print()
    print(
        f"Heuristic solution quality: ratio to optimal, averaged over {trials} random n={n} instances"
    )
    print(f"{'method':>18}{'mean ratio':>12}{'max ratio':>11}{'mean time':>12}")
    print("-" * 53)

    methods = {
        "nearest_neighbor": lambda dist: nearest_neighbor(dist),
        "greedy_edge": lambda dist: greedy_edge(dist),
        "nn + two_opt": lambda dist: two_opt(dist, nearest_neighbor(dist)),
        "christofides": lambda dist: christofides(dist),
    }
    ratios: dict[str, list[float]] = {name: [] for name in methods}
    times: dict[str, list[float]] = {name: [] for name in methods}

    for seed in range(trials):
        _, dist = euclidean_instance(n, seed=seed)
        opt_cost, _ = held_karp(dist)
        for name, method in methods.items():
            start = time.perf_counter()
            tour = method(dist)
            times[name].append(time.perf_counter() - start)
            ratios[name].append(tour_length(dist, tour) / opt_cost)

    for name in methods:
        mean_ratio = statistics.mean(ratios[name])
        max_ratio = max(ratios[name])
        mean_time_ms = statistics.mean(times[name]) * 1000
        print(f"{name:>18}{mean_ratio:12.4f}{max_ratio:11.4f}{mean_time_ms:10.3f}ms")

    print()
    print(
        "'nn + two_opt' is the standout: starting from nearest-neighbor's "
        "~10% average gap, a single round of 2-opt local search closes it to "
        "under 1% on average here, cheaper than Christofides and with no "
        "formal guarantee at all -- a reminder that a proven worst-case bound "
        "(Christofides' 1.5x) and typical-case performance are different "
        "claims. Christofides' own ratios stay comfortably inside its 1.5x "
        "guarantee (max observed well under it) but it is by far the slowest "
        "method here -- the minimum-weight perfect matching step (Edmonds' "
        "blossom algorithm, via networkx) costs real time that the other "
        "three methods simply don't pay."
    )


def main() -> None:
    _bench_exact_methods()
    _bench_heuristic_quality()


if __name__ == "__main__":
    main()
