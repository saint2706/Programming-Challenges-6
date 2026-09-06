"""Wall-clock benchmark: Kruskal vs Prim vs Boruvka across graph densities.

Textbook complexity says Kruskal is O(E log E), Prim (binary heap) is
O(E log V), and Boruvka is O(E log V) with better parallelism. None of that
predicts a *single-threaded Python* ranking, because Boruvka's O(log V)
sequential rounds each re-scan every remaining edge -- overhead the other
two never pay. This script measures where that overhead actually lands.

    uv run python benchmark.py
    uv run python benchmark.py --quick
"""

from __future__ import annotations

import argparse
import random
import time

from mst import ALGORITHMS, solve


def random_graph(n: int, density: float, seed: int) -> list[tuple[int, int, float]]:
    rng = random.Random(seed)
    return [
        (u, v, rng.randint(1, 1_000_000))
        for u in range(n)
        for v in range(u + 1, n)
        if rng.random() < density
    ]


def time_algorithm(algo, n: int, edges, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        solve(algo, n, edges)
        best = min(best, time.perf_counter() - start)
    return best


def run(sizes: list[int], densities: dict[str, float], repeats: int) -> None:
    header = f"{'n':>6} {'density':>10}" + "".join(f"{name:>12}" for name in ALGORITHMS)
    print(header)
    for label, density in densities.items():
        for n in sizes:
            edges = random_graph(n, density, seed=hash((label, n)) & 0xFFFFFFFF)
            row = f"{n:6d} {label:>10}"
            for algo in ALGORITHMS.values():
                t = time_algorithm(algo, n, edges, repeats)
                row += f"{t * 1000:11.2f}m"
            print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="smaller sizes, fewer repeats"
    )
    args = parser.parse_args()

    if args.quick:
        run([50, 200], {"sparse (V edges)": 0.02, "dense (V^2/4)": 0.5}, repeats=2)
    else:
        densities = {
            "sparse ~V": None,  # filled per-n below
            "medium ~VlogV": None,
            "dense ~V^2/4": 0.5,
        }
        sizes = [100, 300, 600, 1000]
        print(
            "Sparse/medium densities are recomputed per n to target ~V and ~V log V edges."
        )
        header = f"{'n':>6} {'density':>16}" + "".join(
            f"{name:>12}" for name in ALGORITHMS
        )
        print(header)
        for n in sizes:
            max_edges = n * (n - 1) / 2
            targets = {
                "sparse (~V edges)": n / max_edges,
                "medium (~VlogV)": (n * max(1, n.bit_length())) / max_edges,
                "dense (V^2/4)": 0.5,
            }
            for label, density in targets.items():
                density = min(density, 1.0)
                edges = random_graph(n, density, seed=hash((label, n)) & 0xFFFFFFFF)
                row = f"{n:6d} {label:>16}"
                for algo in ALGORITHMS.values():
                    t = time_algorithm(algo, n, edges, repeats=3)
                    row += f"{t * 1000:11.2f}m"
                print(row)
