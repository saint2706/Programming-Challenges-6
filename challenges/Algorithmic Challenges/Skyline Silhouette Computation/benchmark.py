"""Benchmarks: sweep_line vs sweep_line_bst vs divide_and_conquer vs brute_force.

All three real methods are O(n log n), so the interesting question is not
"which wins asymptotically" -- it's whether the constant factors favor one of
them depending on *structure*, the way other benchmark-heavy challenges in
this repo (Josephus, Count Inversions, MST Visualizer) found real crossovers
once someone measured instead of assumed. Two axes are varied:

1. **Size**, at fixed density, to confirm all three scale like n log n and to
   see how far brute_force's O(n^2)-ish scan can be pushed before it stops
   being informative.
2. **Density**: "sparse" buildings are laid out on non-overlapping segments
   (skyline size grows linearly with n, the active set sees almost no
   contention), "downtown" buildings are packed into a narrow span with heavy
   overlap (skyline size stays small relative to n, most buildings never
   reach the top of the active set or a merge boundary).

``sweep_line`` and ``sweep_line_bst`` are the same sweep over the same events
-- only the data structure tracking "tallest active building" differs (a
lazy-deletion max-heap vs a ``sortedcontainers.SortedList`` with true
deletion), so this is also a direct test of that one structural choice, not
just another size/density sweep.

    uv run --with sortedcontainers python benchmark.py
    uv run --with sortedcontainers python benchmark.py --quick
"""

from __future__ import annotations

import argparse
import gc
import random
import time

from skyline import (
    Building,
    brute_force,
    divide_and_conquer,
    sweep_line,
    sweep_line_bst,
)

METHODS = {
    "brute": brute_force,
    "sweep": sweep_line,
    "sweep_bst": sweep_line_bst,
    "dc": divide_and_conquer,
}

#: Above this size, brute_force is skipped -- it would dominate the run.
BRUTE_CEILING = 4_000


def timed(fn, *args, repeat: int = 3) -> tuple[float, object]:
    """Best-of-`repeat` wall time with the GC quiet."""
    best, result = float("inf"), None
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeat):
            start = time.perf_counter()
            result = fn(*args)
            best = min(best, time.perf_counter() - start)
    finally:
        if enabled:
            gc.enable()
    return best, result


def sparse_buildings(n: int, seed: int) -> list[Building]:
    """n non-overlapping buildings, back to back -- a "suburb" skyline."""
    rng = random.Random(seed)
    out = []
    x = 0
    for _ in range(n):
        width = rng.randint(1, 6)
        out.append((x, x + width, rng.randint(1, 50)))
        x += width
    return out


def downtown_buildings(n: int, seed: int) -> list[Building]:
    """n buildings packed into a narrow span -- heavy overlap, few survive."""
    rng = random.Random(seed)
    span = max(10, n // 20)
    out = []
    for _ in range(n):
        a, b = rng.randint(0, span), rng.randint(0, span)
        l, r = (a, b) if a < b else (b, a + 1)
        out.append((l, r, rng.randint(1, 1000)))
    return out


# ---------------------------------------------------------------------------
# 1. Throughput vs n, at each density
# ---------------------------------------------------------------------------


def bench_scaling(sizes: list[int]) -> None:
    for label, gen in (("sparse", sparse_buildings), ("downtown", downtown_buildings)):
        print(f"\n== throughput, {label} buildings, seconds (lower is better) ==")
        header = f"{'n':>10} {'points':>8} " + " ".join(f"{m:>11}" for m in METHODS)
        print(header)
        print("-" * len(header))
        for n in sizes:
            buildings = gen(n, seed=n)
            expected = None
            cells = []
            n_points = None
            for name, fn in METHODS.items():
                if name == "brute" and n > BRUTE_CEILING:
                    cells.append(f"{'--':>11}")
                    continue
                secs, got = timed(fn, buildings, repeat=1 if n > 200_000 else 3)
                n_points = n_points or len(got)
                if expected is None:
                    expected = got
                assert got == expected, f"{name} disagreed at n={n} ({label})"
                cells.append(f"{secs:>11.4f}")
            print(f"{n:>10} {n_points:>8} " + " ".join(cells))
        print(
            "\nEvery timed method returned the identical key-point list at every n "
            "(asserted, not hoped)."
        )


# ---------------------------------------------------------------------------
# 2. sweep vs sweep_bst vs dc head to head, across density
# ---------------------------------------------------------------------------


def bench_density(n: int) -> None:
    print(
        f"\n== sweep_line vs sweep_line_bst vs divide_and_conquer, n = {n}, by density =="
    )
    header = (
        f"{'density':>10} {'points':>8} {'sweep':>10} {'sweep_bst':>10} "
        f"{'dc':>10} {'winner':>10}"
    )
    print(header)
    print("-" * len(header))
    densities = {
        "sparse": sparse_buildings(n, seed=1),
        "downtown": downtown_buildings(n, seed=1),
    }
    for label, buildings in densities.items():
        t_sweep, sweep_result = timed(sweep_line, buildings, repeat=5)
        t_bst, bst_result = timed(sweep_line_bst, buildings, repeat=5)
        t_dc, dc_result = timed(divide_and_conquer, buildings, repeat=5)
        assert sweep_result == bst_result == dc_result
        winner = min(
            ("sweep", t_sweep), ("sweep_bst", t_bst), ("dc", t_dc), key=lambda p: p[1]
        )[0]
        print(
            f"{label:>10} {len(sweep_result):>8} {t_sweep:>10.4f} {t_bst:>10.4f} "
            f"{t_dc:>10.4f} {winner:>10}"
        )


# ---------------------------------------------------------------------------
# 3. Sensitivity to overlap: span sweep at fixed n
# ---------------------------------------------------------------------------


def bench_span_sweep(n: int = 40_000) -> None:
    print(f"\n== overlap sweep at fixed n = {n}: span controls density ==")
    header = (
        f"{'span':>10} {'points':>8} {'sweep':>10} {'sweep_bst':>10} "
        f"{'dc':>10} {'winner':>10}"
    )
    print(header)
    print("-" * len(header))
    for span in [50, 200, 1_000, 5_000, 20_000, 100_000]:
        rng = random.Random(span)
        buildings = []
        for _ in range(n):
            a, b = rng.randint(0, span), rng.randint(0, span)
            l, r = (a, b) if a < b else (b, a + 1)
            buildings.append((l, r, rng.randint(1, 1000)))
        t_sweep, s_result = timed(sweep_line, buildings, repeat=3)
        t_bst, b_result = timed(sweep_line_bst, buildings, repeat=3)
        t_dc, d_result = timed(divide_and_conquer, buildings, repeat=3)
        assert s_result == b_result == d_result
        winner = min(
            ("sweep", t_sweep), ("sweep_bst", t_bst), ("dc", t_dc), key=lambda p: p[1]
        )[0]
        print(
            f"{span:>10} {len(s_result):>8} {t_sweep:>10.4f} {t_bst:>10.4f} "
            f"{t_dc:>10.4f} {winner:>10}"
        )
    print(
        "\nSmall span -> heavy overlap -> few key points survive; large span -> nearly"
        " disjoint buildings -> the skyline is almost as big as the input."
    )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick", action="store_true", help="smaller inputs, ~10s total"
    )
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--only", nargs="+", choices=["scaling", "density", "span"])
    args = parser.parse_args(argv)

    if args.sizes:
        sizes = args.sizes
    elif args.quick:
        sizes = [1_000, 10_000, 100_000]
    else:
        sizes = [1_000, 10_000, 100_000, 1_000_000]

    selected = set(args.only) if args.only else None

    def run(name, fn, *a):
        if selected is None or name in selected:
            fn(*a)

    run("scaling", bench_scaling, sizes)
    run("density", bench_density, sizes[-1])
    run("span", bench_span_sweep, min(sizes[-1], 40_000))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
