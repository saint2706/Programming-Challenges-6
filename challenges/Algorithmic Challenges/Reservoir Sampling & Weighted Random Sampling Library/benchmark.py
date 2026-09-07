"""Throughput: reservoir_r vs reservoir_l as n grows, and a_res vs a naive baseline.

Two comparisons:

1. ``reservoir_r`` vs ``reservoir_l`` for a fixed small k as the stream length
   n grows across several orders of magnitude. Algorithm L is O(k*(1 +
   log(n/k))) against Algorithm R's O(n), so the win should widen as n grows
   -- this measures whether it actually does, and by how much, rather than
   assuming the asymptotics translate directly into wall-clock time.

2. ``a_res`` and ``reservoir_chao`` (both stream one item at a time, O(k)
   memory) vs a naive baseline that collects the whole stream into a list
   and repeatedly does weighted choice + removal (O(n) memory, and O(n*k)
   time). This isolates the memory advantage: the streaming samplers should
   scale with n more gently and, past some n, stop losing to the naive
   method on time as well, once the naive method's O(n) list-copy-per-removal
   starts to dominate. It also puts a_res and reservoir_chao side by side --
   both are O(n) per pass (a_res pays O(log k) per item for the heap,
   reservoir_chao pays O(1) -- no heap at all, just a running float), so
   reservoir_chao should come out at least as fast, usually faster.

    uv run python benchmark.py
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

from reservoir import a_res, reservoir_chao, reservoir_l, reservoir_r


def _time_once(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def _naive_weighted_sample(
    items_weights: list[tuple[object, float]], k: int, rng: random.Random
) -> list[object]:
    """Collect everything, then repeated weighted choice + removal. O(n) memory."""
    pool_items = [iw[0] for iw in items_weights]
    pool_weights = [iw[1] for iw in items_weights]
    chosen = []
    for _ in range(min(k, len(pool_items))):
        total = sum(pool_weights)
        target = rng.uniform(0, total)
        upto = 0.0
        for idx, w in enumerate(pool_weights):
            upto += w
            if upto >= target:
                chosen.append(pool_items.pop(idx))
                pool_weights.pop(idx)
                break
    return chosen


def _bench_uniform() -> None:
    print("reservoir_r vs reservoir_l -- fixed k, growing n")
    print(f"{'n':>12}{'k':>6}{'reservoir_r':>14}{'reservoir_l':>14}{'ratio (R/L)':>14}")
    print("-" * 60)
    k = 50
    for n in (10_000, 100_000, 1_000_000, 5_000_000):
        seed = 0
        t_r = _time_once(
            lambda n=n, seed=seed: reservoir_r(range(n), k, rng=random.Random(seed))
        )
        t_l = _time_once(
            lambda n=n, seed=seed: reservoir_l(range(n), k, rng=random.Random(seed))
        )
        ratio = t_r / t_l if t_l else float("inf")
        print(f"{n:>12,}{k:>6}{t_r:13.4f}s{t_l:13.4f}s{ratio:13.2f}x")
    print()
    print("reservoir_l does a coin flip's worth of work only on the items it")
    print("actually keeps or skips past in bulk, while reservoir_r pays for every")
    print("single element -- the ratio should trend upward as n grows past k.")


def _bench_weighted() -> None:
    print()
    print("a_res / reservoir_chao (streaming, O(k) memory) vs naive weighted sample")
    print("(O(n) memory)")
    print(
        f"{'n':>10}{'k':>6}{'a_res':>12}{'chao':>12}{'naive':>12}"
        f"{'ratio (naive/a_res)':>22}{'ratio (naive/chao)':>21}"
    )
    print("-" * 95)
    k = 20
    for n in (2_000, 10_000, 50_000, 150_000):
        items = [(i, (i % 97) + 1.0) for i in range(n)]
        seed = 0
        t_ares = _time_once(
            lambda items=items, seed=seed: a_res(
                iter(items), k, rng=random.Random(seed)
            )
        )
        t_chao = _time_once(
            lambda items=items, seed=seed: reservoir_chao(
                iter(items), k, rng=random.Random(seed)
            )
        )
        t_naive = _time_once(
            lambda items=items, seed=seed: _naive_weighted_sample(
                items, k, random.Random(seed)
            )
        )
        ratio_ares = t_naive / t_ares if t_ares else float("inf")
        ratio_chao = t_naive / t_chao if t_chao else float("inf")
        print(
            f"{n:>10,}{k:>6}{t_ares:11.4f}s{t_chao:11.4f}s{t_naive:11.4f}s"
            f"{ratio_ares:21.2f}x{ratio_chao:20.2f}x"
        )
    print()
    print("The naive baseline is O(n*k) time (a full weighted draw plus an O(n)")
    print("list removal, k times) on top of O(n) memory for the materialized")
    print("stream. a_res is a single O(n log k) streaming pass with O(k) memory,")
    print("reservoir_chao a single O(n) pass with O(k) memory and no heap at all")
    print("(just a running float and a coin flip per item) -- the gap to naive")
    print("should widen with n for both, and they're the only two of the three")
    print("that can run at all when the stream doesn't fit in memory.")


def main() -> None:
    _bench_uniform()
    _bench_weighted()


if __name__ == "__main__":
    main()
