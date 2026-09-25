"""Space, false-positive rate, and throughput across all six filters.

    uv run python benchmark.py

Five comparisons, each isolating one claim from the README:

1. Space vs. measured false-positive rate, all six structures at the same
   target rate -- the brief's literal "compare FP rates" ask.
2. Among the three structures that actually support deletion (Counting
   Bloom, Cuckoo, Semi-Sorted Cuckoo), how much space each needs -- the
   quantitative "cuckoo filters are practically better than [counting]
   Bloom" result.
3. Cuckoo filter's achievable load factor as a function of bucket size,
   reproducing the shape of Fan et al.'s own Figure (bigger buckets pack
   tighter before insertions start failing).
4. Semi-sorted vs. standard cuckoo buckets: the space this compression
   saves against the insert/lookup throughput it costs.
5. Xor vs. Binary Fuse filters across a range of n: Binary Fuse's space
   advantage is asymptotic (its size-factor formula is calibrated against a
   1,000,000-item reference scale), so this also shows *where* the
   crossover against Xor actually happens, not just that it eventually
   does.
"""

from __future__ import annotations

import math
import time

from cuckoo_filter import (
    BinaryFuseFilter,
    BloomFilter,
    CountingBloomFilter,
    CuckooFilter,
    SemiSortedCuckooFilter,
    XorFilter,
)


def _measure_fp_rate(contains_fn, prefix: str, trials: int) -> float:
    hits = sum(1 for i in range(trials) if contains_fn(f"{prefix}{i}"))
    return hits / trials


def bench_space_vs_fp_rate() -> None:
    print(
        "1. Space vs. measured false-positive rate (n=50,000, absent-key trials=100,000)"
    )
    print(
        f"{'structure':<24}{'target':>10}{'measured':>12}{'bytes':>10}{'bits/key':>10}{'delete':>8}"
    )
    print("-" * 74)
    n = 50_000
    trials = 100_000
    for target in (0.10, 0.02, 0.01, 0.001):
        bf = BloomFilter(n, target)
        cbf = CountingBloomFilter(n, target)
        cf = CuckooFilter(n, fp_rate=target, seed=1)
        ssf = SemiSortedCuckooFilter(n, fp_rate=target, seed=1)
        for i in range(n):
            item = f"item-{i}"
            bf.add(item)
            cbf.add(item)
            cf.insert(item)
            ssf.insert(item)

        fp_bits = max(1, math.ceil(math.log2(1 / target)))
        xf = XorFilter((f"item-{i}" for i in range(n)), fingerprint_bits=fp_bits)
        bff = BinaryFuseFilter(
            (f"item-{i}" for i in range(n)), fingerprint_bits=fp_bits
        )

        rows = [
            ("BloomFilter", bf, lambda x, o=bf: x in o, "no"),
            ("CountingBloomFilter", cbf, lambda x, o=cbf: x in o, "yes"),
            ("CuckooFilter", cf, lambda x, o=cf: x in o, "yes"),
            ("SemiSortedCuckooFilter", ssf, lambda x, o=ssf: x in o, "yes"),
            ("XorFilter", xf, lambda x, o=xf: x in o, "no"),
            ("BinaryFuseFilter", bff, lambda x, o=bff: x in o, "no"),
        ]
        for name, obj, contains, deletes in rows:
            measured = _measure_fp_rate(contains, "absent-", trials)
            bits_per_key = 8 * obj.nbytes / n
            print(
                f"{name:<24}{target:>10.3%}{measured:>12.4%}"
                f"{obj.nbytes:>10}{bits_per_key:>10.2f}{deletes:>8}"
            )
        print()


def bench_deletion_capable_space() -> None:
    print("2. Space among deletion-capable structures only (n=50,000, fp_rate=1%)")
    n, fp_rate = 50_000, 0.01
    bf = BloomFilter(n, fp_rate)  # reference point: cannot delete, included for scale
    cbf = CountingBloomFilter(n, fp_rate)
    cf = CuckooFilter(n, fp_rate=fp_rate, seed=1)
    ssf = SemiSortedCuckooFilter(n, fp_rate=fp_rate, seed=1)
    print(f"{'structure':<24}{'bytes':>10}{'x plain Bloom':>16}")
    print("-" * 50)
    for name, obj in [
        ("BloomFilter (no delete)", bf),
        ("CountingBloomFilter", cbf),
        ("CuckooFilter", cf),
        ("SemiSortedCuckooFilter", ssf),
    ]:
        print(f"{name:<24}{obj.nbytes:>10}{obj.nbytes / bf.nbytes:>15.2f}x")
    print()


def bench_load_factor_by_bucket_size() -> None:
    print("3. Cuckoo filter: achievable load factor before the first insert failure")
    print(f"{'bucket_size':>12}{'buckets*size':>14}{'inserted':>10}{'load_factor':>13}")
    print("-" * 50)
    n = 50_000
    for b in (2, 4, 8):
        cf = CuckooFilter(capacity=n, bucket_size=b, fp_rate=0.01, seed=b)
        capacity = cf.num_buckets * cf.bucket_size
        i = 0
        while cf.insert(f"x-{i}"):
            i += 1
            if i > capacity * 1.2:
                break
        print(f"{b:>12}{capacity:>14}{len(cf):>10}{cf.load_factor:>13.3%}")
    print()


def bench_semi_sort_savings() -> None:
    print("4. Semi-sorted vs. standard cuckoo buckets: space saved vs. throughput cost")
    n = 100_000
    print(
        f"{'fp_rate':>10}{'standard B':>12}{'semi-sort B':>13}"
        f"{'space saved':>13}{'insert slowdown':>17}"
    )
    print("-" * 68)
    for fp_rate in (0.1, 0.01, 0.001):
        cf = CuckooFilter(n, fp_rate=fp_rate, seed=2)
        ssf = SemiSortedCuckooFilter(n, fp_rate=fp_rate, seed=2)
        items = [f"item-{i}" for i in range(n)]

        t0 = time.perf_counter()
        for item in items:
            cf.insert(item)
        t_standard = time.perf_counter() - t0

        t0 = time.perf_counter()
        for item in items:
            ssf.insert(item)
        t_semi = time.perf_counter() - t0

        saved = 1 - ssf.nbytes / cf.nbytes
        slowdown = t_semi / t_standard
        print(
            f"{fp_rate:>10.3%}{cf.nbytes:>12}{ssf.nbytes:>13}{saved:>13.2%}{slowdown:>16.2f}x"
        )
    print()


def bench_xor_vs_binary_fuse() -> None:
    print(
        "5. Xor vs. Binary Fuse: space and construction time across n (fingerprint_bits=8)"
    )
    print(
        f"{'n':>10}{'xor bytes':>12}{'bfuse bytes':>13}"
        f"{'bfuse/xor':>11}{'xor build':>12}{'bfuse build':>13}"
    )
    print("-" * 71)
    for n in (2_000, 20_000, 100_000, 300_000):
        items = [f"item-{i}" for i in range(n)]

        t0 = time.perf_counter()
        xf = XorFilter(items, fingerprint_bits=8)
        t_xor = time.perf_counter() - t0

        t0 = time.perf_counter()
        bff = BinaryFuseFilter(items, fingerprint_bits=8)
        t_bfuse = time.perf_counter() - t0

        ratio = bff.nbytes / xf.nbytes
        print(
            f"{n:>10,}{xf.nbytes:>12,}{bff.nbytes:>13,}"
            f"{ratio:>11.3f}{t_xor:>11.3f}s{t_bfuse:>12.3f}s"
        )
    print()


if __name__ == "__main__":
    bench_space_vs_fp_rate()
    bench_deletion_capable_space()
    bench_load_factor_by_bucket_size()
    bench_semi_sort_savings()
    bench_xor_vs_binary_fuse()
