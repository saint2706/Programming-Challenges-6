"""Benchmarks: the amortisation bound, the separator's real cost, and L vs k.

Five questions, each with a table:

1. **Is the Z-array actually linear?** Against the O(n^2) definition, and in
   extension counts rather than seconds -- the machine-independent form of
   the same claim.
2. **What does the concatenation cost?** `z_search` and `z_search_concat`
   compute the same answer; one of them copies the text first. Timed, and
   measured with `tracemalloc` so the memory claim is a number.
3. **Does streaming hold?** Peak memory of `z_search_stream` against text
   size, which should be flat.
4. **L against k.** The multi-pattern claim: scans scale with the number of
   trie leaves, not the number of patterns. Swept over dictionary shapes,
   with Aho-Corasick alongside so the comparison is honest about where the
   Z-array stops being the right tool.
5. **Main-Lorentz.** O(n log n) runs against the quadratic count of squares
   they encode.

    uv run python benchmark.py
    uv run python benchmark.py --quick
"""

from __future__ import annotations

import argparse
import gc
import math
import random
import string
import time
import tracemalloc

from zalgorithm import (
    AhoCorasick,
    MultiZMatcher,
    count_tandem_repeats,
    naive_z_array,
    tandem_repeat_runs,
    z_array,
    z_array_counted,
    z_search,
    z_search_concat,
    z_search_stream,
)


def timed(fn, *args, repeat: int = 3):
    """Best-of-`repeat` wall time with the GC quiet. Returns (seconds, result)."""
    best, result = math.inf, None
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


def peak_kib(fn, *args) -> tuple[float, object]:
    """Peak *additional* allocation of one call, in KiB."""
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    result = fn(*args)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return (peak - base) / 1024, result


def random_text(n: int, alphabet: str, rng: random.Random) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


# ---------------------------------------------------------------------------


def bench_construction(quick: bool) -> None:
    print("\n" + "=" * 74)
    print("1. Z-array construction: linear against the definition")
    print("=" * 74)
    rng = random.Random(1)
    sizes = [1000, 4000, 16000] if quick else [1000, 4000, 16000, 64000, 256000]
    print(f"\n{'n':>8} {'random-26':>11} {'naive':>11} {'ratio':>8} "
          f"{'a^n':>11} {'naive a^n':>11} {'ratio':>8}")
    for n in sizes:
        text = random_text(n, string.ascii_lowercase, rng)
        run = "a" * n
        t_fast, _ = timed(z_array, text)
        t_run, _ = timed(z_array, run)
        if n <= (16000 if quick else 64000):
            t_naive, _ = timed(naive_z_array, text, repeat=1)
            t_naive_run, _ = timed(naive_z_array, run, repeat=1)
            print(f"{n:8d} {t_fast:11.4f} {t_naive:11.4f} {t_naive / t_fast:7.1f}x "
                  f"{t_run:11.4f} {t_naive_run:11.4f} {t_naive_run / t_run:7.1f}x")
        else:
            print(f"{n:8d} {t_fast:11.4f} {'skipped':>11} {'--':>8} "
                  f"{t_run:11.4f} {'skipped':>11} {'--':>8}")
    print("\nThe naive column on random text stays cheap for the same reason the")
    print("naive palindrome scan does: on a large alphabet the inner loop almost")
    print("never runs. On a^n it is the real quadratic, and the gap is unbounded.")

    print("\n  Extension counts -- the amortisation proof, without a clock (n = 20000):")
    n = 20000
    inputs = {
        "all one character": "a" * n,
        "alternating ab": "ab" * (n // 2),
        "abacaba fractal": ("abacaba" * (n // 7 + 1))[:n],
        "one big border": "a" * (n // 2) + "b" + "a" * (n // 2 - 1),
        "random binary": random_text(n, "ab", rng),
        "random 26 letters": random_text(n, string.ascii_lowercase, rng),
    }
    print(f"\n  {'input':<20} {'extensions':>11} {'bound n':>9} {'naive comparisons':>19}")
    for name, s in inputs.items():
        _, ext = z_array_counted(s)
        naive = _naive_comparisons(s)
        print(f"  {name:<20} {ext:11,d} {len(s):9,d} {naive:19,d}")
    print("\n  Never above the bound. That column *is* the proof: every extension")
    print("  step pushes the right edge of the box one place right, and it never")
    print("  goes back.")


def _naive_comparisons(s: str) -> int:
    total = 0
    n = len(s)
    for i in range(1, n):
        k = 0
        while i + k < n and s[k] == s[i + k]:
            k += 1
            total += 1
        total += 1
    return total


def bench_search(quick: bool) -> None:
    print("\n" + "=" * 74)
    print("2. Searching: what the textbook concatenation costs")
    print("=" * 74)
    rng = random.Random(2)
    sizes = [50_000, 200_000] if quick else [50_000, 200_000, 800_000]
    print(f"\n{'n':>9} {'m':>5} {'z_search':>10} {'concat':>10} {'speedup':>8} "
          f"{'peak KiB (search)':>18} {'peak KiB (concat)':>18}")
    for n in sizes:
        text = random_text(n, "acgt", rng)
        for m in (8, 40):
            pat = random_text(m, "acgt", rng)
            t_direct, hits_a = timed(lambda: list(z_search(pat, text)))
            t_concat, hits_b = timed(lambda: list(z_search_concat(pat, text)))
            assert hits_a == hits_b
            mem_direct, _ = peak_kib(lambda: list(z_search(pat, text)))
            mem_concat, _ = peak_kib(lambda: list(z_search_concat(pat, text)))
            print(f"{n:9,d} {m:5d} {t_direct:10.4f} {t_concat:10.4f} "
                  f"{t_concat / t_direct:7.2f}x {mem_direct:18,.0f} {mem_concat:18,.0f}")
    print("\nSame answer both ways. The concatenation allocates a copy of the text")
    print("plus an int per position of it; the direct scan allocates O(m).")

    print("\n  Streaming peak memory (pattern m = 32), which should not move:")
    print(f"\n  {'n':>10} {'peak KiB':>10} {'matches':>8}")
    for n in (20_000, 80_000, 320_000):
        pat = "acgtacgtacgtacgtacgtacgtacgtacgt"
        text = random_text(n, "acgt", rng)
        mem, hits = peak_kib(lambda: list(z_search_stream(pat, iter(text))))
        print(f"  {n:10,d} {mem:10.1f} {len(hits):8d}")
    print("\n  Flat, because the box only moves right and the buffer never holds")
    print("  more than m + 1 characters. An unbounded stream is searchable.")


def bench_multi(quick: bool) -> None:
    print("\n" + "=" * 74)
    print("3. Multi-pattern: scans scale with trie leaves, not pattern count")
    print("=" * 74)
    rng = random.Random(3)
    n = 100_000 if quick else 400_000

    def naive_multi(pats, txt):
        return sorted(
            (pos, i) for i, p in enumerate(pats) for pos in z_search(p, txt)
        )

    stems = ["run", "walk", "read", "sing", "jump", "play", "work", "call"]
    words = []
    for stem in stems:
        words += [stem, stem + "s", stem + "er", stem + "ers", stem + "ing"]
    # A corpus the word list actually occurs in, otherwise `occ` is 0 and the
    # reporting cost -- the part Aho-Corasick pays too -- never shows up.
    corpus = []
    while sum(len(w) + 1 for w in corpus) < n:
        corpus.append(rng.choice(words) if rng.random() < 0.4 else random_text(rng.randint(2, 7), "abcdefgh", rng))
    prose = " ".join(corpus)[:n]

    base = random_text(24, "abcde", rng)
    cases = [
        # (name, patterns, text). L = 1: every pattern is a prefix of the next.
        ("prefix chain (L = 1)", [base[:k] for k in range(3, 25)], random_text(n, "abcde", rng)),
        # L = k: pairwise incomparable, nothing for a chain to share.
        ("antichain (L = k)", [random_text(6, "abcde", rng) for _ in range(22)],
         random_text(n, "abcde", rng)),
        # The realistic middle: shared stems, some words prefixes of others.
        ("word list (stems)", words, prose),
    ]

    print(f"\ntext = {n:,} characters\n")
    print(f"{'dictionary':<24} {'k':>4} {'L':>4} {'MultiZ':>9} {'k-scan':>9} "
          f"{'Aho-Cor.':>9} {'occ':>8}")
    for name, pats, text in cases:
        mz = MultiZMatcher(pats)
        ac = AhoCorasick(pats)
        t_mz, got_mz = timed(lambda: sorted((p, i) for i, p in mz.search(text)), repeat=1)
        t_naive, got_naive = timed(lambda: naive_multi(pats, text), repeat=1)
        t_ac, got_ac = timed(lambda: sorted((p, i) for i, p in ac.search(text)), repeat=1)
        assert got_mz == got_naive == got_ac, name
        print(f"{name:<24} {len(pats):4d} {mz.chain_count:4d} {t_mz:9.4f} "
              f"{t_naive:9.4f} {t_ac:9.4f} {len(got_mz):8,d}")
    print("\nRead the L column first. On the prefix chain, 22 patterns cost one")
    print("scan, and the chain matcher is an order of magnitude faster than the")
    print("obvious per-pattern loop -- the k in O(k*n) really did become L. On the")
    print("antichain there is nothing to share, L = k, and the singleton-chain fast")
    print("path makes it exactly the per-pattern loop again: never worse, which is")
    print("the point. Aho-Corasick wins the wall clock in every row, and should:")
    print("one dict lookup per character beats even one tight list scan. What the")
    print("chain decomposition buys is that a Z-array *stays* usable as k grows,")
    print("with no automaton to build and no memory proportional to the dictionary.")


def bench_tandem(quick: bool) -> None:
    print("\n" + "=" * 74)
    print("4. Main-Lorentz: O(n log n) runs, encoding up to O(n^2) squares")
    print("=" * 74)
    rng = random.Random(4)
    sizes = [2000, 8000] if quick else [2000, 8000, 32000]
    print(f"\n{'input':<20} {'n':>7} {'seconds':>9} {'runs':>10} {'squares':>14} "
          f"{'runs / n log n':>15}")
    for n in sizes:
        cases = {
            "a^n": "a" * n,
            "fibonacci word": _fib_word(n),
            "random binary": random_text(n, "ab", rng),
            "random 26 letters": random_text(n, string.ascii_lowercase, rng),
        }
        for name, s in cases.items():
            t, runs = timed(tandem_repeat_runs, s, repeat=1)
            squares = count_tandem_repeats(s)
            nlogn = n * math.log2(n)
            print(f"{name:<20} {n:7,d} {t:9.4f} {len(runs):10,d} {squares:14,d} "
                  f"{len(runs) / nlogn:15.4f}")
    print("\nThe runs column stays proportional to n log n; the squares column on")
    print("a^n is floor(n/2)*ceil(n/2), which is why the run encoding is not a")
    print("convenience but the only way the function can meet its own bound.")


def _fib_word(n: int) -> str:
    a, b = "b", "a"
    while len(b) < n:
        a, b = b, b + a
    return b[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Z-algorithm benchmarks")
    parser.add_argument("--quick", action="store_true", help="smaller sizes")
    parser.add_argument(
        "--only",
        choices=("construction", "search", "multi", "tandem"),
        help="run a single section",
    )
    args = parser.parse_args()
    sections = {
        "construction": bench_construction,
        "search": bench_search,
        "multi": bench_multi,
        "tandem": bench_tandem,
    }
    for name, fn in sections.items():
        if args.only in (None, name):
            fn(args.quick)
    print()


if __name__ == "__main__":
    main()
