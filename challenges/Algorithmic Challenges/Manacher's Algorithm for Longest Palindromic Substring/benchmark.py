"""Benchmarks: O(n) against O(n^2), and where the naive method is fine anyway.

The brief asks for Manacher against the naive baseline. The interesting part
is not that linear beats quadratic -- it does, enormously -- but *where*, and
why the answer depends so heavily on the input:

1. **Scaling on random text.** On a large alphabet, palindromes are short, so
   the naive expand-around-centre loop almost never runs and its O(n^2) never
   materialises. It stays competitive far past where the complexity classes
   say it should not.
2. **Scaling on the worst case.** A run of equal characters makes every one
   of the 2n-1 naive expansions run to full width. Here the gap is the real
   n / 2 factor, and it grows without bound.
3. **Alphabet size.** The sweep between those two, which is the actual
   variable: palindrome density is what decides whether O(n^2) is O(n^2).
4. **Step counts, not clocks.** Expansions performed by each method, which is
   the machine-independent statement of the same result and matches the
   amortisation proof exactly.
5. **The structures.** Eertree construction, O(1) substring queries, and the
   O(n log n) versus O(n^2) palindromic factorisation.

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

from palindromes import (
    Eertree,
    PalindromeIndex,
    count_distinct_palindromes,
    count_palindromic_substrings,
    dp_longest_palindrome_span,
    longest_palindrome_span,
    min_palindromic_partition,
    naive_longest_palindrome_span,
    palindromic_partition,
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


def random_text(n: int, alphabet: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(alphabet) for _ in range(n))


# ---------------------------------------------------------------------------
# 1 & 2. Scaling, on the best and worst inputs for the naive method
# ---------------------------------------------------------------------------


def bench_scaling(sizes: list[int]) -> None:
    for label, build, ceiling in [
        ("random over 26 letters (naive's best case)",
         lambda n: random_text(n, string.ascii_lowercase, seed=n), 10**9),
        ("a run of one character (naive's worst case)",
         lambda n: "a" * n, 200_000),
    ]:
        print(f"\n== {label} ==")
        header = (f"{'n':>10} {'manacher':>11} {'naive':>11} {'ratio':>8} "
                  f"{'longest':>9}")
        print(header)
        print("-" * len(header))
        for n in sizes:
            text = build(n)
            fast, span = timed(longest_palindrome_span, text, repeat=1 if n > 10**5 else 3)
            if n <= ceiling:
                slow, naive_span = timed(naive_longest_palindrome_span, text,
                                         repeat=1 if n > 10**5 else 3)
                assert naive_span[1] - naive_span[0] == span[1] - span[0]
                ratio = f"{slow / fast:>8.1f}x"
                slow_cell = f"{slow:>11.4f}"
            else:
                ratio, slow_cell = f"{'--':>8}", f"{'skipped':>11}"
            print(f"{n:>10} {fast:>11.4f} {slow_cell} {ratio} {span[1] - span[0]:>9}")

    print("\nThe two tables are the same two implementations. On random text over a")
    print("large alphabet the naive method is not merely competitive, it is *faster*:")
    print("the longest palindrome is O(log n), so its inner loop barely runs, while")
    print("Manacher pays mirror-lookup bookkeeping at every centre for a saving it")
    print("never collects. On a run of equal characters the ratio is proportional to")
    print("n and unbounded. An O(n^2) algorithm is only O(n^2) on the inputs that")
    print("make it so -- and the reason to ship the linear one anyway is that its")
    print("loss is bounded and the naive method's is not.")


# ---------------------------------------------------------------------------
# 3. Alphabet size is the hidden variable
# ---------------------------------------------------------------------------


def bench_alphabet(n: int = 100_000) -> None:
    print(f"\n== palindrome density vs alphabet size, n = {n} ==")
    header = (f"{'|alphabet|':>11} {'longest':>9} {'occurrences':>13} "
              f"{'distinct':>9} {'manacher':>10} {'naive':>10} {'ratio':>8}")
    print(header)
    print("-" * len(header))
    for size in (1, 2, 3, 4, 8, 26, 256):
        alphabet = "".join(chr(97 + i % 26) + chr(65 + i // 26) for i in range(size))[:size]
        alphabet = alphabet or "a"
        text = random_text(n, alphabet, seed=size)
        fast, span = timed(longest_palindrome_span, text, repeat=2)
        slow, _ = timed(naive_longest_palindrome_span, text, repeat=1)
        print(f"{size:>11} {span[1] - span[0]:>9} "
              f"{count_palindromic_substrings(text):>13,} "
              f"{count_distinct_palindromes(text):>9,} "
              f"{fast:>10.4f} {slow:>10.4f} {slow / fast:>7.1f}x")
    print("\nThe expected longest palindrome in random text over an alphabet of size")
    print("k is about 2*log_k(n), so it collapses the moment k > 1 -- and the naive")
    print("method's cost collapses with it. This is not a gradient, it is a cliff:")
    print("the entire difference lives between one letter and two. Past that the")
    print("naive method is consistently a little *faster* than Manacher, because")
    print("Manacher's mirror bookkeeping costs more than the expansions it saves.")


# ---------------------------------------------------------------------------
# 4. Step counts: the machine-independent version of the same claim
# ---------------------------------------------------------------------------


def count_manacher_steps(s: str) -> int:
    """Inner-loop character comparisons Manacher performs. Must be <= 2n."""
    n = len(s)
    d1, d2 = [0] * n, [0] * n
    steps = 0
    l, r = 0, -1
    for i in range(n):
        k = 1 if i > r else min(d1[l + r - i], r - i + 1)
        while i - k >= 0 and i + k < n and s[i - k] == s[i + k]:
            k += 1
            steps += 1
        d1[i] = k
        if i + k - 1 > r:
            l, r = i - k + 1, i + k - 1
    l, r = 0, -1
    for i in range(n):
        k = 0 if i > r else min(d2[l + r - i + 1], r - i + 1)
        while i - k - 1 >= 0 and i + k < n and s[i - k - 1] == s[i + k]:
            k += 1
            steps += 1
        d2[i] = k
        if i + k - 1 > r:
            l, r = i - k, i + k - 1
    return steps


def count_naive_steps(s: str) -> int:
    n = len(s)
    steps = 0
    for centre in range(2 * n - 1):
        i, j = centre // 2, centre // 2 + centre % 2
        while i >= 0 and j < n and s[i] == s[j]:
            i -= 1
            j += 1
            steps += 1
    return steps


def bench_steps(n: int = 20_000) -> None:
    print(f"\n== expansion steps (machine-independent), n = {n} ==")
    header = (f"{'input':>24} {'manacher':>10} {'bound 2n':>10} {'naive':>12} "
              f"{'naive/n':>9}")
    print(header)
    print("-" * len(header))
    inputs = {
        "all one character": "a" * n,
        "alternating ab": "ab" * (n // 2),
        "random binary": random_text(n, "ab", seed=1),
        "random 26 letters": random_text(n, string.ascii_lowercase, seed=2),
        "one giant palindrome": "x" * (n // 2) + "y" + "x" * (n // 2 - 1),
        "abacaba fractal": _fractal(n),
    }
    for label, text in inputs.items():
        m, nv = count_manacher_steps(text), count_naive_steps(text)
        print(f"{label:>24} {m:>10,} {2 * len(text):>10,} {nv:>12,} {nv / len(text):>9.1f}")
    print("\nThe `manacher` column never exceeds `bound 2n`, which is the amortisation")
    print("argument as a measurement: every inner-loop step pushes the right boundary")
    print("one place right, and it never moves left or past the end. The naive column")
    print("has no such bound -- `naive/n` is its effective per-character cost, and on")
    print("the degenerate inputs it is proportional to n.")


def _fractal(n: int) -> str:
    """a, aba, abacaba, ... truncated to n: palindromes at every scale."""
    s = "a"
    for c in string.ascii_lowercase[1:]:
        if len(s) >= n:
            break
        s = s + c + s
    return s[:n]


# ---------------------------------------------------------------------------
# 5. The structures built on top
# ---------------------------------------------------------------------------


def bench_structures(n: int = 200_000) -> None:
    print(f"\n== derived structures, n = {n} ==")
    for label, text in [
        ("random binary", random_text(n, "ab", seed=3)),
        ("random 26 letters", random_text(n, string.ascii_lowercase, seed=4)),
        ("all one character", "a" * n),
    ]:
        print(f"\n  {label}:")
        for name, fn in [
            ("manacher_odd_even (via index)", lambda t=text: PalindromeIndex(t)),
            ("count_palindromic_substrings", lambda t=text: count_palindromic_substrings(t)),
            ("Eertree build", lambda t=text: Eertree(t)),
            ("count_distinct_palindromes", lambda t=text: count_distinct_palindromes(t)),
            ("min_palindromic_partition", lambda t=text: min_palindromic_partition(t)),
        ]:
            secs, _ = timed(fn, repeat=1)
            print(f"    {name:>30}: {secs:7.3f}s  ({secs / n * 1e9:6.0f} ns/char)")

    idx = PalindromeIndex(random_text(n, "ab", seed=5))
    rng = random.Random(6)
    queries = [(lambda a, b: (min(a, b), max(a, b)))(rng.randrange(n), rng.randrange(n))
               for _ in range(200_000)]
    secs, _ = timed(lambda: [idx.is_palindrome(i, j) for i, j in queries], repeat=3)
    print(f"\n  is_palindrome: {secs / len(queries) * 1e9:.0f} ns/query over "
          f"{len(queries)} random spans")
    print("  (that is one list index and one comparison -- the O(n) build is what")
    print("   buys it, and it is the same build the longest-substring answer needs)")


def bench_partition(sizes: list[int]) -> None:
    print("\n== palindromic factorisation: O(n log n) vs O(n^2) ==")
    header = f"{'n':>8} {'series links':>14} {'DP + O(1) test':>16} {'ratio':>8} {'pieces':>8}"
    print(header)
    print("-" * len(header))
    for n in sizes:
        text = random_text(n, "ab", seed=n)
        fast, pieces = timed(min_palindromic_partition, text, repeat=2)
        if n <= 20_000:
            slow, parts = timed(palindromic_partition, text, repeat=1)
            assert len(parts) == pieces
            cells = f"{slow:>16.4f} {slow / fast:>7.1f}x"
        else:
            cells = f"{'skipped':>16} {'--':>8}"
        print(f"{n:>8} {fast:>14.4f} {cells} {pieces:>8}")
    print("\nThe DP is O(n^2) even with O(1) palindrome tests from the radii array;")
    print("the series-link method is O(n log n) because the palindromic suffixes of")
    print("any position fall into O(log n) arithmetic progressions -- a consequence")
    print("of the Fine and Wilf periodicity lemma, which the eertree materialises")
    print("as its series links. `palindromic_partition` keeps the DP only because")
    print("reconstructing the actual pieces needs the predecessor chain.")


# ---------------------------------------------------------------------------


def bench_dp_memory(n: int = 4000) -> None:
    print(f"\n== the O(n^2)-space DP, n = {n} ==")
    secs, span = timed(dp_longest_palindrome_span, "ab" * (n // 2), repeat=1)
    fast, _ = timed(longest_palindrome_span, "ab" * (n // 2), repeat=3)
    print(f"  dp_longest_palindrome_span: {secs:.3f}s, table is {n}^2 = {n*n:,} booleans")
    print(f"  longest_palindrome_span:    {fast:.4f}s, arrays are 2n = {2*n:,} ints")
    print(f"  ratio: {secs / fast:.0f}x slower, {n // 2:,}x more memory")
    print("\n  The DP is the version most tutorials teach first. At n = 100000 its")
    print("  table would be 10^10 booleans -- it is not a slower way to get the")
    print("  answer, it is a way of not getting the answer at all.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smaller inputs, ~20s")
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument(
        "--only", nargs="+",
        choices=["scaling", "alphabet", "steps", "structures", "partition", "dp"],
    )
    args = parser.parse_args(argv)

    if args.sizes:
        sizes = args.sizes
    elif args.quick:
        sizes = [1_000, 10_000, 50_000]
    else:
        sizes = [1_000, 10_000, 100_000, 1_000_000]
    selected = set(args.only) if args.only else None

    def run(name, fn, *a):
        if selected is None or name in selected:
            fn(*a)

    big = min(sizes[-1], 200_000)
    run("scaling", bench_scaling, sizes)
    run("alphabet", bench_alphabet, min(sizes[-1], 100_000))
    run("steps", bench_steps, 5_000 if args.quick else 20_000)
    run("structures", bench_structures, big)
    run("partition", bench_partition, [s for s in sizes if s <= 100_000])
    run("dp", bench_dp_memory, 2_000 if args.quick else 4_000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
