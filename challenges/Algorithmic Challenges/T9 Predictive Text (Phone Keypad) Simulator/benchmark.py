"""Benchmarks: full-sequence lookup vs incremental typing, three ways.

Two separate questions, because they have different answers:

1. **One-shot full-sequence lookup**, as vocabulary size grows. `HashmapT9`
   should win here -- O(1) average against the trie's O(L) descent and
   naive's O(n*L) scan -- and the point of measuring it is to confirm that
   plain advantage before asking the more interesting question.
2. **Incremental typing**: the cost of getting suggestions after *each* of
   k keystrokes, which is the actual T9 UX. `TrieT9`'s `TypingSession` only
   ever follows one more edge per key. `HashmapT9` has no prefix structure at
   all, so getting a prefix answer out of it means scanning every key it
   holds (`predict_slow`) -- and naive means rescanning the whole vocabulary.
   This is where the trie's tree structure pays for itself, even though it
   loses question 1.

    uv run --with wordfreq python benchmark.py
    uv run --with wordfreq python benchmark.py --sizes 5000 20000 --quick
"""

from __future__ import annotations

import argparse
import gc
import math
import random
import time
from collections.abc import Callable

from t9 import (
    HashmapT9,
    TrieT9,
    build_vocabulary,
    naive_lookup,
    naive_prefix_lookup,
    word_to_digits,
)


def timed(fn: Callable[[], object], repeat: int = 5) -> float:
    """Best-of-`repeat` wall time in seconds, with the GC held quiet."""
    best = math.inf
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeat):
            start = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - start)
    finally:
        if was_enabled:
            gc.enable()
    return best


# ---------------------------------------------------------------------------
# 1. Full-sequence lookup latency, as vocabulary size grows
# ---------------------------------------------------------------------------


def bench_full_lookup(sizes: list[int], queries: int, repeat: int) -> None:
    print("Full-sequence lookup: one query, whole digit sequence given up front")
    print(f"{'n words':>10}{'naive':>14}{'hashmap':>14}{'trie':>14}   winner")
    print("-" * 70)
    rng = random.Random(0)
    for size in sizes:
        vocab = build_vocabulary(size)
        words = [w for w, _f in vocab]
        probes = [word_to_digits(rng.choice(words)) for _ in range(queries)]
        hashmap = HashmapT9(vocab)
        trie = TrieT9(vocab)

        t_naive = timed(
            lambda v=vocab, p=probes: [naive_lookup(v, d) for d in p], repeat=repeat
        )
        t_hash = timed(
            lambda h=hashmap, p=probes: [h.lookup(d) for d in p], repeat=repeat
        )
        t_trie = timed(
            lambda t=trie, p=probes: [t.lookup_exact(d) for d in p], repeat=repeat
        )

        per = {
            "naive": t_naive / queries,
            "hashmap": t_hash / queries,
            "trie": t_trie / queries,
        }
        winner = min(per, key=per.__getitem__)
        print(
            f"{size:>10,}{per['naive'] * 1e6:>12.2f}us{per['hashmap'] * 1e6:>12.2f}us"
            f"{per['trie'] * 1e6:>12.2f}us   {winner}"
        )
    print()
    print("naive is always last, by orders of magnitude, and gets worse with n --")
    print("exactly the O(n*L) rescan its docstring promises. HashmapT9 and TrieT9")
    print("are both sub-2us here (one dict lookup vs. descending ~5-8 trie levels)")
    print("so which one 'wins' at this scale is mostly measurement noise; the real")
    print("separation between them shows up below, on incremental typing.")


# ---------------------------------------------------------------------------
# 2. Incremental typing: cost per keystroke, and cost of the whole word
# ---------------------------------------------------------------------------


def bench_incremental_typing(
    vocab_size: int, sample_words: int, top_k: int, repeat: int
) -> None:
    vocab = build_vocabulary(vocab_size)
    words = [w for w, _f in vocab]
    rng = random.Random(1)
    probes = [word_to_digits(w) for w in rng.sample(words, sample_words)]

    hashmap = HashmapT9(vocab)
    trie = TrieT9(vocab, top_k=top_k)

    def type_out_trie() -> None:
        for sig in probes:
            session = trie.session()
            for digit in sig:
                session.type_digit(digit)

    def type_out_hashmap() -> None:
        for sig in probes:
            for i in range(1, len(sig) + 1):
                hashmap.predict_slow(sig[:i], limit=top_k)

    def type_out_naive() -> None:
        for sig in probes:
            for i in range(1, len(sig) + 1):
                naive_prefix_lookup(vocab, sig[:i], limit=top_k)

    total_keystrokes = sum(len(sig) for sig in probes)

    print()
    print(
        f"Incremental typing: {sample_words} real words, {total_keystrokes} keystrokes total"
    )
    print(f"(vocabulary size {vocab_size:,}, trie top_k={top_k})")
    print(f"{'method':<12}{'total time':>14}{'per keystroke':>18}")
    print("-" * 46)
    for name, fn in (
        ("trie", type_out_trie),
        ("hashmap", type_out_hashmap),
        ("naive", type_out_naive),
    ):
        t = timed(fn, repeat=repeat)
        print(f"{name:<12}{t * 1000:>12.3f}ms{(t / total_keystrokes) * 1e6:>16.2f}us")

    print()
    print("The trie's per-keystroke cost should be flat regardless of vocabulary")
    print("size (it only ever follows one more edge); hashmap and naive both pay")
    print("for a fresh scan on every single keystroke of every word.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[2_000, 10_000, 30_000],
        help="vocabulary sizes",
    )
    parser.add_argument(
        "--queries", type=int, default=300, help="full-lookup queries per size"
    )
    parser.add_argument(
        "--repeat", type=int, default=5, help="best-of-N timing repeats"
    )
    parser.add_argument(
        "--incremental-vocab",
        type=int,
        default=30_000,
        help="vocab size for the typing benchmark",
    )
    parser.add_argument(
        "--sample-words",
        type=int,
        default=40,
        help="words to type out for the incremental benchmark",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--quick", action="store_true", help="smaller repeat/sample counts"
    )
    args = parser.parse_args(argv)

    repeat = 2 if args.quick else args.repeat
    queries = 100 if args.quick else args.queries
    sample_words = 15 if args.quick else args.sample_words

    bench_full_lookup(args.sizes, queries, repeat)
    bench_incremental_typing(args.incremental_vocab, sample_words, args.top_k, repeat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
