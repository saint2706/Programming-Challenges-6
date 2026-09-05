# Exact String Search Benchmark

**Category:** Algorithmic Challenges
**Difficulty:** I (brief: "Naive vs KMP vs Boyer-Moore vs Rabin-Karp, same corpus")

**Status:** Implemented (Python)

Four algorithms on one corpus is the brief. Running it produces a result the
brief does not anticipate: **the ranking among those four barely matters,
because all four lose to `str.find` by one to two orders of magnitude.**

So this is two benchmarks, kept apart on purpose:

| Question | Answer lives in | What it measures |
| --- | --- | --- |
| How many characters does each method *look at*? | §1, §3, §4 | The algorithm |
| How long does it *take*? | §2, §5 | The interpreter |

They disagree, and the disagreement is the finding. Boyer-Moore inspects the
fewest characters of anything here — about `n/14` of them on 128-character
English patterns — and is among the *slowest* in seconds, because each skip it
computes costs several bytecodes while the naive loop's comparison costs one.
Skipping work only pays when the work you skip is more expensive than the
arithmetic that decides to skip it.

Eleven methods are implemented: the four from the brief, plus Horspool,
Sunday, Two-Way (Crochemore-Perrin), Shift-Or, a Boyer-Moore variant *without*
the Galil rule so its contribution can be isolated, `str.find` in a loop, and
one that is not in any textbook — see §5.

## Instrumentation without a second copy of the code

Every algorithm reaches the text only through `text[i]` and `len(text)`.
So `count_accesses` wraps the text in a proxy that counts `__getitem__` and
runs the *unmodified* algorithm:

```python
class _CountingSequence:
    def __getitem__(self, index):
        self.count += 1
        return self._data[index]
```

One implementation each, exact instrumentation, and no second copy to drift
out of sync. `test_counting_proxy_sees_every_access` checks the proxied run
returns the same matches for all nine countable methods.
`bitparallel` and `builtin` are excluded, not forgotten: their scanning
happens inside C primitives that consume the whole text at once, so
"characters inspected" is not a quantity they expose and reporting one would
be inventing a comparable number where none exists.

## 1. What the complexity classes look like when you measure them

Accesses divided by `n`, so `1.00x` means "read the text once".

```
  english prose, n = 59,999
      m        naive          kmp  boyer-moore     horspool       sunday   rabin-karp      two-way
      2        1.10x        1.00x        1.12x        1.20x        0.89x        2.15x        1.00x
      8        1.07x        1.00x        0.48x        0.57x        0.39x        2.00x        0.96x
     32        1.02x        1.00x        0.16x        0.16x        0.15x        2.00x        1.00x
    128        1.08x        1.00x        0.07x        0.11x        0.09x        2.00x        1.00x

  bytes (256), n = 60,000
    128        1.00x        1.00x        0.02x        0.02x        0.02x        2.00x        1.00x

  one letter (1), n = 4,000
    128      123.94x        1.00x        1.00x      124.90x      124.90x      125.90x        1.00x
```

Everything textbook is visible and exact:

- **KMP is pinned at 1.00x**, everywhere, on every corpus. That is its whole
  character: it never re-reads and it never skips, so it cannot go above 1.00
  (well, above 2.00 with the `while` retries — `test_kmp_never_exceeds_two_n_accesses`)
  and it cannot go below.
- **The Boyer-Moore family falls with `m` and with alphabet size**, to 0.02x
  on 256-symbol input. That is the sublinear average case, and it is the
  entire reason the family exists.
- **Rabin-Karp is pinned at 2.00x** — one read to add a character to the
  rolling hash, one to remove it — regardless of everything else.
- **The unary row is where four of them go quadratic** and two do not.

## 2. The same grid in seconds, which says something else

```
  english prose, n = 400,000
      m        naive          kmp  boyer-moore     horspool       sunday   rabin-karp      two-way     shift-or  bitparallel      builtin
      8      0.0219s      0.0189s      0.0118s      0.0069s      0.0074s      0.1094s      0.0215s      0.0295s      0.0040s      0.0003s
     64      0.0184s      0.0160s      0.0036s      0.0023s      0.0027s      0.1096s      0.0219s      0.0327s      0.0114s      0.0001s

  dna (4), n = 400,000
      8      0.0226s      0.0195s      0.0170s      0.0119s      0.0128s      0.1179s      0.0381s      0.0288s      0.0025s      0.0007s
     64      0.0223s      0.0195s      0.0096s      0.0108s      0.0132s      0.1146s      0.0195s      0.0334s      0.0037s      0.0005s
```

Four things the access table did not predict:

1. **`builtin` wins every row**, by 4× to 100×. CPython's `fastsearch` is
   Crochemore-Perrin Two-Way for long patterns with a Horspool skip loop and a
   Bloom filter for short ones — so the `builtin` column is roughly the
   `two-way` and `horspool` columns written in C. The gap between them is the
   interpreter, not the algorithm.
2. **Horspool beats Boyer-Moore** on prose and DNA, despite inspecting more
   characters. Horspool is Boyer-Moore with the good-suffix table deleted; the
   deleted table produces a minority of the skips and costs a lookup plus a
   `max` at every mismatch. Deleting it was the right call in 1980 and still is.
3. **Rabin-Karp is last by 5×** while inspecting only `2n` characters, because
   each of those characters costs an `ord`, a multiply, a subtract and a
   modulo. Access counts are not a proxy for time when the per-access work
   differs by 20×.
4. **Two-Way is slower than naive in Python.** Its selling point is O(1) space
   with no preprocessing table — priceless in a C library called for every
   `in` and `find`, worth nothing when the table you avoided allocating was a
   Python dict you were going to build once.

## 3. Sublinearity, against the bound

Yao (1979) proved that any exact matcher must inspect
Ω(n·log_σ(m)/m) characters in expectation. The last column is that shape:

```
      m        naive          kmp  boyer-moore     horspool       sunday      two-way  log_s(m)/m
      2       1.111x       1.000x       1.176x       1.201x       0.872x       1.000x      0.1091
      8       1.084x       1.000x       0.399x       0.426x       0.346x       1.000x      0.0818
     32       1.181x       1.000x       0.175x       0.188x       0.179x       1.000x      0.0341
    128       1.139x       1.000x       0.109x       0.109x       0.105x       0.999x      0.0119
    512       1.015x       1.000x       0.074x       0.097x       0.086x       0.996x      0.0038
```

The Boyer-Moore family tracks the bound's *shape* within a constant of about
20 and keeps falling; naive, KMP and Two-Way sit at 1.0 forever, because
neither has any mechanism for not looking at a character. That is the honest
statement of what the skip tables buy — and the constant is why it does not
translate into wall-clock wins in an interpreter.

## 4. The worst cases, constructed rather than hoped for

Each of the three is built by a function in the module, so the claim is
reproducible rather than asserted.

**(a) `naive_adversary`: `a^(m-1) b` inside `a^n`.** Every alignment matches
`m-1` characters and then fails, and none is a match.

```
         n          naive       /n          kmp      /n   boyer-moore      /n
    32,000      1,023,008     32.0       32,000     1.0        63,938     2.0
```

**(b) `boyer_moore_adversary`: `a^m` inside `a^n`.** Every alignment is a full
match, so the scan reads all `m` characters and the good-suffix shift after a
match is the period, which is 1.

```
         n     BM + Galil       /n   BM, no Galil       /n    ratio
    32,000         32,000      1.0      1,023,008     32.0    32.0x
```

The **Galil rule** is one integer of state: after a match the pattern shifts by
its own period `p`, so the leading `m − p` characters of the next alignment are
*already known* to match and the scan can stop at index `m − p`. That is the
whole difference between O(n) and O(nm), and it is missing from most
Boyer-Moore code you will find.

**(c) `rabin_karp_adversary`: hash flooding.** Pick two characters whose
ordinals are congruent mod the modulus — `chr(1)` and `chr(1 + mod)`. Each
contributes the same residue at every position, so *any* string over those two
characters hashes like *any other* of the same length, and in particular like
the all-`chr(1)` pattern. Every window collides.

Colliding is not enough on its own: verification stops at the first mismatch,
so a random mix fails in O(1). The text is therefore blocks of
`low^(m-1) + high`, so every window agrees with the pattern on ~`m/2`
characters before failing. Collision *plus* a long common prefix is what makes
it quadratic.

```
         n        mod 127       /n     mod 2^61-1       /n    ratio
    32,000        591,968     18.5         64,992      2.0     9.1x
```

Same text, same algorithm; only the modulus changed. That is the point: a
fixed small modulus in a public codebase is a denial-of-service surface. The
default here is `2^61 − 1`, wide enough that no two code points are congruent
so the construction cannot even be built (`test_rabin_karp_adversary_rejects_impossible_moduli`);
`rabin_karp_randomized_search` draws base and modulus per call, which moves the
quadratic case from a property of the *input* to a property of an unlucky draw
that no attacker can influence.

Note also that Rabin-Karp here is **always exact**. The rolling hash is a
filter, not the answer — every hit is verified. Collisions cost time, never
correctness, and `test_rabin_karp_is_exact_under_a_tiny_modulus` runs it at
`mod=7` to prove it.

## 5. Transposing the loop: the method that is not in the textbook

Every algorithm above has the same shape — a loop over the text, in the
interpreter. In CPython that loop is the cost, so the way to win is to not
have it.

For each *distinct* character `c` of the pattern, build one integer `B[c]`
whose bit `i` is set exactly when `text[i] == c`. Then

```
M = AND over k in 0..m-1 of (B[pattern[k]] >> k)
```

and bit `j` of `M` is set iff `text[j+k] == pattern[k]` for every `k` — that
is, iff the pattern occurs at `j`. Shifting right by `k` also drops the bits
that would run off the end, so there is no boundary case.

This is Shift-Or with the loop transposed: the text axis lives *inside* the
machine word instead of in the loop, so the number of interpreted operations
is O(σ_P + m) **regardless of n**. The work is O((σ_P + m)·n/w) bit
operations, all inside CPython's bignum routines. More total bit operations,
vastly fewer interpreter dispatches — a trade that is a clear win in Python
and would be pointless in C.

The masks are built with C primitives too: `bytes.translate` into a string of
`'0'`/`'1'` and then `int(s, 2)`, which is linear because base 2 is a power of
two. A `str` is re-encoded to Latin-1 first when it fits — index-preserving,
so no offset remapping — and falls back to `split`/`join` when it does not.
Reporting the positions skips whole zero bytes with a regex scan for
`[^\x00]`, so even the output loop is mostly in C.

```
  (a) Sparse matches: a random 16-mer in random DNA.
           n   bitparallel       builtin       two-way      horspool      occ
   1,000,000       0.0086s       0.0018s       0.0569s       0.0335s        1
   4,000,000       0.0319s       0.0056s       skipped       skipped        1

  (b) Dense overlapping matches: a^8 inside a^n.
           n   bitparallel       builtin   speedup          occ
   1,000,000       0.0509s       0.1257s     2.47x      999,993
   4,000,000       0.2743s       0.6124s     2.23x    3,999,993
```

On sparse matches `builtin` still wins — but the interpreted matchers are
another 5-10× behind, so among things you can *write* this is the one to
write. On dense overlapping matches the ordering flips: `str.find` reports one
match per interpreted iteration and this reports all of them from one integer.
It is the only method here that ever beats the C baseline.

The cost model is `σ_P` mask builds plus `m` shift-and-AND operations, which
makes it best on small alphabets — DNA, binary, log levels, byte protocols —
and worst on a long pattern of mostly-distinct characters (the `bytes (256)`,
`m = 64` cell in §2, where it pays 64 mask builds and loses to everything).

## Edge cases the tests pin down

- The **empty pattern** matches at each of the `n + 1` gaps, `n` included — the
  `str.find("") == 0` convention. All eleven agree.
- **Overlapping occurrences are all reported**, in increasing order. This is
  not the default for several textbook presentations, which shift by `m` after
  a match; `test_overlapping_matches_all_reported` and the periodic-pattern
  tests cover the cases where that goes wrong.
- **Every method is a generator**, so `next()` costs one match rather than all
  of them — checked against a 100,000-character text.
- **Periodic patterns** (`abab`, `aabaab`, `abaaba`) get their own test,
  because they are where the Two-Way periodic branch and the Galil rule both
  live and both have off-by-one traps.
- **Any sequence works** for the nine pure-Python methods: `str`, `bytes`,
  `list`, `tuple`. `bitparallel` and `builtin` fall back to Two-Way for
  sequences with no C primitive to exploit.
- **Unicode beyond Latin-1** forces `bitparallel` off its fast path onto
  `split`/`join`; astral-plane characters and CJK are both tested.
- **Latin-1 narrowing preserves indices** — one byte per code point — so
  `"café au lait, café noir"` reports 0 and 14, not byte offsets.
- Exhaustive agreement: every method against brute force on **every binary
  pattern up to length 4 × every binary text up to length 8**, and every
  ternary pattern up to length 3 × every ternary text up to length 6.

## Where this is actually used

**The practical answer is: use the one in your standard library.** Section 2
is unambiguous, and the reason to know the algorithms anyway is that the
standard library's version keeps running out.

- **When you cannot use `find`.** Streaming input that never exists as one
  buffer; a custom equality (case-insensitive, IUPAC nucleotide codes where
  `N` matches anything, tokens rather than characters); a `memoryview` over
  shared memory. Then you are writing the loop, and which loop matters.
- **When the pattern is reused thousands of times.** `find` rebuilds its
  factorization on every call. A compiled matcher amortises preprocessing —
  which is exactly why `re.compile` exists.
- **grep and friends.** GNU grep's speed is famously Boyer-Moore plus "avoid
  looking at bytes you do not have to"; `ripgrep` and Hyperscan replace the
  skip loop with SIMD but keep the same shape. The skip-table family is the
  reason a full-text search over a gigabyte finishes.
- **Genomics.** `dna (4)` in the tables is not decoration. Read alignment,
  primer and restriction-site location, and adapter trimming are exact
  matching over a 4-symbol alphabet, at scale — the regime where the
  bit-parallel method's `σ_P` cost model is at its best, and where `bowtie`
  and `bwa` use bit-parallel and FM-index techniques for the same reason.
- **Intrusion detection and DPI.** Snort and Suricata match thousands of
  signatures against every packet; that is Aho-Corasick territory (challenge
  9) built on the same primitives, and hash-flooding resistance (§4c) is a
  live concern rather than a thought experiment.

**The security lesson generalises past Rabin-Karp.** Any algorithm whose
performance depends on a *public, fixed* hash is attackable: the same argument
sank Python's own `dict` in 2011 (CVE-2012-1150) and produced
`PYTHONHASHSEED`. The fix is the same each time — widen the hash, or make it
per-process and unpredictable — and it is cheaper than the incident.

**And the honest closing note:** for one search over one buffer, everything in
this module is the wrong answer and `str.find` is the right one. The value of
having written them is knowing *which* one to reach for on the day `find` does
not fit, and being able to say why rather than guessing.

## Files

| File | What it is |
| --- | --- |
| `stringsearch.py` | Eleven matchers, the counting proxy, three adversary generators, CLI |
| `test_stringsearch.py` | 69 tests: exhaustive agreement over binary and ternary inputs, the worst cases as assertions, Unicode, CLI |
| `benchmark.py` | Accesses and seconds over five corpora, sublinearity against Yao's bound, the three worst cases, the bit-parallel sweep |

## Running it

```bash
uv run python stringsearch.py --demo
uv run python stringsearch.py --verify
uv run python stringsearch.py --list
uv run python stringsearch.py --pattern ana --text banana
uv run python stringsearch.py --pattern ana --text banana --algorithm boyer-moore

uv run --with pytest pytest -q                 # all 69
uv run --with pytest pytest -q -m "not slow"
uv run python benchmark.py --quick
uv run python benchmark.py --only worstcase
```

No third-party dependencies at all — the module is pure standard library.

## Sources

- Knuth, Morris & Pratt, ["Fast Pattern Matching in Strings"](https://doi.org/10.1137/0206024), *SIAM J. Computing* 6(2), 1977
- Boyer & Moore, ["A Fast String Searching Algorithm"](https://doi.org/10.1145/359842.359859), *CACM* 20(10), 1977
- Galil, ["On improving the worst case running time of the Boyer-Moore string matching algorithm"](https://doi.org/10.1145/359146.359148), *CACM* 22(9), 1979 — the one integer in §4b
- Horspool, ["Practical fast searching in strings"](https://doi.org/10.1002/spe.4380100608), *Software: Practice and Experience* 10(6), 1980
- Karp & Rabin, ["Efficient randomized pattern-matching algorithms"](https://doi.org/10.1147/rd.312.0249), *IBM J. Research and Development* 31(2), 1987
- Sunday, ["A very fast substring search algorithm"](https://doi.org/10.1145/79173.79184), *CACM* 33(8), 1990
- Crochemore & Perrin, ["Two-way string-matching"](https://doi.org/10.1145/116825.116845), *JACM* 38(3), 1991
- Baeza-Yates & Gonnet, ["A new approach to text searching"](https://doi.org/10.1145/135239.135243), *CACM* 35(10), 1992 — Shift-Or
- Yao, ["The complexity of pattern matching for a random string"](https://doi.org/10.1137/0208029), *SIAM J. Computing* 8(3), 1979 — the Ω(n log_σ(m)/m) bound in §3
- Faro & Lecroq, ["The Exact String Matching Problem: a Comprehensive Experimental Evaluation"](https://arxiv.org/abs/1012.2547) (2010), and the [SMART](https://www.dmi.unict.it/faro/smart/) testbed
- CPython, [`Objects/stringlib/fastsearch.h`](https://github.com/python/cpython/blob/main/Objects/stringlib/fastsearch.h) and [`stringlib_find_two_way_notes.txt`](https://github.com/python/cpython/blob/main/Objects/stringlib/stringlib_find_two_way_notes.txt) — what `builtin` actually runs
