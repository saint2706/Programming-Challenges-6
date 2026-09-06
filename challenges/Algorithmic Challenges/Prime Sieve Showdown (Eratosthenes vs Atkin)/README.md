# Prime Sieve Showdown (Eratosthenes vs Atkin)

**Category:** Algorithmic Challenges
**Difficulty:** B (brief: "benchmark both to 10^8, compare memory and wall time")

**Status:** Implemented (Python)

The Sieve of Atkin has strictly better asymptotics than the Sieve of
Eratosthenes — O(N / log log N) operations against O(N log log N). It is also
comprehensively slower in practice, at every size a laptop can reach. This is a
benchmark built to answer *why*, without stacking the deck.

## The verdict first

N = 10^8, Python 3.11 on x86-64, one run each, measured by `benchmark.py`:

| Implementation                    | Family       | Time       | Peak RSS   | bytes/int | Memory | π(N)      |
| --------------------------------- | ------------ | ---------- | ---------- | --------- | ------ | --------- |
| Eratosthenes, wheel-30 segmented  | eratosthenes | **0.34 s** | **6.6 MB** | 0.069     | O(√N)  | 5,761,455 |
| Eratosthenes, wheel-30 packed     | eratosthenes | 0.82 s     | 25.9 MB    | 0.272     | O(N)   | 5,761,455 |
| Eratosthenes, odds-only NumPy     | eratosthenes | 0.82 s     | 47.8 MB    | 0.502     | O(N)   | 5,761,455 |
| Eratosthenes, odds-only bytearray | eratosthenes | 1.24 s     | 66.8 MB    | 0.700     | O(N)   | 5,761,455 |
| Atkin, mod-60 packed NumPy        | atkin        | 3.54 s     | 216.3 MB   | 2.268     | O(N)   | 5,761,455 |
| Atkin, mod-60 packed              | atkin        | 20.99 s    | 25.5 MB    | 0.267     | O(N)   | 5,761,455 |

Atkin's best showing is **10× slower** than Eratosthenes' best, and needs **33×
the memory** to get there. At 10^9 the gap holds in time and widens sharply in
space:

| Implementation                   | Time       | Peak RSS   |
| -------------------------------- | ---------- | ---------- |
| Eratosthenes, wheel-30 segmented | **3.71 s** | **8.9 MB** |
| Eratosthenes, wheel-30 packed    | 9.75 s     | 258.9 MB   |
| Eratosthenes, odds-only NumPy    | 10.75 s    | 477.0 MB   |
| Atkin, mod-60 packed NumPy       | 37.93 s    | 1.9 GB     |

This is not a Python artifact. [primesieve](https://github.com/kimwalisch/primesieve/blob/master/doc/ALGORITHMS.md),
the fastest sieve anyone has written in C++, uses a **segmented Eratosthenes
with wheel factorization** and beats primegen, the fastest published Atkin
implementation. The pattern reproduces here because the reasons are structural,
not incidental.

## Why Atkin loses

**1. Its operations are more expensive than Eratosthenes'.** Eratosthenes'
inner loop is `p += step` and a store — in Python, a strided slice assignment
that becomes one `memset`-shaped C loop. Atkin's inner loop evaluates a
quadratic form, reduces mod 60, tests set membership, computes a packed index,
and flips a bit. Beating an O(log log N) factor requires the constant to be
within that factor, and it isn't close.

**2. Its memory access is scattered; Eratosthenes' is sequential.** Striking
multiples of `p` walks memory forward at a fixed stride — perfectly
prefetchable. Atkin's `n = 4x² + y²` jumps across the array as `y` grows.
Modern hardware pays for that, and the reduced instruction count doesn't buy it
back.

**3. Vectorizing Atkin costs memory.** The NumPy version is 6× faster than the
pure-Python one, but at 2.27 bytes per integer against 0.267 — the temporary
`int64` arrays holding candidate `n` values dwarf the sieve itself. At 10^9
that's 1.9 GB. Speed bought with an order of magnitude more memory is a bad
trade for a sieve, whose whole selling point at scale is that it fits.

**4. Segmentation, which is where the real win lives, favors Eratosthenes.**
Eratosthenes segments trivially: keep base primes up to √N and a rolling
offset per prime. That single change is worth 2.4× in time and 3.9× in memory
at 10^8 — and 2.6× and 29× at 10^9, because the flat array keeps growing and
the segment does not. Atkin can be segmented too (it was born segmented), but
each segment has to re-solve the quadratic forms over its own range, so the
bookkeeping is far heavier per unit of output.

## Making it a fair fight

A benchmark where one side is a textbook transcription and the other is tuned
proves nothing. Both got real engineering:

**Eratosthenes**

- Odds-only (0.5 bytes/int) → wheel-30, one byte per number coprime to 2, 3, 5
  (0.267 bytes/int).
- The identity that makes wheel-30 practical in Python: within a fixed residue
  class `r` mod 30, the multiples of `p` land at *constant stride 8p*.

  ```
  idx(p·(30k + r)) == 8·p·k + idx(p·r)
  ```

  So each prime needs eight strided slice assignments instead of one — but each
  covers 8/30 of the range instead of 1/2, cutting bytes written by 47%. This is
  verified as a property test, not just asserted.
- Segmented variant with per-prime, per-residue rolling offsets.

**Atkin**

- The full mod-60 form, not the simplified mod-12 one: three quadratic forms
  over the sixteen residues coprime to 60, which between them partition every
  residue class a prime above 5 can occupy.
- The same 0.267 bytes/int packing, so the memory comparison is apples to
  apples.
- Parity pruning — each form only produces odd `n`, so only half the `y` values
  are worth enumerating. That halving is most of what keeps it competitive at
  all.
- The squarefree correction pass, which is what makes the parity-of-
  representations argument actually yield primes.

## The one trick that makes NumPy Atkin viable

Atkin *flips* a flag per representation, and flips don't survive fancy
indexing — `flags[idx] ^= True` with duplicate indices in `idx` silently keeps
only one flip. The usual fix, `np.bitwise_xor.at`, is an unbuffered ufunc and
roughly an order of magnitude slower than vectorized code.

The way out: **for a fixed `x`, a form's `n` values are pairwise distinct**
(distinct `y` ⟹ distinct `y²`). So looping over `x` and vectorizing over `y`
makes plain fancy indexing correct — no read-modify-write hazard — and it runs
at full vector speed. That's the difference between a NumPy Atkin that's 6×
faster than pure Python and one that is slower than it.

## Two results that surprised me

**Pure Python beat NumPy.** The segmented `bytearray` sieve is 0.34 s at 10^8;
the NumPy one is 0.82 s. Strided slice assignment on a `bytearray` *is* a C
loop, so there was never an interpreter overhead to remove — and the
`bytearray` version carries a better algorithm (wheel-30 + segmentation) into
that same C loop. Reaching for NumPy is not automatically the optimization.

**The optimal segment size is 64× larger than the textbook says.** C sieves use
L1/L2-sized segments (32–256 KB). Measured here at 10^8:

| Segment   | Time       |
| --------- | ---------- |
| 32 KiB    | 2.78 s     |
| 128 KiB   | 0.85 s     |
| 512 KiB   | 0.31 s     |
| **2 MiB** | **0.25 s** |
| 8 MiB     | 0.35 s     |

An 11× inversion, because in Python the per-segment bookkeeping is *interpreted*
while the striking is `memcpy` in C. Cache locality only starts to matter once
the loop overhead is gone. Ported to C, this advice flips straight back.

There is an honest crossover: below about 2·10^7, segmentation *costs* memory
(two 2 MiB buffers) rather than saving it. It wins from 10^8 up, which is where
it matters.

## One micro-optimization worth stealing

The striking loop needs a run of zero bytes to assign into a strided slice:

```python
sieve[start::step] = bytes(count)          # allocates `count` bytes, every strike
sieve[start::step] = zeros[:count]         # zeros is a memoryview -- O(1) slice
```

Slicing a `memoryview` is a view, not a copy: measured **17× cheaper** than
slicing `bytes`, and it took the segmented sieve from 0.44 s to 0.25 s at 10^8.
The buffer only needs `size / 56` bytes, because the smallest wheel prime is 7
and its stride is 8·7 = 56.

## Measuring memory without lying

Naive in-process benchmarking gets sieve memory wrong twice over. Peak RSS is a
process-wide high-water mark, so a segmented sieve run after a flat one
inherits the flat one's watermark and looks equally hungry. And the allocator
hands a freed 100 MB buffer straight back to the next run, which then never
touches the kernel and looks unfairly fast.

So every measurement runs in a **fresh child process**, which measures its own
footprint two ways and keeps the larger:

- a background thread sampling `/proc/self/statm` every millisecond, and
- the `ru_maxrss` delta.

Neither alone works. `ru_maxrss` is blind to anything below CPython's own
startup peak — a 2.6 MB sieve reads as *exactly zero*, which is how the first
version of this harness quietly reported nonsense. Sampling is blind to spikes
shorter than its interval. Together they agree with theory: wheel-30 measures
0.272 bytes/int against a predicted 8/30 = 0.267.

## Correctness

Every implementation is checked against:

- a trial-division oracle, at **every** limit from 0 to 300 — where off-by-one
  wheel bugs live;
- limits sitting exactly on and around the wheel moduli 30, 60, 210, 2310;
- published π(10^k) values through 10^7 (and 10^8/10^9 in the benchmark itself).

Plus structural property tests: that `W30`/`W60` really are the coprime
residues, that the wheel index identity holds for arbitrary `p`, `r`, `k`, and
that Atkin's three form-residue sets are disjoint and cover the wheel.

## Where this is actually used

**Screening candidates before a primality test.** This one runs in production
constantly, and it is not "generate all primes below N". RSA and Diffie–Hellman
key generation pick a random odd number and test it; a Miller–Rabin round is
expensive, so implementations first trial-divide by a table of small primes to
discard the large majority of candidates that have a tiny factor. Go's
`crypto/rand.Prime` and OpenSSL both carry such a table, and a sieve is how you
build one. The sieve runs once ever; the saving runs on every keygen.

**Windowed sieving, for a range you cannot reach from zero.** `primes_in_range`
is the piece with real reach: sieving [10¹², 10¹² + 10⁶] costs O(√N) memory and
no time proportional to 10¹². That is how distributed prime searches
(PrimeGrid), prime-gap and twin-prime verification, and any "find a prime near
this specific bit length" parameter search actually work.

**Small-prime bases everywhere else in number theory.** Pollard's rho, the
quadratic sieve, ECM and factor-base construction all begin with "give me the
primes below B". So does choosing prime moduli for hash tables, rolling hashes
and CRC polynomials.

**But the transferable result is the benchmark itself.** Three things here
generalize well past primes:

- *Better asymptotics lost, and the reason was memory.* Atkin performs fewer
  operations and runs 10× slower, because its inner loop touches memory in a
  pattern the cache hates while Eratosthenes' is a strided write that compiles
  to something `memset`-shaped. The same shape of result turns up again in
  [Count Inversions](../Count%20Inversions%20in%20an%20Array/), where an
  O(n log n) method loses to an O(n log² n) one for precisely the same reason.
  Complexity classes rank algorithms; the memory hierarchy ranks
  implementations.
- *Segmentation is cache blocking.* The segmented sieve wins because each
  segment fits in L2 — the identical idea as tiled matrix multiplication and
  chunked file processing. It is not a prime-numbers trick.
- *Measuring memory honestly needs a subprocess.* Peak RSS sampled inside the
  process that did the allocating is polluted by the interpreter and by the
  allocator's freelists. Spawning a child and reading its high-water mark is the
  technique, and it applies to any memory claim you want believed.

## Run it

```bash
cd "challenges/Algorithmic Challenges/Prime Sieve Showdown (Eratosthenes vs Atkin)"

uv run --with numpy python sieves.py                       # self-check
uv run --with numpy python benchmark.py                    # the table above, N = 10^8
uv run --with numpy python benchmark.py --limit 1e9 --only era-segmented
uv run --with numpy python benchmark.py --markdown         # README-ready output
uv run --with numpy python benchmark.py --list

uv run --with pytest --with numpy pytest -q -m "not slow"   # 141 tests
uv run --with pytest --with numpy pytest -q                 # + 7 slow ones (to 10^9)
```

NumPy is **optional** — the pure-Python tier is the reference implementation and
needs nothing but the standard library. The harness skips the NumPy entries with
a printed reason rather than failing, and it also skips any implementation
declared too slow for the requested N (pure Atkin above 10^8) instead of
appearing to hang.

For actually using primes rather than counting them:

```python
from sieves import primes_below, primes_in_range, iter_primes

primes_below(100)                 # [2, 3, 5, ..., 97]
next(iter_primes(10**9))          # lazy, segmented, O(√N) memory
primes_in_range(10**12, 10**12 + 100)
# [1000000000039, 1000000000061, 1000000000063, 1000000000091]  -- in 0.5 s
```

`primes_in_range` is what a segmented sieve is actually *for*, and the one thing
`primes_below` cannot do: the primes just above 10^12 cost O(√hi) memory and
time proportional to the width of the window, not to `hi`. It shares its
striking arithmetic with `eratosthenes_segmented` through one helper rather than
duplicating it, and it is checked against `primes_below` over every window with
lo < 300 and width < 120 — about 36,000 of them.

## Robustness

A bogus `limit` used to surface as a `TypeError` from inside a `range` call
three frames down. Every entry point now normalizes it up front: integral floats
are accepted, because `1e8` is how people actually write this bound (and how the
CLI parses `--limit`), and anything else fails with a message naming the
parameter. Negative limits are empty results, not errors.

## Files

| File             | What it is                                                                                      |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| `sieves.py`      | Six implementations plus the shared wheel tables and a registry with per-implementation caveats |
| `benchmark.py`   | Subprocess-isolated harness: wall time, peak RSS, bytes/integer, correctness                    |
| `test_sieves.py` | 148 tests — oracle agreement, wheel boundaries, structural invariants, harness behavior         |

## Sources

- [primesieve — ALGORITHMS.md](https://github.com/kimwalisch/primesieve/blob/master/doc/ALGORITHMS.md)
- [Prime Number Sieving — A Systematic Review with Performance Analysis](https://www.mdpi.com/1999-4893/17/4/157)
