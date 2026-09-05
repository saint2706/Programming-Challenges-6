"""Six prime sieves, built to be compared honestly.

The headline question is whether the Sieve of Atkin's better asymptotics --
O(N / log log N) operations against Eratosthenes' O(N log log N) -- actually
buys anything. The answer, at every scale a laptop can reach, is no; see the
README for the numbers and the reasons.

To make that a fair fight rather than a strawman, both algorithms get a
properly engineered implementation:

* Eratosthenes gets odds-only, wheel-30 packing (one byte per number coprime to
  2, 3, 5 -- 0.267 bytes per integer), and a segmented variant with O(sqrt N)
  memory.
* Atkin gets the full mod-60 form with all three quadratic forms, the same
  0.267 bytes-per-integer packing (16 residues coprime to 60), and the
  squarefree correction pass.

Every implementation has the same signature -- ``f(limit) -> int``, returning
pi(limit) -- because *counting* is the benchmark that lets each one show its
true memory profile. A sieve forced to hand back a 5.7-million-element Python
list is being benchmarked on list construction, not on sieving. Use
``primes_below`` when you actually want the primes.

Run directly to self-check every implementation against known pi(x) values:

    uv run --with numpy python sieves.py
"""

from __future__ import annotations

from bisect import bisect_right
from math import isqrt
from typing import Callable, Iterator

try:  # NumPy is optional: the pure-Python tier is the reference implementation.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only on a bare interpreter
    _np = None

__all__ = [
    "eratosthenes_simple",
    "eratosthenes_wheel30",
    "eratosthenes_segmented",
    "atkin",
    "eratosthenes_numpy",
    "atkin_numpy",
    "primes_below",
    "primes_in_range",
    "iter_primes",
    "IMPLEMENTATIONS",
    "PI_REFERENCE",
    "HAVE_NUMPY",
]

HAVE_NUMPY = _np is not None

# pi(10^k), for correctness checks. Values are standard and easy to verify
# against any table of prime counts.
PI_REFERENCE = {
    10**1: 4,
    10**2: 25,
    10**3: 168,
    10**4: 1229,
    10**5: 9592,
    10**6: 78498,
    10**7: 664579,
    10**8: 5761455,
    10**9: 50847534,
}


# ---------------------------------------------------------------------------
# Wheel tables
# ---------------------------------------------------------------------------

# Residues coprime to 30 = 2*3*5. Eight of every thirty integers survive, so a
# byte-per-residue array costs 8/30 = 0.267 bytes per integer.
W30 = (1, 7, 11, 13, 17, 19, 23, 29)
POS30 = [-1] * 30
for _i, _r in enumerate(W30):
    POS30[_r] = _i

# Residues coprime to 60 = 2^2*3*5, which is the modulus Atkin's quadratic
# forms are stated over. Sixteen of every sixty -- the same 0.267 bytes per
# integer, so the two algorithms' memory numbers are directly comparable.
W60 = (1, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 49, 53, 59)
POS60 = [-1] * 60
for _i, _r in enumerate(W60):
    POS60[_r] = _i

# The three quadratic forms only ever produce primes in these residue classes.
ATKIN_FORM1 = frozenset({1, 13, 17, 29, 37, 41, 49, 53})  # n = 4x^2 + y^2
ATKIN_FORM2 = frozenset({7, 19, 31, 43})  # n = 3x^2 + y^2
ATKIN_FORM3 = frozenset({11, 23, 47, 59})  # n = 3x^2 - y^2, x > y


def _size30(limit: int) -> int:
    """How many integers in [1, limit] are coprime to 30."""
    return (limit // 30) * 8 + bisect_right(W30, limit % 30)


def _size60(limit: int) -> int:
    """How many integers in [1, limit] are coprime to 60."""
    return (limit // 60) * 16 + bisect_right(W60, limit % 60)


def _small_pi(limit: int) -> int:
    """pi(limit) for limit < 7, where the wheels have nothing to say."""
    return sum(1 for p in (2, 3, 5) if p <= limit)


def _as_limit(limit) -> int:
    """Normalize and validate a sieve bound.

    Integral floats are accepted because ``1e8`` is how people actually write
    this bound (the CLI parses ``--limit`` that way too); anything else fails
    here with a readable message rather than deep inside a ``range`` call.
    """
    if isinstance(limit, bool) or not isinstance(limit, (int, float)):
        raise TypeError(
            f"limit must be an integer, got {type(limit).__name__}: {limit!r}"
        )
    if isinstance(limit, float):
        if limit != int(limit):
            raise ValueError(f"limit must be a whole number, got {limit!r}")
        limit = int(limit)
    return limit


def _wheel_value(index: int) -> int:
    """The integer a wheel-30 index stands for."""
    return (index // 8) * 30 + W30[index % 8]


def _first_hits(base_primes: list[int], from_index: int) -> list[list[int]]:
    """For each base prime, the eight wheel indices where its striking begins.

    Row ``j`` holds one entry per residue class: the smallest index that is
    both at or above ``p^2`` and at or above ``from_index``. Splitting this out
    is what lets the segmented sieve and :func:`primes_in_range` share the same
    arithmetic instead of two copies that can drift apart.
    """
    rows: list[list[int]] = []
    for p in base_primes:
        step = 8 * p
        row = []
        for r in W30:
            pr = p * r
            k0 = max(0, -((r - p) // 30))
            hit = (pr // 30) * 8 + POS30[pr % 30] + step * k0
            if hit < from_index:
                hit += step * (-((hit - from_index) // step))
            row.append(hit)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Eratosthenes
# ---------------------------------------------------------------------------


def eratosthenes_simple(limit: int) -> int:
    """Odds-only Sieve of Eratosthenes over a ``bytearray``.

    The baseline every Python programmer writes, and it is genuinely hard to
    beat: ``sieve[start::step] = b"\\x00" * n`` pushes the inner loop into C.
    Costs 0.5 bytes per integer.
    """
    limit = _as_limit(limit)
    if limit < 2:
        return 0
    if limit == 2:
        return 1

    # Index i represents the odd number 2i + 1.
    n = (limit - 1) // 2 + 1
    sieve = bytearray(b"\x01") * n
    sieve[0] = 0  # 1 is not prime

    for i in range(1, (isqrt(limit) - 1) // 2 + 1):
        if sieve[i]:
            p = 2 * i + 1
            start = (p * p) // 2  # p*p is odd, so this is its index
            sieve[start::p] = bytes(len(range(start, n, p)))
    # ``.count(1)`` rather than ``sum()``: memchr instead of a Python-level
    # iteration, worth 0.30 s of the 1.2 s this takes at 10^8. The baseline has
    # to be a *fair* baseline or the comparison is worthless.
    return 1 + sieve.count(1)  # +1 for the prime 2


def eratosthenes_wheel30(limit: int) -> int:
    """Wheel-30 Sieve of Eratosthenes, one byte per number coprime to 30.

    The key identity that makes this practical in Python: in the wheel
    representation, the multiples of ``p`` that share a residue class ``r``
    (mod 30) land at *constant stride* ``8p``::

        idx(p * (30k + r)) == 8*p*k + idx(p*r)

    So each prime needs eight slice assignments instead of one -- but each
    covers 8/30 of the range instead of 1/2, so the total number of bytes
    written drops by 47%. Whether that wins depends on the ratio of slice-call
    overhead to slice length, which is exactly what the benchmark measures.
    """
    limit = _as_limit(limit)
    if limit < 7:
        return _small_pi(limit)

    size = _size30(limit)
    sieve = bytearray(b"\x01") * size
    sieve[0] = 0  # 1

    # The smallest wheel prime is 7 and its stride is 8*7 = 56, so no single
    # strike ever needs more than size/56 zeros. Slicing a memoryview of that
    # one buffer is O(1); slicing ``bytes`` would allocate on every strike, and
    # measured ~17x more expensive.
    zeros = memoryview(bytes(size // 56 + 2))

    for i in range(1, size):
        p = (i // 8) * 30 + W30[i % 8]
        if p * p > limit:
            break
        if not sieve[i]:
            continue
        step = 8 * p
        for r in W30:
            pr = p * r
            # First multiple at or above p^2: p*(30k + r) >= p^2 <=> 30k+r >= p.
            k0 = max(0, -((r - p) // 30))
            start = (pr // 30) * 8 + POS30[pr % 30] + step * k0
            if start < size:
                sieve[start::step] = zeros[: -((start - size) // step)]

    return 3 + sieve.count(1)  # +3 for 2, 3, 5


def eratosthenes_segmented(limit: int, segment_size: int | None = None) -> int:
    """Segmented wheel-30 sieve: O(sqrt N) memory instead of O(N).

    This is the shape every serious sieve uses (primesieve included). Memory
    stops depending on ``limit``, and each segment is small enough to stay in
    cache while it is being struck -- which is where the real speedup comes
    from at large N, not from the reduced instruction count.

    ``segment_size`` is in wheel bytes; each byte covers 30/8 integers.

    The default is deliberately *not* the L1/L2-sized segment a C sieve wants.
    Measured on this machine at 10^8, a 32 KiB segment costs 2.78 s and a 2 MiB
    segment 0.25 s -- an 11x inversion of the usual advice, because in Python
    the per-segment bookkeeping is interpreted while the striking is memcpy in
    C. The cache argument only starts to matter once the loop overhead is gone.
    """
    limit = _as_limit(limit)
    if limit < 7:
        return _small_pi(limit)

    root = isqrt(limit)
    base_primes = [p for p in _wheel30_primes(root) if p >= 7]

    total_size_hint = _size30(limit)
    if segment_size is None:
        segment_size = min(1 << 21, max(1 << 15, total_size_hint))
    segment_size = max(segment_size, 8)

    total_size = total_size_hint
    count = 3  # 2, 3, 5

    # next_hit[j][r] is the absolute wheel index of the next multiple of
    # base_primes[j] in residue class W30[r] that is still unstruck.
    next_hit = _first_hits(base_primes, 0)

    # One segment buffer, refilled in place from a memoryview -- no allocation
    # per segment, and none per strike either.
    ones = memoryview(bytes(b"\x01") * segment_size)
    zeros = memoryview(bytes(segment_size // 56 + 2))
    segment = bytearray(segment_size)

    for lo in range(0, total_size, segment_size):
        hi = min(lo + segment_size, total_size)
        width = hi - lo
        segment[:width] = ones[:width]
        if width < segment_size:
            # Only the final segment is short; zero the tail so the stale ones
            # left by the previous segment are not counted.
            segment[width:] = bytes(segment_size - width)
        if lo == 0:
            segment[0] = 0  # 1 is not prime

        for j, p in enumerate(base_primes):
            step = 8 * p
            row = next_hit[j]
            for r in range(8):
                cur = row[r]
                if cur >= hi:
                    continue
                local = cur - lo
                segment[local::step] = zeros[: -((local - segment_size) // step)]
                # Advance past this whole segment arithmetically, not by looping.
                row[r] = cur + step * (-((cur - hi) // step))
        count += segment.count(1)

    return count


def _wheel30_primes(limit: int) -> list[int]:
    """Primes up to ``limit`` as a list -- used only for base primes."""
    if limit < 2:
        return []
    n = (limit - 1) // 2 + 1
    sieve = bytearray(b"\x01") * n
    sieve[0] = 0
    for i in range(1, (isqrt(limit) - 1) // 2 + 1):
        if sieve[i]:
            p = 2 * i + 1
            start = (p * p) // 2
            sieve[start::p] = bytes(len(range(start, n, p)))
    return [2] + [2 * i + 1 for i in range(1, n) if sieve[i]]


# ---------------------------------------------------------------------------
# Atkin
# ---------------------------------------------------------------------------


def atkin(limit: int) -> int:
    """Sieve of Atkin, full mod-60 form, packed to 0.267 bytes per integer.

    Three quadratic forms decide primality by the *parity* of the number of
    representations::

        n = 4x^2 + y^2   flips n when n mod 60 in {1,13,17,29,37,41,49,53}
        n = 3x^2 + y^2   flips n when n mod 60 in {7,19,31,43}
        n = 3x^2 - y^2   flips n when n mod 60 in {11,23,47,59} and x > y

    A number left flagged after all three passes is prime *if* it is also
    squarefree, so a final pass clears multiples of every p^2.

    Parity note: each form only produces odd n, so y (form 1) or the opposite
    parity of x (forms 2 and 3) is the only half worth enumerating. That
    halving is not cosmetic -- it is most of what keeps this competitive.
    """
    limit = _as_limit(limit)
    if limit < 7:
        return _small_pi(limit)

    size = _size60(limit)
    flags = bytearray(size)
    root = isqrt(limit)

    def flip(n: int) -> None:
        i = (n // 60) * 16 + POS60[n % 60]
        flags[i] ^= 1

    # Form 1: n = 4x^2 + y^2, y odd.
    x = 1
    while 4 * x * x + 1 <= limit:
        fx = 4 * x * x
        y = 1
        while fx + y * y <= limit:
            n = fx + y * y
            if n % 60 in ATKIN_FORM1:
                flip(n)
            y += 2
        x += 1

    # Form 2: n = 3x^2 + y^2, y of the opposite parity to x.
    x = 1
    while 3 * x * x + 1 <= limit:
        fx = 3 * x * x
        y = 2 if x % 2 else 1
        while fx + y * y <= limit:
            n = fx + y * y
            if n % 60 in ATKIN_FORM2:
                flip(n)
            y += 2
        x += 1

    # Form 3: n = 3x^2 - y^2 with x > y and opposite parities. For a given x
    # the smallest n is 3x^2 - (x-1)^2 = 2x^2 + 2x - 1, which bounds the loop;
    # walking y downward makes n increase, so the inner loop stops on the
    # first n above the limit.
    x = 2
    while 2 * x * x + 2 * x - 1 <= limit:
        fx = 3 * x * x
        y = x - 1
        if (x - y) % 2 == 0:
            y -= 1
        while y >= 1 and fx - y * y <= limit:
            n = fx - y * y
            if n % 60 in ATKIN_FORM3:
                flip(n)
            y -= 2
        x += 1

    # Squarefree correction: a flagged n is prime only if p^2 does not divide
    # it for any prime p, so strike the multiples of every flagged p^2.
    count = 3  # 2, 3, 5
    for i in range(1, size):
        n = (i // 16) * 60 + W60[i % 16]
        if n > root:
            break
        if flags[i]:
            sq = n * n
            for m in range(sq, limit + 1, sq):
                if POS60[m % 60] >= 0:
                    flags[(m // 60) * 16 + POS60[m % 60]] = 0

    return count + sum(flags)


# ---------------------------------------------------------------------------
# NumPy tier -- same algorithms, vectorized inner loops
# ---------------------------------------------------------------------------


def _require_numpy() -> None:
    if _np is None:  # pragma: no cover
        raise RuntimeError("this implementation needs numpy: uv add numpy")


def eratosthenes_numpy(limit: int) -> int:
    """Odds-only Eratosthenes over a NumPy boolean array. 0.5 bytes/integer."""
    _require_numpy()
    limit = _as_limit(limit)
    if limit < 2:
        return 0
    if limit == 2:
        return 1
    n = (limit - 1) // 2 + 1
    sieve = _np.ones(n, dtype=bool)
    sieve[0] = False
    for i in range(1, (isqrt(limit) - 1) // 2 + 1):
        if sieve[i]:
            p = 2 * i + 1
            sieve[(p * p) // 2 :: p] = False
    return 1 + int(_np.count_nonzero(sieve))


def atkin_numpy(limit: int) -> int:
    """Atkin with the quadratic forms vectorized over ``y``.

    The trick that makes this work without a slow ``ufunc.at``: for a *fixed*
    x, the n values a form produces are pairwise distinct, so plain fancy
    indexing (``flags[idx] ^= True``) is a correct flip -- no read-modify-write
    hazard, and it runs at full vector speed.
    """
    _require_numpy()
    limit = _as_limit(limit)
    if limit < 7:
        return _small_pi(limit)

    np = _np
    size = _size60(limit)
    flags = np.zeros(size, dtype=bool)

    pos60 = np.full(60, -1, dtype=np.int64)
    for i, r in enumerate(W60):
        pos60[r] = i
    form_lut = np.zeros((3, 60), dtype=bool)
    for j, residues in enumerate((ATKIN_FORM1, ATKIN_FORM2, ATKIN_FORM3)):
        for r in residues:
            form_lut[j, r] = True

    def flip(n, form: int) -> None:
        n = n[form_lut[form][n % 60]]
        if n.size:
            flags[(n // 60) * 16 + pos60[n % 60]] ^= True

    # Form 1: n = 4x^2 + y^2 with y odd.
    x = 1
    while 4 * x * x + 1 <= limit:
        fx = 4 * x * x
        y_max = isqrt(limit - fx)
        y = np.arange(1, y_max + 1, 2, dtype=np.int64)
        if y.size:
            flip(fx + y * y, 0)
        x += 1

    # Form 2: n = 3x^2 + y^2, y of the opposite parity to x.
    x = 1
    while 3 * x * x + 1 <= limit:
        fx = 3 * x * x
        y_max = isqrt(limit - fx)
        start = 2 if x % 2 else 1
        y = np.arange(start, y_max + 1, 2, dtype=np.int64)
        if y.size:
            flip(fx + y * y, 1)
        x += 1

    # Form 3: n = 3x^2 - y^2 with x > y and opposite parities.
    x = 2
    while 3 * x * x - (x - 1) * (x - 1) <= limit:
        fx = 3 * x * x
        # 3x^2 - y^2 <= limit  <=>  y^2 >= 3x^2 - limit
        y_min = isqrt(max(0, fx - limit - 1)) + 1 if fx > limit else 1
        y_hi = x - 1
        if (x - y_hi) % 2 == 0:
            y_hi -= 1
        if (x - y_min) % 2 == 0:
            y_min += 1
        if y_hi >= y_min:
            y = np.arange(y_min, y_hi + 1, 2, dtype=np.int64)
            if y.size:
                flip(fx - y * y, 2)
        x += 1

    # Squarefree correction.
    root = isqrt(limit)
    idx = np.flatnonzero(flags)
    vals = (idx // 16) * 60 + np.asarray(W60, dtype=np.int64)[idx % 16]
    for n in vals[vals <= root].tolist():
        sq = n * n
        m = np.arange(sq, limit + 1, sq, dtype=np.int64)
        m = m[pos60[m % 60] >= 0]
        if m.size:
            flags[(m // 60) * 16 + pos60[m % 60]] = False

    return 3 + int(flags.sum())


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def primes_below(limit: int) -> list[int]:
    """The primes in [2, limit] as a list. For when you want primes, not a count."""
    return _wheel30_primes(_as_limit(limit))


def primes_in_range(lo: int, hi: int) -> list[int]:
    """Primes in the inclusive range ``[lo, hi]`` without sieving from zero.

    This is what a segmented sieve is actually *for*, and it is the one thing
    ``primes_below`` cannot do: the primes just above 10^12 cost O(sqrt(hi))
    memory and time proportional to the width of the window, not to ``hi``.

    >>> primes_in_range(10**12, 10**12 + 100)
    [1000000000039, 1000000000061, 1000000000063, 1000000000091]
    """
    lo, hi = _as_limit(lo), _as_limit(hi)
    if hi < lo or hi < 2:
        return []
    out = [p for p in (2, 3, 5) if lo <= p <= hi]
    if hi < 7:
        return out

    lo = max(lo, 7)
    # There are ``_size30(n)`` integers coprime to 30 in [1, n], so the first
    # such integer at or above ``lo`` sits at wheel index ``_size30(lo - 1)``.
    start = _size30(lo - 1)
    stop = _size30(hi)
    if stop <= start:
        return out

    base_primes = [p for p in _wheel30_primes(isqrt(hi)) if p >= 7]
    next_hit = _first_hits(base_primes, start)

    width = stop - start
    segment_size = min(1 << 21, max(1 << 12, width))
    ones = memoryview(bytes(b"\x01") * segment_size)
    zeros = memoryview(bytes(segment_size // 56 + 2))
    segment = bytearray(segment_size)

    for base in range(start, stop, segment_size):
        end = min(base + segment_size, stop)
        span = end - base
        segment[:span] = ones[:span]
        for j, p in enumerate(base_primes):
            step = 8 * p
            row = next_hit[j]
            for r in range(8):
                cur = row[r]
                if cur >= end:
                    continue
                local = cur - base
                segment[local::step] = zeros[: -((local - segment_size) // step)]
                row[r] = cur + step * (-((cur - end) // step))
        k = segment.find(1, 0, span)
        while k >= 0:
            out.append(_wheel_value(base + k))
            k = segment.find(1, k + 1, span)
    return out


def iter_primes(limit: int) -> Iterator[int]:
    """Stream primes in [2, limit] with O(sqrt N) memory."""
    if limit >= 2:
        yield 2
    if limit >= 3:
        yield 3
    if limit >= 5:
        yield 5
    if limit < 7:
        return

    root = isqrt(limit)
    base = [p for p in _wheel30_primes(root) if p >= 7]
    segment_size = max(1 << 15, _size30(root))
    total = _size30(limit)

    next_hit = []
    for p in base:
        step = 8 * p
        next_hit.append(
            [
                (p * r // 30) * 8 + POS30[(p * r) % 30] + step * max(0, -((r - p) // 30))
                for r in W30
            ]
        )

    ones = memoryview(bytes(b"\x01") * segment_size)
    zeros = memoryview(bytes(segment_size // 56 + 2))
    segment = bytearray(segment_size)

    for lo in range(0, total, segment_size):
        hi = min(lo + segment_size, total)
        width = hi - lo
        segment[:width] = ones[:width]
        if lo == 0:
            segment[0] = 0
        for j, p in enumerate(base):
            step = 8 * p
            row = next_hit[j]
            for r in range(8):
                cur = row[r]
                if cur >= hi:
                    continue
                local = cur - lo
                segment[local::step] = zeros[: -((local - segment_size) // step)]
                row[r] = cur + step * (-((cur - hi) // step))
        # ``index`` skips runs of composites in C rather than in the loop.
        k = segment.find(1, 0, width)
        while k >= 0:
            i = lo + k
            yield (i // 8) * 30 + W30[i % 8]
            k = segment.find(1, k + 1, width)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class Implementation:
    """One sieve, plus the honest caveats the benchmark needs to know."""

    def __init__(
        self,
        key: str,
        label: str,
        fn: Callable[[int], int],
        family: str,
        bytes_per_int: float,
        memory: str,
        needs_numpy: bool = False,
        max_limit: int | None = None,
    ) -> None:
        self.key = key
        self.label = label
        self.fn = fn
        self.family = family
        self.bytes_per_int = bytes_per_int
        self.memory = memory
        self.needs_numpy = needs_numpy
        self.max_limit = max_limit

    def available(self) -> tuple[bool, str]:
        if self.needs_numpy and not HAVE_NUMPY:
            return False, "numpy not installed"
        return True, ""

    def suitable(self, limit: int) -> tuple[bool, str]:
        ok, why = self.available()
        if not ok:
            return ok, why
        if self.max_limit is not None and limit > self.max_limit:
            return False, f"too slow above {self.max_limit:.0e} (would take minutes)"
        return True, ""


IMPLEMENTATIONS: dict[str, Implementation] = {
    impl.key: impl
    for impl in [
        Implementation(
            "era-simple", "Eratosthenes, odds-only bytearray",
            eratosthenes_simple, "eratosthenes", 0.5, "O(N)",
        ),
        Implementation(
            "era-wheel30", "Eratosthenes, wheel-30 packed",
            eratosthenes_wheel30, "eratosthenes", 8 / 30, "O(N)",
        ),
        Implementation(
            "era-segmented", "Eratosthenes, wheel-30 segmented",
            eratosthenes_segmented, "eratosthenes", 0.0, "O(sqrt N)",
        ),
        Implementation(
            "atkin", "Atkin, mod-60 packed",
            atkin, "atkin", 16 / 60, "O(N)",
            max_limit=10**8,  # ~19 s at 10^8; 10^9 would be most of an hour
        ),
        Implementation(
            "era-numpy", "Eratosthenes, odds-only NumPy",
            eratosthenes_numpy, "eratosthenes", 0.5, "O(N)", needs_numpy=True,
        ),
        Implementation(
            "atkin-numpy", "Atkin, mod-60 packed NumPy",
            atkin_numpy, "atkin", 16 / 60, "O(N)", needs_numpy=True,
        ),
    ]
}


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _self_check() -> int:
    failures = 0
    limits = [0, 1, 2, 3, 4, 5, 6, 7, 10, 30, 31, 60, 61, 100, 1000, 10**5, 10**6]
    reference = {n: len(primes_below(n)) for n in limits}

    for impl in IMPLEMENTATIONS.values():
        ok, why = impl.available()
        if not ok:
            print(f"  [skip] {impl.label}: {why}")
            continue
        bad = [(n, impl.fn(n), reference[n]) for n in limits if impl.fn(n) != reference[n]]
        if bad:
            failures += 1
            print(f"  [FAIL] {impl.label}: {bad[:3]}")
        else:
            print(f"  [ok  ] {impl.label}: pi(x) correct for {len(limits)} limits")

    # The prime *lists* must agree too, not just the counts.
    ref = primes_below(10**5)
    streamed = list(iter_primes(10**5))
    if streamed != ref:
        failures += 1
        print("  [FAIL] iter_primes disagrees with primes_below")
    else:
        print(f"  [ok  ] iter_primes streams all {len(ref)} primes below 10^5")

    for n, expected in PI_REFERENCE.items():
        if n > 10**6:
            continue
        got = eratosthenes_segmented(n)
        if got != expected:
            failures += 1
            print(f"  [FAIL] pi(10^{len(str(n)) - 1}) = {got}, expected {expected}")
    print("  [ok  ] matches published pi(10^k) values up to 10^6")

    print("all self-checks passed" if not failures else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
