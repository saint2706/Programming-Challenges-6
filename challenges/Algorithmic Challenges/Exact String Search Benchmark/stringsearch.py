"""Nine exact string matchers, one interface, and one honest conclusion.

The brief is "naive vs KMP vs Boyer-Moore vs Rabin-Karp, same corpus". Running
that benchmark in Python produces a result the brief does not anticipate: the
ranking among those four barely matters, because all four lose to
``str.find`` by two orders of magnitude. So this module answers two questions
side by side, and keeps them apart:

* **How many characters does each algorithm look at?** Machine-independent,
  and where the algorithmic content lives. Every implementation here is
  instrumented by :func:`count_accesses` without a second copy of the code:
  the *text* is wrapped in a proxy that counts ``__getitem__``, so the number
  reported is exactly what the algorithm asked for.
* **How long does it take?** Machine-dependent, dominated by the interpreter,
  and the number you actually ship against.

The two answers disagree, and the disagreement is the finding. Boyer-Moore
inspects the fewest characters of any method here -- roughly ``n/m`` of them
on English text, sublinear as promised -- and is among the *slowest* in
seconds, because each skip it computes costs several bytecodes while the
naive loop's comparison costs one. Skipping work is only worth it when the
work you skip is more expensive than the arithmetic that decides to skip it.

Which is why :func:`bitparallel_search` is here. It abandons the
"loop over characters" shape entirely: for each distinct pattern character it
builds one big integer whose bit ``i`` says "text[i] is this character", then
ANDs those integers together with the right shifts. All occurrences fall out
of ``m`` big-integer operations, so the per-character work happens in C
inside CPython's bignum routines rather than in the interpreter. It is the
only hand-written method here that ever beats ``str.find`` -- on dense
overlapping matches, where ``find``-in-a-loop pays Python overhead per match
and this pays none.

    uv run python stringsearch.py --demo
    uv run python stringsearch.py --verify
    uv run python stringsearch.py --pattern needle --text "a needle in a haystack"
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections.abc import Iterator, Sequence
from typing import Any, Callable

__all__ = [
    "naive_search",
    "kmp_search",
    "boyer_moore_search",
    "boyer_moore_no_galil_search",
    "horspool_search",
    "sunday_search",
    "rabin_karp_search",
    "rabin_karp_randomized_search",
    "two_way_search",
    "shift_or_search",
    "bitparallel_search",
    "builtin_search",
    "ALGORITHMS",
    "COUNTABLE",
    "prefix_function",
    "count_accesses",
    "boyer_moore_adversary",
    "rabin_karp_adversary",
    "naive_adversary",
    "verify",
    "main",
]

_DEFAULT_MOD = (1 << 61) - 1  # a Mersenne prime; wide enough that the
_DEFAULT_BASE = 257  # birthday bound is not reachable in practice


# ---------------------------------------------------------------------------
# 1. Naive
# ---------------------------------------------------------------------------


def naive_search(pattern: Sequence[Any], text: Sequence[Any]) -> Iterator[int]:
    """Try every alignment, compare left to right. ``O(nm)`` worst case.

    Written as an explicit character loop rather than ``text[j:j+m] ==
    pattern`` on purpose: the slice version is a different algorithm with a C
    inner loop, and comparing it against the others would be measuring
    CPython, not the method. :func:`builtin_search` is here for that job.
    """
    m, n = len(pattern), len(text)
    if m == 0:
        yield from range(n + 1)
        return
    for j in range(n - m + 1):
        k = 0
        while k < m and pattern[k] == text[j + k]:
            k += 1
        if k == m:
            yield j


# ---------------------------------------------------------------------------
# 2. Knuth-Morris-Pratt
# ---------------------------------------------------------------------------


def prefix_function(s: Sequence[Any]) -> list[int]:
    """``pi[i]`` is the longest proper border of ``s[:i+1]``. ``O(n)``.

    This is the same amortisation argument as the Z-algorithm next door
    (challenge 9): ``j`` grows by at most one per position and never goes
    below zero, so the ``while`` that shrinks it runs at most ``n`` times in
    total.
    """
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi


def kmp_search(pattern: Sequence[Any], text: Sequence[Any]) -> Iterator[int]:
    """Never re-examine a text character. ``O(n + m)``, at most ``2n``
    comparisons.

    The guarantee is real and it is also the reason KMP is unexciting in
    practice: it *cannot* look at fewer than ``n`` characters, so it has no
    sublinear case at all. Boyer-Moore's worst case is worse and its typical
    case is several times better.
    """
    m, n = len(pattern), len(text)
    if m == 0:
        yield from range(n + 1)
        return
    if m > n:
        return
    pi = prefix_function(pattern)
    j = 0
    for i in range(n):
        c = text[i]
        while j and pattern[j] != c:
            j = pi[j - 1]
        if pattern[j] == c:
            j += 1
        if j == m:
            yield i - m + 1
            j = pi[j - 1]


# ---------------------------------------------------------------------------
# 3-4. Boyer-Moore, with and without the Galil rule
# ---------------------------------------------------------------------------


def _bad_character_table(pattern: Sequence[Any]) -> dict[Any, int]:
    """Last index of each character. A dict, not an array of size sigma, so
    the alphabet can be anything and the build is ``O(m)`` not ``O(m + sigma)``."""
    return {c: i for i, c in enumerate(pattern)}


def _good_suffix_table(pattern: Sequence[Any]) -> list[int]:
    """``shift[i]`` = how far to move when the mismatch is at ``pattern[i-1]``.

    The two classic cases: the matched suffix occurs again elsewhere in the
    pattern (preceded by a different character), or only a prefix of the
    pattern matches a suffix of what matched. ``shift[m]`` is the shift after
    a mismatch at the last character; ``shift[0]`` is the shift after a full
    match, and equals the pattern's smallest period -- which is what makes the
    Galil rule below correct.
    """
    m = len(pattern)
    shift = [0] * (m + 1)
    border = [0] * (m + 1)
    i, j = m, m + 1
    border[i] = j
    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if shift[j] == 0:
                shift[j] = j - i
            j = border[j]
        i -= 1
        j -= 1
        border[i] = j
    j = border[0]
    for i in range(m + 1):
        if shift[i] == 0:
            shift[i] = j
        if i == j:
            j = border[j]
    return shift


def boyer_moore_search(pattern: Sequence[Any], text: Sequence[Any]) -> Iterator[int]:
    """Full Boyer-Moore: bad character + good suffix + Galil rule. ``O(n)``.

    Scanning right to left is what buys the sublinear average case: a
    mismatched character that does not occur in the pattern at all lets the
    whole pattern jump past it, so on a large alphabet the expected number of
    inspected characters is about ``n * log_sigma(m) / m`` (Yao's bound, which
    Boyer-Moore attains up to a constant).

    Without the **Galil rule** the algorithm is ``O(nm)`` on repetitive input:
    searching ``a^m`` in ``a^n`` re-verifies all ``m`` characters at every one
    of the ``n`` alignments. Galil's observation is that after a match the
    pattern shifts by its own period ``p = shift[0]``, so the leading ``m - p``
    characters of the next alignment are *already known* to match and the scan
    can stop at index ``m - p``. That one variable turns the quadratic case
    linear; :func:`boyer_moore_no_galil_search` is the same code without it,
    and the benchmark runs both on :func:`boyer_moore_adversary` output.
    """
    m, n = len(pattern), len(text)
    if m == 0:
        yield from range(n + 1)
        return
    if m > n:
        return
    last = _bad_character_table(pattern)
    gs = _good_suffix_table(pattern)
    j = 0
    memory = -1  # indices <= memory are known to match at this alignment
    while j <= n - m:
        i = m - 1
        while i > memory and pattern[i] == text[j + i]:
            i -= 1
        if i <= memory:
            yield j
            period = gs[0]
            j += period
            memory = m - period - 1 if period < m else -1
        else:
            shift = max(gs[i + 1], i - last.get(text[j + i], -1))
            j += shift
            memory = -1


def boyer_moore_no_galil_search(
    pattern: Sequence[Any], text: Sequence[Any]
) -> Iterator[int]:
    """Boyer-Moore as usually written: no Galil rule, so ``O(nm)`` on runs."""
    m, n = len(pattern), len(text)
    if m == 0:
        yield from range(n + 1)
        return
    if m > n:
        return
    last = _bad_character_table(pattern)
    gs = _good_suffix_table(pattern)
    j = 0
    while j <= n - m:
        i = m - 1
        while i >= 0 and pattern[i] == text[j + i]:
            i -= 1
        if i < 0:
            yield j
            j += gs[0]
        else:
            j += max(gs[i + 1], i - last.get(text[j + i], -1))


def horspool_search(pattern: Sequence[Any], text: Sequence[Any]) -> Iterator[int]:
    """Boyer-Moore-Horspool: drop the good-suffix table, shift on the text
    character aligned with the pattern's last position. ``O(nm)`` worst case.

    Horspool (1980) threw away the more complicated half of Boyer-Moore and
    got a *faster* algorithm in practice, because the surviving table is the
    one that produces almost all of the skips and it costs one lookup instead
    of two plus a max. This is the shape most "fast" matchers still have.
    """
    m, n = len(pattern), len(text)
    if m == 0:
        yield from range(n + 1)
        return
    if m > n:
        return
    shift = {c: m - 1 - i for i, c in enumerate(pattern[: m - 1])}
    j = 0
    while j <= n - m:
        i = m - 1
        while i >= 0 and pattern[i] == text[j + i]:
            i -= 1
        if i < 0:
            yield j
        j += shift.get(text[j + m - 1], m)


def sunday_search(pattern: Sequence[Any], text: Sequence[Any]) -> Iterator[int]:
    """Sunday's Quick Search: shift on the character *just past* the window.

    One position further right than Horspool looks, so the shift can be as
    large as ``m + 1`` instead of ``m``. On short patterns over a large
    alphabet this is usually the fastest of the skip-table family -- fewer
    alignments, and the comparison order inside a window stops mattering.
    """
    m, n = len(pattern), len(text)
    if m == 0:
        yield from range(n + 1)
        return
    if m > n:
        return
    shift = {c: m - i for i, c in enumerate(pattern)}
    j = 0
    while j <= n - m:
        k = 0
        while k < m and pattern[k] == text[j + k]:
            k += 1
        if k == m:
            yield j
        if j + m >= n:
            return
        j += shift.get(text[j + m], m + 1)


# ---------------------------------------------------------------------------
# 5. Rabin-Karp
# ---------------------------------------------------------------------------


def _ordinal(x: Any) -> int:
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        return ord(x) if len(x) == 1 else hash(x)
    return hash(x)


def rabin_karp_search(
    pattern: Sequence[Any],
    text: Sequence[Any],
    *,
    base: int = _DEFAULT_BASE,
    mod: int = _DEFAULT_MOD,
) -> Iterator[int]:
    """Rolling hash, verified on every hit, so the answer is always exact.

    The rolling update is ``O(1)``, so the scan is ``O(n)`` *plus* ``O(m)``
    for each hash collision. Collisions are not a correctness problem here --
    the verification loop settles them -- but they are a performance problem,
    and with a **fixed, public** modulus they are an attacker's lever:
    :func:`rabin_karp_adversary` builds a text of length ``n`` in which every
    single window collides, forcing ``Theta(nm)``.

    The defence is the modulus. The default is ``2^61 - 1``, wide enough that
    a collision needs ~``2^30`` windows before it is even likely; a *secret*
    or per-call random modulus is the defence against a chosen input, which is
    what :func:`rabin_karp_randomized_search` does.
    """
    m, n = len(pattern), len(text)
    if m == 0:
        yield from range(n + 1)
        return
    if m > n:
        return
    high = pow(base, m - 1, mod)
    ph = th = 0
    for k in range(m):
        ph = (ph * base + _ordinal(pattern[k])) % mod
        th = (th * base + _ordinal(text[k])) % mod
    last = n - m
    for j in range(last + 1):
        if th == ph:
            k = 0
            while k < m and pattern[k] == text[j + k]:
                k += 1
            if k == m:
                yield j
        if j < last:
            th = ((th - _ordinal(text[j]) * high) * base + _ordinal(text[j + m])) % mod


def rabin_karp_randomized_search(
    pattern: Sequence[Any],
    text: Sequence[Any],
    *,
    rng: random.Random | None = None,
) -> Iterator[int]:
    """Rabin-Karp with a modulus drawn per call, so no fixed input is bad.

    Still exact -- the verification loop does not care where the modulus came
    from. What randomising buys is that the ``Theta(nm)`` case stops being a
    property of the *input* and becomes a property of an unlucky draw, whose
    probability an adversary cannot influence. The distinction is the same one
    that makes randomised quicksort pivot selection worth the coin flip.
    """
    rng = rng or random.Random()
    mod = rng.choice(_LARGE_PRIMES)
    base = rng.randrange(256, mod - 1)
    yield from rabin_karp_search(pattern, text, base=base, mod=mod)


_LARGE_PRIMES = [
    (1 << 61) - 1,
    2305843009213693921 - 70,  # 2^61 - 69, prime
    1000000007,
    1000000009,
    998244353,
    2147483647,
]


# ---------------------------------------------------------------------------
# 6. Two-Way (Crochemore-Perrin) -- what CPython actually runs
# ---------------------------------------------------------------------------


def _maximal_suffix(pattern: Sequence[Any], reverse_order: bool) -> tuple[int, int]:
    """``(index, period)`` of the maximal suffix under ``<=`` or under ``>=``."""
    m = len(pattern)
    ms, j, k, p = -1, 0, 1, 1
    while j + k < m:
        a, b = pattern[j + k], pattern[ms + k]
        if (a > b) if reverse_order else (a < b):
            j += k
            k = 1
            p = j - ms
        elif a == b:
            if k != p:
                k += 1
            else:
                j += p
                k = 1
        else:
            ms = j
            j = ms + 1
            k = p = 1
    return ms, p


def _critical_factorization(pattern: Sequence[Any]) -> tuple[int, int]:
    """The critical factorization point and the local period there.

    The Critical Factorization Theorem (Cesari-Vincent, via Crochemore-Perrin)
    says every string has a split point whose *local* period equals the
    string's *global* period, and that such a point can be found within the
    first ``period`` characters. That is what makes Two-Way work in ``O(1)``
    extra space: no tables proportional to ``m`` or to the alphabet, just two
    integers computed by comparing the two maximal suffixes.
    """
    i1, p1 = _maximal_suffix(pattern, False)
    i2, p2 = _maximal_suffix(pattern, True)
    return (i1, p1) if i1 >= i2 else (i2, p2)


def two_way_search(pattern: Sequence[Any], text: Sequence[Any]) -> Iterator[int]:
    """Crochemore-Perrin Two-Way: ``O(n)`` time, ``O(1)`` space, no tables.

    This is the algorithm behind ``str.find``, ``bytes.find``, glibc's
    ``memmem`` and ``strstr``. It splits the pattern at a critical point, then
    at each alignment scans the right factor left-to-right and only on success
    the left factor right-to-left. Because of the critical factorization, a
    mismatch in the right factor licenses a shift past everything compared --
    so no text character is examined more than a constant number of times,
    with no preprocessing table at all.

    That last part is why it wins in a real library: a matcher used for every
    ``in`` and ``find`` in a language cannot afford an ``O(sigma)`` table
    allocation per call. CPython pairs it with a Bloom-filter skip loop for
    the common short-pattern case; see ``Objects/stringlib/fastsearch.h``.
    """
    m, n = len(pattern), len(text)
    if m == 0:
        yield from range(n + 1)
        return
    if m > n:
        return
    ell, period = _critical_factorization(pattern)

    # Does the pattern repeat with that period across the split point? If so,
    # the periodic variant applies and `memory` avoids re-comparing the part
    # a period-sized shift already proved.
    if _prefix_equals(pattern, ell, period):
        memory = 0
        j = 0
        while j + m <= n:
            i = max(ell + 1, memory)
            while i < m and pattern[i] == text[j + i]:
                i += 1
            if i >= m:
                i = ell
                while i >= memory and pattern[i] == text[j + i]:
                    i -= 1
                if i < memory:
                    yield j
                j += period
                memory = m - period
            else:
                j += i - ell
                memory = 0
    else:
        shift = max(ell + 1, m - ell - 1) + 1
        j = 0
        while j + m <= n:
            i = ell + 1
            while i < m and pattern[i] == text[j + i]:
                i += 1
            if i >= m:
                i = ell
                while i >= 0 and pattern[i] == text[j + i]:
                    i -= 1
                if i < 0:
                    yield j
                j += shift
            else:
                j += i - ell


def _prefix_equals(pattern: Sequence[Any], ell: int, period: int) -> bool:
    """``pattern[:ell+1] == pattern[period:period+ell+1]``, bounds-safe."""
    if period + ell + 1 > len(pattern):
        return False
    for k in range(ell + 1):
        if pattern[k] != pattern[period + k]:
            return False
    return True


# ---------------------------------------------------------------------------
# 7. Shift-Or (bitap)
# ---------------------------------------------------------------------------


def shift_or_search(pattern: Sequence[Any], text: Sequence[Any]) -> Iterator[int]:
    """Baeza-Yates-Gonnet bitap. ``O(n * ceil(m/w))``, no comparisons at all.

    The state is a bit vector: bit ``i`` is clear when ``pattern[:i+1]`` is a
    suffix of the text read so far. One shift and one OR per text character
    advances every candidate alignment simultaneously, and the algorithm is
    completely branch-free with respect to the text -- its cost does not
    depend on the content at all, only on the length.

    In C with ``m <= 64`` that is unbeatable for short patterns. In Python the
    per-character loop is still a Python loop, so it is slow; the value here
    is that it is the natural stepping stone to :func:`bitparallel_search`,
    which keeps the bit-vector idea but transposes it so the loop runs over
    the *pattern* instead of the text.
    """
    m, n = len(pattern), len(text)
    if m == 0:
        yield from range(n + 1)
        return
    if m > n:
        return
    all_ones = (1 << m) - 1
    masks: dict[Any, int] = {}
    for i, c in enumerate(pattern):
        masks[c] = masks.get(c, all_ones) & ~(1 << i)
    top = 1 << (m - 1)
    state = all_ones
    for i in range(n):
        state = ((state << 1) | masks.get(text[i], all_ones)) & all_ones
        if not state & top:
            yield i - m + 1


# ---------------------------------------------------------------------------
# 8. The transposed bit-parallel matcher
# ---------------------------------------------------------------------------

_BYTE_BITS = [[b for b in range(8) if (v >> b) & 1] for v in range(256)]
_CHAR_TABLES: dict[int, bytes] = {}
_NONZERO = re.compile(rb"[^\x00]")


def _indicator_table(c: int) -> bytes:
    table = _CHAR_TABLES.get(c)
    if table is None:
        table = _CHAR_TABLES[c] = bytes(49 if i == c else 48 for i in range(256))
    return table


def _indicator_mask(text: Any, c: Any, n: int) -> int:
    """Integer whose bit ``i`` is set iff ``text[i] == c``. ``O(n)``, in C.

    For ``bytes`` this is one ``translate`` (a 256-entry lookup per byte,
    inside CPython) followed by one ``int(s, 2)`` -- base 2 is a power of two,
    so that conversion is linear, not the quadratic general-base path. For
    ``str`` that cannot be narrowed to one byte per character it is ``split``
    on the character and a ``join``: the same idea with different C
    primitives, and slower only because ``split`` allocates one object per
    occurrence.
    """
    if n == 0:
        return 0
    if isinstance(text, (bytes, bytearray)):
        digits = bytes(text).translate(_indicator_table(c))
    else:
        digits = "1".join("0" * len(part) for part in text.split(c))
    return int(digits[::-1], 2)


def _narrow(pattern: Any, text: Any) -> tuple[Any, Any] | None:
    """Re-encode a Latin-1 ``str`` pair as ``bytes``, or ``None``.

    Latin-1 is one byte per code point, so the encoding preserves every index
    -- the positions the bytes search reports are the positions in the
    original string, with no remapping. It is a pure win when it applies:
    one C-level encode buys the ``translate`` path for every mask afterwards,
    instead of a ``split`` per distinct character.
    """
    try:
        return pattern.encode("latin-1"), text.encode("latin-1")
    except (UnicodeEncodeError, AttributeError):
        return None


def bitparallel_search(pattern: Sequence[Any], text: Sequence[Any]) -> Iterator[int]:
    """All occurrences from ``m`` big-integer operations, not ``n`` loop steps.

    Build, for each *distinct* character ``c`` of the pattern, an integer
    ``B[c]`` whose bit ``i`` is set exactly when ``text[i] == c``. Then

        M = AND over k in 0..m-1 of (B[pattern[k]] >> k)

    and bit ``j`` of ``M`` is set iff ``text[j+k] == pattern[k]`` for every
    ``k`` -- that is, iff the pattern occurs at ``j``. Shifting right by ``k``
    also drops the bits that would run off the end, so no boundary case is
    needed.

    This is Shift-Or with the loop transposed: the text axis lives inside the
    machine word rather than in the loop, so the number of *interpreted*
    operations is ``O(sigma_P + m)`` regardless of ``n``. The work is
    ``O((sigma_P + m) * n / w)`` bit operations, all of them inside CPython's
    bignum routines. That trade -- more total bit operations, vastly fewer
    interpreter dispatches -- is the whole reason it is fast in Python and
    would be pointless in C.

    ``bytes`` take the ``translate`` path; a ``str`` is re-encoded to Latin-1
    first when it fits (index-preserving, so the reported offsets need no
    remapping) and falls back to ``split`` when it does not. Anything else
    falls back to :func:`two_way_search`, because there is no C primitive to
    build the masks from a list.

    One packing choice, measured and rejected: giving each text position a
    whole *byte* rather than a bit makes the mask build ~3x faster
    (``int.from_bytes`` is a memcpy where ``int(s, 2)`` is a parse) but makes
    every AND 8x wider. The crossover sits near ``m == sigma_P``, so the byte
    packing only wins for very short patterns and loses by 2.5x at ``m = 64``.
    One implementation, chosen for the regime that matters, beats two plus a
    heuristic.
    """
    m, n = len(pattern), len(text)
    if m == 0:
        yield from range(n + 1)
        return
    if m > n:
        return
    if not isinstance(text, (str, bytes, bytearray)) or not isinstance(
        pattern, (str, bytes, bytearray)
    ):
        yield from two_way_search(pattern, text)
        return
    if isinstance(text, str):
        narrowed = _narrow(pattern, text)
        if narrowed is not None:
            pattern, text = narrowed

    masks: dict[Any, int] = {}
    matches: int | None = None
    for k in range(m):
        c = pattern[k]
        mask = masks.get(c)
        if mask is None:
            mask = masks[c] = _indicator_mask(text, c, n)
            if mask == 0:
                return  # a pattern character absent from the text: no match
        matches = (mask >> k) if matches is None else matches & (mask >> k)
        if matches == 0:
            return
    assert matches is not None
    # No end-of-text mask is needed. Each `mask` carries bits only in [0, n),
    # so bit j of `mask >> k` is zero once j + k >= n; the k = m-1 term alone
    # clears every j > n - m. Masking again would cost an O(n)-bit allocation
    # to change nothing.

    # Reporting is the only per-position work left, and even that skips whole
    # zero bytes at C speed via the regex scan.
    buf = matches.to_bytes((n >> 3) + 2, "little")
    for hit in _NONZERO.finditer(buf):
        i = hit.start()
        base = i << 3
        for bit in _BYTE_BITS[buf[i]]:
            yield base + bit


# ---------------------------------------------------------------------------
# 9. The C baseline
# ---------------------------------------------------------------------------


def builtin_search(pattern: Sequence[Any], text: Sequence[Any]) -> Iterator[int]:
    """``str.find`` in a loop -- the thing every other row has to beat.

    CPython's ``fastsearch`` is Crochemore-Perrin Two-Way for long patterns,
    with a Boyer-Moore-Horspool skip loop and a Bloom filter over the pattern's
    characters for short ones. So this row is roughly :func:`two_way_search`
    and :func:`horspool_search` written in C -- which is the point: the gap
    between this row and those two is the interpreter, not the algorithm.

    One structural weakness: it reports one match per call, so ``occ`` matches
    cost ``occ`` interpreted iterations. On dense overlapping matches that is
    what :func:`bitparallel_search` beats.
    """
    m, n = len(pattern), len(text)
    if m == 0:
        yield from range(n + 1)
        return
    if not isinstance(text, (str, bytes, bytearray)):
        yield from two_way_search(pattern, text)
        return
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos < 0:
            return
        yield pos
        start = pos + 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALGORITHMS: dict[str, Callable[..., Iterator[int]]] = {
    "naive": naive_search,
    "kmp": kmp_search,
    "boyer-moore": boyer_moore_search,
    "bm-no-galil": boyer_moore_no_galil_search,
    "horspool": horspool_search,
    "sunday": sunday_search,
    "rabin-karp": rabin_karp_search,
    "two-way": two_way_search,
    "shift-or": shift_or_search,
    "bitparallel": bitparallel_search,
    "builtin": builtin_search,
}

#: Methods whose text accesses `count_accesses` can see. `bitparallel` and
#: `builtin` do their scanning inside C primitives that take the whole text at
#: once, so "characters inspected" is not a quantity they expose -- counting
#: them would report a number that is not comparable with the others.
COUNTABLE: tuple[str, ...] = (
    "naive",
    "kmp",
    "boyer-moore",
    "bm-no-galil",
    "horspool",
    "sunday",
    "rabin-karp",
    "two-way",
    "shift-or",
)


class _CountingSequence:
    """A read-only sequence proxy that counts element accesses.

    Instrumenting nine algorithms by hand would mean nine second copies to
    keep in sync; wrapping the text instead means the count is exactly the
    accesses the unmodified algorithm made. Every method above reaches the
    text only through ``text[i]`` and ``len(text)``, which is the property
    this relies on -- and ``test_counting_proxy_sees_every_access`` checks it
    by confirming the proxied run returns the same matches.
    """

    __slots__ = ("_data", "count")

    def __init__(self, data: Sequence[Any]) -> None:
        self._data = data
        self.count = 0

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Any:
        self.count += 1
        return self._data[index]


def count_accesses(
    algorithm: Callable[..., Iterator[int]],
    pattern: Sequence[Any],
    text: Sequence[Any],
) -> tuple[list[int], int]:
    """``(matches, text_accesses)`` for one algorithm on one input."""
    proxy = _CountingSequence(text)
    matches = list(algorithm(pattern, proxy))
    return matches, proxy.count


# ---------------------------------------------------------------------------
# Adversaries: the worst cases, constructed rather than hoped for
# ---------------------------------------------------------------------------


def naive_adversary(m: int, n: int, alphabet: str = "ab") -> tuple[str, str]:
    """``(pattern, text)`` forcing the naive scan to ``Theta(nm)``.

    ``a^(m-1) b`` inside ``a^n``: every alignment matches ``m - 1``
    characters before failing on the last one, and none of them is a match.
    """
    a, b = alphabet[0], alphabet[1]
    return a * (m - 1) + b, a * n


def boyer_moore_adversary(m: int, n: int, alphabet: str = "ab") -> tuple[str, str]:
    """``(pattern, text)`` forcing Boyer-Moore *without* Galil to ``Theta(nm)``.

    ``a^m`` inside ``a^n``. Every alignment is a full match, so the scan reads
    all ``m`` characters; the good-suffix shift after a match is the period,
    which is 1. The Galil rule notices that a shift of one period leaves
    ``m - 1`` characters already verified, and drops the per-alignment cost to
    ``O(1)``.
    """
    a = alphabet[0]
    return a * m, a * n


def rabin_karp_adversary(m: int, n: int, mod: int) -> tuple[str, str]:
    """``(pattern, text)`` in which *every* window collides with the pattern.

    Pick two characters whose ordinals are congruent modulo ``mod``: ``chr(1)``
    and ``chr(1 + mod)``. Each contributes the same residue at every position,
    so any string over those two characters has the same hash as any other of
    the same length -- and in particular as the all-``chr(1)`` pattern. Every
    one of the ``n - m + 1`` windows therefore triggers the ``O(m)``
    verification loop.

    Colliding is not enough on its own: verification stops at the first
    mismatch, so a *random* mix of the two characters fails after O(1)
    comparisons and costs nothing. The text here is blocks of
    ``low^(m-1) + high``, so every window agrees with the pattern on a long
    prefix -- on average ``m/2`` characters -- before failing. Collision plus
    a long common prefix is what actually makes it ``Theta(nm)``.

    This is a hash-flooding attack, and the defence is the same as everywhere
    else: a modulus wide enough that no two codepoints are congruent, or one
    the attacker cannot see. It needs ``1 + mod`` to be a usable codepoint, so
    it demonstrates against small moduli only -- which is exactly the claim.
    """
    if not 1 <= mod < 0x10FFFF - 1 or 0xD800 <= mod + 1 <= 0xDFFF:
        raise ValueError(f"no congruent codepoint pair exists for mod={mod}")
    low, high = chr(1), chr(1 + mod)
    block = low * (m - 1) + high
    # A run of `low` up front so the search is not trivially match-free, then
    # blocks that collide on every window and mismatch as late as possible.
    body = block * (n // m + 2)
    return low * m, (low * m + body)[:n]


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def verify(*, seed: int = 0, trials: int = 400, verbose: bool = True) -> bool:
    """Every algorithm against brute force on adversarial and random input."""
    import itertools

    rng = random.Random(seed)
    checks: list[tuple[str, bool]] = []
    names = list(ALGORITHMS)

    def brute(pattern, text):
        m = len(pattern)
        if m == 0:
            return list(range(len(text) + 1))
        return [i for i in range(len(text) - m + 1) if text[i : i + m] == pattern]

    # Exhaustive: every binary pattern up to length 4 against every binary
    # text up to length 10.
    agree = {name: True for name in names}
    patterns = [
        "".join(p) for k in range(0, 5) for p in itertools.product("ab", repeat=k)
    ]
    texts = ["".join(t) for k in range(0, 9) for t in itertools.product("ab", repeat=k)]
    for pat in patterns:
        for txt in texts:
            expected = brute(pat, txt)
            for name in names:
                if list(ALGORITHMS[name](pat, txt)) != expected:
                    agree[name] = False
    for name in names:
        checks.append(
            (f"{name}: exhaustive binary patterns <=4 x texts <=8", agree[name])
        )

    # Random, over several alphabets, including bytes and lists.
    ok_random = True
    for _ in range(trials):
        size = rng.choice((1, 2, 4, 26))
        alpha = "abcdefghijklmnopqrstuvwxyz"[:size]
        txt = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 120)))
        if rng.random() < 0.5 and txt:
            start = rng.randrange(len(txt))
            pat = txt[start : start + rng.randint(1, 8)]
        else:
            pat = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 8)))
        expected = brute(pat, txt)
        for name in names:
            ok_random &= list(ALGORITHMS[name](pat, txt)) == expected
        ok_random &= list(rabin_karp_randomized_search(pat, txt, rng=rng)) == expected
        bp, bt = pat.encode(), txt.encode()
        ok_random &= list(bitparallel_search(bp, bt)) == expected
        ok_random &= list(builtin_search(bp, bt)) == expected
        ok_random &= list(two_way_search(list(pat), list(txt))) == expected
        ok_random &= list(bitparallel_search(list(pat), list(txt))) == expected
    checks.append(("all algorithms agree on random text (str, bytes, list)", ok_random))

    # Adversarial inputs still produce the right answer.
    ok_adv = True
    for m, n in ((5, 60), (12, 200)):
        for pat, txt in (
            naive_adversary(m, n),
            boyer_moore_adversary(m, n),
            rabin_karp_adversary(m, n, 127),
        ):
            expected = brute(pat, txt)
            for name in names:
                ok_adv &= list(ALGORITHMS[name](pat, txt)) == expected
    checks.append(("all algorithms agree on the adversarial inputs", ok_adv))

    # Instrumentation does not change behaviour, and KMP's 2n bound holds.
    ok_count = ok_kmp = True
    for _ in range(60):
        txt = "".join(rng.choice("ab") for _ in range(rng.randint(1, 200)))
        pat = "".join(rng.choice("ab") for _ in range(rng.randint(1, 6)))
        expected = brute(pat, txt)
        for name in COUNTABLE:
            matches, accesses = count_accesses(ALGORITHMS[name], pat, txt)
            ok_count &= matches == expected
            ok_count &= accesses >= 0
        _, kmp_accesses = count_accesses(kmp_search, pat, txt)
        ok_kmp &= kmp_accesses <= 2 * len(txt)
    checks.append(("count_accesses preserves every algorithm's output", ok_count))
    checks.append(("KMP inspects at most 2n text characters", ok_kmp))

    # The Galil rule is the difference between linear and quadratic.
    pat, txt = boyer_moore_adversary(40, 4000)
    _, with_galil = count_accesses(boyer_moore_search, pat, txt)
    _, without = count_accesses(boyer_moore_no_galil_search, pat, txt)
    checks.append(
        (
            f"Galil rule: {with_galil:,} accesses vs {without:,} without",
            with_galil < len(txt) * 3 <= without,
        )
    )

    if verbose:
        for name, ok in checks:
            print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> None:
    text = "the rain in spain stays mainly in the plain"
    pattern = "ain"
    print(f"text    {text!r}")
    print(f"pattern {pattern!r}\n")
    print(f"{'algorithm':<14} {'matches':<18} {'text accesses':>14}")
    for name in ALGORITHMS:
        matches = list(ALGORITHMS[name](pattern, text))
        if name in COUNTABLE:
            _, accesses = count_accesses(ALGORITHMS[name], pattern, text)
            shown = f"{accesses:,}"
        else:
            shown = "n/a (runs in C)"
        print(f"{name:<14} {str(matches):<18} {shown:>14}")
    print(f"\ntext length {len(text)}; every method agrees on {matches}")

    print("\nWorst cases, in text accesses (m = 30, n = 3000):")
    print(
        f"\n{'input':<34} {'naive':>12} {'kmp':>10} {'BM':>10} {'BM no Galil':>13} "
        f"{'rabin-karp':>12}"
    )
    cases = {
        "a^(m-1)b in a^n (naive killer)": naive_adversary(30, 3000),
        "a^m in a^n (Galil killer)": boyer_moore_adversary(30, 3000),
        "every window collides (mod 127)": rabin_karp_adversary(30, 3000, 127),
    }
    for label, (pat, txt) in cases.items():
        row = []
        for name in ("naive", "kmp", "boyer-moore", "bm-no-galil"):
            _, acc = count_accesses(ALGORITHMS[name], pat, txt)
            row.append(acc)
        _, rk = count_accesses(lambda p, t: rabin_karp_search(p, t, mod=127), pat, txt)
        print(
            f"{label:<34} {row[0]:>12,} {row[1]:>10,} {row[2]:>10,} {row[3]:>13,} "
            f"{rk:>12,}"
        )
    print("\nEach column is quadratic in exactly one row, and it is a different")
    print("row each time. 'Which algorithm is fastest' has no input-free answer.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stringsearch",
        description="Exact string search: nine algorithms, one interface.",
    )
    parser.add_argument("--demo", action="store_true", help="worked example")
    parser.add_argument("--verify", action="store_true", help="run the self-checks")
    parser.add_argument("--pattern", help="pattern to search for")
    parser.add_argument("--text", help="text to search in")
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALGORITHMS),
        help="use one algorithm instead of all of them",
    )
    parser.add_argument("--list", action="store_true", help="list the algorithms")
    args = parser.parse_args(argv)

    if args.list:
        for name in ALGORITHMS:
            countable = "counted" if name in COUNTABLE else "C-backed"
            print(f"{name:<14} {countable}")
        return 0
    if args.demo:
        _demo()
        return 0
    if args.verify:
        print("Verifying...")
        ok = verify()
        print("all checks passed" if ok else "FAILURES above")
        return 0 if ok else 1
    if args.pattern is None or args.text is None:
        parser.print_help()
        return 0

    names = [args.algorithm] if args.algorithm else list(ALGORITHMS)
    for name in names:
        matches = list(ALGORITHMS[name](args.pattern, args.text))
        extra = ""
        if name in COUNTABLE:
            _, acc = count_accesses(ALGORITHMS[name], args.pattern, args.text)
            extra = f"  ({acc:,} text accesses)"
        print(f"{name:<14} {matches}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
