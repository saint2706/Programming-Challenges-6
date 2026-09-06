"""Manacher's algorithm, and what the radii array is actually good for.

The brief is "O(n) longest palindromic substring, compared against the naive
O(n^2)". Manacher (1975) delivers that, but the substring is the least
interesting thing it computes. What the algorithm really produces is the
*complete* palindromic structure of the string -- for every one of the 2n+1
possible centres, the radius of the longest palindrome around it -- and the
longest substring is one `max` over that array. Once you have the array you
also get, in O(1) or O(n):

    is_palindrome(i, j)          O(1) for any substring, after an O(n) build
    count_palindromic_substrings O(n)   occurrences, which can be ~n^2/4 of them
    longest_palindromic_prefix   O(n)   -> shortest_palindrome_by_prepending
    all_maximal_palindromes      O(n)   the n distinct "cores"

Three implementation choices here differ from the version everyone writes.

**No transformed string.** The usual trick interleaves separators --
"abc" -> "#a#b#c#" -- to make even-length palindromes odd. That is a real
2n+1 allocation, and it forces the input to be a string. This module runs the
two-array formulation (`d1` for odd centres, `d2` for even ones) directly on
the input, so it allocates 2n small ints instead of a 2n+1 sequence *plus*
2n+1 ints, and it works on any sequence: `str`, `bytes`, `list`, `tuple`,
a list of grapheme clusters. `palindrome_radii` still hands you the classic
2n+1 array when you want it, derived from d1/d2 by
`rad[2i+1] = 2*d1[i]-1`, `rad[2i] = 2*d2[i]`.

**The separator is not a correctness hazard -- the boundary sentinels are.**
It is widely repeated that the "#" trick breaks if "#" occurs in the input.
It does not: Manacher only ever compares positions of equal parity, so a real
"#" at an odd index is never compared with a separator at an even one. What
*does* break is the `$...^` boundary-guard variant, which assumes two
characters that cannot appear in the input. This module has neither, so there
is no character it cannot handle -- and `test_palindromes.py` checks "#",
"$" and "^" inputs explicitly.

**Manacher is not the end of the story.** For "how many *distinct*
palindromes" (rather than occurrences), the radii array is the wrong tool:
there are O(n) distinct palindromic substrings but counting them needs the
`Eertree` (Rubinchik & Shur 2015), which is here too, online and in O(n). It
also drives `min_palindromic_partition` in O(n log n) via series links, which
beats the O(n^2) DP that the radii array supports.

    uv run python palindromes.py --demo
    uv run python palindromes.py --verify
    uv run python palindromes.py "A man, a plan, a canal: Panama" --relaxed
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from collections.abc import Iterator, Sequence
from typing import Any

__all__ = [
    "PalindromeIndex",
    "Eertree",
    "manacher_odd_even",
    "palindrome_radii",
    "longest_palindrome",
    "longest_palindrome_span",
    "count_palindromic_substrings",
    "all_maximal_palindromes",
    "distinct_palindromes",
    "count_distinct_palindromes",
    "longest_palindromic_prefix",
    "longest_palindromic_suffix",
    "shortest_palindrome_by_prepending",
    "min_palindromic_partition",
    "palindromic_partition",
    "naive_longest_palindrome_span",
    "dp_longest_palindrome_span",
    "brute_force_longest_palindrome_span",
    "graphemes",
    "relaxed_view",
    "verify",
    "main",
]


# ---------------------------------------------------------------------------
# The core: d1 / d2 radii, no transformed string
# ---------------------------------------------------------------------------


def manacher_odd_even(s: Sequence[Any]) -> tuple[list[int], list[int]]:
    """Return ``(d1, d2)``: the odd and even palindromic radii of ``s``.

    ``d1[i]`` is the number of odd-length palindromes centred at ``i``, which
    is also the radius counting the centre, so the longest one is
    ``s[i-d1[i]+1 : i+d1[i]]`` and has length ``2*d1[i]-1``. Every ``d1[i]``
    is at least 1, because a single character is a palindrome.

    ``d2[i]`` is the number of even-length palindromes centred *between*
    ``i-1`` and ``i``; the longest is ``s[i-d2[i] : i+d2[i]]``, of length
    ``2*d2[i]``. ``d2[0]`` is always 0 -- there is no gap before the first
    character.

    Both loops are the same idea. Keep the palindrome ``[l, r]`` that reaches
    furthest right. For a new centre ``i`` inside it, the mirror centre
    ``l + r - i`` has already been solved, and by the symmetry of ``[l, r]``
    the answer at ``i`` is at least the mirror's answer -- clamped to
    ``r - i + 1``, because past ``r`` the symmetry is not known to hold. Then
    extend by brute force from there.

    **Why that is O(n).** The clamp means the brute-force loop can only run
    when the palindrome at ``i`` reaches at least ``r``, and every iteration
    of it pushes ``r`` one further right. ``r`` never decreases and never
    exceeds ``n``, so the total work in the inner loops across the whole scan
    is at most ``n`` per array -- the outer loop is n steps, the inner loops
    are n steps *in total*, and the whole thing is O(n) rather than the
    O(n^2) that a nested loop suggests. (`test_palindromes.py` asserts this
    bound directly by counting expansions.)

    Works on any sequence whose elements support ``==``: str, bytes, list,
    tuple, or a list of grapheme clusters.
    """
    n = len(s)
    d1 = [0] * n
    d2 = [0] * n

    # Odd-length palindromes, centred on a character.
    l, r = 0, -1
    for i in range(n):
        k = 1 if i > r else min(d1[l + r - i], r - i + 1)
        while i - k >= 0 and i + k < n and s[i - k] == s[i + k]:
            k += 1
        d1[i] = k
        if i + k - 1 > r:
            l, r = i - k + 1, i + k - 1

    # Even-length palindromes, centred between i-1 and i.
    l, r = 0, -1
    for i in range(n):
        k = 0 if i > r else min(d2[l + r - i + 1], r - i + 1)
        while i - k - 1 >= 0 and i + k < n and s[i - k - 1] == s[i + k]:
            k += 1
        d2[i] = k
        if i + k - 1 > r:
            l, r = i - k, i + k - 1

    return d1, d2


def palindrome_radii(s: Sequence[Any]) -> list[int]:
    """The classic 2n+1 radii array over the "#"-interleaved string, never built.

    ``rad[c]`` is the length in ``s`` of the longest palindrome centred at
    virtual position ``c``: odd ``c = 2i+1`` sits on character ``i``, even
    ``c = 2i`` sits in the gap before it. The palindrome itself is
    ``s[(c - rad[c]) // 2 : (c + rad[c]) // 2]``.

    Handy because a single array indexes both parities uniformly -- the
    substring ``s[i:j]`` is a palindrome exactly when ``rad[i+j] >= j-i``,
    which is what makes :meth:`PalindromeIndex.is_palindrome` O(1).
    """
    d1, d2 = manacher_odd_even(s)
    n = len(s)
    rad = [0] * (2 * n + 1)
    for i in range(n):
        rad[2 * i] = 2 * d2[i]
        rad[2 * i + 1] = 2 * d1[i] - 1
    return rad


def longest_palindrome_span(s: Sequence[Any]) -> tuple[int, int]:
    """``(start, end)`` of a longest palindromic substring; leftmost on a tie.

    The empty string gets ``(0, 0)``, which is the empty palindrome -- the
    honest answer, and the one that keeps ``s[start:end]`` valid for every
    input rather than needing a special case at the call site.
    """
    d1, d2 = manacher_odd_even(s)
    best_start, best_len = 0, 0
    for i, radius in enumerate(d1):
        length = 2 * radius - 1
        start = i - radius + 1
        if length > best_len or (length == best_len and start < best_start):
            best_start, best_len = start, length
    for i, radius in enumerate(d2):
        if radius == 0:
            continue
        length = 2 * radius
        start = i - radius
        if length > best_len or (length == best_len and start < best_start):
            best_start, best_len = start, length
    return best_start, best_start + best_len


def longest_palindrome(s: Sequence[Any]) -> Any:
    """The longest palindromic substring itself, leftmost on a tie.

    Returns a slice of ``s``, so a ``str`` in gives a ``str`` out and a list
    of grapheme clusters gives a list back.
    """
    start, end = longest_palindrome_span(s)
    return s[start:end]


def count_palindromic_substrings(s: Sequence[Any]) -> int:
    """How many (start, end) pairs are palindromic. O(n), and it can be huge.

    Counts *occurrences*, not distinct strings: "aaaa" has 10 palindromic
    substrings but only 4 distinct ones. A string of n equal characters has
    n(n+1)/2 of them, which is why no O(n) algorithm can enumerate them all --
    but summing the radii counts them without enumerating, since each centre
    contributes exactly its radius many nested palindromes.

    Use :func:`count_distinct_palindromes` for the other question; the answer
    there is always at most n, by a theorem of Droubay, Justin and Pirillo.
    """
    d1, d2 = manacher_odd_even(s)
    return sum(d1) + sum(d2)


def all_maximal_palindromes(s: Sequence[Any]) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` for the maximal palindrome at each centre.

    "Maximal" means not extendable at that centre -- these are the n cores
    every other palindromic substring is nested inside, so this is the compact
    O(n) representation of a set that can have O(n^2) members. Zero-length
    ones are skipped; ties and duplicates are not filtered, because two
    centres can legitimately produce the same span in a run of equal
    characters.
    """
    d1, d2 = manacher_odd_even(s)
    for i, radius in enumerate(d1):
        yield i - radius + 1, i + radius
    for i, radius in enumerate(d2):
        if radius:
            yield i - radius, i + radius


def longest_palindromic_prefix(s: Sequence[Any]) -> int:
    """Length of the longest prefix of ``s`` that is a palindrome.

    Reads straight off the radii: a palindrome is a prefix exactly when its
    left end is 0, i.e. ``c == rad[c]``.
    """
    rad = palindrome_radii(s)
    return max((r for c, r in enumerate(rad) if c == r), default=0)


def longest_palindromic_suffix(s: Sequence[Any]) -> int:
    """Length of the longest suffix of ``s`` that is a palindrome."""
    rad = palindrome_radii(s)
    end = 2 * len(s)
    return max((r for c, r in enumerate(rad) if c + r == end), default=0)


def shortest_palindrome_by_prepending(s: Sequence[Any]) -> Any:
    """The shortest palindrome obtainable by adding characters to the front.

    Take the longest palindromic *prefix* and reflect everything after it.
    That is optimal: whatever you prepend, the result's second half must
    contain all of ``s``, so the overlap you avoid re-adding is exactly a
    palindromic prefix, and the longest one avoids the most.
    """
    k = longest_palindromic_prefix(s)
    tail = s[k:]
    return tail[::-1] + s


# ---------------------------------------------------------------------------
# The index: build once, query O(1)
# ---------------------------------------------------------------------------


class PalindromeIndex:
    """Precomputed palindromic structure of a sequence. O(n) build, O(1) queries.

    >>> idx = PalindromeIndex("abacaba")
    >>> idx.longest()
    'abacaba'
    >>> idx.is_palindrome(0, 3), idx.is_palindrome(0, 4)
    (True, False)
    >>> idx.count()
    12
    """

    __slots__ = ("_s", "_d1", "_d2", "_rad")

    def __init__(self, s: Sequence[Any]) -> None:
        self._s = s
        self._d1, self._d2 = manacher_odd_even(s)
        n = len(s)
        rad = [0] * (2 * n + 1)
        for i in range(n):
            rad[2 * i] = 2 * self._d2[i]
            rad[2 * i + 1] = 2 * self._d1[i] - 1
        self._rad = rad

    def is_palindrome(self, start: int, end: int) -> bool:
        """Is ``s[start:end]`` a palindrome? O(1). Empty and single spans are.

        The whole trick: a substring's virtual centre is ``start + end`` and
        its radius is ``end - start``, so one array lookup settles it, with no
        separate cases for odd and even lengths.
        """
        n = len(self._s)
        if not 0 <= start <= end <= n:
            raise IndexError(f"span [{start}, {end}) out of range for length {n}")
        return self._rad[start + end] >= end - start

    def radius(self, centre: int) -> int:
        """``rad[centre]`` over the 2n+1 virtual centres."""
        return self._rad[centre]

    def longest(self) -> Any:
        start, end = self.longest_span()
        return self._s[start:end]

    def longest_span(self) -> tuple[int, int]:
        best_start, best_len = 0, 0
        for i, r in enumerate(self._d1):
            length, start = 2 * r - 1, i - r + 1
            if length > best_len or (length == best_len and start < best_start):
                best_start, best_len = start, length
        for i, r in enumerate(self._d2):
            if r:
                length, start = 2 * r, i - r
                if length > best_len or (length == best_len and start < best_start):
                    best_start, best_len = start, length
        return best_start, best_start + best_len

    def longest_at_centre(self, centre: int) -> tuple[int, int]:
        """The maximal palindrome around a virtual centre, as ``(start, end)``."""
        r = self._rad[centre]
        return (centre - r) // 2, (centre + r) // 2

    def count(self) -> int:
        """Palindromic substring occurrences."""
        return sum(self._d1) + sum(self._d2)

    def odd_even(self) -> tuple[list[int], list[int]]:
        return list(self._d1), list(self._d2)

    def __len__(self) -> int:
        return len(self._s)

    def __repr__(self) -> str:
        return f"PalindromeIndex(len={len(self._s)}, longest={self.longest_span()})"


# ---------------------------------------------------------------------------
# Eertree: the structure Manacher cannot replace
# ---------------------------------------------------------------------------


class Eertree:
    """Palindromic tree (Rubinchik & Shur, 2015). Online, O(n) time and space.

    Manacher answers "what palindromes occur, and where". The eertree answers
    "what *distinct* palindromes occur", which the radii array cannot: a
    string has at most n distinct palindromic substrings (Droubay-Justin-
    Pirillo), but they are not the maximal ones, and reading them off the
    radii would take O(n^2).

    The structure is two trees sharing nodes: edges labelled ``c`` from a node
    ``p`` to the node ``cpc``, and suffix links to the longest proper
    palindromic suffix. The two roots have length -1 and 0; the imaginary
    length -1 root is what makes single characters fall out of the same rule
    as everything else, with no special case.

    Amortised O(1) per character: each `add` walks up suffix links, and the
    depth of the longest palindromic suffix grows by at most 1 per character,
    so the total walking is bounded by n.

        >>> t = Eertree("ababa")
        >>> t.count_distinct()
        5
        >>> sorted(t.distinct())
        ['a', 'aba', 'ababa', 'b', 'bab']
    """

    __slots__ = (
        "_s",
        "_len",
        "_link",
        "_trans",
        "_suffix",
        "_occ",
        "_diff",
        "_series",
        "_ends",
    )

    ROOT_IMAGINARY = 0  # length -1
    ROOT_EMPTY = 1  # length 0

    def __init__(self, s: Sequence[Any] = ()) -> None:
        self._s: list[Any] = []
        self._len = [-1, 0]
        self._link = [0, 0]
        self._trans: list[dict[Any, int]] = [{}, {}]
        self._occ = [0, 0]
        self._diff = [0, 0]
        self._series = [0, 0]
        # Where each node's palindrome first ended. A node is created the first
        # time its palindrome appears, and it appears as a suffix of the prefix
        # built so far, so the creation index is a genuine occurrence -- one
        # int per node instead of a search to recover the span later.
        self._ends = [0, 0]
        self._suffix = self.ROOT_EMPTY
        self.extend(s)

    # -- construction -------------------------------------------------------

    def add(self, c: Any) -> bool:
        """Append one element. Returns True if it created a new distinct palindrome."""
        s = self._s
        s.append(c)
        pos = len(s) - 1
        cur = self._find_link(self._suffix, pos, c)

        existing = self._trans[cur].get(c)
        if existing is not None:
            self._suffix = existing
            self._occ[existing] += 1
            return False

        node = len(self._len)
        self._len.append(self._len[cur] + 2)
        self._trans.append({})
        self._occ.append(1)
        self._ends.append(pos + 1)
        if self._len[node] == 1:
            self._link.append(self.ROOT_EMPTY)
        else:
            anchor = self._find_link(self._link[cur], pos, c)
            self._link.append(self._trans[anchor][c])

        # Series links: group palindromic suffixes by the gap to their own
        # suffix link. Runs of equal gaps form arithmetic progressions, and
        # there are only O(log n) distinct gaps among the suffixes of any
        # position -- which is what makes min_palindromic_partition O(n log n).
        link = self._link[node]
        diff = self._len[node] - self._len[link]
        self._diff.append(diff)
        self._series.append(link if diff != self._diff[link] else self._series[link])

        self._trans[cur][c] = node
        self._suffix = node
        return True

    def extend(self, items: Sequence[Any]) -> None:
        for c in items:
            self.add(c)

    def _find_link(self, start: int, pos: int, c: Any) -> int:
        """Walk suffix links until ``c + palindrome + c`` fits inside ``s[:pos+1]``."""
        s = self._s
        node = start
        while True:
            back = pos - self._len[node] - 1
            if back >= 0 and s[back] == c:
                return node
            node = self._link[node]

    # -- queries ------------------------------------------------------------

    def count_distinct(self) -> int:
        """Distinct palindromic substrings. Always <= len(s), with equality
        exactly for the "rich" strings."""
        return len(self._len) - 2

    def distinct(self) -> Iterator[Any]:
        """Yield every distinct palindromic substring. O(total output length)."""
        for node in range(2, len(self._len)):
            start, end = self._span(node)
            yield self._materialise(start, end)

    def occurrences(self) -> dict[Any, int]:
        """Map each distinct palindrome to how many times it occurs in ``s``.

        Node counts as built only record occurrences where the palindrome is
        the *longest* palindromic suffix. Every other occurrence is a suffix of
        some longer palindrome, so one reverse pass down the suffix links
        totals them -- nodes are created in increasing length order, so
        iterating backwards visits children before parents.
        """
        counts = list(self._occ)
        for node in range(len(counts) - 1, 1, -1):
            counts[self._link[node]] += counts[node]
        out: dict[Any, int] = {}
        for node in range(2, len(self._len)):
            start, end = self._span(node)
            out[self._materialise(start, end)] = counts[node]
        return out

    def longest_span(self) -> tuple[int, int]:
        """``(start, end)`` of a longest distinct palindrome, leftmost on a tie."""
        best = (0, 0)
        for node in range(2, len(self._len)):
            start, end = self._span(node)
            if end - start > best[1] - best[0]:
                best = (start, end)
        return best

    def _span(self, node: int) -> tuple[int, int]:
        """``(start, end)`` of this node's palindrome's first occurrence."""
        end = self._ends[node]
        return end - self._len[node], end

    def __len__(self) -> int:
        return len(self._s)

    def __repr__(self) -> str:
        return f"Eertree(len={len(self._s)}, distinct={self.count_distinct()})"

    def _materialise(self, start: int, end: int) -> Any:
        chunk = self._s[start:end]
        return "".join(chunk) if chunk and isinstance(chunk[0], str) else tuple(chunk)


# ---------------------------------------------------------------------------
# Palindromic factorisation
# ---------------------------------------------------------------------------


def min_palindromic_partition(s: Sequence[Any]) -> int:
    """Fewest palindromes ``s`` can be cut into. O(n log n) via eertree series links.

    The O(n^2) DP is "``dp[j] = 1 + min(dp[i])`` over palindromic ``s[i:j]``",
    and with :class:`PalindromeIndex` each palindrome test is O(1), so the DP
    is O(n^2) with a tiny constant. This does better by exploiting the
    structure of the set of palindromic suffixes at a position: they fall into
    O(log n) arithmetic progressions of lengths (a consequence of the Fine and
    Wilf periodicity lemma), and the eertree's *series links* name exactly
    those progressions. Each progression's minimum is maintained incrementally
    in ``g``, so a position costs O(log n) instead of O(n).

    Returns 0 for the empty sequence.
    """
    n = len(s)
    if n == 0:
        return 0
    tree = _PartitionEertree()
    dp = [0] + [n + 1] * n
    g: list[int] = [0, 0]

    for i, c in enumerate(s):
        tree.add(c, g)
        pos = i + 1
        node = tree.suffix
        while tree.length[node] > 0:
            series = tree.series[node]
            g[node] = dp[pos - tree.length[series] - tree.diff[node]]
            if tree.diff[node] == tree.diff[tree.link[node]]:
                g[node] = min(g[node], g[tree.link[node]])
            if g[node] + 1 < dp[pos]:
                dp[pos] = g[node] + 1
            node = series
    return dp[n]


class _PartitionEertree:
    """A stripped eertree exposing the arrays `min_palindromic_partition` needs.

    Kept separate from :class:`Eertree` on purpose: the partition DP wants the
    raw parallel arrays in the hot loop, and the public class wants to also
    track occurrences and spans. Fusing them would slow both.
    """

    __slots__ = ("s", "length", "link", "trans", "diff", "series", "suffix")

    def __init__(self) -> None:
        self.s: list[Any] = []
        self.length = [-1, 0]
        self.link = [0, 0]
        self.trans: list[dict[Any, int]] = [{}, {}]
        self.diff = [0, 0]
        self.series = [0, 0]
        self.suffix = 1

    def add(self, c: Any, g: list[int]) -> None:
        self.s.append(c)
        pos = len(self.s) - 1
        cur = self._walk(self.suffix, pos, c)
        existing = self.trans[cur].get(c)
        if existing is not None:
            self.suffix = existing
            return
        node = len(self.length)
        self.length.append(self.length[cur] + 2)
        self.trans.append({})
        if self.length[node] == 1:
            self.link.append(1)
        else:
            anchor = self._walk(self.link[cur], pos, c)
            self.link.append(self.trans[anchor][c])
        link = self.link[node]
        d = self.length[node] - self.length[link]
        self.diff.append(d)
        self.series.append(link if d != self.diff[link] else self.series[link])
        g.append(0)
        self.trans[cur][c] = node
        self.suffix = node

    def _walk(self, node: int, pos: int, c: Any) -> int:
        s = self.s
        while True:
            back = pos - self.length[node] - 1
            if back >= 0 and s[back] == c:
                return node
            node = self.link[node]


def palindromic_partition(s: Sequence[Any]) -> list[Any]:
    """One partition of ``s`` into the fewest possible palindromes.

    The O(n^2) DP with O(1) palindrome tests, kept because reconstructing the
    actual cut needs the predecessor chain, and n^2 with a Manacher-backed
    O(1) test is fast enough for any string you would want to read. Verified
    against :func:`min_palindromic_partition` on every input the tests throw.
    """
    n = len(s)
    if n == 0:
        return []
    idx = PalindromeIndex(s)
    dp = [0] + [n + 1] * n
    cut = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j):
            if dp[i] + 1 < dp[j] and idx.is_palindrome(i, j):
                dp[j], cut[j] = dp[i] + 1, i
    pieces = []
    j = n
    while j:
        pieces.append(s[cut[j] : j])
        j = cut[j]
    return pieces[::-1]


def distinct_palindromes(s: Sequence[Any]) -> list[Any]:
    """Every distinct palindromic substring, shortest first then alphabetical."""
    return sorted(Eertree(s).distinct(), key=lambda p: (len(p), p))


def count_distinct_palindromes(s: Sequence[Any]) -> int:
    """How many distinct palindromic substrings ``s`` has. O(n), and always <= n."""
    return Eertree(s).count_distinct()


# ---------------------------------------------------------------------------
# Baselines, for the comparison the brief asks for
# ---------------------------------------------------------------------------


def naive_longest_palindrome_span(s: Sequence[Any]) -> tuple[int, int]:
    """Expand around all 2n-1 centres. O(n^2) time, O(1) space.

    The honest baseline: same answer, same tie-breaking, and genuinely fast
    for short strings because it allocates nothing. It degrades to its worst
    case exactly on the strings Manacher handles best -- a run of n equal
    characters makes every expansion run the full width.
    """
    n = len(s)
    best_start, best_len = 0, 0
    for centre in range(2 * n - 1):
        i, j = centre // 2, centre // 2 + centre % 2
        while i >= 0 and j < n and s[i] == s[j]:
            i -= 1
            j += 1
        i, j = i + 1, j
        if j - i > best_len or (j - i == best_len and i < best_start):
            best_start, best_len = i, j - i
    return best_start, best_start + best_len


def dp_longest_palindrome_span(s: Sequence[Any]) -> tuple[int, int]:
    """Classic O(n^2) time, O(n^2) space DP over all substrings.

    Included because it is the version most tutorials teach first, and because
    its memory is the reason not to use it: at n = 100000 the table is 10^10
    booleans. It is here to be measured, not used.
    """
    n = len(s)
    if n == 0:
        return 0, 0
    table = [[False] * n for _ in range(n)]
    best_start, best_len = 0, 1
    for i in range(n):
        table[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and (length == 2 or table[i + 1][j - 1]):
                table[i][j] = True
                if length > best_len:
                    best_start, best_len = i, length
    return best_start, best_start + best_len


def brute_force_longest_palindrome_span(s: Sequence[Any]) -> tuple[int, int]:
    """O(n^3): check every substring against its own reverse. The oracle.

    Slow enough to be obviously correct, which is the only property required
    of a test oracle.
    """
    n = len(s)
    best = (0, 0)
    for i in range(n):
        for j in range(i + 1, n + 1):
            chunk = s[i:j]
            if j - i > best[1] - best[0] and list(chunk) == list(chunk)[::-1]:
                best = (i, j)
    return best


# ---------------------------------------------------------------------------
# Unicode: what counts as "the same character read backwards"
# ---------------------------------------------------------------------------

_MARKS = frozenset({"Mn", "Mc", "Me"})
_ZWJ = "‍"


def graphemes(text: str) -> list[str]:
    """Split into user-perceived characters, so reversal does not break accents.

    Reversing "e\\u0301" codepoint-wise gives "\\u0301e" -- a combining acute
    with nothing to combine with, which is not the same character and often
    not even renderable. Palindrome checks on accented text therefore have to
    run over clusters, not codepoints. A pragmatic subset of UAX #29:
    base + combining marks, ZWJ sequences, CRLF, and regional-indicator pairs.
    """
    if not text:
        return []
    out: list[str] = []
    buf = text[0]
    ri_run = 1 if _is_ri(text[0]) else 0
    for ch in text[1:]:
        if unicodedata.category(ch) in _MARKS or ch == _ZWJ or buf.endswith(_ZWJ):
            buf += ch
            ri_run = 0
        elif buf == "\r" and ch == "\n":
            buf += ch
            ri_run = 0
        elif _is_ri(ch) and ri_run % 2 == 1:
            buf += ch
            ri_run += 1
        else:
            out.append(buf)
            buf = ch
            ri_run = 1 if _is_ri(ch) else 0
    out.append(buf)
    return out


def _is_ri(ch: str) -> bool:
    return "\U0001f1e6" <= ch <= "\U0001f1ff"


def relaxed_view(text: str) -> tuple[list[str], list[int]]:
    """Alphanumeric, case-folded, NFC clusters plus a map back to original indices.

    The "A man, a plan, a canal: Panama" reading of "palindrome", where case
    and punctuation do not count. Returning the index map is the part that is
    usually skipped and usually wanted: without it you can tell the user *that*
    there is a 21-character palindrome but not *where* it is in their string.

    ``indices[k]`` is the index in ``text`` where cluster ``k`` starts.
    """
    clusters = graphemes(unicodedata.normalize("NFC", text))
    kept: list[str] = []
    indices: list[int] = []
    offset = 0
    for cluster in clusters:
        if cluster[:1].isalnum():
            kept.append(cluster.casefold())
            indices.append(offset)
        offset += len(cluster)
    return kept, indices


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def verify(*, seed: int = 0, trials: int = 400, verbose: bool = True) -> bool:
    """Cross-check Manacher against the O(n^3) oracle and the identities."""
    import itertools
    import random

    rng = random.Random(seed)
    cases: list[str] = [
        "",
        "a",
        "aa",
        "ab",
        "aba",
        "abba",
        "aaaa",
        "#",
        "$^#",
        "abacaba",
        "forgeeksskeegfor",
        "racecar",
    ]
    # Every binary string up to length 12: the densest palindrome structure there is.
    for length in range(0, 11):
        cases.extend("".join(b) for b in itertools.product("ab", repeat=length))
    for _ in range(trials):
        alphabet = rng.choice(["a", "ab", "abc", "abcdefghij"])
        cases.append("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30))))

    ok = True

    def fail(msg: str) -> None:
        nonlocal ok
        ok = False
        if verbose:
            print(f"  MISMATCH {msg}", file=sys.stderr)

    for case in cases:
        expected = brute_force_longest_palindrome_span(case)
        length = expected[1] - expected[0]
        for name, fn in [
            ("manacher", longest_palindrome_span),
            ("naive", naive_longest_palindrome_span),
            ("dp", dp_longest_palindrome_span),
        ]:
            got = fn(case)
            if got[1] - got[0] != length:
                fail(f"{name} length on {case!r}: {got} vs {expected}")
            elif case and got != expected:
                fail(f"{name} span on {case!r}: {got} vs {expected}")

        idx = PalindromeIndex(case)
        n = len(case)
        for i in range(n + 1):
            for j in range(i, n + 1):
                if idx.is_palindrome(i, j) != (case[i:j] == case[i:j][::-1]):
                    fail(f"is_palindrome({i},{j}) on {case!r}")

        occurrences = sum(
            1
            for i in range(n)
            for j in range(i + 1, n + 1)
            if case[i:j] == case[i:j][::-1]
        )
        if count_palindromic_substrings(case) != occurrences:
            fail(f"occurrence count on {case!r}")

        distinct = {
            case[i:j]
            for i in range(n)
            for j in range(i + 1, n + 1)
            if case[i:j] == case[i:j][::-1]
        }
        if count_distinct_palindromes(case) != len(distinct):
            fail(f"distinct count on {case!r}")
        if set(Eertree(case).distinct()) != distinct:
            fail(f"distinct set on {case!r}")

        if len(case) <= 24:
            pieces = palindromic_partition(case)
            if "".join(pieces) != case:
                fail(f"partition does not reassemble {case!r}")
            if any(p != p[::-1] for p in pieces):
                fail(f"partition piece is not a palindrome in {case!r}")
            if len(pieces) != min_palindromic_partition(case):
                fail(
                    f"partition size on {case!r}: {len(pieces)} vs "
                    f"{min_palindromic_partition(case)}"
                )

    if verbose:
        print(
            f"verify: {len(cases)} strings, Manacher + eertree + partitions vs "
            f"brute force -- {'OK' if ok else 'FAILED'}"
        )
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _report(text: str, relaxed: bool) -> None:
    if relaxed:
        units, indices = relaxed_view(text)
        start, end = longest_palindrome_span(units)
        print(f"input:   {text!r}")
        print(f"relaxed: {''.join(units)!r}   (alphanumeric, case-folded)")
        if end > start:
            origin = indices[start]
            stop = (
                indices[end - 1] + len(graphemes(text)[end - 1])
                if end <= len(indices)
                else len(text)
            )
            print(
                f"longest: {''.join(units[start:end])!r}  "
                f"({end - start} units, from original index {origin})"
            )
            print(f"         as written: {text[origin:stop]!r}")
        else:
            print("longest: '' (no alphanumeric characters)")
        return

    units = graphemes(text)
    idx = PalindromeIndex(units)
    start, end = idx.longest_span()
    print(f"input:   {text!r}  ({len(units)} grapheme clusters)")
    print(f"longest: {''.join(units[start:end])!r}  at [{start}, {end})")
    print(f"occurrences of palindromic substrings: {idx.count()}")
    print(
        f"distinct palindromic substrings:       {count_distinct_palindromes(units)}"
        f"   (at most {len(units)}, always)"
    )
    parts = palindromic_partition(units)
    print(
        f"minimum palindromic partition ({len(parts)} pieces): "
        f"{[''.join(p) for p in parts]}"
    )


def _demo() -> None:
    for text in [
        "forgeeksskeegfor",
        "abacaba",
        "aaaa",
        "abcde",
        "",
        "A man, a plan, a canal: Panama",
    ]:
        _report(text, relaxed=False)
        print()
    print("-- the same phrase, ignoring case and punctuation --")
    _report("A man, a plan, a canal: Panama", relaxed=True)
    print()
    print("-- combining marks: 'e' + U+0301 reversed is not the same character --")
    accented = "ab́a"  # a, b-with-acute, a
    print(f"codepoints: {longest_palindrome(accented)!r}   (wrong: splits the accent)")
    print(f"clusters:   {''.join(longest_palindrome(graphemes(accented)))!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Longest palindromic substring, in linear time."
    )
    parser.add_argument("text", nargs="*", help="the string (reads stdin if omitted)")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="ignore case and non-alphanumeric characters",
    )
    args = parser.parse_args(argv)

    if args.verify:
        return 0 if verify() else 1
    if args.demo:
        _demo()
        return 0

    if args.text:
        text = " ".join(args.text)
    elif not sys.stdin.isatty():
        text = sys.stdin.read().rstrip("\n")
    else:
        parser.error("give a string, --demo, or --verify")
    _report(text, args.relaxed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
