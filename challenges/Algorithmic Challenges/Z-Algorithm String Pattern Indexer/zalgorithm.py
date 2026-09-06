"""The Z-array, without the separator, and what it is actually an index *for*.

The brief is "build the Z-array, use it for multi-pattern matching". Both
halves turn out to hide a better answer than the textbook one.

**The separator is a myth, and the concatenation is a waste.** Every
tutorial searches for ``P`` in ``T`` by building ``P + '#' + T`` and reading
off the positions where ``Z == len(P)``, warning that ``'#'`` must not occur
in the input. Two things are wrong with that:

1. The separator is not needed for *correctness*. For ``i >= m``,
   ``Z[i] >= m`` holds exactly when ``S[i:i+m] == P``, which is exactly a
   match at text offset ``i - m``. No separator, same answer -- proved in the
   module docstring of :func:`z_search_concat` and checked by
   ``test_concatenation_without_separator_still_correct``.
2. The concatenation itself is the real cost: ``O(n + m)`` extra memory to
   search an ``n``-character text. :func:`z_search` runs the same recurrence
   over a *virtual* concatenation and allocates ``O(m)``. Because the box
   only ever moves right, it also never re-reads a text character it has
   passed -- so :func:`z_search_stream` searches an iterator of unbounded
   length in ``O(m)`` memory, which the concatenation cannot do at all.

**Multi-pattern matching is where the Z-array is supposed to lose.** The
honest statement is that ``k`` patterns cost ``k`` scans, ``O(k*n + M)``,
against Aho-Corasick's ``O(n + M + occ)``. But ``k`` is the wrong count.
A single scan against pattern ``P`` decides *every* pattern that is a prefix
of ``P`` at the same time, because ``Q`` occurs at ``j`` iff
``lcp(P, T[j:]) >= |Q|``. So the scans needed are the chains of the
"is-a-prefix-of" partial order, and the minimum number of chains covering it
is exactly the number of **leaves** of the pattern trie -- see
:class:`MultiZMatcher`, which is ``O(L*n + M)`` with ``L <= k``, and the
tight-bound proof in the README. On a dictionary of prefix-closed terms
(``"a"``, ``"ab"``, ``"abc"``, ...) that is one scan instead of ``k``.

The array is also an index in its own right. From one ``O(n)`` build:

    all_borders / smallest_period / string_power      O(n)
    prefix_occurrence_counts                          O(n)
    prefix_function  <->  z_array                     O(n), both directions
    count_distinct_substrings                         O(n^2), online
    tandem_repeats (Main-Lorentz)                     O(n log n)

    uv run python zalgorithm.py --demo
    uv run python zalgorithm.py --verify
    uv run python zalgorithm.py needle "haystack with a needle in it"
    uv run python zalgorithm.py --multi he she his hers -- "ushers"
"""

from __future__ import annotations

import argparse
import heapq
import sys
from bisect import bisect_right
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

__all__ = [
    "z_array",
    "z_array_counted",
    "naive_z_array",
    "z_match_lengths",
    "z_search",
    "z_search_stream",
    "z_search_concat",
    "ZPatternIndex",
    "MultiZMatcher",
    "AhoCorasick",
    "prefix_function",
    "prefix_from_z",
    "z_from_prefix",
    "z_from_prefix_direct",
    "string_from_prefix",
    "all_borders",
    "longest_border",
    "smallest_period",
    "string_power",
    "prefix_occurrence_counts",
    "count_distinct_substrings",
    "tandem_repeat_runs",
    "count_tandem_repeats",
    "tandem_repeats",
    "verify",
    "main",
]


# ---------------------------------------------------------------------------
# The core recurrence
# ---------------------------------------------------------------------------


def z_array(s: Sequence[Any]) -> list[int]:
    """Return ``z`` where ``z[i] = len(lcp(s, s[i:]))``, and ``z[0] = len(s)``.

    ``z[0]`` is genuinely ``n`` -- the whole string is its own longest common
    prefix -- and defining it that way rather than the common ``0`` placeholder
    is what makes :func:`prefix_from_z` and :func:`z_from_prefix` exact
    inverses with no special case at index 0.

    Works on any sequence whose elements support ``==``: ``str``, ``bytes``,
    ``list``, ``tuple``, a list of grapheme clusters.

    The invariant is the *box* ``[l, r)``: the interval reaching furthest right
    that is known to equal a prefix, ``s[l:r] == s[:r-l]``. For ``i`` inside
    it, ``s[i:r]`` mirrors ``s[i-l:r-l]``, so ``z[i-l]`` is already the answer
    whenever it is strictly smaller than ``r - i``; when it is not, the box
    stops short of the truth and the only way forward is character comparison
    -- which then pushes ``r`` right by exactly as many steps as it costs.
    ``r`` never decreases and never exceeds ``n``, so the comparison loop runs
    at most ``n`` times in total. :func:`z_array_counted` measures that.
    """
    n = len(s)
    if n == 0:
        return []
    z = [0] * n
    z[0] = n
    l = r = 0
    for i in range(1, n):
        k = 0
        if i < r:
            k = z[i - l]
            if k < r - i:
                z[i] = k
                continue
            k = r - i
        while i + k < n and s[k] == s[i + k]:
            k += 1
        z[i] = k
        l, r = i, i + k
    return z


def z_array_counted(s: Sequence[Any]) -> tuple[list[int], int]:
    """``(z_array(s), extensions)`` -- the amortisation bound as a measurement.

    ``extensions`` counts successful character comparisons in the extension
    loop. The claim proved above is ``extensions <= n``; the tests assert it
    on the inputs built to break it.
    """
    n = len(s)
    if n == 0:
        return [], 0
    z = [0] * n
    z[0] = n
    l = r = 0
    extensions = 0
    for i in range(1, n):
        k = 0
        if i < r:
            k = z[i - l]
            if k < r - i:
                z[i] = k
                continue
            k = r - i
        while i + k < n and s[k] == s[i + k]:
            k += 1
            extensions += 1
        z[i] = k
        l, r = i, i + k
    return z, extensions


def naive_z_array(s: Sequence[Any]) -> list[int]:
    """The ``O(n^2)`` definition, for cross-checking the linear version."""
    n = len(s)
    z = [0] * n
    if n:
        z[0] = n
    for i in range(1, n):
        k = 0
        while i + k < n and s[k] == s[i + k]:
            k += 1
        z[i] = k
    return z


# ---------------------------------------------------------------------------
# Searching: the same recurrence over a concatenation that is never built
# ---------------------------------------------------------------------------


def z_match_lengths(pattern: Sequence[Any], text: Sequence[Any]) -> list[int]:
    """``out[j] = len(lcp(pattern, text[j:]))`` for every ``j``, in ``O(n + m)``.

    This is the Z-array of ``pattern + text`` restricted to the text half, with
    neither the concatenation nor the separator. The box ``[l, r)`` now lives
    in ``text`` and mirrors against ``z_array(pattern)``:

        text[l:r] == pattern[:r-l]  =>  text[j:r] == pattern[j-l:r-l]

    so ``z[j-l]`` answers position ``j`` outright whenever it is smaller than
    ``r - j``. Space is ``O(m)`` for the pattern's own Z-array (plus the
    ``O(n)`` result), against ``O(n + m)`` for the concatenation.

    Values are naturally capped at ``m``, so the result doubles as a
    "longest prefix of the pattern occurring here" table -- which is what
    :class:`MultiZMatcher` reads to settle a whole chain of patterns at once.
    """
    m, n = len(pattern), len(text)
    out = [0] * n
    if m == 0 or n == 0:
        return out
    z = z_array(pattern)
    l = r = 0
    for j in range(n):
        k = 0
        if j < r:
            k = z[j - l]
            if k < r - j:
                out[j] = k
                continue
            k = r - j
        while k < m and j + k < n and pattern[k] == text[j + k]:
            k += 1
        out[j] = k
        l, r = j, j + k
    return out


def z_search(pattern: Sequence[Any], text: Sequence[Any]) -> Iterator[int]:
    """Yield every start offset where ``pattern`` occurs in ``text``.

    Overlapping occurrences are all reported, in increasing order. ``O(n + m)``
    time, ``O(m)`` working space, and no result list is materialised.

    The empty pattern occurs at each of the ``n + 1`` gap positions, ``n``
    included -- the same convention as ``str.find`` (``"abc".find("") == 0``)
    and ``re.finditer``, and the one that keeps ``text[j:j+m]`` valid at every
    reported ``j``.
    """
    m, n = len(pattern), len(text)
    if m == 0:
        yield from range(n + 1)
        return
    if m > n:
        return
    z = z_array(pattern)
    l = r = 0
    for j in range(n - m + 1):
        k = 0
        if j < r:
            k = z[j - l]
            if k < r - j:
                continue
            k = r - j
        while k < m and j + k < n and pattern[k] == text[j + k]:
            k += 1
        if k == m:
            yield j
        l, r = j, j + k


def z_search_stream(pattern: Sequence[Any], stream: Iterable[Any]) -> Iterator[int]:
    """:func:`z_search` over an iterator, in ``O(m)`` memory and one pass.

    The box only ever moves right, and the extension loop only ever reads at
    or past ``r``, so the algorithm never needs a text character older than
    the current position. That makes it a genuine streaming matcher: a buffer
    of at most ``m + 1`` elements suffices no matter how long the stream is.
    The concatenation formulation cannot do this -- it needs the whole text
    resident before it starts.

    ``stream`` may be a file object iterated by character, a generator, or a
    chunked reader flattened with ``itertools.chain``.
    """
    m = len(pattern)
    it = iter(stream)
    if m == 0:
        # One gap before every element and one after the last.
        yield 0
        for pos, _ in enumerate(it, start=1):
            yield pos
        return

    z = z_array(pattern)
    buf: deque[Any] = deque()
    base = 0  # text index of buf[0]
    exhausted = False

    def ensure(count: int) -> bool:
        """Make ``len(buf) >= count`` if the stream allows. True if it does."""
        nonlocal exhausted
        while len(buf) < count and not exhausted:
            try:
                buf.append(next(it))
            except StopIteration:
                exhausted = True
        return len(buf) >= count

    j = 0
    l = r = 0
    while True:
        if not ensure(j - base + 1):
            return  # j is past the end of the text
        k = 0
        if j < r:
            k = z[j - l]
            if k < r - j:
                j += 1
                while base < j:
                    buf.popleft()
                    base += 1
                continue
            k = r - j
        while k < m and ensure(j - base + k + 1) and pattern[k] == buf[j - base + k]:
            k += 1
        if k == m:
            yield j
        l, r = j, j + k
        j += 1
        while base < j:
            buf.popleft()
            base += 1


def z_search_concat(
    pattern: Sequence[Any],
    text: Sequence[Any],
    *,
    separator: Any = None,
) -> Iterator[int]:
    """The textbook version: build ``P + sep + T``, read off ``Z == m``.

    Kept because it is the version everyone writes and because it settles the
    folklore. With ``separator=None`` **no separator is inserted at all**, and
    the result is still exactly correct:

        for i >= m:   Z[i] >= m  <=>  S[i:i+m] == S[:m] == P  <=>  match at i-m

    The implication runs both ways by the definition of ``Z``, and ``i >= m``
    forces the occurrence to start at or after ``T[0]``, so nothing straddles
    the join. The separator is not a correctness device; it only caps ``Z`` at
    ``m`` inside the text region, which this function never relies on.

    What it *does* cost is ``O(n + m)`` memory for a copy of the text.
    :func:`z_search` is the same algorithm without that. Use this one to check
    that one, not in anger.
    """
    m, n = len(pattern), len(text)
    if m == 0:
        yield from range(n + 1)
        return
    if isinstance(pattern, str) and isinstance(text, str):
        joined: Sequence[Any] = (
            pattern + ("" if separator is None else separator) + text
        )
    elif isinstance(pattern, (bytes, bytearray)) and isinstance(
        text, (bytes, bytearray)
    ):
        mid = b"" if separator is None else bytes(separator)
        joined = bytes(pattern) + mid + bytes(text)
    else:
        joined = list(pattern) + ([] if separator is None else [separator]) + list(text)
    offset = m + (0 if separator is None else 1)
    z = z_array(joined)
    for i in range(offset, len(joined)):
        if z[i] >= m:
            yield i - offset


class ZPatternIndex:
    """A pattern compiled once, searched against many texts.

    The Z-array of the pattern is the whole index: ``O(m)`` ints, built once,
    reused for every ``search``. It also answers "how long a prefix of the
    pattern starts at ``pattern[i]``" in ``O(1)``, which is the primitive
    behind :attr:`borders` and :attr:`period`.
    """

    __slots__ = ("pattern", "z", "_borders", "_period")

    def __init__(self, pattern: Sequence[Any]) -> None:
        self.pattern = pattern
        self.z = z_array(pattern)
        self._borders: list[int] | None = None
        self._period: int | None = None

    def __len__(self) -> int:
        return len(self.pattern)

    def search(self, text: Sequence[Any]) -> Iterator[int]:
        """Every start offset of the pattern in ``text``, overlaps included."""
        return z_search(self.pattern, text)

    def search_stream(self, stream: Iterable[Any]) -> Iterator[int]:
        """:meth:`search` over an iterator, in ``O(m)`` memory and one pass."""
        return z_search_stream(self.pattern, stream)

    def find(self, text: Sequence[Any]) -> int:
        """First occurrence, or ``-1``. Stops the scan at the first hit."""
        for pos in z_search(self.pattern, text):
            return pos
        return -1

    def count(self, text: Sequence[Any]) -> int:
        """Number of (possibly overlapping) occurrences."""
        return sum(1 for _ in z_search(self.pattern, text))

    def match_lengths(self, text: Sequence[Any]) -> list[int]:
        """``lcp(pattern, text[j:])`` at every ``j`` -- partial matches too."""
        return z_match_lengths(self.pattern, text)

    @property
    def borders(self) -> list[int]:
        """All border lengths, ascending. A border is a proper prefix that is
        also a suffix."""
        if self._borders is None:
            self._borders = all_borders(self.pattern, z=self.z)
        return self._borders

    @property
    def period(self) -> int:
        """Smallest ``p`` with ``pattern[i] == pattern[i+p]`` for all valid ``i``."""
        if self._period is None:
            self._period = smallest_period(self.pattern, z=self.z)
        return self._period

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ZPatternIndex({self.pattern!r})"


# ---------------------------------------------------------------------------
# Multi-pattern: one scan per chain of the prefix order, not one per pattern
# ---------------------------------------------------------------------------


def _hashable(seq: Sequence[Any]) -> Any:
    if isinstance(seq, (str, bytes, tuple)):
        return seq
    if isinstance(seq, bytearray):
        return bytes(seq)
    return tuple(seq)


class MultiZMatcher:
    """Multi-pattern search with one Z-scan per *chain*, not per pattern.

    A single scan of ``text`` against a pattern ``P`` produces
    ``lcp(P, T[j:])`` at every ``j``. That number settles not just ``P`` but
    every pattern ``Q`` that is a prefix of ``P``, since ``Q`` occurs at ``j``
    exactly when ``lcp(P, T[j:]) >= |Q|``. So a scan resolves a *chain* of the
    "is-a-proper-prefix-of" partial order, and the number of scans needed is
    the minimum number of chains covering the pattern set.

    That order is a forest (a pattern's parent is its longest proper prefix in
    the set), so its root-to-leaf paths are ``L`` chains that cover it, where
    ``L`` is the number of leaves of the pattern trie. ``L`` is also a lower
    bound: the leaves are pairwise incomparable, and two incomparable patterns
    can never share a scan, so no cover is smaller. Hence ``L`` scans exactly
    -- Dilworth's theorem, with the maximum antichain in hand.

    Cost is ``O(L*n + M)`` for ``M = sum(len(p))``, against ``O(k*n + M)`` for
    the obvious loop and ``O(n + M + occ)`` for :class:`AhoCorasick`. ``L <= k``
    always, with equality iff no pattern is a prefix of another -- so this is
    never worse than the naive loop and is ``k`` times better on a
    prefix-closed dictionary. It does not beat Aho-Corasick asymptotically and
    is not meant to; see the README for where each one actually wins.

    Duplicate patterns are matched once each (all their indices are reported).
    """

    def __init__(self, patterns: Iterable[Sequence[Any]]) -> None:
        self.patterns: list[Sequence[Any]] = list(patterns)
        self._empty_indices: list[int] = []
        by_key: dict[Any, list[int]] = {}
        for idx, pat in enumerate(self.patterns):
            if len(pat) == 0:
                self._empty_indices.append(idx)
                continue
            by_key.setdefault(_hashable(pat), []).append(idx)
        self._by_key = by_key
        self._chains: list[tuple[Sequence[Any], list[tuple[int, Any]]]] = []
        if by_key:
            self._build_chains()

    # -- trie, then one chain per leaf ---------------------------------------

    def _build_chains(self) -> None:
        # Trie over the distinct patterns. Nodes are indices into parallel
        # lists; `terminal[v]` is the pattern key ending at v, if any.
        children: list[dict[Any, int]] = [{}]
        terminal: list[Any] = [None]
        for key in self._by_key:
            node = 0
            for ch in key:
                nxt = children[node].get(ch)
                if nxt is None:
                    nxt = len(children)
                    children.append({})
                    terminal.append(None)
                    children[node][ch] = nxt
                node = nxt
            terminal[node] = key

        # Iterative DFS. Terminals collected on the way down are handed to the
        # first child's subtree, so every pattern lands in exactly one chain.
        # A leaf of the trie is always a terminal, so each chain's scan pattern
        # is a real pattern -- never a synthetic one.
        stack: list[tuple[int, list[Any], bool]] = [(0, [], True)]
        while stack:
            node, pending, inherit = stack.pop()
            here = list(pending) if inherit else []
            if terminal[node] is not None:
                here.append(terminal[node])
            kids = list(children[node].values())
            if not kids:
                leaf_key = terminal[node]
                members = [(len(k), k) for k in here]
                members.sort()
                self._chains.append((leaf_key, members))
                continue
            # `reversed` so the *first* child (in insertion order) is popped
            # first and therefore inherits `here`.
            for pos, kid in enumerate(reversed(kids)):
                stack.append((kid, here, pos == len(kids) - 1))

    @property
    def chain_count(self) -> int:
        """``L`` -- the number of Z-scans one :meth:`search` performs."""
        return len(self._chains)

    @property
    def chains(self) -> list[list[Sequence[Any]]]:
        """The chains themselves, shortest pattern first. Diagnostic."""
        return [[key for _, key in members] for _, members in self._chains]

    # -- searching ------------------------------------------------------------

    def search(self, text: Sequence[Any]) -> Iterator[tuple[int, int]]:
        """Yield ``(pattern_index, start_offset)`` for every occurrence.

        Results arrive grouped by chain. Use :meth:`finditer` for text order.
        """
        n = len(text)
        for idx in self._empty_indices:
            for pos in range(n + 1):
                yield idx, pos
        for leaf, members in self._chains:
            for pos, idx in self._chain_stream(leaf, members, text):
                yield idx, pos

    def finditer(self, text: Sequence[Any]) -> Iterator[tuple[int, int]]:
        """:meth:`search`, merged into ascending text order.

        Each chain already emits its members in ascending ``(position, index)``,
        so this is a ``heapq.merge`` of ``L`` sorted streams -- no full sort and
        no list of all occurrences in memory.
        """
        n = len(text)
        streams = []
        if self._empty_indices:
            streams.append(
                (pos, idx) for pos in range(n + 1) for idx in self._empty_indices
            )
        for leaf, members in self._chains:
            streams.append(self._chain_stream(leaf, members, text))
        for pos, idx in heapq.merge(*streams):
            yield idx, pos

    def _chain_stream(
        self,
        leaf: Sequence[Any],
        members: list[tuple[int, Any]],
        text: Sequence[Any],
    ) -> Iterator[tuple[int, int]]:
        """One Z-scan, every member of the chain settled from it."""
        if len(members) == 1:
            # A chain of one is just a single-pattern search, and `z_search`
            # never materialises the length table -- so the L = k case (no
            # pattern is a prefix of another) costs exactly what the obvious
            # per-pattern loop costs, never more.
            indices = self._by_key[members[0][1]]
            for pos in z_search(leaf, text):
                for idx in indices:
                    yield pos, idx
            return
        # Members ascend by length, so the prefix of the list that matches at
        # `pos` is decided by one bisect rather than a walk per member: the
        # scan stays O(n) plus O(occ), not O(n * chain length).
        lengths = z_match_lengths(leaf, text)
        cutoffs = [length for length, _ in members]
        keys = [key for _, key in members]
        by_key = self._by_key

        for pos, got in enumerate(lengths):
            if not got:
                continue
            take = bisect_right(cutoffs, got)
            if not take:
                continue
            here: list[int] = []
            for key in keys[:take]:
                here.extend(by_key[key])
            # Sorted so each chain's stream is ordered by (position, index) --
            # the key heapq.merge assumes in finditer.
            here.sort()
            for idx in here:
                yield pos, idx

    def count(self, text: Sequence[Any]) -> int:
        """Total number of occurrences across all patterns."""
        return sum(1 for _ in self.search(text))


class AhoCorasick:
    """The ``O(n + M + occ)`` baseline, for the comparison to be honest.

    Goto/fail/output automaton (Aho & Corasick 1975) with compressed output
    links. It is here so the README's claim -- that the Z-array is the wrong
    tool once ``L`` is large -- is a measurement rather than an assertion.
    """

    def __init__(self, patterns: Iterable[Sequence[Any]]) -> None:
        self.patterns = list(patterns)
        self._goto: list[dict[Any, int]] = [{}]
        self._fail: list[int] = [0]
        self._out: list[list[int]] = [[]]
        self._out_link: list[int] = [-1]
        for idx, pat in enumerate(self.patterns):
            node = 0
            for ch in pat:
                nxt = self._goto[node].get(ch)
                if nxt is None:
                    nxt = len(self._goto)
                    self._goto.append({})
                    self._fail.append(0)
                    self._out.append([])
                    self._out_link.append(-1)
                    self._goto[node][ch] = nxt
                node = nxt
            self._out[node].append(idx)
        self._build_links()

    def _build_links(self) -> None:
        queue = deque()
        for nxt in self._goto[0].values():
            self._fail[nxt] = 0
            queue.append(nxt)
        while queue:
            node = queue.popleft()
            f = self._fail[node]
            self._out_link[node] = f if self._out[f] else self._out_link[f]
            for ch, nxt in self._goto[node].items():
                probe = f
                while probe and ch not in self._goto[probe]:
                    probe = self._fail[probe]
                self._fail[nxt] = self._goto[probe].get(ch, 0)
                queue.append(nxt)

    def search(self, text: Sequence[Any]) -> Iterator[tuple[int, int]]:
        """Yield ``(pattern_index, start_offset)``, in ascending end position."""
        empties = [i for i, p in enumerate(self.patterns) if len(p) == 0]
        for idx in empties:
            for pos in range(len(text) + 1):
                yield idx, pos
        node = 0
        goto, fail, out, out_link = self._goto, self._fail, self._out, self._out_link
        for i, ch in enumerate(text):
            while node and ch not in goto[node]:
                node = fail[node]
            node = goto[node].get(ch, 0)
            probe = node if out[node] else out_link[node]
            while probe != -1:
                for idx in out[probe]:
                    yield idx, i - len(self.patterns[idx]) + 1
                probe = out_link[probe]


# ---------------------------------------------------------------------------
# The Z-array and the prefix function are the same information
# ---------------------------------------------------------------------------


def prefix_function(s: Sequence[Any]) -> list[int]:
    """KMP's failure function: ``pi[i]`` is the longest proper border of
    ``s[:i+1]``. ``O(n)``, here for the conversions to be checked against."""
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


def prefix_from_z(z: Sequence[int]) -> list[int]:
    """Prefix function from the Z-array alone, in ``O(n)`` and with no string.

    ``z[i] = L`` says ``s[i:i+L]`` is a prefix, so the prefix of length ``j+1``
    is a border of ``s[:i+j+1]`` for every ``j < L``. Writing the *largest*
    such ``j`` first and stopping at the first slot already filled means each
    ``pi`` entry is written exactly once -- which is what makes the nested
    loop linear rather than quadratic.
    """
    n = len(z)
    pi = [0] * n
    for i in range(1, n):
        for j in range(z[i] - 1, -1, -1):
            if pi[i + j] > 0:
                break
            pi[i + j] = j + 1
    return pi


def string_from_prefix(pi: Sequence[int]) -> list[int]:
    """Rebuild *a* string with prefix function ``pi``, in ``O(n)``.

    The prefix function pins a string down to renaming of the alphabet, and
    the reconstruction is forced at every step: ``pi[i] > 0`` says
    ``s[i] == s[pi[i]-1]``, and ``pi[i] == 0`` says ``s[i]`` differs from every
    candidate the KMP loop would have tried. Handing out a **fresh symbol**
    in the second case satisfies that with no search: fresh symbols create no
    equalities, and every equality this builds is one ``pi`` already asserted.

    Returns integer symbols, so the alphabet is unbounded and no character is
    reserved. ``string_from_prefix(prefix_function(s))`` is ``s`` relabelled,
    not ``s`` itself -- ``"aab"`` and ``"xxy"`` have the same prefix function
    and are indistinguishable from it.
    """
    n = len(pi)
    s = [0] * n
    fresh = 1
    for i in range(1, n):
        if pi[i] > 0:
            s[i] = s[pi[i] - 1]
        else:
            s[i] = fresh
            fresh += 1
    return s


def z_from_prefix(pi: Sequence[int]) -> list[int]:
    """Z-array from the prefix function alone, in ``O(n)``.

    Rebuild the string (up to renaming) and run the Z-algorithm on it. That
    is a legitimate conversion, not a dodge: both arrays are invariant under
    renaming the alphabet, so any string with prefix function ``pi`` has the
    Z-array of every such string. The two ``O(n)`` steps compose to ``O(n)``,
    and there is no case analysis to get wrong.

    :func:`z_from_prefix_direct` is the transfer-in-place version, kept
    because it is the one usually quoted -- and because the two cross-check
    each other in the test suite.
    """
    return z_array(string_from_prefix(pi))


def z_from_prefix_direct(pi: Sequence[int]) -> list[int]:
    """Z-array from the prefix function without rebuilding a string. ``O(n)``.

    Each ``pi[i] > 0`` places one known value, ``z[i - pi[i] + 1] = pi[i]``
    (the longest border ending at ``i`` starts there). Distinct ``i`` mapping
    to the same start carry distinct -- and monotone -- ``pi``, so plain
    assignment already keeps the maximum. The second pass propagates each
    known value rightward inside its own box.

    The transfer rule is ``z[i+j] = min(z[j], z[i] - j)``, and the ``min``
    is not decoration: the abbreviated ``z[i+j] = z[i] - j`` that circulates
    for this conversion reports ``z = [4, 0, 2, 1]`` for ``"abab"``, whose
    real Z-array is ``[4, 0, 2, 0]`` -- position 3 is ``"b"`` and shares no
    prefix with ``"abab"`` at all. ``test_z_from_prefix_needs_the_min`` pins
    that down. The ``break`` covers the remaining case, where the first pass
    already knows a larger value than the box can justify; restarting at that
    position is what keeps the whole thing linear.
    """
    n = len(pi)
    if n == 0:
        return []
    z = [0] * n
    for i in range(1, n):
        if pi[i] > 0:
            z[i - pi[i] + 1] = pi[i]
    z[0] = n
    i = 1
    while i < n:
        j = i
        if z[i] > 0:
            for k in range(1, z[i]):
                if z[i + k] > z[i] - k:
                    break
                z[i + k] = min(z[k], z[i] - k)
                j = i + k
        i = j + 1
    return z


# ---------------------------------------------------------------------------
# What one Z-array tells you about the string
# ---------------------------------------------------------------------------


def all_borders(s: Sequence[Any], *, z: Sequence[int] | None = None) -> list[int]:
    """Every border length of ``s``, ascending. ``O(n)``.

    ``s`` has a border of length ``n - i`` exactly when the suffix starting at
    ``i`` is a prefix, i.e. ``i + z[i] == n``. Reading them off the Z-array
    takes one pass; the prefix-function route needs the ``pi`` chain instead.
    """
    n = len(s)
    if n == 0:
        return []
    if z is None:
        z = z_array(s)
    return [n - i for i in range(n - 1, 0, -1) if i + z[i] == n]


def longest_border(s: Sequence[Any], *, z: Sequence[int] | None = None) -> int:
    """Length of the longest proper border, or 0. ``O(n)``."""
    borders = all_borders(s, z=z)
    return borders[-1] if borders else 0


def smallest_period(s: Sequence[Any], *, z: Sequence[int] | None = None) -> int:
    """Smallest ``p`` with ``s[i] == s[i+p]`` for all ``i + p < n``. ``O(n)``.

    ``p = n - longest_border``, which the Z-array gives as the smallest ``i``
    with ``i + z[i] == n``. Note this is the *weak* period: ``p`` need not
    divide ``n`` (``"aabaa"`` has period 3 and length 5). Use
    :func:`string_power` for the exact-repetition question.
    """
    n = len(s)
    if n == 0:
        return 0
    if z is None:
        z = z_array(s)
    for i in range(1, n):
        if i + z[i] == n:
            return i
    return n


def string_power(
    s: Sequence[Any], *, z: Sequence[int] | None = None
) -> tuple[int, int]:
    """``(period, exponent)`` for the largest ``e`` with ``s == base * e``.

    ``("abcabcabc")`` gives ``(3, 3)``; ``("aabaa")`` gives ``(5, 1)`` because
    its period 3 does not divide 5. This is the "smallest generating unit"
    question, and it needs the divisibility check that :func:`smallest_period`
    deliberately does not make.
    """
    n = len(s)
    if n == 0:
        return 0, 0
    p = smallest_period(s, z=z)
    if n % p:
        return n, 1
    return p, n // p


def prefix_occurrence_counts(
    s: Sequence[Any], *, z: Sequence[int] | None = None
) -> list[int]:
    """``counts[L]`` = occurrences of ``s[:L]`` inside ``s``, for ``L`` in
    ``0..n``. ``O(n)``.

    Every occurrence of a prefix of length ``L`` at position ``i > 0`` is
    counted by ``z[i] >= L``, so bucketing the ``z`` values and taking a suffix
    sum answers all ``n`` questions at once. ``counts[0]`` is ``n + 1`` (the
    empty prefix sits in every gap); position 0 contributes the ``+1`` to every
    non-empty length.
    """
    n = len(s)
    counts = [0] * (n + 1)
    if n == 0:
        return [1]
    if z is None:
        z = z_array(s)
    for i in range(1, n):
        if z[i]:
            counts[z[i]] += 1
    for length in range(n - 1, 0, -1):
        counts[length] += counts[length + 1]
    for length in range(1, n + 1):
        counts[length] += 1
    counts[0] = n + 1
    return counts


def count_distinct_substrings(s: Sequence[Any]) -> int:
    """Number of distinct non-empty substrings, in ``O(n^2)`` and online.

    Appending ``s[k]`` adds ``k + 1`` new suffixes; the ones already seen are
    exactly those that also occur earlier, and the longest of those has length
    ``max(z_array(reverse(s[:k+1])))``. So each step is one Z-array of the
    reversed prefix. Quadratic overall -- a suffix automaton does this in
    ``O(n)`` -- but it is *online*, which the linear method has to work for.
    """
    total = 0
    rev: list[Any] = []
    for k, ch in enumerate(s):
        rev.insert(0, ch)
        z = z_array(rev)
        longest = max(z[1:], default=0)
        total += (k + 1) - longest
    return total


# ---------------------------------------------------------------------------
# Main-Lorentz: all tandem repeats in O(n log n), on four Z-arrays a level
# ---------------------------------------------------------------------------


def tandem_repeat_runs(s: Sequence[Any]) -> list[tuple[int, int, int]]:
    """All tandem repeats (squares) of ``s`` as ``(first_start, count, period)``.

    A tandem repeat is a substring equal to ``ww``; ``period`` is ``len(w)``,
    and the run means squares of that period start at every position in
    ``first_start .. first_start + count - 1``.

    Main & Lorentz (1984): split the string, recurse, then find the repeats
    that *cross* the split. Crossing repeats of a given period sit in
    contiguous runs, each decided by two longest-common-extension queries --
    one forward, one backward -- and those are exactly Z-array lookups. Four
    Z-arrays per level, ``O(n)`` each, ``O(log n)`` levels: ``O(n log n)``.

    The run encoding is not a convenience. ``"a" * n`` contains
    ``floor(n/2) * ceil(n/2)`` squares -- quadratically many -- so any function
    returning them individually cannot be ``O(n log n)``. The runs are
    ``O(n log n)`` and lose nothing; :func:`tandem_repeats` expands them when
    the caller really wants each one.
    """
    seq = list(s)
    out: list[tuple[int, int, int]] = []
    _find_repeats(seq, 0, out)
    out.sort()
    return out


def _find_repeats(s: list[Any], shift: int, out: list[tuple[int, int, int]]) -> None:
    n = len(s)
    if n < 2:
        return
    nu = n // 2
    nv = n - nu
    u, v = s[:nu], s[nu:]
    ru, rv = u[::-1], v[::-1]

    _find_repeats(u, shift, out)
    _find_repeats(v, shift + nu, out)

    # Four longest-common-extension tables. The textbook builds z2/z3 as
    # `z_array(v + '#' + u)` and `z_array(ru + '#' + rv)`; `z_match_lengths`
    # is the same table without the sentinel or the copy, which also means
    # this works on sequences with no spare alphabet symbol.
    z1 = z_array(ru)
    z2 = z_match_lengths(v, u)
    z3 = z_match_lengths(ru, rv)
    z4 = z_array(v)

    for cntr in range(n):
        if cntr < nu:
            left = True
            length = nu - cntr
            k1 = z1[nu - cntr] if nu - cntr < len(z1) else 0
            k2 = z2[cntr]
        else:
            left = False
            length = cntr - nu + 1
            k1 = z3[nv - 1 - (cntr - nu)]
            k2 = z4[cntr - nu + 1] if cntr - nu + 1 < len(z4) else 0
        if k1 + k2 < length:
            continue
        lo = max(1, length - k2)
        hi = min(length, k1)
        if left:
            hi = min(hi, length - 1)
        if lo > hi:
            continue
        # pos decreases by 1 as l1 increases by 1, in both branches, so the
        # positions form one contiguous run.
        if left:
            first = shift + cntr - hi
        else:
            first = shift + cntr - length - hi + 1
        out.append((first, hi - lo + 1, length))


def count_tandem_repeats(s: Sequence[Any]) -> int:
    """How many ``(start, period)`` squares ``s`` contains. ``O(n log n)``."""
    return sum(count for _, count, _ in tandem_repeat_runs(s))


def tandem_repeats(s: Sequence[Any]) -> list[tuple[int, int]]:
    """Every square as ``(start, period)``, sorted. Output can be ``O(n^2)``."""
    seen = []
    for first, count, period in tandem_repeat_runs(s):
        seen.extend((first + d, period) for d in range(count))
    seen.sort()
    return seen


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def verify(*, seed: int = 0, trials: int = 300, verbose: bool = True) -> bool:
    """Randomised cross-checks of every claim the module makes. ``True`` if
    all hold."""
    import itertools
    import random

    rng = random.Random(seed)
    checks: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        checks.append((name, ok))

    # Exhaustive over short binary strings: the fast and slow paths agree,
    # and the two representations of the same information round-trip.
    ok_z = ok_pi = ok_zpi = ok_border = ok_restore = True
    for length in range(0, 13):
        for bits in itertools.product("ab", repeat=length):
            s = "".join(bits)
            z = z_array(s)
            ok_z &= z == naive_z_array(s)
            pi = prefix_function(s)
            ok_pi &= prefix_from_z(z) == pi
            ok_zpi &= z_from_prefix(pi) == z
            ok_zpi &= z_from_prefix_direct(pi) == z
            ok_restore &= prefix_function(string_from_prefix(pi)) == pi
            expect = [b for b in range(1, length) if s[:b] == s[length - b :]]
            ok_border &= all_borders(s) == expect
    check("z_array == naive on all binary strings to length 12", ok_z)
    check("prefix_from_z == prefix_function", ok_pi)
    check("z_from_prefix (both routes) == z_array", ok_zpi)
    check("string_from_prefix round-trips its prefix function", ok_restore)
    check("all_borders matches the definition", ok_border)

    # Search: three implementations, one answer.
    ok_search = ok_stream = ok_concat = True
    for _ in range(trials):
        alpha = "ab" if rng.random() < 0.7 else "abcde"
        text = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 60)))
        pat = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 5)))
        brute = [
            i for i in range(len(text) - len(pat) + 1) if text[i : i + len(pat)] == pat
        ]
        ok_search &= list(z_search(pat, text)) == brute
        ok_stream &= list(z_search_stream(pat, iter(text))) == brute
        ok_concat &= list(z_search_concat(pat, text)) == brute
        ok_concat &= list(z_search_concat(pat, text, separator="\x00")) == brute
    check("z_search == brute force", ok_search)
    check("z_search_stream == brute force", ok_stream)
    check("z_search_concat (with and without separator) == brute force", ok_concat)

    # Multi-pattern: the chain matcher, Aho-Corasick and brute force agree.
    ok_multi = ok_chains = True
    for _ in range(trials // 2 or 1):
        text = "".join(rng.choice("abc") for _ in range(rng.randint(0, 40)))
        pats = [
            "".join(rng.choice("abc") for _ in range(rng.randint(1, 4)))
            for _ in range(rng.randint(1, 6))
        ]
        brute = sorted(
            (i, p)
            for p in range(len(pats))
            for i in range(len(text) - len(pats[p]) + 1)
            if text[i : i + len(pats[p])] == pats[p]
        )
        mz = MultiZMatcher(pats)
        ok_multi &= sorted((pos, idx) for idx, pos in mz.search(text)) == brute
        ok_multi &= (
            sorted((pos, idx) for idx, pos in AhoCorasick(pats).search(text)) == brute
        )
        ok_multi &= [(pos, idx) for idx, pos in mz.finditer(text)] == sorted(
            (pos, idx) for idx, pos in mz.search(text)
        )
        distinct = {p for p in pats}
        maximal = {
            p for p in distinct if not any(q != p and q.startswith(p) for q in distinct)
        }
        ok_chains &= mz.chain_count == len(maximal)
    check("MultiZMatcher == AhoCorasick == brute force", ok_multi)
    check("chain count equals the number of maximal patterns", ok_chains)

    # Periods, powers, prefix counts, distinct substrings, tandem repeats.
    ok_period = ok_counts = ok_distinct = ok_tandem = True
    for _ in range(trials):
        s = "".join(rng.choice("ab") for _ in range(rng.randint(0, 22)))
        n = len(s)
        p = smallest_period(s)
        ok_period &= all(s[i] == s[i + p] for i in range(n - p))
        ok_period &= p == n or not any(
            all(s[i] == s[i + q] for i in range(n - q)) for q in range(1, p)
        )
        counts = prefix_occurrence_counts(s)
        ok_counts &= all(
            counts[L] == sum(1 for i in range(n - L + 1) if s[i : i + L] == s[:L])
            for L in range(0, n + 1)
        )
        ok_distinct &= count_distinct_substrings(s) == len(
            {s[i:j] for i in range(n) for j in range(i + 1, n + 1)}
        )
        brute_sq = sorted(
            (i, w)
            for w in range(1, n // 2 + 1)
            for i in range(n - 2 * w + 1)
            if s[i : i + w] == s[i + w : i + 2 * w]
        )
        ok_tandem &= tandem_repeats(s) == brute_sq
    check("smallest_period is a period and is smallest", ok_period)
    check("prefix_occurrence_counts matches the definition", ok_counts)
    check("count_distinct_substrings matches a set of slices", ok_distinct)
    check("tandem_repeat_runs expands to every square, exactly once", ok_tandem)

    # The amortisation bound, on the inputs designed to break it.
    ok_linear = True
    for s in (
        "a" * 4000,
        "ab" * 2000,
        ("abacaba" * 600),
        "a" * 2000 + "b" + "a" * 2000,
    ):
        _, ext = z_array_counted(s)
        ok_linear &= ext <= len(s)
    check("extension loop total <= n on adversarial inputs", ok_linear)

    if verbose:
        for name, ok in checks:
            print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> None:
    s = "aabxaabxcaabxaabxay"
    z = z_array(s)
    print(f"s = {s!r}")
    print("i    :", " ".join(f"{i:2d}" for i in range(len(s))))
    print("s[i] :", " ".join(f"{c:>2}" for c in s))
    print("z[i] :", " ".join(f"{v:2d}" for v in z))
    print()
    print(f"borders            {all_borders(s)}")
    print(f"smallest period    {smallest_period(s)}")
    print(f"string power       {string_power('abcabcabc')}  (of 'abcabcabc')")
    print(f"prefix counts      {prefix_occurrence_counts(s)[:6]} ... (lengths 0..5)")
    print(f"distinct substrings {count_distinct_substrings(s)}")
    print()

    text = "she sells seashells by the seashore"
    pats = ["she", "sea", "seashell", "seashells", "sells", "he", "hell"]
    mz = MultiZMatcher(pats)
    print(f"text     {text!r}")
    print(f"patterns {pats}")
    print(f"chains   {mz.chain_count} scan(s) for {len(pats)} patterns: {mz.chains}")
    for idx, pos in mz.finditer(text):
        print(f"  @{pos:2d}  {pats[idx]!r}")
    print()

    for word in ("abaabaabaaba", "aabaab", "mississippi"):
        runs = tandem_repeat_runs(word)
        print(
            f"tandem repeats of {word!r}: {count_tandem_repeats(word)} squares, "
            f"{len(runs)} runs -> {runs}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="zalgorithm",
        description="Z-array construction, pattern search, and multi-pattern search.",
    )
    parser.add_argument("pattern", nargs="?", help="pattern to search for")
    parser.add_argument("text", nargs="?", help="text to search in")
    parser.add_argument("--demo", action="store_true", help="worked example")
    parser.add_argument("--verify", action="store_true", help="run the self-checks")
    parser.add_argument(
        "--multi",
        nargs="+",
        metavar="PATTERN",
        help="patterns for a multi-pattern search; text follows after --",
    )
    parser.add_argument(
        "--z", action="store_true", help="print the Z-array of the text"
    )
    args = parser.parse_args(argv)

    if args.demo:
        _demo()
        return 0
    if args.verify:
        print("Verifying...")
        ok = verify()
        print("all checks passed" if ok else "FAILURES above")
        return 0 if ok else 1
    if args.multi:
        text = args.pattern if args.text is None else args.text
        if text is None:
            parser.error("--multi needs a text: --multi p1 p2 -- TEXT")
        mz = MultiZMatcher(args.multi)
        print(f"{len(args.multi)} patterns -> {mz.chain_count} Z-scan(s)")
        for idx, pos in mz.finditer(text):
            print(f"{pos:6d}  {args.multi[idx]!r}")
        return 0
    if args.pattern is None:
        parser.print_help()
        return 0
    if args.z and args.text is None:
        print(z_array(args.pattern))
        return 0
    if args.text is None:
        parser.error("give a text to search, or use --demo / --verify / --z")

    hits = list(z_search(args.pattern, args.text))
    print(f"{len(hits)} occurrence(s) of {args.pattern!r}")
    for pos in hits:
        print(
            f"{pos:6d}  {args.text[max(0, pos - 10) : pos + len(args.pattern) + 10]!r}"
        )
    if args.z:
        print("Z:", z_array(args.text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
