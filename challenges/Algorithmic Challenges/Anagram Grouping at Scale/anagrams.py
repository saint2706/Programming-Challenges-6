"""Anagram grouping at scale: five canonical keys, and what "at scale" costs.

Two words are anagrams iff their character multisets are equal, so grouping
is just grouping by *some* canonical form of the multiset. Every solution to
this problem is that one line. What actually differs -- and what the "at
scale" in the title is about -- is the key you pick, because at ten million
words the dictionary of keys is bigger than the words themselves.

    key                build            key size            exact?
    ------------------------------------------------------------------
    sorted string      O(L log L)       L chars             yes
    count vector       O(L)  (bincount) |Sigma| ints        yes
    prime product      O(L^2 / 64)      ~4.9 bits/char*     yes (unique factorisation)
    multiset hash      O(L)             16 bytes            yes, after verification
    (* for the 26-letter alphabet with frequency-ordered primes; derivation in the README)

The interesting one is the last. A 128-bit additive multiset hash is a
homomorphism from the free commutative monoid over the alphabet into
Z/2^128 -- H(A + B) = H(A) + H(B) -- so it is computable in one pass,
updatable in O(1) per character insertion *or deletion*, and its key is 16
bytes no matter how long the word is. It is not collision-free, so this
module never trusts it: `group_anagrams(..., method="hash")` buckets by the
hash and then splits each bucket by exact multiset equality. Verification is
O(total length), which is *cheaper* than sorting every word, so the exact
answer comes out of a strictly smaller and strictly faster pipeline.

Three things here are less obvious than they look, and all three are tested:

* **Byte-level sorting is wrong for non-ASCII.** The obvious speed-up is
  `bytes(sorted(word.encode()))` -- but UTF-8 byte multisets are not injective
  on character multisets. "ã©" and "é£" both encode to the byte multiset
  {A3, A9, C2, C3} and are not anagrams, so that key silently merges them.
  Nothing here ever sorts encoded bytes; every key is built from characters.
* **Normalisation is part of the problem statement.** "café" and "café"
  are the same word to a reader and different multisets to `sorted`. See
  `Normalizer`, which composes NFC/NFD/NFKC/NFKD, case folding, grapheme
  clustering and character filtering.
* **Duplicate words are data, not noise.** A corpus with "listen" twice has a
  group of three, not two, unless you ask for `unique=True`.

For corpora that do not fit in memory there are two escape hatches, both
exact: `group_anagrams_external` (chunk, spill sorted runs to disk, k-way
merge, group the merged stream -- memory is O(runs + largest group)) and
`shard_of` / `group_anagrams_parallel` (the multiset hash partitions the
corpus into independent shards, so N cores do N-way disjoint work).

Run directly for a demo, a self-check, or to group a word list:

    uv run python anagrams.py --demo
    uv run python anagrams.py --verify
    uv run python anagrams.py /usr/share/dict/words --min-size 4 --top 10
"""

from __future__ import annotations

import argparse
import os
import pickle
import string
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "Normalizer",
    "AnagramIndex",
    "key_sorted",
    "key_counter",
    "key_primes",
    "key_bincount",
    "multiset_hash",
    "char_value",
    "group_anagrams",
    "group_anagrams_external",
    "group_anagrams_parallel",
    "are_anagrams",
    "shard_of",
    "verify",
    "main",
]

_MASK64 = (1 << 64) - 1
_MASK128 = (1 << 128) - 1

#: Methods accepted by :func:`group_anagrams`, cheapest-key-first.
METHODS = ("auto", "sorted", "counter", "primes", "bincount", "hash")


# ---------------------------------------------------------------------------
# Normalisation -- deciding what "the same letter" means before counting
# ---------------------------------------------------------------------------

_MARK_CATEGORIES = frozenset({"Mn", "Mc", "Me"})
_ZWJ = "‍"


def _is_mark(ch: str) -> bool:
    return unicodedata.category(ch) in _MARK_CATEGORIES


def _is_regional_indicator(ch: str) -> bool:
    return "\U0001f1e6" <= ch <= "\U0001f1ff"


def graphemes(text: str) -> list[str]:
    """Split ``text`` into user-perceived characters (grapheme clusters).

    A pragmatic subset of UAX #29: a cluster is a base character followed by
    any combining marks, with ZWJ sequences (emoji families), CRLF and
    regional-indicator pairs (flags) held together. It deliberately does not
    implement the full Indic conjunct or emoji-modifier tables -- the point is
    that "e" + U+0301 must count as *one* atom, not two, so that "éa" and
    "eá" are not reported as anagrams of each other.
    """
    if not text:
        return []
    out: list[str] = []
    buf = text[0]
    prev_ri = _is_regional_indicator(text[0])
    ri_run = 1 if prev_ri else 0
    for ch in text[1:]:
        if _is_mark(ch) or ch == _ZWJ or buf.endswith(_ZWJ):
            buf += ch
            prev_ri = False
            ri_run = 0
            continue
        if buf == "\r" and ch == "\n":
            buf += ch
            prev_ri = False
            ri_run = 0
            continue
        if _is_regional_indicator(ch) and prev_ri and ri_run % 2 == 1:
            buf += ch
            ri_run += 1
            continue
        out.append(buf)
        buf = ch
        prev_ri = _is_regional_indicator(ch)
        ri_run = 1 if prev_ri else 0
    out.append(buf)
    return out


@dataclass(frozen=True)
class Normalizer:
    """Turns a word into the sequence of atoms whose multiset defines equality.

    ``form``
        Unicode normalisation form, or ``None`` to leave the text alone. The
        default NFC is what makes "cafe\\u0301" and "caf\\u00e9" the same word;
        without it they differ by a codepoint and land in different groups.
    ``casefold``
        ``str.casefold`` rather than ``str.lower``: it maps "\\u00df" to "ss" and
        both Greek sigmas to the same letter, which ``lower`` does not.
    ``graphemes``
        Count grapheme clusters instead of codepoints. Needed for scripts where
        NFC cannot precompose, e.g. "z\\u0301" stays two codepoints forever.
    ``ignore``
        Characters to drop: a string/set of characters, or a predicate. Use
        :meth:`phrase` for the classic "Dormitory" = "Dirty Room" behaviour.

    Instances are frozen and hashable, so they can be reused as cache keys.
    """

    form: str | None = "NFC"
    casefold: bool = False
    graphemes: bool = False
    ignore: str | frozenset[str] | Callable[[str], bool] | None = None

    def __post_init__(self) -> None:
        if self.form is not None and self.form not in ("NFC", "NFD", "NFKC", "NFKD"):
            raise ValueError(
                f"form must be one of NFC/NFD/NFKC/NFKD or None, got {self.form!r}"
            )
        if isinstance(self.ignore, str):
            object.__setattr__(self, "ignore", frozenset(self.ignore))

    # -- presets ------------------------------------------------------------

    @classmethod
    def exact(cls) -> "Normalizer":
        """No normalisation at all: raw codepoint multisets."""
        return cls(form=None)

    @classmethod
    def phrase(cls) -> "Normalizer":
        """Case-insensitive, alphanumeric-only: for multi-word phrase anagrams."""
        return cls(form="NFKC", casefold=True, graphemes=True, ignore=_not_alnum)

    # -- use ----------------------------------------------------------------

    def units(self, word: str) -> Sequence[str]:
        """The atoms of ``word``: a ``str`` (codepoints) or a ``list[str]``."""
        if not isinstance(word, str):
            raise TypeError(f"expected str, got {type(word).__name__}")
        text = unicodedata.normalize(self.form, word) if self.form else word
        if self.casefold:
            text = text.casefold()
            # Case folding can decompose (ẞ -> ss), so re-normalise to keep the
            # promise that the output is in `form`.
            if self.form:
                text = unicodedata.normalize(self.form, text)
        atoms: Sequence[str] = graphemes(text) if self.graphemes else text
        drop = self.ignore
        if drop is None:
            return atoms
        keep = (lambda a: not drop(a)) if callable(drop) else (lambda a: a not in drop)
        return [a for a in atoms if keep(a)]


def _not_alnum(atom: str) -> bool:
    """True for atoms that :meth:`Normalizer.phrase` throws away."""
    return not atom[:1].isalnum()


#: Used whenever a caller does not supply one. NFC only: accents are preserved
#: and case is significant, which is the least surprising default for words.
DEFAULT_NORMALIZER = Normalizer()


# ---------------------------------------------------------------------------
# Canonical keys -- five ways to name a multiset
# ---------------------------------------------------------------------------


def key_sorted(atoms: Sequence[str]) -> str:
    """Sorted atoms, joined. O(L log L), key size L.

    The baseline every other key is measured against, and the one to reach for
    unless a benchmark says otherwise: ``sorted`` is Timsort in C, so the
    log factor is bought at a constant Python cannot beat with a loop.

    Note it sorts *atoms*, never bytes. Sorting the UTF-8 encoding would be
    faster and wrong -- see the module docstring for the counterexample.
    """
    return "".join(sorted(atoms))


def key_counter(atoms: Sequence[str]) -> tuple[tuple[str, int], ...]:
    """``((atom, count), ...)`` sorted by atom. O(L + s log s) for s distinct atoms.

    Wins over :func:`key_sorted` exactly when words are long and the alphabet
    is small (DNA, protein sequences, log tokens), because the key shrinks from
    L to the number of *distinct* atoms.
    """
    return tuple(sorted(Counter(atoms).items()))


#: Primes assigned smallest-first to the most frequent English letters, which
#: minimises the expected bit length of a prime-product key (see README).
_ENGLISH_FREQUENCY_ORDER = "etaoinshrdlcumwfgypbvkjxqz"
_FIRST_PRIMES = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
)
PRIME_TABLE: dict[str, int] = dict(zip(_ENGLISH_FREQUENCY_ORDER, _FIRST_PRIMES))


def key_primes(atoms: Sequence[str], table: dict[str, int] = PRIME_TABLE) -> int:
    """Product of per-letter primes. Exact by unique factorisation.

    The textbook "clever" key, included so the benchmark can show why it is a
    trap: the product of L primes is an L-limb bignum, so building it costs
    O(L^2 / 64) word operations, and hashing it for the dictionary costs
    another O(L / 64). It only exists for a small fixed alphabet -- there is no
    sane prime for U+1F600 -- so it raises rather than silently mis-grouping.
    """
    acc = 1
    for atom in atoms:
        p = table.get(atom)
        if p is None:
            raise ValueError(
                f"{atom!r} is outside the prime table (alphabet: {''.join(sorted(table))!r}); "
                "use method='sorted' or 'hash' for unrestricted text"
            )
        acc *= p
    return acc


def key_bincount(atoms: Sequence[str]) -> bytes:
    """Fixed-width count vector over Latin-1, via ``numpy.bincount``. O(L) in C.

    Genuinely O(L) with no Python-level loop, and essentially flat in L: 2.3us
    per word from L = 2 to L = 256, against 26us for `key_sorted` at L = 256.
    The cost is a fixed 1 KiB key, so it only pays past L ~ 40, where the
    measured lines cross. Grouping long DNA reads or protein sequences by
    composition is the case it was written for.

    Requires numpy and a text that encodes to Latin-1 (codepoints < 256).
    """
    import numpy as np  # local: numpy stays an optional dependency

    text = atoms if isinstance(atoms, str) else "".join(atoms)
    raw = np.frombuffer(text.encode("latin-1"), dtype=np.uint8)
    return np.bincount(raw, minlength=256).astype(np.uint32).tobytes()


# ---------------------------------------------------------------------------
# The 128-bit additive multiset hash
# ---------------------------------------------------------------------------


def _splitmix64(x: int) -> int:
    """SplitMix64 finaliser: a bijection on 64 bits with good avalanche."""
    z = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


_char_cache: dict[str, int] = {}


def char_value(atom: str) -> int:
    """The random 128-bit value an atom contributes to :func:`multiset_hash`.

    Derived deterministically from the atom (so runs are reproducible and
    shards computed on different machines agree) by hashing its codepoints
    through SplitMix64 twice with different tweaks and concatenating.
    """
    v = _char_cache.get(atom)
    if v is None:
        seed = 0
        for ch in atom:
            seed = _splitmix64(seed ^ ord(ch))
        v = (_splitmix64(seed) << 64) | _splitmix64(seed ^ 0xA5A5A5A5A5A5A5A5)
        _char_cache[atom] = v
    return v


def multiset_hash(atoms: Sequence[str]) -> int:
    """Additive 128-bit hash of the atom multiset: sum of `char_value` mod 2^128.

    This is MSet-Add-Hash (Clarke et al., *Incremental Multiset Hash Functions
    and Their Application to Memory Integrity Checking*, ASIACRYPT 2003)
    instantiated at 128 bits. Its defining property is the homomorphism

        H(A + B) = H(A) + H(B)   (mod 2^128)

    which buys three things at once: the hash of a word is the sum over its
    letters in any order, adding or deleting a letter is one 128-bit add or
    subtract, and the hash of a concatenation is the sum of the parts, so a
    corpus can be hashed in parallel chunks and combined.

    Collisions between m distinct multisets are birthday-bounded by about
    m^2 / 2^129 for non-adversarial input -- 5e-21 at m = 10^9. Being
    *additive* it is linear, so an adversary who knows the table can construct
    collisions by solving a subset-sum; nothing here relies on it not being
    possible, because `group_anagrams(method="hash")` verifies every bucket
    against the exact multiset before returning.
    """
    total = 0
    for atom, n in Counter(atoms).items():
        total += char_value(atom) * n
    return total & _MASK128


def shard_of(word: str, shards: int, *, normalizer: Normalizer | None = None) -> int:
    """Which of ``shards`` partitions ``word`` belongs to.

    Anagrams always land in the same shard, so shards can be grouped
    independently and their results concatenated -- the basis of
    :func:`group_anagrams_parallel` and of any map-reduce version of this.
    """
    if shards < 1:
        raise ValueError(f"shards must be at least 1, got {shards}")
    norm = normalizer or DEFAULT_NORMALIZER
    return multiset_hash(norm.units(word)) % shards


def are_anagrams(a: str, b: str, *, normalizer: Normalizer | None = None) -> bool:
    """Exact pairwise test. O(len(a) + len(b))."""
    norm = normalizer or DEFAULT_NORMALIZER
    return Counter(norm.units(a)) == Counter(norm.units(b))


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

_KEY_FUNCS: dict[str, Callable[[Sequence[str]], Any]] = {
    "sorted": key_sorted,
    "counter": key_counter,
    "primes": key_primes,
    "bincount": key_bincount,
}


def _resolve(method: str, sample: Sequence[str] | None) -> str:
    """Pick a concrete method for ``method="auto"``.

    Short words go to `sorted` (a C sort of a handful of characters is
    unbeatable); long ones go to `counter`, whose key is bounded by the number
    of distinct atoms rather than the length. The threshold is measured, not
    guessed: `benchmark.py --only keys` puts the crossover at L ~ 100
    (L = 64: 5.1us vs 7.0us for `counter`; L = 128: 11.4us vs 9.3us).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method != "auto":
        return method
    if sample is None or len(sample) <= 100:
        return "sorted"
    return "counter"


def group_anagrams(
    words: Iterable[str],
    *,
    normalizer: Normalizer | None = None,
    method: str = "auto",
    unique: bool = False,
    min_size: int = 1,
    sort_groups: bool = False,
) -> list[list[str]]:
    """Group ``words`` into lists of mutual anagrams. Exact for every method.

    Groups come back in order of first appearance, and words inside a group in
    input order, so the output is deterministic for a given input -- worth
    having when you are diffing runs over a 10-million-word corpus. Duplicate
    words are kept (a corpus with "listen" twice yields a group of three)
    unless ``unique=True``.

    ``method`` selects the canonical key; see the module docstring for the
    trade-offs. ``"hash"`` groups by the 128-bit multiset hash and then splits
    each bucket by exact multiset equality, which keeps dictionary keys at a
    fixed 16 bytes without giving up exactness.

    ``min_size`` drops groups smaller than the given size -- the usual reason
    to run this at all is to find the *interesting* groups, and in a 300k-word
    dictionary about 80% of groups are singletons.
    """
    if min_size < 1:
        raise ValueError(f"min_size must be at least 1, got {min_size}")
    norm = normalizer or DEFAULT_NORMALIZER

    buckets: dict[Any, list[str]] = {}
    seen: set[str] | None = set() if unique else None
    keyfunc: Callable[[Sequence[str]], Any] | None = None

    for word in words:
        if seen is not None:
            if word in seen:
                continue
            seen.add(word)
        atoms = norm.units(word)
        if method == "hash":
            key: Any = multiset_hash(atoms)
        else:
            if keyfunc is None:
                keyfunc = _KEY_FUNCS[_resolve(method, atoms)]
            key = keyfunc(atoms)
        bucket = buckets.get(key)
        if bucket is None:
            buckets[key] = [word]
        else:
            bucket.append(word)

    groups = list(buckets.values())
    if method == "hash":
        groups = _split_hash_collisions(groups, norm)
    if min_size > 1:
        groups = [g for g in groups if len(g) >= min_size]
    if sort_groups:
        groups.sort(key=lambda g: (-len(g), g[0]))
    return groups


def _split_hash_collisions(
    groups: list[list[str]], norm: Normalizer
) -> list[list[str]]:
    """Re-split hash buckets by exact multiset equality.

    O(total length): building a Counter per word is linear, and this is the
    only place the exact multiset is ever materialised. Single-word buckets --
    the overwhelming majority -- skip the work entirely, and multi-word buckets
    take a fast path that checks the common case (no collision) with one pass
    before falling back to a per-word split.
    """
    out: list[list[str]] = []
    for group in groups:
        if len(group) == 1:
            out.append(group)
            continue
        first = Counter(norm.units(group[0]))
        if all(Counter(norm.units(w)) == first for w in group[1:]):
            out.append(group)
            continue
        # A real collision. Rare enough that the slow path can be dumb.
        exact: dict[tuple[tuple[str, int], ...], list[str]] = {}
        for word in group:
            exact.setdefault(key_counter(norm.units(word)), []).append(word)
        out.extend(exact.values())
    return out


# ---------------------------------------------------------------------------
# Out-of-core grouping: chunk, spill, k-way merge
# ---------------------------------------------------------------------------


def group_anagrams_external(
    words: Iterable[str],
    *,
    normalizer: Normalizer | None = None,
    chunk_size: int = 1_000_000,
    tmpdir: str | os.PathLike[str] | None = None,
    min_size: int = 1,
) -> Iterator[list[str]]:
    """Group a corpus far larger than memory. Yields groups lazily, exactly.

    The standard external-sort shape: buffer ``chunk_size`` (key, word) pairs,
    sort the buffer, spill it to a temp file as a sorted run, and at the end
    merge all runs with ``heapq.merge`` and cut the merged stream at key
    boundaries. Peak memory is one chunk during the spill phase and
    O(runs + largest group) during the merge -- so a corpus of any size fits,
    as long as a single anagram group does.

    Groups come out in canonical-key order rather than first-appearance order:
    ordering by appearance would require holding the whole corpus. Words within
    a group keep input order, because the chunk sort is stable and the merge
    breaks ties by run age.

    Records are pickled per (key, index, word) triple, which is self-delimiting
    and encoding-agnostic -- text formats would need escaping for words that
    contain the delimiter or a newline, and words containing newlines are
    exactly the sort of thing a real corpus contains.
    """
    import heapq

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")
    norm = normalizer or DEFAULT_NORMALIZER

    with tempfile.TemporaryDirectory(dir=tmpdir, prefix="anagram-runs-") as workdir:
        runs: list[Path] = []
        buf: list[tuple[str, int, str]] = []
        counter = 0

        def spill() -> None:
            buf.sort()
            path = Path(workdir) / f"run-{len(runs):05d}.pkl"
            with path.open("wb") as fh:
                for record in buf:
                    pickle.dump(record, fh, protocol=pickle.HIGHEST_PROTOCOL)
            runs.append(path)
            buf.clear()

        for word in words:
            buf.append((key_sorted(norm.units(word)), counter, word))
            counter += 1
            if len(buf) >= chunk_size:
                spill()

        # Nothing spilled: the corpus fit in one chunk, so skip the disk round trip.
        if not runs:
            buf.sort()
            yield from _cut_runs(iter(buf), min_size)
            return
        if buf:
            spill()

        handles = [p.open("rb") for p in runs]
        try:
            merged = heapq.merge(*(_read_run(fh) for fh in handles))
            yield from _cut_runs(merged, min_size)
        finally:
            for fh in handles:
                fh.close()


def _read_run(fh: Any) -> Iterator[tuple[str, int, str]]:
    """Stream pickled records back out of one spilled run."""
    while True:
        try:
            yield pickle.load(fh)
        except EOFError:
            return


def _cut_runs(
    records: Iterator[tuple[str, int, str]], min_size: int
) -> Iterator[list[str]]:
    """Cut a key-sorted record stream into groups at key boundaries."""
    group: list[str] = []
    current: str | None = None
    for key, _seq, word in records:
        if key != current:
            if group and len(group) >= min_size:
                yield group
            group = []
            current = key
        group.append(word)
    if group and len(group) >= min_size:
        yield group


# ---------------------------------------------------------------------------
# Sharded grouping: the multiset hash partitions the corpus for free
# ---------------------------------------------------------------------------


def _shard_worker(payload: tuple[list[str], str, bool, bool, Any]) -> list[list[str]]:
    """Module-level so it survives pickling into a process pool."""
    words, form, casefold, use_graphemes, ignore = payload
    norm = Normalizer(
        form=form or None, casefold=casefold, graphemes=use_graphemes, ignore=ignore
    )
    return group_anagrams(words, normalizer=norm, method="hash")


def group_anagrams_parallel(
    words: Iterable[str],
    *,
    normalizer: Normalizer | None = None,
    workers: int | None = None,
    min_size: int = 1,
) -> list[list[str]]:
    """Group across processes by sharding on the multiset hash.

    Anagrams share a hash, so they share a shard: the shards are disjoint
    sub-problems and their results concatenate with no merge step at all. That
    is the whole reason to hash rather than sort for partitioning -- a sorted
    key would work too but costs L bytes per word to ship to the worker.

    Falls back to the serial path for small inputs or when a process pool
    cannot start (restricted sandboxes, ``fork`` unavailable), because paying
    interpreter-startup cost to group ten thousand words is a loss.
    """
    norm = normalizer or DEFAULT_NORMALIZER
    words = list(words)
    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    if workers <= 1 or len(words) < 50_000:
        return group_anagrams(words, normalizer=norm, method="hash", min_size=min_size)

    shards: list[list[str]] = [[] for _ in range(workers)]
    for word in words:
        shards[multiset_hash(norm.units(word)) % workers].append(word)

    payloads = [
        (s, norm.form or "", norm.casefold, norm.graphemes, norm.ignore) for s in shards
    ]
    try:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_shard_worker, payloads))
    except (OSError, ValueError, ImportError, NotImplementedError):
        results = [_shard_worker(p) for p in payloads]

    return [g for shard in results for g in shard if len(g) >= min_size]


# ---------------------------------------------------------------------------
# An index you can query, rather than a one-shot grouping
# ---------------------------------------------------------------------------


class AnagramIndex:
    """Incremental anagram index: add words, then ask what anagrams anything.

    Keys are 128-bit hashes, and each bucket keeps the exact multiset of its
    first member so that collisions are caught on insertion rather than
    papered over. Lookup is O(len(query)) -- no scan of the corpus, and no
    canonical key longer than 16 bytes retained per group.

        >>> idx = AnagramIndex(["listen", "silent", "enlist", "google"])
        >>> sorted(idx.lookup("tinsel"))
        ['enlist', 'listen', 'silent']
        >>> idx.lookup("nothing")
        []
    """

    __slots__ = ("_normalizer", "_buckets", "_size")

    def __init__(
        self,
        words: Iterable[str] = (),
        *,
        normalizer: Normalizer | None = None,
    ) -> None:
        self._normalizer = normalizer or DEFAULT_NORMALIZER
        # hash -> list of (exact multiset, words); the list has one entry
        # unless a 128-bit collision has actually happened.
        self._buckets: dict[
            int, list[tuple[tuple[tuple[str, int], ...], list[str]]]
        ] = {}
        self._size = 0
        self.extend(words)

    def add(self, word: str) -> None:
        atoms = self._normalizer.units(word)
        digest = multiset_hash(atoms)
        entries = self._buckets.get(digest)
        if entries is None:
            self._buckets[digest] = [(key_counter(atoms), [word])]
        else:
            exact = key_counter(atoms)
            for stored, members in entries:
                if stored == exact:
                    members.append(word)
                    break
            else:
                entries.append((exact, [word]))
        self._size += 1

    def extend(self, words: Iterable[str]) -> None:
        for word in words:
            self.add(word)

    def lookup(self, word: str, *, include_self: bool = True) -> list[str]:
        """Every indexed word that is an anagram of ``word``, in insertion order."""
        atoms = self._normalizer.units(word)
        entries = self._buckets.get(multiset_hash(atoms))
        if not entries:
            return []
        exact = key_counter(atoms)
        for stored, members in entries:
            if stored == exact:
                return (
                    list(members) if include_self else [w for w in members if w != word]
                )
        return []

    def groups(self, *, min_size: int = 1) -> list[list[str]]:
        """All groups, largest first, ties broken by the first word."""
        out = [
            list(members)
            for entries in self._buckets.values()
            for _stored, members in entries
            if len(members) >= min_size
        ]
        out.sort(key=lambda g: (-len(g), g[0]))
        return out

    def largest(self, k: int = 1) -> list[list[str]]:
        """The ``k`` biggest groups."""
        return self.groups()[:k]

    def stats(self) -> dict[str, int | float]:
        """Corpus shape, and the collision count -- which should always be 0."""
        sizes = [len(m) for entries in self._buckets.values() for _s, m in entries]
        collisions = sum(len(e) - 1 for e in self._buckets.values())
        return {
            "words": self._size,
            "groups": len(sizes),
            "singletons": sum(1 for s in sizes if s == 1),
            "largest_group": max(sizes, default=0),
            "hash_collisions": collisions,
            "mean_group_size": (self._size / len(sizes)) if sizes else 0.0,
        }

    def __len__(self) -> int:
        return self._size

    def __contains__(self, word: str) -> bool:
        return bool(self.lookup(word))

    def __repr__(self) -> str:
        return f"AnagramIndex({self._size} words, {len(self._buckets)} buckets)"


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def _canonical(groups: Iterable[Iterable[str]]) -> set[tuple[str, ...]]:
    """Order-insensitive form of a grouping, for comparing two methods."""
    return {tuple(sorted(g)) for g in groups}


def verify(*, seed: int = 0, trials: int = 200, verbose: bool = True) -> bool:
    """Check every method against a brute-force O(n^2) oracle. Returns True on success.

    The oracle is the definition itself: two words are anagrams iff their
    ``Counter``s are equal, checked over all pairs. Everything else in this
    module is an optimisation of that, so it is the only thing worth trusting.
    """
    import random

    rng = random.Random(seed)
    alphabets = [string.ascii_lowercase[:3], string.ascii_lowercase[:8], "acgt", "ab"]
    ok = True

    for trial in range(trials):
        alphabet = alphabets[trial % len(alphabets)]
        words = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 8)))
            for _ in range(rng.randint(0, 40))
        ]
        expected = _brute_force(words)
        for method in ("sorted", "counter", "hash"):
            got = _canonical(group_anagrams(words, method=method))
            if got != expected:
                ok = False
                if verbose:
                    print(f"  MISMATCH method={method} words={words}", file=sys.stderr)
        # The prime key only exists for the lowercase alphabet.
        if set(alphabet) <= set(PRIME_TABLE):
            if _canonical(group_anagrams(words, method="primes")) != expected:
                ok = False
                if verbose:
                    print(f"  MISMATCH method=primes words={words}", file=sys.stderr)
        if _canonical(group_anagrams_external(words)) != expected:
            ok = False
            if verbose:
                print(f"  MISMATCH external words={words}", file=sys.stderr)
        if _canonical(AnagramIndex(words).groups()) != expected:
            ok = False
            if verbose:
                print(f"  MISMATCH index words={words}", file=sys.stderr)

    if verbose:
        print(
            f"verify: {trials} random corpora, all methods vs brute force -- "
            f"{'OK' if ok else 'FAILED'}"
        )
    return ok


def _brute_force(words: Sequence[str]) -> set[tuple[str, ...]]:
    """O(n^2 L) grouping straight from the definition. The oracle."""
    counters = [Counter(unicodedata.normalize("NFC", w)) for w in words]
    assigned = [-1] * len(words)
    groups: list[list[str]] = []
    for i, word in enumerate(words):
        if assigned[i] >= 0:
            continue
        assigned[i] = len(groups)
        members = [word]
        for j in range(i + 1, len(words)):
            if assigned[j] < 0 and counters[j] == counters[i]:
                assigned[j] = len(groups)
                members.append(words[j])
        groups.append(members)
    return _canonical(groups)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_words(path: str) -> list[str]:
    if path == "-":
        return [line.rstrip("\n") for line in sys.stdin if line.strip()]
    return [
        line.rstrip("\n")
        for line in Path(path)
        .read_text(encoding="utf-8", errors="replace")
        .splitlines()
        if line.strip()
    ]


DEMO_WORDS = [
    "listen",
    "silent",
    "enlist",
    "inlets",
    "tinsel",
    "evil",
    "vile",
    "live",
    "veil",
    "levi",
    "stressed",
    "desserts",
    "café",
    "café",
    "facé",  # NFC folds the first two together
    "Dormitory",
    "Dirty Room",  # only anagrams under Normalizer.phrase()
    "google",
    "elgoog",
    "unique",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Group words into anagram classes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Run directly")[-1],
    )
    parser.add_argument(
        "path", nargs="?", help="word list, one per line ('-' for stdin)"
    )
    parser.add_argument(
        "--demo", action="store_true", help="run on a built-in word list"
    )
    parser.add_argument(
        "--verify", action="store_true", help="cross-validate every method"
    )
    parser.add_argument("--method", default="auto", choices=METHODS)
    parser.add_argument(
        "--min-size", type=int, default=2, help="hide groups smaller than this"
    )
    parser.add_argument(
        "--top", type=int, default=20, help="show at most this many groups"
    )
    parser.add_argument(
        "--phrase",
        action="store_true",
        help="case-insensitive, ignore non-alphanumerics",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="use the out-of-core path (bounded memory)",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args(argv)

    if args.verify:
        return 0 if verify() else 1

    if args.demo:
        words = DEMO_WORDS
    elif args.path:
        words = _read_words(args.path)
    else:
        parser.error("give a word list, --demo, or --verify")

    norm = Normalizer.phrase() if args.phrase else DEFAULT_NORMALIZER

    if args.external:
        groups = list(
            group_anagrams_external(
                words,
                normalizer=norm,
                chunk_size=args.chunk_size,
                min_size=args.min_size,
            )
        )
        groups.sort(key=lambda g: (-len(g), g[0]))
    else:
        groups = group_anagrams(
            words,
            normalizer=norm,
            method=args.method,
            min_size=args.min_size,
            sort_groups=True,
        )

    total = sum(len(g) for g in groups)
    print(
        f"{len(words)} words -> {len(groups)} groups of size >= {args.min_size} "
        f"({total} words)"
    )
    for group in groups[: args.top]:
        print(f"  {len(group):>4}  {', '.join(group)}")
    if len(groups) > args.top:
        print(f"  ... and {len(groups) - args.top} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
