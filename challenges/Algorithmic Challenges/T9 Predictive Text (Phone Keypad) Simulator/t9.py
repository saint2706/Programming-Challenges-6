"""T9 predictive text: three ways to turn a digit sequence into ranked words.

Standard phone keypad: 2=abc 3=def 4=ghi 5=jkl 6=mno 7=pqrs 8=tuv 9=wxyz. T9
("Text on 9 keys") disambiguates the many words that share a digit sequence by
real-world frequency: 4663 is GOOD, HOME, GOOF, HOOF and more, and the
dictionary decides which the typist probably meant. This is a ladder of three
methods for that lookup, each earning its place:

    method        build       one full query   incremental (k keystrokes)
    ------------------------------------------------------------------------
    naive_lookup  --          O(n*L)           O(n*L) per keystroke, O(k*n*L) total
    HashmapT9     O(n*L)      O(1) average     O(n) per keystroke (must rescan keys)
    TrieT9        O(n*L)      O(L)             O(1) amortised per keystroke

n = vocabulary size, L = average word length, k = digits typed.

The interesting design tension is the last column. A hashmap keyed by the
*full* digit signature is excellent at "what words spell exactly 4663?" -- one
dict lookup, done. But T9's actual UX is "the user has typed 4, then 46, then
466, then 4663, show live suggestions after every key" -- and a hashmap has no
way to answer "which keys start with this prefix?" except scanning every key
it holds, because a hash function deliberately destroys prefix locality: sig
"4663" and sig "46639" hash nowhere near each other. Fixing that by also
storing every prefix as its own dict entry works, but that dictionary-of-all-
prefixes *is a trie*, just flattened into a hash table instead of walked as a
tree -- so building it costs the same O(n*L), and each entry now duplicates
work its own prefix's entry already did.

A trie keyed by digit gets prefix locality for free: node depth *is* prefix
length, so "typed one more digit" is "follow one more edge", and each node can
cache its own top-k words (by frequency) once at build time. Typing a digit is
then one dict-of-9 lookup plus reading a pre-sorted list off the node -- no
rescan, no re-sort, no re-derivation of anything. That is the whole reason
this module bothers with a trie instead of just shipping the hashmap.

Multi-tap disambiguation -- several real words sharing one digit signature --
is not a special case here, it is the normal case: every lookup returns a
frequency-ranked *list*, and picking [0] is what a T9 phone does when you
don't cycle through candidates yourself.

That corpus-frequency ranking is fixed at build time, but real T9 phones were
adaptive: pick a non-default candidate once (say "home" over "good" on 4663)
and the phone remembers, so the next 4663 offers "home" first. `AdaptiveT9`
wraps `HashmapT9` or `TrieT9` with exactly that: a `select(digits, word)` call
records the choice, keyed by the exact digit signature, and re-ranks that
signature's candidate list by (times selected, original frequency rank) on
every later `lookup`/`predict` -- no re-derivation of the vocabulary or the
underlying structure, just a small `dict[str, Counter[str]]` of nudges.

Word/frequency data comes from the `wordfreq` package (Zipf-scale, real
corpus frequencies), restricted to pure a-z words -- see `build_vocabulary`.

Run directly for a demo, a live-typing simulation, adaptive re-ranking, or
self-verification:

    uv run --with wordfreq python t9.py --demo
    uv run --with wordfreq python t9.py --type 4663
    uv run --with wordfreq python t9.py --demo-adaptive
    uv run --with wordfreq python t9.py --verify
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence

__all__ = [
    "DEFAULT_VOCAB_SIZE",
    "KEYPAD",
    "LETTER_TO_DIGIT",
    "AdaptiveT9",
    "HashmapT9",
    "TrieT9",
    "TypingSession",
    "Vocabulary",
    "build_vocabulary",
    "main",
    "naive_lookup",
    "naive_prefix_lookup",
    "verify",
    "word_to_digits",
]

#: The standard ITU/T9 keypad letter groupings. No letters on 0 or 1.
KEYPAD: dict[str, str] = {
    "2": "abc",
    "3": "def",
    "4": "ghi",
    "5": "jkl",
    "6": "mno",
    "7": "pqrs",
    "8": "tuv",
    "9": "wxyz",
}

#: The inverse mapping: one lowercase letter -> the digit that types it.
LETTER_TO_DIGIT: dict[str, str] = {
    letter: digit for digit, letters in KEYPAD.items() for letter in letters
}

DEFAULT_VOCAB_SIZE = 30_000

#: A word list paired with real-world Zipf frequencies, in no particular order
#: (every method here sorts internally rather than trusting caller order).
Vocabulary = list[tuple[str, float]]


def _validate_digits(digits: str) -> None:
    if not digits:
        raise ValueError("digit sequence must not be empty")
    bad = sorted({d for d in digits if d not in KEYPAD})
    if bad:
        raise ValueError(
            f"invalid keypad digit(s) {bad}; each digit must be one of "
            f"{sorted(KEYPAD)} (0 and 1 carry no letters)"
        )


def word_to_digits(word: str) -> str:
    """The keypad signature of ``word``: one digit per letter.

    Requires a pure a-z word (after lowercasing) -- there is no sane keypad
    digit for an apostrophe, hyphen, or non-Latin letter, so this raises
    rather than silently dropping characters.
    """
    try:
        return "".join(LETTER_TO_DIGIT[ch] for ch in word.lower())
    except KeyError as exc:
        raise ValueError(
            f"{word!r} contains {exc.args[0]!r}, which has no keypad digit "
            "(only a-z words are supported)"
        ) from exc


# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------


def build_vocabulary(size: int = DEFAULT_VOCAB_SIZE, lang: str = "en") -> Vocabulary:
    """The top ``size`` English words usable on a T9 keypad, with Zipf frequency.

    Reproducible construction: walk ``wordfreq.iter_wordlist(lang)`` (already
    frequency-descending), keep the first ``size`` distinct words that are pure
    ASCII a-z after lowercasing -- this drops numerals ("00"), hyphenated and
    apostrophed forms, and anything outside the Latin alphabet, since none of
    those have an unambiguous keypad digit -- then look up each one's actual
    Zipf frequency (`wordfreq.zipf_frequency`) rather than trusting rank order,
    and sort by (frequency descending, word ascending) so ties break
    deterministically.
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    from wordfreq import iter_wordlist, zipf_frequency

    words: list[str] = []
    seen: set[str] = set()
    for raw in iter_wordlist(lang):
        candidate = raw.lower()
        if candidate.isascii() and candidate.isalpha() and candidate not in seen:
            seen.add(candidate)
            words.append(candidate)
            if len(words) >= size:
                break
    vocab = [(w, zipf_frequency(w, lang)) for w in words]
    vocab.sort(key=lambda wf: (-wf[1], wf[0]))
    return vocab


# ---------------------------------------------------------------------------
# 1. Naive linear scan -- the correctness oracle
# ---------------------------------------------------------------------------


def naive_lookup(
    vocabulary: Vocabulary, digits: str, limit: int | None = None
) -> list[str]:
    """Every word whose signature is exactly ``digits``, ranked by frequency.

    O(n*L): converts and compares every word in the vocabulary. Trusts nothing
    about the caller's ordering -- it is the oracle every other method here is
    tested against, so it sorts its own results from scratch.
    """
    _validate_digits(digits)
    matches = [(w, f) for w, f in vocabulary if word_to_digits(w) == digits]
    matches.sort(key=lambda wf: (-wf[1], wf[0]))
    words = [w for w, _f in matches]
    return words[:limit] if limit is not None else words


def naive_prefix_lookup(
    vocabulary: Vocabulary, prefix: str, limit: int | None = None
) -> list[str]:
    """Every word whose signature *starts with* ``prefix``, ranked by frequency.

    The oracle for :meth:`TrieT9.predict` -- O(n*L), rescanning the whole
    vocabulary for every call, which is exactly the cost the trie avoids.
    """
    _validate_digits(prefix)
    matches = [(w, f) for w, f in vocabulary if word_to_digits(w).startswith(prefix)]
    matches.sort(key=lambda wf: (-wf[1], wf[0]))
    words = [w for w, _f in matches]
    return words[:limit] if limit is not None else words


# ---------------------------------------------------------------------------
# 2. Hashmap over complete signatures -- O(1) average for a full query
# ---------------------------------------------------------------------------


class HashmapT9:
    """A ``dict[signature, words]`` for O(1) average full-sequence lookup.

    Building it is one pass over the vocabulary: O(n*L) to compute every
    signature, same as the trie. What it *cannot* do cheaply is answer a
    prefix query -- "what might this become as more digits arrive" -- because
    a hash table intentionally scatters similar keys apart. Asking it for a
    prefix means testing every key it holds, ``str.startswith``, one at a
    time: see :meth:`predict_slow`, kept here specifically to make that cost
    visible in the benchmark rather than asserted in prose.
    """

    __slots__ = ("_index",)

    def __init__(self, vocabulary: Vocabulary) -> None:
        buckets: dict[str, list[tuple[str, float]]] = {}
        for word, freq in vocabulary:
            buckets.setdefault(word_to_digits(word), []).append((word, freq))
        self._index: dict[str, list[str]] = {}
        for sig, entries in buckets.items():
            entries.sort(key=lambda wf: (-wf[1], wf[0]))
            self._index[sig] = [w for w, _f in entries]

    def lookup(self, digits: str, limit: int | None = None) -> list[str]:
        """O(1) average: one dict lookup, no scan."""
        _validate_digits(digits)
        words = self._index.get(digits, [])
        return words[:limit] if limit is not None else list(words)

    def predict_slow(self, prefix: str, limit: int | None = None) -> list[str]:
        """Prefix query the only way a hashmap can do one: scan every key.

        O(number of distinct signatures), every call, no matter how short or
        long ``prefix`` is -- this is the method the module docstring and the
        benchmark point at when they say the hashmap can't do incremental
        typing cheaply.
        """
        _validate_digits(prefix)
        matches: list[tuple[str, float]] = []
        for sig, words in self._index.items():
            if sig.startswith(prefix):
                for w in words:
                    matches.append((w, 0.0))
        # Re-derive frequency order by falling back on stored per-signature
        # order isn't enough across signatures, so this needs the words'
        # actual rank -- which this class doesn't retain, another cost of the
        # flat hashmap design. Words are returned signature-bucket order.
        words_out = [w for w, _f in matches]
        return words_out[:limit] if limit is not None else words_out

    def groups(self) -> list[tuple[str, list[str]]]:
        """Every (signature, words) pair, largest group first."""
        pairs = sorted(self._index.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        return [(sig, list(words)) for sig, words in pairs]

    def __len__(self) -> int:
        return sum(len(v) for v in self._index.values())

    def __contains__(self, digits: str) -> bool:
        return digits in self._index


# ---------------------------------------------------------------------------
# 3. Trie over digits -- O(L) full query, O(1) amortised per keystroke
# ---------------------------------------------------------------------------


class _TrieNode:
    __slots__ = ("children", "exact", "top")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.exact: list[tuple[str, float]] = []  # words whose signature ends here
        self.top: list[tuple[str, float]] = []  # best top_k under this node

    def get_child(self, digit: str) -> _TrieNode | None:
        return self.children.get(digit)


class TrieT9:
    """A trie keyed by digit, each node caching its top-k words by frequency.

    Two things a trie gives away for free that a hashmap has to pay for:

    * **Descent is the query.** A node at depth d *is* the set of everything
      reachable by typing those d digits, so "what if the user types one more
      digit" is "follow one more edge" -- O(1) amortised, not a rescan.
    * **Top-k composes.** The true top-k of a node's whole subtree equals the
      top-k of (its own exact words, union each child's already-computed
      top-k) -- because any element that is globally top-k is certainly
      top-k within whichever child it lives in. So each node's cache is built
      from its children's caches in O(children * top_k) instead of touching
      every descendant word again.

    ``lookup_exact`` mirrors :func:`naive_lookup` / :meth:`HashmapT9.lookup`
    exactly (unbounded, for cross-verification). ``predict`` is the
    incremental-typing method, bounded to ``top_k`` -- realistic for a UI that
    only ever shows a handful of suggestions, and the reason building the trie
    is worth it at all.
    """

    __slots__ = ("root", "top_k")

    def __init__(self, vocabulary: Vocabulary, top_k: int = 10) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        self.top_k = top_k
        self.root = _TrieNode()
        for word, freq in vocabulary:
            node = self.root
            for digit in word_to_digits(word):
                child = node.get_child(digit)
                if child is None:
                    child = node.children[digit] = _TrieNode()
                node = child
            node.exact.append((word, freq))
        self._finalize(self.root)

    def _finalize(self, node: _TrieNode) -> None:
        """Post-order: sort this node's exact words, then compose top-k."""
        node.exact.sort(key=lambda wf: (-wf[1], wf[0]))
        pool = list(node.exact)
        for child in node.children.values():
            self._finalize(child)
            pool.extend(child.top)
        pool.sort(key=lambda wf: (-wf[1], wf[0]))
        node.top = pool[: self.top_k]

    def _descend(self, digits: str) -> _TrieNode | None:
        node = self.root
        for digit in digits:
            node = node.get_child(digit)
            if node is None:
                return None
        return node

    def lookup_exact(self, digits: str, limit: int | None = None) -> list[str]:
        """Every word whose full signature is exactly ``digits``. O(L) to descend."""
        _validate_digits(digits)
        node = self._descend(digits)
        words = [w for w, _f in node.exact] if node else []
        return words[:limit] if limit is not None else words

    def predict(self, digits: str, limit: int | None = None) -> list[str]:
        """Top suggestions for everything reachable from this prefix. O(L)."""
        _validate_digits(digits)
        node = self._descend(digits)
        if node is None:
            return []
        k = min(limit, self.top_k) if limit is not None else self.top_k
        return [w for w, _f in node.top[:k]]

    def session(self) -> TypingSession:
        """A live-typing session pinned to this trie's root."""
        return TypingSession(self)


class TypingSession:
    """A phone typing one digit at a time: O(1) amortised suggestions per key.

    This is the object that makes "incremental" literal: it holds a pointer
    into the trie and *moves* it one edge per keystroke, rather than
    re-descending from the root (which would already be O(L) per key, and is
    exactly what :meth:`TrieT9.predict` does when called fresh each time). Use
    whichever is convenient -- they agree, since they read the same nodes --
    but ``TypingSession`` is the one with no per-keystroke redundant work at
    all, which is the property :meth:`HashmapT9.predict_slow` cannot match.
    """

    __slots__ = ("digits", "node", "trie")

    def __init__(self, trie: TrieT9) -> None:
        self.trie = trie
        self.node: _TrieNode | None = trie.root
        self.digits = ""

    def type_digit(self, digit: str) -> list[str]:
        """Advance by one keystroke; return the updated suggestions."""
        if digit not in KEYPAD:
            raise ValueError(f"invalid keypad digit {digit!r}; must be one of 2-9")
        self.digits += digit
        if self.node is not None:
            self.node = self.node.get_child(digit)
        return self.suggestions()

    def suggestions(self, limit: int | None = None) -> list[str]:
        """Suggestions for whatever has been typed so far, without advancing."""
        if self.node is None:
            return []
        k = min(limit, self.trie.top_k) if limit is not None else self.trie.top_k
        return [w for w, _f in self.node.top[:k]]

    def reset(self) -> None:
        self.node = self.trie.root
        self.digits = ""


# ---------------------------------------------------------------------------
# 4. Adaptive re-ranking -- a thin per-user layer on top of either backend
# ---------------------------------------------------------------------------


class AdaptiveT9:
    """Nudges corpus-frequency ranking toward what *this* user actually picks.

    Real T9 phones didn't just rank by static dictionary frequency -- picking
    a non-default candidate taught the phone, so the next time you typed the
    same digits it offered your word first (see the Wikipedia T9 article's
    account of "next word prediction" and adaptive disambiguation). This class
    is that behaviour, factored out as a wrapper rather than baked into
    `HashmapT9`/`TrieT9`: it holds no vocabulary of its own, just a
    `dict[signature, Counter[word]]` of how many times each word has been
    selected for that exact digit signature, and re-sorts whatever the
    wrapped backend returns by `(-times_selected, original_rank)` -- selected
    words float to the front, in how-often-picked order, and everything else
    keeps the backend's original frequency order behind them.

    The boost is scoped to the *exact* signature it was recorded against.
    Selecting "home" after typing all four digits of 4663 re-ranks future
    `lookup("4663")`/`predict("4663")` calls; it does not retroactively
    re-rank the shorter prefixes ("4", "46", "466") seen on the way there --
    doing that would mean re-deriving where "home" falls within every
    shorter prefix's own candidate list too, which is a different (and
    heavier) feature than the one being demonstrated here.

    `select(digits, word)` raises `ValueError` if `word` is not actually a
    candidate for `digits` (i.e. its signature isn't `digits`) -- recording a
    selection that could never have been offered is a caller bug, not
    something to silently accept.
    """

    __slots__ = ("_backend", "_selections")

    def __init__(self, backend: HashmapT9 | TrieT9) -> None:
        self._backend = backend
        self._selections: dict[str, Counter[str]] = {}

    def _base_candidates(self, digits: str) -> list[str]:
        """Unbounded, backend-native ranking, before any adaptive boost."""
        if isinstance(self._backend, TrieT9):
            return self._backend.lookup_exact(digits)
        return self._backend.lookup(digits)

    def _rerank(self, digits: str, candidates: list[str]) -> list[str]:
        counts = self._selections.get(digits)
        if not counts:
            return list(candidates)
        original_rank = {word: i for i, word in enumerate(candidates)}
        return sorted(candidates, key=lambda w: (-counts.get(w, 0), original_rank[w]))

    def lookup(self, digits: str, limit: int | None = None) -> list[str]:
        """Exact-signature candidates, adaptively re-ranked."""
        _validate_digits(digits)
        ranked = self._rerank(digits, self._base_candidates(digits))
        return ranked[:limit] if limit is not None else ranked

    def predict(self, digits: str, limit: int | None = None) -> list[str]:
        """Prefix suggestions, adaptively re-ranked. Requires a `TrieT9` backend."""
        if not isinstance(self._backend, TrieT9):
            raise TypeError(
                "AdaptiveT9.predict() requires a TrieT9 backend "
                "(HashmapT9 has no prefix structure to predict from)"
            )
        _validate_digits(digits)
        ranked = self._rerank(digits, self._backend.predict(digits))
        return ranked[:limit] if limit is not None else ranked

    def select(self, digits: str, word: str) -> None:
        """Record that ``word`` was chosen for ``digits``, boosting it going forward."""
        _validate_digits(digits)
        candidates = self._base_candidates(digits)
        if word not in candidates:
            raise ValueError(
                f"{word!r} is not a candidate for digits {digits!r} "
                f"(candidates: {candidates or '(none)'})"
            )
        self._selections.setdefault(digits, Counter())[word] += 1

    def selection_counts(self, digits: str) -> Counter[str]:
        """A copy of the recorded selection counts for ``digits`` (empty if none)."""
        _validate_digits(digits)
        return Counter(self._selections.get(digits, Counter()))


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def verify(
    vocabulary: Vocabulary, *, trials: int = 400, seed: int = 0, top_k: int = 10
) -> list[str]:
    """Check HashmapT9 and TrieT9 against the naive_lookup oracle. Returns failures."""
    import random

    rng = random.Random(seed)
    problems: list[str] = []
    hashmap = HashmapT9(vocabulary)
    trie = TrieT9(vocabulary, top_k=top_k)
    words_only = [w for w, _f in vocabulary]

    def check_full(digits: str) -> None:
        expected = naive_lookup(vocabulary, digits)
        got_hash = hashmap.lookup(digits)
        got_trie = trie.lookup_exact(digits)
        if got_hash != expected:
            problems.append(
                f"HashmapT9.lookup({digits!r}) = {got_hash}, expected {expected}"
            )
        if got_trie != expected:
            problems.append(
                f"TrieT9.lookup_exact({digits!r}) = {got_trie}, expected {expected}"
            )

    # Real word signatures -- guaranteed at least one match.
    for word in rng.sample(words_only, min(trials, len(words_only))):
        check_full(word_to_digits(word))

    # Purely random digit strings -- exercises the "no match" path too.
    for _ in range(trials):
        length = rng.randint(1, 8)
        digits = "".join(rng.choice(list(KEYPAD)) for _ in range(length))
        check_full(digits)

    # predict() vs the naive prefix oracle, at the trie's own top_k so the
    # comparison is apples to apples (predict truncates to top_k by design).
    for word in rng.sample(words_only, min(trials // 4, len(words_only))):
        sig = word_to_digits(word)
        prefix = sig[: rng.randint(1, len(sig))]
        expected = naive_prefix_lookup(vocabulary, prefix, limit=top_k)
        got = trie.predict(prefix, limit=top_k)
        if got != expected:
            problems.append(f"TrieT9.predict({prefix!r}) = {got}, expected {expected}")

    # Incremental typing must agree with a fresh predict() at every prefix
    # length -- the whole point of TypingSession is that it's the *same*
    # answer, reached without re-descending from the root.
    for word in rng.sample(words_only, min(trials // 4, len(words_only))):
        sig = word_to_digits(word)
        session = trie.session()
        for i, digit in enumerate(sig, 1):
            got = session.type_digit(digit)
            expected = trie.predict(sig[:i])
            if got != expected:
                problems.append(
                    f"TypingSession after {sig[:i]!r} = {got}, expected {expected}"
                )
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _pick_demo_digits(hashmap: HashmapT9, count: int = 5) -> list[str]:
    """A few signatures with genuine multi-word collisions, data-driven.

    Always leads with 4663 -- GOOD / HOME / GOOF / HOOF is the textbook T9
    example, and it is a real collision (not a contrived one): all four are
    common a-z words that happen to share every digit.
    """
    picks = [sig for sig, words in hashmap.groups() if len(words) >= 2]
    ordered = ["4663"] + [s for s in picks if s != "4663"]
    return ordered[:count]


def _print_candidates(
    vocabulary: Vocabulary,
    hashmap: HashmapT9,
    trie: TrieT9,
    digits: str,
    top: int,
) -> None:
    naive = naive_lookup(vocabulary, digits, limit=top)
    fast = hashmap.lookup(digits, limit=top)
    print(f"  digits {digits}: {naive if naive else '(no matches)'}")
    if fast != naive:
        print(f"    ! HashmapT9 disagrees: {fast}")
    exact = trie.lookup_exact(digits, limit=top)
    if exact != naive:
        print(f"    ! TrieT9.lookup_exact disagrees: {exact}")


def _demo_adaptive(vocabulary: Vocabulary, hashmap: HashmapT9, top: int) -> None:
    """Show a non-default selection bumping a word to the front, concretely."""
    digits = "4663"
    adaptive = AdaptiveT9(hashmap)
    before = adaptive.lookup(digits, limit=top)
    default = before[0]
    # Pick some other real candidate to "select", so the demo is a genuine
    # non-default choice rather than re-selecting whatever was already first.
    runner_up = next((w for w in before if w != default), None)
    print("Adaptive re-ranking: corpus frequency picks the default, but a user")
    print("who keeps choosing something else should see that choice stick.")
    print(f"  digits {digits}, before any selection: {before}")
    if runner_up is None:
        print("    (only one candidate at this vocab size -- nothing to select)")
        return
    adaptive.select(digits, runner_up)
    after = adaptive.lookup(digits, limit=top)
    print(f"  selected {runner_up!r} once (was rank {before.index(runner_up)})")
    print(f"  digits {digits}, after one selection:  {after}")
    assert after[0] == runner_up, "selected word should now rank first"
    print(f"    -> {runner_up!r} is now first, without touching the vocabulary")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="T9 predictive text: naive scan vs hashmap vs digit trie.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Run directly")[-1],
    )
    parser.add_argument("digits", nargs="?", help="a full digit sequence to look up")
    parser.add_argument("--type", metavar="DIGITS", help="simulate typing DIGITS live")
    parser.add_argument("--demo", action="store_true", help="show multi-tap examples")
    parser.add_argument(
        "--demo-adaptive",
        action="store_true",
        help="show a non-default selection re-ranking future lookups",
    )
    parser.add_argument("--verify", action="store_true", help="cross-validate methods")
    parser.add_argument(
        "--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="words to load"
    )
    parser.add_argument("--top-k", type=int, default=10, help="trie top-k per node")
    parser.add_argument("--top", type=int, default=8, help="candidates to display")
    args = parser.parse_args(argv)

    if not (args.digits or args.type or args.demo or args.demo_adaptive or args.verify):
        parser.error(
            "give a digit sequence, --type, --demo, --demo-adaptive, or --verify"
        )

    print(
        f"loading top {args.vocab_size:,} English words (wordfreq)...", file=sys.stderr
    )
    vocabulary = build_vocabulary(args.vocab_size)

    if args.verify:
        problems = verify(vocabulary, top_k=args.top_k)
        if problems:
            print(f"{len(problems)} disagreement(s):")
            for p in problems[:20]:
                print("  " + p)
            return 1
        print(
            f"all methods agree over {args.vocab_size:,} words, real and random "
            "digit sequences, prefix queries, and incremental typing"
        )
        return 0

    hashmap = HashmapT9(vocabulary)
    trie = TrieT9(vocabulary, top_k=args.top_k)

    if args.demo:
        print("Multi-tap disambiguation: several words sharing one signature,")
        print("ranked by real-world frequency (highest first):")
        for digits in _pick_demo_digits(hashmap):
            _print_candidates(vocabulary, hashmap, trie, digits, args.top)

    if args.demo_adaptive:
        _demo_adaptive(vocabulary, hashmap, args.top)

    if args.type:
        _validate_digits(args.type)
        print(f"typing {args.type!r} one digit at a time:")
        session = trie.session()
        for i, digit in enumerate(args.type, 1):
            suggestions = session.type_digit(digit)
            typed = args.type[:i]
            shown = suggestions[: args.top] if suggestions else "(no matches yet)"
            print(f"  {typed:<10} -> {shown}")

    if args.digits:
        _print_candidates(vocabulary, hashmap, trie, args.digits, args.top)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
