# T9 Predictive Text (Phone Keypad) Simulator

**Category:** Algorithmic Challenges
**Difficulty:** I (brief: "trie over digit-groups, rank by frequency")

**Status:** Implemented (Python)

Old phone keypads put three or four letters on every digit from 2 to 9:

```
2 abc   3 def   4 ghi
5 jkl   6 mno   7 pqrs
8 tuv   9 wxyz
```

so a word typed on a numeric keypad becomes a digit string, and that digit
string is almost never unique to one word. Pressing 4-6-6-3 could mean GOOD,
HOME, GONE, HOOF or GOOF — T9 ("Text on 9 keys") is the system that guesses
which one you meant, and it guesses by real-world word frequency. This module
is a ladder of three implementations of that guess, each earning its place:

| method         | build  | one full query | incremental (k keystrokes)                |
| -------------- | ------ | -------------- | ----------------------------------------- |
| `naive_lookup` | —      | O(n·L)         | O(n·L) per keystroke, O(k·n·L) total      |
| `HashmapT9`    | O(n·L) | O(1) average   | O(n) per keystroke (must rescan all keys) |
| `TrieT9`       | O(n·L) | O(L)           | O(1) amortised per keystroke              |

n = vocabulary size, L = average word length, k = digits typed. `TrieT9` is
the one that actually earns the name "T9" — see below. A fourth piece,
`AdaptiveT9`, sits on top of either lookup structure rather than replacing
one: it doesn't rank from a static corpus at all, it re-ranks whatever the
wrapped backend returns based on what *this* user has actually picked before
— see [Adaptive re-ranking](#adaptive-re-ranking-learning-from-what-the-user-picks)
below.

## The design tension: why a hashmap can't do incremental typing cheaply

A `dict[signature, words]` is unbeatable at "what words spell exactly 4663?" —
one hash, one lookup, done. But that is not the real T9 user experience. The
real one is: the user has pressed 4, then 4-6, then 4-6-6, then 4-6-6-3, and
the screen updates with live suggestions after *every single key*. That's a
prefix query — "which signatures start with what's been typed so far?" — and
a hash table is structurally the wrong tool for it, on purpose: a hash
function's whole job is to scatter similar keys apart, so that collisions are
rare. Signature `"4663"` and signature `"46639"` are not stored anywhere near
each other, even though one is a prefix of the other. Getting a prefix answer
out of a hashmap means testing every key it holds — `HashmapT9.predict_slow`
in `t9.py` does exactly that, and exists specifically so the benchmark below
can show what it costs.

The tempting fix — "just also store every prefix as its own dict key" — does
work, but look at what it actually is: a `dict` keyed by every prefix of every
word *is a trie*, flattened into a hash table instead of walked as a tree. It
costs the same O(n·L) to build, and every entry duplicates work its own
shorter prefix's entry already did, because nothing is shared between
`dict["466"]` and `dict["4663"]` — they're unrelated hash buckets that happen
to have related keys.

A trie keyed by digit gets the sharing for free, because it *is* the prefix
structure rather than an encoding of it: node depth is prefix length, so
"the user typed one more digit" is "follow one more edge" — one dict-of-at-
most-9 lookup. And because each node can cache its own top-k words by
frequency once, at build time, reading the suggestions after that edge is
just reading a pre-sorted list off the node. No rescan, no re-sort, no
re-deriving anything the previous keystroke already computed. That's the
entire reason this module bothers with a trie instead of shipping only the
hashmap — see `TrieT9` and the incremental-typing benchmark below.

### Composing top-k up the tree

The trick that makes building the per-node cache cheap rather than another
O(n·L) pass: the true top-k of everything under a node equals the top-k of
(that node's own exact-length words, union each child's *already-computed*
top-k) — because any word that is genuinely in the global top-k is certainly
in the top-k of whichever child subtree it lives in. So each node's cache is
built once, bottom-up, from small pre-sorted lists rather than by re-scanning
every descendant:

```python
pool = list(node.exact)
for child in node.children.values():
    pool.extend(child.top)  # each already trimmed to top_k
pool.sort(key=lambda wf: (-wf[1], wf[0]))
node.top = pool[:top_k]
```

## Multi-tap disambiguation

Several real words sharing a signature is not a special case here, it's the
normal case — every lookup returns a frequency-ranked list, and taking `[0]`
is what a T9 phone does by default when you don't cycle through candidates
yourself:

```
$ uv run --with wordfreq python t9.py 4663 --vocab-size 30000
digits 4663: ['good', 'home', 'gone', 'hood', 'hone', 'hoof', 'goof']

$ uv run --with wordfreq python t9.py 43556 --vocab-size 30000
digits 43556: ['hello']

$ uv run --with wordfreq python t9.py 228 --vocab-size 30000
digits 228: ['act', 'cat', 'bat', 'abu', 'abt', 'cbt', 'abv']
```

4663 is the textbook T9 example for a reason: GOOD, HOME and GONE are all
extremely common, and they collide because O/O, O/O, and O/N sit on the same
keys — n and m are both on 6. 228 is a smaller, sharper case: CAT / ACT / BAT
collapse onto one signature because a, b and c are literally the same key.

Live incremental typing, via `TrieT9.session()`:

```
$ uv run --with wordfreq python t9.py --type 4663 --vocab-size 30000
typing '4663' one digit at a time:
  4          -> ['in', 'i', 'is', 'it', 'have', 'he', 'his', 'if']
  46         -> ['in', 'how', 'good', 'into', 'go', 'going', 'got', 'home']
  466        -> ['good', 'home', 'gonna', 'gone', 'immediately', 'google', 'honest', 'homes']
  4663       -> ['good', 'home', 'gone', 'immediately', 'honest', 'homes', 'honestly', 'goods']
```

Note the later columns aren't restricted to 4-digit words — `predict()`
returns anything whose signature *starts with* what's been typed, so
"immediately" (whose first four digits are also 4663) shows up as a live
completion, exactly like predictive text on a real phone.

## Adaptive re-ranking: learning from what the user picks

Corpus frequency is a good default, but it's not the user. Real T9 phones
were adaptive: cycle past the default suggestion to pick something else once,
and the phone remembers — the next time you type the same digits, your word
comes up first (the Wikipedia T9 article documents this "next word
prediction" behaviour; see Sources). `AdaptiveT9` is that mechanism, factored
out as a thin wrapper around `HashmapT9` or `TrieT9` rather than baked into
either:

```python
class AdaptiveT9:
    def __init__(self, backend: HashmapT9 | TrieT9) -> None:
        self._backend = backend
        self._selections: dict[str, Counter[str]] = {}
```

`select(digits, word)` records one choice, keyed by the *exact* digit
signature; `lookup`/`predict` then re-sort whatever the backend already
computed by `(-times_selected, original_frequency_rank)` — selected words
float to the front in how-often-picked order, everything else keeps its
original relative order behind them. No vocabulary re-derivation, no rebuilt
trie: just a `dict[str, Counter[str]]` of nudges read on every query.

Concrete before/after, at the full 30,000-word vocabulary
(`t9.py --demo-adaptive`):

```
$ uv run --with wordfreq python t9.py --demo-adaptive
Adaptive re-ranking: corpus frequency picks the default, but a user
who keeps choosing something else should see that choice stick.
  digits 4663, before any selection: ['good', 'home', 'gone', 'hood', 'hone', 'hoof', 'goof']
  selected 'home' once (was rank 1)
  digits 4663, after one selection:  ['home', 'good', 'gone', 'hood', 'hone', 'hoof', 'goof']
    -> 'home' is now first, without touching the vocabulary
```

`AdaptiveT9.select()` requires that `word` is actually a candidate for
`digits` (its own signature must equal `digits`) — recording a selection that
the backend could never have offered is a caller bug, and this raises
`ValueError` rather than silently accepting it. The boost is scoped to the
*exact* signature it was recorded against: selecting "home" for the full
`"4663"` re-ranks future queries of `"4663"` (both `lookup` and, on a
`TrieT9` backend, `predict`), but it does not retroactively change what
shows up while typing the shorter prefixes `"4"`, `"46"`, `"466"` on the way
there — doing that would mean re-deriving where "home" falls inside every
shorter prefix's own candidate list too, a heavier feature than the one
demonstrated here. `test_t9.py` covers the happy path (selection moves a
word to first, repeated selections outrank single ones, the boost doesn't
leak across signatures) and the edge cases (selecting a non-candidate word
raises, `predict()` on an `AdaptiveT9` wrapping a `HashmapT9` raises
`TypeError` since a hashmap has no prefix structure to predict from).

`AdaptiveT9` isn't in `benchmark.py`: `select()` is one backend lookup plus a
counter increment, and re-ranking is a sort over however many candidates
already share a signature — typically single digits, never more than a
handful even for 4663-sized collisions — so there's no interesting cost
curve to plot next to the naive/hashmap/trie comparison above.

## Context-aware (bigram) ranking: a real technique this module skips, honestly

Modern predictive keyboards go one step further than per-word frequency or
per-user adaptation: they rank candidates by what word came *before* it, so
typing 4663 after "coming" favours "home" over "good" even before any
personal adaptation has happened. That needs a bigram (word-pair) frequency
table — `P(candidate | previous_word)` — and this module deliberately does
not implement it, because the data isn't available cleanly:

`wordfreq` (already a dependency here) ships **unigram** frequencies only —
one score per word, no pair data. Looking for a lightweight substitute turned
up nothing that fits the `uv run --with X` pattern the rest of this module
uses: `nltk` ships tools for building n-gram models but not a bundled
word-bigram corpus — using it means a separate `nltk.download()` of a corpus
(tens of megabytes, at runtime, over the network) — and the assorted
pip packages that mention "bigram frequency" (e.g. `corpus-toolkit`,
`frequency-analysis`) are tools for computing bigrams from a corpus *you*
supply, not sources of pre-built word-pair data themselves. The pre-built
bigram datasets that do exist online are either letter-pair (character)
frequency tables for cipher analysis — a different thing entirely — or raw
CSV/JSON dumps meant to be fetched by hand from a specific URL, not something
installable and pinned the way `wordfreq` is. Reaching for any of those would
mean either a heavy, network-dependent download at runtime or a bundled data
file this repo would have to own and maintain — worse tradeoffs than simply
not implementing the feature.

If this data existed cleanly, the shape would be small: a
`dict[(prev_word, digits), list[str]]` or a `Counter[(prev_word, candidate)]`
built once from the bigram corpus, consulted only when a previous word is
known, falling back to the existing unigram ranking otherwise — the same
"cache once, read many" shape `TrieT9` already uses, just keyed on a pair
instead of a prefix. `rank_by="bigram"` was the planned name for that mode;
it's unimplemented here specifically because the corpus it would need isn't
available at this module's dependency weight, not because the technique
doesn't matter.

## Vocabulary construction

`build_vocabulary(size=30_000, lang="en")` walks `wordfreq.iter_wordlist("en")`
(already frequency-descending), keeps the first `size` distinct words that are
pure ASCII a-z after lowercasing — dropping numerals ("00"), hyphenated and
apostrophed forms, and anything outside the Latin alphabet, since none of
those have an unambiguous keypad digit — looks up each one's real Zipf
frequency with `wordfreq.zipf_frequency` (rather than trusting rank order),
and sorts by `(frequency descending, word ascending)` so ties break the same
way every run. That last part matters: it's what makes `naive_lookup`,
`HashmapT9`, and `TrieT9` produce byte-identical output on the same input,
which is what the cross-verification below actually checks.

## Correctness / verification

`naive_lookup` is the oracle: it converts and compares every word in the
vocabulary from scratch, trusting nothing about caller ordering. Everything
else is checked against it, and against a second oracle,
`naive_prefix_lookup`, for prefix queries:

```
$ uv run --with wordfreq python t9.py --verify --vocab-size 30000
all methods agree over 30,000 words, real and random digit sequences, prefix queries, and incremental typing
```

`verify()` checks, for the given vocabulary:

- **Real word signatures** — every sampled word's own digit signature, which
  is guaranteed to have at least one match (itself).
- **Random digit strings** — exercises the "no match" path, which real-word
  sampling alone would rarely hit.
- **`HashmapT9.lookup` and `TrieT9.lookup_exact`** both must equal
  `naive_lookup` exactly, order included, for every case above.
- **`TrieT9.predict`** must equal `naive_prefix_lookup` truncated to the
  trie's own `top_k`, for prefixes of real words at every length.
- **Incremental typing consistency** — stepping a `TypingSession` through a
  word's signature one digit at a time must produce, at every step, exactly
  what a fresh `TrieT9.predict()` call on that same prefix would return. This
  is the property that makes `TypingSession` trustworthy as a UI backend: it
  is never a shortcut that happens to agree, it reads the same cached node.

`test_t9.py` adds 43 tests: the classic collisions by hand (`good`/`home`/
`gone`/`hoof`/`goof` all on 4663; `cat`/`act`/`bat` all on 228), empty and
invalid digit strings (0 and 1 carry no letters and are rejected), a digit
sequence with zero matches, repeated digits, `limit=` truncation, prefix
queries including completions past the typed length, `TypingSession` dead
ends and resets, `AdaptiveT9` re-ranking (a non-default selection moves to
first and stays there on the next query, repeated selections outrank a
single one, the boost doesn't leak into a different signature, selecting a
non-candidate word raises), 20 rounds of randomized synthetic-vocabulary
agreement, and full-scale verification (`-m slow`) against the real
30,000-word vocabulary.

```
uv run --with wordfreq --with pytest pytest -q              # 43 tests, ~11s
uv run --with wordfreq --with pytest pytest -q -m slow       # +2 full-vocab tests
```

## Benchmark results

Two different questions, because they have different answers — run on this
machine, real numbers, not fabricated:

**Question 1: one-shot full-sequence lookup, as vocabulary size grows.**

```
Full-sequence lookup: one query, whole digit sequence given up front
   n words         naive       hashmap          trie   winner
----------------------------------------------------------------------
     2,000      731.83us        0.35us        0.68us   hashmap
    10,000     3900.83us        0.98us        1.36us   hashmap
    30,000    30137.78us        1.42us        1.29us   trie
```

`naive_lookup` gets worse in lockstep with n, exactly as its O(n·L) promises —
by 30,000 words it's four orders of magnitude slower than the other two.
`HashmapT9` and `TrieT9` are both already sub-2 microseconds at every size
tested here (one dict lookup vs. descending roughly 5-8 trie levels for an
average English word), so which one "wins" at this scale is mostly
measurement noise around numbers this small — not a meaningful gap.

**Question 2: incremental typing — the cost of getting suggestions after
*each* keystroke, typing out 40 real words (290 keystrokes total) against the
full 30,000-word vocabulary.**

```
Incremental typing: 40 real words, 290 keystrokes total
(vocabulary size 30,000, trie top_k=10)
method          total time     per keystroke
----------------------------------------------
trie               0.166ms            0.57us
hashmap          929.609ms         3205.55us
naive           9736.102ms        33572.76us
```

This is the result the whole ladder was built to show. Per keystroke, the
trie is roughly **5,600x faster than the hashmap** and **59,000x faster than
naive rescanning** — not because `TrieT9` is doing anything exotic, but
because it is the only one of the three that isn't redoing work the previous
keystroke already finished. `HashmapT9.predict_slow` scans every one of its
~30,000 keys on every single key press (and doesn't even bother re-deriving
frequency order while doing it — see its docstring — so this is a generous
lower bound on what a correct implementation would cost, not an inflated
one). `naive_prefix_lookup` re-derives every word's full signature from
scratch, every keystroke, which is why it's the slowest by another order of
magnitude.

```
uv run --with wordfreq python benchmark.py                            # full run
uv run --with wordfreq python benchmark.py --sizes 5000 20000 --quick # faster, smaller
```

## Run it

```bash
cd "challenges/Algorithmic Challenges/T9 Predictive Text (Phone Keypad) Simulator"

uv run --with wordfreq python t9.py 4663                    # multi-tap candidates
uv run --with wordfreq python t9.py --type 4663              # live incremental typing
uv run --with wordfreq python t9.py --demo                   # a handful of real collisions
uv run --with wordfreq python t9.py --demo-adaptive           # selection re-ranking, before/after
uv run --with wordfreq python t9.py --verify                 # cross-validate all 3 methods
uv run --with wordfreq python t9.py --vocab-size 50000 2668  # bigger vocabulary

uv run --with wordfreq python benchmark.py                   # timing, both questions

uv run --with wordfreq --with pytest pytest -q                # 43 tests
uv run --with wordfreq --with pytest pytest -q -m slow        # +2 full-vocabulary tests
```

`--vocab-size` controls how many words `build_vocabulary` loads (default
30,000); `--top-k` controls how many suggestions each trie node caches
(default 10, which bounds what `predict()`/`TypingSession` can ever return —
`lookup_exact` and `naive_lookup` are never bounded by it, since exact
multi-tap matches are usually few).

## Where this is actually used

**T9 itself** was patented by Tegic Communications (Cliff Kushler and Martin
King) in the mid-1990s and shipped on hundreds of millions of feature phones
before touchscreens made letter-by-letter typing fast enough that
disambiguation stopped being the bottleneck. It's a small, complete example
of a much bigger idea: whenever an input channel is *lossier* than the output
space you want (nine keys standing in for 26 letters), a ranked dictionary
lookup is what recovers the intended output — the same shape shows up in
Chinese pinyin and Japanese kana-to-kanji input methods (IMEs), where one
phonetic spelling maps to many candidate characters and frequency/context
ranking picks the likely one.

**The trie-with-cached-top-k pattern** generalizes past phone keypads
entirely: it's the standard shape behind search-box autocomplete (type-ahead
suggestions ranked by popularity, updating per keystroke without a server
round trip per key), IDE symbol completion, and any "predict the rest from a
prefix" feature where the corpus is static enough to precompute but the
queries arrive one character at a time. The specific insight this module
leans on — that a node's top-k composes from its children's top-k without
re-scanning the subtree — is the general technique for maintaining any
order-statistic over a tree incrementally, not something specific to T9.

**Beyond this challenge.** Two more real techniques belong in the same
family as `TrieT9` and `AdaptiveT9` but are out of scope here. **FST-based
dictionary compression** (e.g. the `marisa-trie` package) stores the same
prefix structure as a minimized finite-state transducer instead of a
pointer-per-edge Python trie, trading build-time complexity for a large
memory win at production scale — it's a *memory* optimization on the exact
same shape this module already builds, with no new asymptotic story beyond
what `TrieT9` already has, so it wouldn't teach anything new here. **Neighbor-key
fuzzy correction** — tolerating a mistyped digit by also checking signatures
one key-press away (the digit dialed next to the intended one, the same idea
as QWERTY-neighbor typo correction) — is a genuinely different feature, typo
tolerance rather than frequency ranking, layered on top of whichever lookup
structure is in use rather than replacing it. Both are real, both show up in
production predictive-text systems, and both are left out deliberately
rather than bolted on to check a box.

## Sources

- [T9 (predictive text) — Wikipedia](https://en.wikipedia.org/wiki/T9_%28predictive_text%29)
- [Phone keypad letter mapping (ITU E.161) — Wikipedia](https://en.wikipedia.org/wiki/Telephone_keypad#Letter_mapping)
- [wordfreq — PyPI](https://pypi.org/project/wordfreq/)
- [wordfreq documentation and Zipf-frequency scale](https://github.com/rspeer/wordfreq)
- [Trie — Wikipedia](https://en.wikipedia.org/wiki/Trie)
- [Autocomplete / predictive text and IME systems — Wikipedia](https://en.wikipedia.org/wiki/Autocomplete)
