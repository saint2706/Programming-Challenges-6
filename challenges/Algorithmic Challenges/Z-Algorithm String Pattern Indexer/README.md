# Z-Algorithm String Pattern Indexer

**Category:** Algorithmic Challenges
**Difficulty:** I (brief: "Build the Z-array, use it for multi-pattern matching")

**Status:** Implemented (Python)

The Z-array of a string gives, for every position `i`, the length of the
longest common prefix of the string with the suffix starting at `i`. Building
it takes O(n) by the same amortisation trick as Manacher (challenge 8) and
KMP (challenge 10): keep the interval that reaches furthest right, reuse what
symmetry already proved, and pay for character comparisons only when they push
that interval further.

Both halves of the brief turn out to hide a better answer than the version
every tutorial writes.

## 1. The separator is a myth, and the concatenation is a waste

The textbook search is: build `S = P + '#' + T`, compute `Z(S)`, report every
`i ≥ m+1` with `Z[i] == m`, and warn the reader that `'#'` must not occur in
the input. Two things are wrong with that.

**The separator is not needed for correctness.** For `i ≥ m`,

```
Z[i] ≥ m  ⟺  S[i:i+m] == S[:m] == P  ⟺  P occurs at text offset i − m
```

by the definition of `Z`, in both directions. And `i ≥ m` forces the
occurrence to start at or after `T[0]`, so nothing straddles the join. What
the separator actually buys is that `Z` values inside the text region are
capped at `m` — a tidiness property no correct matcher relies on.
`z_search_concat(pattern, text, separator=None)` runs with no separator at
all, and `test_concatenation_without_separator_still_correct` checks it on
texts full of `#`.

**The concatenation is the real cost.** It copies the whole text and then
allocates an integer per position of the copy. `z_search` runs the identical
recurrence over a *virtual* concatenation: the box `[l, r)` lives in the text
and mirrors against `z_array(pattern)`, which is `O(m)` ints built once.

```
      n     m   z_search     concat  speedup  peak KiB (search)  peak KiB (concat)
800,000     8     0.0726     0.0846    1.17x                  1              7,033
800,000    40     0.0759     0.0915    1.20x                  1              7,032
```

7 MB against 1 KB, for the same answer. The speedup is the smaller half of the
result.

**And it streams.** Follow the box: the extension loop only ever reads at or
past `r`, and `r` never decreases, so the algorithm never wants a text
character it has already passed. `z_search_stream` therefore searches an
iterator of unbounded length in a buffer of at most `m + 1` elements:

```
      n   peak KiB  matches
 20,000        3.0
 80,000        3.0
320,000        3.0
```

Flat. The concatenation formulation cannot do this at all — it needs the whole
text resident before it starts.

## 2. Multi-pattern: the scan count is L, not k

This is where the Z-array is supposed to lose. The honest textbook statement
is that `k` patterns cost `k` scans, O(k·n + M), against Aho-Corasick's
O(n + M + occ).

But `k` is the wrong count. One scan against pattern `P` produces
`lcp(P, T[j:])` at every `j`, and that number settles not just `P` but **every
pattern that is a prefix of `P`**, since `Q` occurs at `j` exactly when
`lcp(P, T[j:]) ≥ |Q|`. A scan resolves a *chain* of the "is-a-proper-prefix-of"
partial order, so the number of scans needed is the minimum number of chains
covering the pattern set.

**That number is exactly the number of leaves of the pattern trie.** The
prefix order is a forest — a pattern's parent is its longest proper prefix in
the set — so its root-to-leaf paths are `L` chains that cover it. `L` is also a
lower bound: the leaves are pairwise incomparable, no two incomparable
patterns can share a scan, and every element lies on some root-to-leaf path,
so any antichain injects into the set of paths. Maximum antichain = minimum
chain cover = `L` (Dilworth, with the extremal antichain in hand).

`MultiZMatcher` is therefore **O(L·n + M) with L ≤ k**, equality exactly when
no pattern is a prefix of another. `test_chain_count_matches_the_antichain_bound`
asserts `chain_count == |maximal patterns|` on 300 random dictionaries.

```
text = 400,000 characters

dictionary                  k    L    MultiZ    k-scan  Aho-Cor.      occ
prefix chain (L = 1)       22    1    0.0435    0.6728    0.0263    3,872
antichain (L = k)          22   22    0.6650    0.6913    0.0375      578
word list (stems)          40   24    0.7988    1.0730    0.0433   53,762
```

Row 1 is the whole point: 22 patterns, one scan, 15× faster than the obvious
loop. Row 2 is the honest other end — nothing to share, so a chain of one
falls back to `z_search` and the matcher costs *exactly* what the per-pattern
loop costs, never more.

And Aho-Corasick wins the wall clock in every row, as it should: one dict
lookup per character beats even one tight pass over a list of ints, and its
`n + M` never becomes `L·n`. The chain decomposition is not a claim to have
beaten it. It is the answer to "the brief says use the Z-array for
multi-pattern matching, so how well can that actually be made to work" —
and the answer is *`k` times better than the obvious way*, with no automaton
to build and no memory proportional to the dictionary. Use Aho-Corasick for a
large dictionary; use this when the dictionary is a handful of prefix-related
terms and you already have a Z-array.

## 3. The array is an index in its own right

Once `z_array(s)` exists, the following are one pass each:

| From the Z-array                                                      | Cost       |
| --------------------------------------------------------------------- | ---------- |
| `all_borders` — every proper prefix that is also a suffix             | O(n)       |
| `smallest_period`, `string_power`                                     | O(n)       |
| `prefix_occurrence_counts` — how often *every* prefix occurs, at once | O(n)       |
| `prefix_function` ⇄ `z_array`, both directions                        | O(n)       |
| `count_distinct_substrings`, online                                   | O(n²)      |
| `tandem_repeat_runs` — every square, Main-Lorentz                     | O(n log n) |

The borders derivation is the one worth stating: `s` has a border of length
`n − i` exactly when `i + z[i] == n`, so reading them off is a single scan
with no `pi`-chain walk.

### The two arrays are the same information

`prefix_from_z` and `z_from_prefix` convert in O(n) **without the string**.
The forward direction is the pretty one: `z[i] = L` says `s[i:i+L]` is a
prefix, hence the prefix of length `j+1` is a border of `s[:i+j+1]` for every
`j < L`; writing the largest `j` first and stopping at the first slot already
filled means every `pi` entry is written exactly once, which is what makes the
nested loop linear.

The reverse direction ships twice. `z_from_prefix` rebuilds a string and runs
the Z-algorithm on it — legitimate, not a dodge, because the prefix function
pins a string down to *renaming of the alphabet* and both arrays are
renaming-invariant. `string_from_prefix` does that in O(n) by handing out a
fresh symbol wherever `pi[i] == 0`; fresh symbols create no equalities, and
every equality it builds is one `pi` already asserted.

`z_from_prefix_direct` is the in-place transfer, kept because it is the one
usually quoted — and because the abbreviated rule that circulates for it,
`z[i+j] = z[i] − j`, is wrong. The correct transfer is
`z[i+j] = min(z[j], z[i] − j)`:

```
"abab":  real Z = [4, 0, 2, 0]
         without the min = [4, 0, 2, 1]      # position 3 is "b"
```

`test_z_from_prefix_needs_the_min` pins that down, and both routes are checked
against `z_array` on every binary string up to length 11.

### Main-Lorentz, and why the output is run-encoded

`tandem_repeat_runs` finds every substring of the form `ww` in O(n log n):
split the string, recurse, then find the repeats that cross the split — which
reduces to two longest-common-extension queries per candidate period, and
those are Z-array lookups. Four Z-arrays per level, O(log n) levels.

Two of those four are usually built as `z_array(v + '#' + u)`. Here they are
`z_match_lengths(v, u)` — the same table with no sentinel and no copy, which
also means the algorithm works on sequences that have no spare alphabet
symbol.

The run encoding is not a convenience. `"a" * n` contains `⌊n/2⌋·⌈n/2⌉`
squares, so any function returning them individually cannot be O(n log n):

```
input                      n   seconds       runs        squares  runs / n log n
a^n                   32,000    0.3914    428,034    256,000,000          0.8938
fibonacci word        32,000    0.2090     42,360        294,292          0.0885
random binary         32,000    0.2288     28,388         31,825          0.0593
random 26 letters     32,000    0.1592      1,303          1,303          0.0027
```

`runs / n log n` stays bounded down the table while `squares` goes quadratic.
`tandem_repeats` expands the runs when the caller really wants each one.

## Linearity, proved and then measured

The box `[l, r)` reaching furthest right satisfies `s[l:r] == s[:r-l]`. For a
new `i` inside it, `s[i:r]` mirrors `s[i-l:r-l]`, so `z[i-l]` is already the
answer when it is strictly less than `r − i`; when it is not, the box stops
short of the truth and character comparison is the only way forward — and
every such comparison pushes `r` one place right. `r` never decreases and
never exceeds `n`, so the extension loop runs **at most n times in total**.

`z_array_counted` returns that count, and the tests assert the bound on the
inputs built to break it:

```
input                 extensions   bound n   naive comparisons
all one character         19,999    20,000         200,009,999
alternating ab            19,998    20,000         100,009,999
abacaba fractal           19,997    20,000          28,595,713
one big border            19,998    20,000         100,009,999
random binary             17,582    20,000              40,106
random 26 letters            772    20,000              20,772
```

The left column never exceeds the bound; the right column has no bound at all.
And the last row is the caveat that keeps the comparison honest — on random
26-letter text the naive O(n²) definition performs about `n` comparisons too,
because the inner loop almost never runs. Against the clock it is *faster*
there. The quadratic only materialises when the alphabet collapses:

```
    n   random-26       naive    ratio         a^n   naive a^n    ratio
 1000      0.0001      0.0001     1.8x      0.0001      0.0209   196.3x
 4000      0.0003      0.0002     0.8x      0.0004      0.3674   844.9x
16000      0.0012      0.0011     0.9x      0.0020      6.0137  3060.9x
64000      0.0044      0.0037     0.8x      0.0077     93.5060 12221.1x
```

0.8× on prose, 12,000× on a run — same two functions. The reason to ship the
linear one is that the 0.8× is a bounded loss and the 12,000× is an unbounded
one.

## Edge cases the tests pin down

- **`z[0] = n`**, not the common `0` placeholder. The whole string *is* its own
  longest common prefix, and defining it honestly is what lets the two
  conversions be exact inverses with no special case at index 0.
- The **empty pattern** matches at each of the `n + 1` gaps, `n` included —
  the `str.find("") == 0` / `re.finditer` convention, and the one that keeps
  `text[j:j+m]` valid at every reported `j`. All three search implementations
  and both multi-pattern matchers agree on it.
- **Overlapping occurrences are all reported**, in increasing order.
- **No character is reserved.** Nothing here is a sentinel, so `#`, `$`, `^`,
  `\x00` and `￿` are ordinary input.
- **Any sequence works**: `str`, `bytes`, `list`, `tuple`, a list of grapheme
  clusters. Astral-plane characters are single characters in Python 3 and work;
  combining marks are separate code points, so a caller wanting grapheme
  semantics passes a list of clusters and the generic core does the rest.
- **Duplicate patterns** in a multi-pattern set are matched once and reported
  under every index they were given.
- **Empty patterns inside a multi-pattern set** are handled outside the trie,
  because a chain of length 0 has no leaf to scan against.
- `MultiZMatcher.finditer` merges `L` sorted streams with `heapq.merge`, so
  text-order output never materialises the full occurrence list — and each
  chain sorts its indices per position, or the merge key would be wrong.
- `count_distinct_substrings` is verified against a literal `set` of slices on
  every binary string up to length 9; `tandem_repeat_runs` against an O(n³)
  oracle on every binary string up to length 11, expanding to each square
  **exactly once**.

## Where this is actually used

**The Z-array's real job is as a subroutine.** It is not usually the thing you
ship; it is the linear-time primitive inside something larger:

- **Main-Lorentz**, above — and tandem repeats are not a curiosity. Short
  tandem repeats (STRs) are the basis of DNA fingerprinting, and expansions of
  trinucleotide repeats cause Huntington's disease and fragile X. Repeat
  finders are standard genomics tooling.
- **The Burrows-Wheeler / suffix-array world** uses Z-style longest-common-
  extension queries throughout; when you only need extensions against *one*
  fixed string, the Z-array is the O(n) answer where a suffix array would be
  O(n log n) to build.
- **String periodicity.** `smallest_period` in one pass is what tells a
  compressor whether a block is a power of a shorter unit, and what tells a
  network protocol parser that a frame is a repeated keep-alive.

**Prefix-structured dictionaries are common enough to matter.** The chain
decomposition wins exactly when the pattern set is closed under prefixes,
which is the shape of: autocomplete candidate sets, URL path-prefix routing
tables, hierarchical tag namespaces (`log.error`, `log.error.db`), and
stemmed word lists. It is worth knowing that on those inputs "one scan per
pattern" is leaving a factor of `k` on the table, even if the eventual answer
is still to use an automaton.

**And the technique transfers further than the problem does.** The box
argument here — keep the rightmost known-good boundary, use the mirror
position to skip work already proved, amortise because the boundary only ever
moves right — is the same argument behind Manacher (challenge 8) and KMP's
failure function (challenge 10). The `prefix_from_z` / `z_from_prefix` pair
makes that "same argument" literal: the two arrays are inter-convertible in
linear time with no reference to the string, so they are two encodings of one
object.

The honest limit: for a *large* dictionary, use Aho-Corasick; for
longest-common-extension between arbitrary suffix pairs, use a suffix
automaton or a suffix array with RMQ. The Z-array is the right tool when there
is one fixed reference string, and it is the cheapest such tool by a wide
margin.

## Files

| File                 | What it is                                                                                                                                          |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `zalgorithm.py`      | `z_array`, separator-free and streaming search, `MultiZMatcher` + `AhoCorasick`, both conversions, borders/periods/prefix counts, Main-Lorentz, CLI |
| `test_zalgorithm.py` | 21,564 tests: exhaustive over all binary strings to length 11, linearity measured, the chain bound, Unicode, CLI                                    |
| `benchmark.py`       | Construction scaling, concatenation memory, streaming, `L` vs `k` vs Aho-Corasick, Main-Lorentz                                                     |

## Running it

```bash
uv run python zalgorithm.py --demo
uv run python zalgorithm.py --verify
uv run python zalgorithm.py needle "a needle in a haystack with a needle"
uv run python zalgorithm.py --multi he she his hers -- "ushers"
uv run python zalgorithm.py --z aabaab

uv run --with pytest pytest -q                 # all 21,564
uv run --with pytest pytest -q -m "not slow"
uv run python benchmark.py --quick
```

No third-party dependencies at all — the module is pure standard library.

## Sources

- Gusfield, *Algorithms on Strings, Trees, and Sequences* (1997), §1.3-1.5 — the Z-algorithm, and the reduction of exact matching to it
- [cp-algorithms: Z-function](https://cp-algorithms.com/string/z-function.html)
- [cp-algorithms: Prefix function](https://cp-algorithms.com/string/prefix-function.html) — the two conversions
- Main & Lorentz, ["An O(n log n) algorithm for finding all repetitions in a string"](https://doi.org/10.1016/0196-6774(84)90021-X), *Journal of Algorithms* 5(3), 1984
- [cp-algorithms: Main-Lorentz](https://cp-algorithms.com/string/main_lorentz.html)
- Aho & Corasick, ["Efficient string matching: an aid to bibliographic search"](https://doi.org/10.1145/360825.360855), *CACM* 18(6), 1975
- Dilworth, "A decomposition theorem for partially ordered sets", *Annals of Mathematics* 51(1), 1950 — the chain-cover bound behind `MultiZMatcher`
