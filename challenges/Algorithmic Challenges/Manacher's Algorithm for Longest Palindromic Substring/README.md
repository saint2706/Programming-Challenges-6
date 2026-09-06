# Manacher's Algorithm for Longest Palindromic Substring

**Category:** Algorithmic Challenges
**Difficulty:** I (brief: "O(n) approach; compare against the naive O(n²) baseline")

**Status:** Implemented (Python)

Manacher (1975) finds the longest palindromic substring in linear time. But the
substring is the *least* interesting thing the algorithm computes. What it
actually produces is the **complete palindromic structure** of the string — for
each of the 2n+1 possible centres, the radius of the longest palindrome around
it — and the longest substring is one `max` over that array.

Once you have the array, you also get:

| From the radii                                                       | Cost                      |
| -------------------------------------------------------------------- | ------------------------- |
| `is_palindrome(i, j)` for **any** substring                          | O(1), after an O(n) build |
| `count_palindromic_substrings` (occurrences, up to ~n²/2 of them)    | O(n)                      |
| `all_maximal_palindromes` — the n cores everything else nests inside | O(n)                      |
| `longest_palindromic_prefix` → `shortest_palindrome_by_prepending`   | O(n)                      |

## Three choices that differ from the version everyone writes

### 1. No transformed string

The usual trick interleaves separators — `"abc"` → `"#a#b#c#"` — so that even
palindromes become odd. That is a real 2n+1 allocation on top of the 2n+1
radii, and it forces the input to be a string.

This module runs the **two-array formulation** (`d1` for odd centres, `d2` for
even) directly on the input. It allocates 2n small ints and nothing else, and
it works on any sequence whose elements support `==`: `str`, `bytes`, `list`,
`tuple`, or a list of grapheme clusters. `palindrome_radii` still hands you the
classic 2n+1 array when you want one, derived rather than built:

```
rad[2i+1] = 2·d1[i] − 1        rad[2i] = 2·d2[i]
```

That uniform array is what makes O(1) substring queries work: `s[i:j]` has
virtual centre `i+j` and radius `j−i`, so

```python
def is_palindrome(i, j):
    return rad[i + j] >= j - i
```

settles both parities with one lookup and no case split.

### 2. The separator is *not* a correctness hazard — the boundary sentinels are

It is widely repeated that the `#` trick breaks if `#` appears in the input.
**It does not.** Manacher only ever compares positions of equal parity (their
indices sum to twice the centre), so a real `#` sitting at an odd index is
never compared against a separator at an even one. The genuinely unsafe variant
is the `$…^` boundary-guard version, which pads the ends with two characters
assumed not to occur in the input in order to skip bounds checks.

This module has neither, so there is no character it cannot handle.
`test_separator_and_sentinel_characters_are_ordinary_input` runs `#`, `$`, `^`,
`\x00` and `|` through it explicitly.

### 3. Manacher is not the end of the story

For **distinct** palindromes rather than occurrences, the radii array is the
wrong tool. A string of length n has at most n distinct palindromic substrings
(Droubay–Justin–Pirillo), but they are not the maximal ones, and reading them
off the radii takes O(n²). That needs the **eertree** (Rubinchik & Shur, 2015),
which is here too — online, O(n) time and space — and which also drives the
O(n log n) palindromic factorisation.

## Linearity, proved and then measured

The amortisation argument: the algorithm keeps the palindrome `[l, r]` reaching
furthest right. For a new centre `i` inside it, the mirror centre `l+r−i` is
already solved, and by symmetry the answer at `i` is at least the mirror's,
clamped to `r−i+1` because past `r` the symmetry is not known to hold. The
clamp means the brute-force extension loop can *only* run when the palindrome
at `i` reaches `r` — and every iteration of it pushes `r` one place right. `r`
never decreases and never exceeds n, so **total inner-loop work across the
entire scan is at most n per array**.

The test suite asserts that directly by counting expansions on the worst inputs
it can construct (`"a"*2000`, `"ab"*1000`, `"abacaba"*300`, one giant
palindrome), and checks the instrumented copy produces identical output to the
production loops. On `"a"*600`:

```
naive expansions:     ~90 000   (quadratic)
Manacher expansions:   ≤ 1 200   (linear)
```

## The benchmark's real finding: O(n²) is only O(n²) sometimes

The brief asks for Manacher against the naive expand-around-centre baseline.
Linear beats quadratic — but *where*, and by how much, depends almost entirely
on the input, and that turns out to be the interesting part.

On **random text over 26 letters** the longest palindrome is about 2·log₂₆(n)
characters, so the naive inner loop almost never runs and its O(n²) never
materialises. The naive method is not merely competitive there — it is
*faster*, because Manacher pays mirror-lookup bookkeeping at every centre for
a saving it never gets to collect:

```
    n   manacher      naive    ratio    longest
 1000     0.0005     0.0004     0.8x          4
10000     0.0047     0.0035     0.7x          7
50000     0.0241     0.0180     0.7x          7
```

On **a run of equal characters** every one of the 2n−1 naive expansions runs to
full width, and the ratio is proportional to n and unbounded:

```
    n   manacher      naive     ratio   longest
 1000     0.0007     0.0300     40.1x      1000
10000     0.0082     3.0949    379.2x     10000
50000     0.0412    77.0224   1869.7x     50000
```

Same two implementations, same table format, and a factor of 2600 between the
conclusions you would draw. The hidden variable is palindrome density, and
sweeping alphabet size at n = 50 000 shows it is not a gradient at all — it is
a cliff, entirely between one letter and two:

| \|Σ\| | longest | occurrences   | distinct | manacher | naive  | ratio     |
| ----- | ------- | ------------- | -------- | -------- | ------ | --------- |
| 1     | 50000   | 1,250,025,000 | 50,000   | 0.042    | 76.887 | **1827×** |
| 2     | 34      | 149,680       | 1,427    | 0.039    | 0.025  | 0.6×      |
| 4     | 25      | 109,598       | 1,026    | 0.034    | 0.022  | 0.7×      |
| 26    | 15      | 86,282        | 924      | 0.029    | 0.021  | 0.7×      |
| 256   | 7       | 56,428        | 778      | 0.025    | 0.018  | 0.7×      |

The expected longest palindrome in random text over an alphabet of size k is
about 2·log_k(n), which collapses immediately once k > 1, and the naive
method's cost collapses with it. A quadratic algorithm is quadratic on the
inputs that make it so — and if you benchmark only on English text you will
conclude, wrongly, that the linear algorithm is not worth having.

The reason to ship Manacher anyway is that the 0.7× is a *bounded* loss and the
1827× is an *unbounded* one. You do not get to choose your input, and only one
of the two methods has a worst case you can state.

### Step counts, since clocks are machine-dependent

The same claim without a stopwatch, at n = 5000:

| Input                | Manacher | bound 2n | naive      | naive/n |
| -------------------- | -------- | -------- | ---------- | ------- |
| all one character    | 9,997    | 10,000   | 12,502,500 | 2500.5  |
| alternating `ab`     | 4,998    | 10,000   | 6,252,500  | 1250.5  |
| one giant palindrome | 9,994    | 10,000   | 6,252,500  | 1250.5  |
| random binary        | 7,333    | 10,000   | 15,078     | 3.0     |
| `abacaba` fractal    | 4,987    | 10,000   | 29,809     | 6.0     |
| random 26 letters    | 395      | 10,000   | 5,396      | 1.1     |

The Manacher column never exceeds the bound — that is the amortisation proof as
a measurement. The naive column has no bound at all, and `naive/n` is its
effective per-character cost.

The `dp_longest_palindrome_span` baseline is included for a different reason:
its O(n²) *space* is why not to use it. At n = 10⁵ the table would be 10¹⁰
booleans. It is not a slower way to get the answer; it is a way of not getting
the answer at all.

## Palindromic factorisation, two ways

`min_palindromic_partition` cuts a string into the fewest possible palindromes
in **O(n log n)**, via the eertree's *series links*. The palindromic suffixes
at any position fall into O(log n) arithmetic progressions of lengths — a
consequence of the Fine–Wilf periodicity lemma — and a series link names
exactly one such progression, so each position costs O(log n) instead of O(n).

`palindromic_partition` keeps the O(n²) DP (with O(1) palindrome tests from the
radii array) because reconstructing the actual pieces needs the predecessor
chain. The two are cross-validated against each other and against a
definition-level O(n³) oracle on **every binary string up to length 12**, plus
Fibonacci words and periodic strings — the inputs where the series structure
actually has many suffixes to compress.

The gap is not subtle, even against a DP whose palindrome test is already O(1):

```
    n   series links   DP + O(1) test    ratio   pieces
 1000         0.0008           0.0725    87.7x      158
10000         0.0082           7.4993   919.6x     1516
50000         0.0369          skipped       --     7575
```

## Unicode: what counts as "the same character read backwards"

Reversing `"e"+U+0301` codepoint-wise gives a combining acute with nothing to
combine with. Palindrome checks on accented text have to run over grapheme
clusters, not codepoints — which the sequence-generic core makes free:

```python
longest_palindrome(graphemes(text))       # clusters, not codepoints
```

`relaxed_view` gives the "A man, a plan, a canal: Panama" reading — NFC,
case-folded, alphanumerics only — **plus an index map back into the original
string**. The map is the part usually skipped and usually wanted: without it
you can report *that* there is a 21-character palindrome but not *where* it is
in the user's text.

## Edge cases the tests pin down

- The **empty string** yields the empty palindrome `(0, 0)` — the honest
  answer, and the one that keeps `s[start:end]` valid with no special case at
  the call site.
- **Ties break leftmost**, consistently across Manacher, the naive method, and
  the DP, so the three are interchangeable rather than merely equinumerous.
- **`d1[i] ≥ 1` everywhere** (a single character is a palindrome) and
  **`d2[0] = 0`** (there is no gap before the first character).
- Every radius stays inside the string; every centre's recovered slice really
  is a palindrome of exactly that length.
- **Astral-plane characters** are single characters in Python 3, and work.
- **ZWJ emoji sequences, regional-indicator flag pairs and CRLF** stay whole
  under `graphemes`.
- `count_distinct_palindromes(s) ≤ len(s)` is checked on every binary string up
  to length 11 — the Droubay–Justin–Pirillo bound — along with the "rich"
  strings that attain it.
- `Eertree` is **online**: adding characters one at a time matches building
  from the whole string at every prefix.

## Where this is actually used

**Biology is the serious application.** Palindromic structure in nucleic acids
is functional, not decorative:

- **Restriction enzyme sites are palindromes.** EcoRI cuts at `GAATTC`, whose
  reverse complement is itself; the enzyme's two-fold symmetric structure binds
  a two-fold symmetric site. Locating these is step one of any digest planning
  or cloning tool.
- **Inverted repeats form hairpins and stem-loops**, which govern transcription
  termination, RNA secondary structure and transposon boundaries. EMBOSS ships
  `palindrome` and `einverted` for exactly this scan.
- **CRISPR arrays** are named for their palindromic repeats.

One honest caveat: biological palindromes are *reverse-complement* palindromes
(`A` pairs with `T`, `C` with `G`), not literal ones. The loop here compares
`s[i-k] == s[i+k]`; the biological version compares
`s[i-k] == complement(s[i+k])` — a one-line change to the same amortization
argument, not a drop-in use of this module. The sequence-generic design means
everything built on top still applies once you make it.

**The radii array is the reusable artifact.** O(1) "is `s[i:j]` a palindrome"
after an O(n) build is a primitive other string algorithms consume: palindromic
factorization (studied in its own right as a compression scheme), text
segmentation, and any dynamic program whose inner test is a palindrome check —
the O(n²) DP in this module is only tolerable because the test became free.

**The technique transfers further than the problem does.** Manacher's core idea
— keep the rightmost known-good boundary, use the mirror position to skip work
already proved, and amortize because that boundary only ever moves right — is
the same argument powering the **Z-algorithm** and **KMP's failure function**
(challenges 9 and 10 in this repository). Learn it once here and three
algorithms stop being tricks.

**And the benchmark result is the practical guidance.** On natural language the
naive O(n²) scan is *faster*, because palindromes in real text are short. The
quadratic blowup needs a near-unary alphabet — long homopolymer runs, which is
to say exactly the sequencing data above, or attacker-chosen input. So: on
prose, either works and the simpler one wins; on genomic or untrusted input,
only the algorithm with a worst case you can state is safe. Choosing between
them without measuring palindrome density is guessing.

The literal use — does this phrase read the same backwards — is where
`relaxed_view` earns its index map. A palindrome feature that cannot point at
*where* in the user's original text the match sits is only half built.

## Files

| File                  | What it is                                                                                            |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| `palindromes.py`      | `manacher_odd_even`, `PalindromeIndex`, `Eertree`, factorisation, Unicode views, three baselines, CLI |
| `test_palindromes.py` | 76 tests: exhaustive over all binary strings to length 12, linearity measured, Unicode, CLI           |
| `benchmark.py`        | Best/worst-case scaling, alphabet sweep, machine-independent step counts, structure timings           |

## Running it

```bash
uv run python palindromes.py --demo
uv run python palindromes.py --verify
uv run python palindromes.py "forgeeksskeegfor"
uv run python palindromes.py --relaxed "A man, a plan, a canal: Panama"

uv run --with pytest pytest -q               # all 76
uv run --with pytest pytest -q -m "not slow"
uv run python benchmark.py --quick
```

No third-party dependencies at all — the module is pure standard library.

## Sources

- Manacher, ["A New Linear-Time 'On-Line' Algorithm for Finding the Smallest Initial Palindrome of a String"](https://doi.org/10.1145/321892.321896), *JACM* 22(3), 1975
- [cp-algorithms: Manacher's algorithm](https://cp-algorithms.com/string/manacher.html) — the d1/d2 formulation used here
- Rubinchik & Shur, ["EERTREE: An Efficient Data Structure for Processing Palindromes in Strings"](https://doi.org/10.1016/j.ejc.2017.07.021), *European J. Combinatorics*, 2018
- Droubay, Justin & Pirillo, "Episturmian words and some constructions of de Luca and Rauzy" (2001) — the ≤ n distinct palindromes bound
- Kosolobov, Rubinchik & Shur, ["Pal^k Is Linear Recognizable Online"](https://arxiv.org/abs/1408.4576) (2015) — series links for palindromic factorisation
