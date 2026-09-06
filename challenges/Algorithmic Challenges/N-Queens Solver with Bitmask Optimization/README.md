# N-Queens Solver with Bitmask Optimization

**Category:** Algorithmic Challenges
**Difficulty:** I (brief: "Bitwise column/diagonal tracking; visualize solutions found")

**Status:** Implemented (Python)

The brief asks for bitmask backtracking and a visualization. Both are here,
plus two things past the brief: a **2-fold mirror-symmetry speedup** on top
of the bitmask search (the actual "beat the naive version" optimization),
and a **direct, empirical computation of the full 8-fold dihedral symmetry**
— counting not just solutions but *fundamentally different* solutions, with
the result checked against Burnside's lemma computed from real data rather
than asserted from memory. That check caught a wrong assumption along the
way — see "A claim that turned out to be wrong" below.

## The bitmask representation

A solution is a permutation `cols` of `range(n)`: `cols[r]` is the queen's
column in row `r`. Three integers track every attacked square after placing
rows `0..row-1`:

- `cols` — bit `c` set means column `c` is occupied.
- `diag1` — the "/" diagonals, shifted left one bit per row descended.
- `diag2` — the "\" diagonals, shifted right one bit per row descended.

The available columns for the next row are `mask & ~(cols | diag1 | diag2)`
in one integer operation — `mask = (1 << n) - 1` — instead of scanning
every placed queen to check attacks. Extracting each set bit with
`bit = available & -available` and clearing it with `available ^= bit` is
the standard two's-complement trick for iterating set bits without a loop
over all `n` positions. This (with slightly different variable names) is
the technique commonly attributed to Jeff Somers' bitwise N-Queens solver,
and is close to the fastest approach practical in a high-level language
without SIMD or hand-written assembly.

## The mirror-symmetry optimization

`count_solutions` doesn't search all `n` choices for row 0 — it searches
only the left half (`range(n // 2)`), and doubles each result, because
mirroring the entire board left-right maps any solution to another valid
solution with row 0's queen in the mirrored column. For odd `n` there is
also a middle column (`n // 2`); solutions starting there are **not**
doubled, because mirroring a middle-column-row-0 solution produces another
solution that *also* starts in the middle column of row 0 — its partner is
already inside that same subtree, so searching the full subtree once
(instead of half of it twice) already counts every pair exactly once.

```
 n     full (s)   halved (s)   speedup  solutions
 8       0.0006       0.0003     2.18x         92
 9       0.0025       0.0012     2.09x        352
10       0.0066       0.0029     2.26x        724
11       0.0447       0.0243     1.84x       2680
12       0.1871       0.0792     2.36x      14200
13       0.8415       0.4623     1.82x      73712
14       5.3626       2.5597     2.10x     365596
15      32.0016      17.2214     1.86x    2279184
```

(`uv run python benchmark.py`.) The speedup clusters around 1.8-2.3x, not
exactly 2x, because halving only row 0's choices doesn't halve the *work* of
searching them evenly — later rows still explore an asymmetric fraction of
the tree depending on which half-column was chosen first. Full 8-fold
symmetry reduction *during* the search (rather than after, as done below)
would in principle approach 8x, but correctly avoiding both over- and
under-counting near-symmetric partial boards mid-search is substantially
more intricate than this 2-fold trick for a proportionally smaller marginal
win — the reduction actually implemented here is the counting-based
approach below, not a deeper pruning rule.

## Full 8-fold symmetry: fundamental solutions and Burnside's lemma

The square has 8 symmetries (the dihedral group D4): identity, three
rotations, and four reflections (horizontal, vertical, both diagonals).
`count_fundamental_solutions(n)` computes, for every solution found by the
full (unhalved) search, its canonical form — the lexicographically smallest
of its 8 images under those symmetries — and counts the distinct canonical
forms. That count matches [OEIS A002562](https://oeis.org/A002562) exactly:

```
 n  total (A000170)  fundamental (A002562)
 0                1                      1
 1                1                      1
 2                0                      0
 3                0                      0
 4                2                      1
 5               10                      2
 6                4                      1
 7               40                      6
 8               92                     12
 9              352                     46
10              724                     92
11             2680                    341
12            14200                   1787
```

**Burnside's lemma** states the number of orbits under a group `G` acting
on a set equals `(1/|G|) * sum(|Fix(g)| for g in G)`, where `Fix(g)` is the
set of elements left unchanged by `g`. `fixed_point_counts(n)` computes
`|Fix(g)|` for each of the 8 symmetries directly (by checking `g(s) == s`
for every found solution `s`), and `test_burnside_lemma_matches_direct_canonicalization_count`
checks that `sum(fixed) / 8` always equals `count_fundamental_solutions(n)`
exactly — two independently-computed numbers that must agree if the
transforms and the canonicalization are both correct.

### A claim that turned out to be wrong

A commonly repeated statement about the N-Queens problem is that no
solution has true 4-fold rotational symmetry (i.e. `Fix(rot90)` is always
empty). The first version of this README's test suite asserted exactly
that, from memory, for `n = 1..9` — and it failed:

```
 n   |Fix(rot90)|
 4          2
 5          2
 6          0
 7          0
 8          0
 9          0
10          0
11          0
12          8
```

Both n=4 solutions are fixed by a 90-degree rotation (rotating either one
by 90 degrees reproduces itself), and so are 2 of the 10 solutions at n=5
and 8 of the 14,200 at n=12 — while n=6 through n=11 genuinely have none.
There is no clean pattern by `n mod 4` here; it is exactly the sort of claim
this repo's README's have flagged before as sounding authoritative while
being falsifiable by five minutes of computation. What *is* provably true,
and is what the test suite asserts now instead, is the group-theoretic fact
that `|Fix(g)| == |Fix(g^-1)|` for any symmetry `g` (since `g(s) = s` iff
`s = g^-1(s)`) — confirmed for `rot90`/`rot270` at every `n` tested.

## Edge cases

- **n = 0**: exactly one solution, the vacuous placement of zero queens —
  matches `A000170(0) = 1` by convention.
- **n = 1**: exactly one solution (the single queen).
- **n = 2, n = 3**: zero solutions each, provable by exhaustion (both are
  too small for any two queens to avoid all rows/columns/diagonals).
- **Determinism**: `all_solutions` returns solutions in a fixed, reproducible
  order (row-0 column ascending, depth-first) with no duplicates, checked
  directly in `test_nqueens.py`.
- **Validity of every intermediate board**, not just completed ones — a
  dedicated test checks that every "place" step's partial board has no
  internal attack, since the whole point of the bitmask is that this is
  guaranteed structurally rather than checked after the fact.

## Pacing: readable without pausing

The first version gave every place/backtrack step a flat 0.1-0.12s
`run_time` with no separate reading pause. For a one-off step that might be
fine; a demo search finding just 4 solutions at n=6 already produces ~250
place/backtrack steps, so at that rate the captions are pure blur — nobody
can read "Place row 2 at column 5" in 0.12 seconds while it's also
mid-transform. `pacing.py` applies the same two-phase fix as the MST and
sorting visualizers in this repo — a short **motion** phase, then a still
**hold** phase sized by subtitle-reading-speed practice — and is testable
with no Manim import:

```
uv run python pacing.py
n=6 limit=4:  258 steps, motion=39.5s hold=57.5s total=97.0s
n=8 limit=3:  349 steps, motion=52.9s hold=77.8s total=130.7s
```

This is also where the repeat-discount matters most in this repo: "Place
row 2 at column 5" and "Place row 4 at column 1" are the same caption
*shape*, but the row and column digits sit in unrelated positions and
almost always change together on a backtracking step. A naive
common-prefix/common-suffix trim breaks on exactly that case — the row
digit ends the prefix match, the column digit sits at the very end and
breaks the suffix match immediately, so the entire middle of the caption
gets (wrongly) counted as unread. `novel_characters` uses
`difflib.SequenceMatcher.get_matching_blocks()` instead, which finds every
matching run regardless of position, so the unchanged "at column"/"row"
text between the two changed digits is correctly recognized as
already-read — see the docstring in `pacing.py` for the full worked
example and `test_pacing.py` for the regression test.
`NQUEENS_HOLD_SCALE=0` drops every hold for fast iteration.

## Files

| File              | What it is                                                                                                                                                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nqueens.py`      | Bitmask backtracking, `count_solutions` (mirror-halved), `solve_bitmask_steps`/`all_solutions` (full search + Step stream), the 8 D4 transforms, `canonical_form`, `count_fundamental_solutions`, `fixed_point_counts` |
| `test_nqueens.py` | Cross-checks against OEIS A000170/A002562, board validity (completed and in-progress), the Burnside's-lemma verification, the corrected 4-fold-symmetry finding                                                        |
| `benchmark.py`    | Full-search vs mirror-halved timing (the table above), plus the cost of fundamental-solution counting vs plain counting                                                                                                |
| `pacing.py`       | Motion/hold timing for the animation, testable without Manim                                                                                                                                                           |
| `test_pacing.py`  | Caption-shape diffing (including the multi-digit-position case above), repeat-discount behavior, bounded total duration despite hundreds of steps                                                                      |
| `visualize.py`    | Manim scene animating the backtracking search live, stopping after a handful of solutions rather than rendering all of them, paced to be readable without pausing                                                      |

## Running it

```bash
uv run python nqueens.py -n 8               # counts + Burnside check for one n
uv run python nqueens.py --verify           # n = 0..12 against OEIS

uv run --with pytest pytest -q              # 60 tests
uv run python benchmark.py --quick          # n = 8..12, fast
uv run python benchmark.py                  # the tables above (n up to 15, ~40s)
uv run python pacing.py                     # animation durations, no rendering

uv run --with manim manim -pql visualize.py NQueensScene

# Fast layout iteration with every reading-hold dropped:
NQUEENS_HOLD_SCALE=0 uv run --with manim manim -pql visualize.py NQueensScene
```

Manim needs **ffmpeg** on the PATH plus the system Cairo/Pango development
libraries; `uv` handles the Python side only. `nqueens.py`, `benchmark.py`,
and the tests are pure standard library and need none of it. Rendered video
lands in `media/videos/visualize/` and is git-ignored.

## Where this is actually used

- **Constraint satisfaction benchmarking.** N-Queens is a standard,
  parameter-free benchmark for CSP solvers, SAT encodings, and local-search
  methods (min-conflicts, simulated annealing) precisely because its
  structure (one queen per row/column, three families of linear
  constraints) generalizes to real scheduling and resource-allocation
  problems without any domain-specific machinery.
- **Bitmask backtracking itself** is the general technique behind fast
  exact-cover and combinatorial-search solvers for small-to-moderate state
  spaces — the same "represent a frontier as a machine word, extract set
  bits" trick underlies fast Sudoku solvers and bitboard-based chess move
  generation.
- **Symmetry reduction in search** (the 2-fold trick here) is the same idea
  used at much larger scale in SAT/CSP solvers via *symmetry breaking
  predicates*, which add constraints that eliminate symmetric duplicate
  branches before the solver ever explores them.

## Sources

- Somers, ["N Queens Bit Solver"](http://jsomers.com/nqueen_demo/nqueens.html) — the bitmask column/diagonal technique
- OEIS [A000170](https://oeis.org/A000170) — number of n-queens solutions
- OEIS [A002562](https://oeis.org/A002562) — number of inequivalent (fundamental) solutions
- Burnside, *Theory of Groups of Finite Order*, 2nd ed., 1911 — the counting lemma (though it predates Burnside, sometimes called the Cauchy-Frobenius lemma)
- Rivin, Vardi & Zabih, ["A Deque-Based Algorithm for the N-Queens Problem"](https://www.researchgate.net/publication/220827794) and related symmetry-classification literature on N-Queens
- Gent, Jefferson & Nightingale, ["Complexity of n-Queens Completion"](https://doi.org/10.1613/jair.5512), *JAIR* 59, 2017 — modern complexity-theoretic treatment
