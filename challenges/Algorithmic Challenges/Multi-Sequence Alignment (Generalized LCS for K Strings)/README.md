# Multi-Sequence Alignment (Generalized LCS for K Strings)

**Category:** Algorithmic Challenges
**Difficulty:** A (brief: "Extend classic 2-string LCS DP to K sequences")

**Status:** Implemented (Python)

The longest common subsequence problem, generalized from 2 strings to K: find
the longest sequence that is a subsequence of all K inputs. The textbook
2-string DP generalizes directly -- one cell per pair of prefix lengths
becomes one cell per K-tuple of prefix lengths -- but that direct
generalization's `prod(n_i)` state space is only practical for a handful of
short sequences. The brief asks for the direct generalization; this
implementation also adds the actual state-of-the-art fix for that blow-up.

| Method                | Time                               | Space         | Optimal?     |
| --------------------- | ---------------------------------- | ------------- | ------------ |
| `brute_force_mlcs`    | O(prod(n_i) \* K)                  | O(prod(n_i))  | Always       |
| `dominant_point_mlcs` | O(D \* Sigma \* K), D << prod(n_i) | O(D)          | Always       |
| `progressive_mlcs`    | O(K) pairwise LCS calls            | O(max(n_i)^2) | No guarantee |

`brute_force_mlcs` and `dominant_point_mlcs` are two independent exact
algorithms, cross-verified against each other below. `progressive_mlcs` is
a fast heuristic included the same way TSP's nearest-neighbor is: to make
"exact is expensive, heuristics trade that away" concrete, including a
worked example of exactly how it goes wrong.

## `brute_force_mlcs`: the direct K-dimensional generalization

`table[(i_1, ..., i_K)]` = the LCS of the length-`i_1`, ..., length-`i_K`
prefixes of the K sequences. The 2-string recurrence generalizes exactly: if
every sequence's prefix ends in the *same* character, taking it is always
safe (it can only help, since it's available in every sequence);
otherwise, try dropping the last character of each sequence in turn and
keep whichever drop leaves the longest result:

```
table[i_1, ..., i_K] =
    table[i_1-1, ..., i_K-1] + c        if seq_1[i_1] = ... = seq_K[i_K] = c
    max over j of table[..., i_j - 1, ...]   otherwise
```

Implemented as memoized recursion (`functools.cache` over index tuples)
rather than explicit nested loops, so it works for any K without
K-specific code. This reaches exactly `prod(n_i)` reachable states -- the
whole K-dimensional grid, whether or not the optimum actually needs most of
it.

## `dominant_point_mlcs`: only visit the states that can matter

The grid brute force visits is mostly wasted effort: for a fixed common
subsequence length `l`, many K-tuples of prefix lengths that achieve length
`l` are strictly worse than another K-tuple on every coordinate
simultaneously, and can therefore be discarded without ever affecting the
final answer. This is the **dominant point method**, originating with Hunt
& Szymanski's 2-string algorithm and generalized to K strings by Hakata &
Imai (1992) -- the method behind the modern MLCS-solver literature (e.g.
Wang, Korkin & Shang's `FAST_LCS`, IJCAI 2009).

A **point** is a K-tuple of prefix lengths `(p_1, ..., p_K)`. Point `p`
**dominates** point `q` when `p_i <= q_i` for every `i`: reaching `p`
leaves at least as much of every sequence still available as reaching `q`
does, so anything `q` can be extended to, `p` can be extended to at least
as well. Among all points reachable via *some* length-`l` common
subsequence, only the dominant (Pareto-minimal) ones can ever be needed --
every non-dominant one is redundant by definition.

The search proceeds level by level instead of grid-cell by grid-cell:

- `L_0 = {(0, ..., 0)}`.
- From each point `p` in `L_l` and each character `c` that occurs
  somewhere after position `p_i` in **every** sequence `i`, the successor
  point is `(next_occ_1[p_1][c], ..., next_occ_K[p_K][c])` -- the earliest
  place each sequence can next supply a `c`. This greedy choice is always
  safe: taking the earliest available match leaves the most sequence left
  for everything after, so it can never make the eventual answer shorter
  than taking a later occurrence would have.
- `L_{l+1}` is the dominant subset of every such successor point, over all
  `p` in `L_l` and all valid `c`.
- The MLCS length is the last level index with a non-empty `L_l`; an actual
  LCS is recovered by walking parent/character pointers from a point in
  that level back to `L_0`.

`next_occ_i[p][c]` -- "smallest position after `p` in sequence `i` holding
character `c`" -- is precomputed once per sequence via a single backward
scan (`O(n * |alphabet|)`), giving O(1) successor lookups instead of
rescanning. The dominance filter itself is the direct O(m^2 \* K) pairwise
check over each level's candidates, not a sorted/incremental skyline
structure -- the sub-quadratic skyline algorithms the MLCS literature
layers on top of this are a further optimization on the *computation* of
each level, separate from the state-space reduction this implementation
demonstrates (visiting `D` dominant points total instead of `prod(n_i)`
grid cells). Reimplementing a full skyline-maintenance data structure was
out of scope here, the same way reimplementing Edmonds' blossom algorithm
was out of scope for TSP's Christofides heuristic.

## `progressive_mlcs`: fast, but an early choice can't be undone

Fold sequences into a running answer one at a time: `cur = LCS(s_1, s_2)`,
then `cur = LCS(cur, s_3)`, and so on. Each step's result is a subsequence
of both its inputs, so by induction the final result is always a valid
common subsequence of all K sequences -- but it commits to one particular
optimal alignment of the first two sequences before ever consulting the
third, which is exactly the well-documented failure mode of progressive
multiple sequence alignment (the same reason tools like ClustalW's guide-
tree approach can't correct an early mistake later).

Concrete instance where this actually happens:

```python
>>> progressive_mlcs(["AABB", "BBAA", "BBBB"])
''
>>> brute_force_mlcs(["AABB", "BBAA", "BBBB"])
'BB'
```

`"AABB"` and `"BBAA"` have *two* tied length-2 optimal pairwise LCS's:
`"AA"` and `"BB"`. The DP's tie-break happens to pick `"AA"` first -- a
perfectly valid optimal answer for the first two strings in isolation --
but `"BBBB"` contains no `A` at all, so folding it in collapses the result
to `""`. The other tied choice, `"BB"`, would have survived and is in fact
the true 3-way optimum. Once `"AA"` is chosen there is no mechanism to go
back and reconsider.

## Correctness

```
$ uv run --with pytest pytest -q
138 passed in 0.2s
```

`dominant_point_mlcs` is checked against `brute_force_mlcs` -- two
structurally unrelated exact algorithms -- across 60 random instances
spanning K=2..5 and alphabets of size 2-4, on length alone (both are also
independently checked to return an actual valid common subsequence of
every input string). `progressive_mlcs` is checked to never *exceed* the
true optimum (which would indicate a bug, since it can only be a valid
common subsequence) across another 60 random instances, plus the specific
`AABB`/`BBAA`/`BBBB` counterexample above as a regression test pinning the
exact gap.

## Benchmarks

```
$ uv run python benchmark.py
State space size: brute force (prod n_i) vs dominant points actually generated (K=4, alphabet='ACGT')
   n    grid cells   dominant points   reduction
------------------------------------------------
   4           625                 1      625.0x
   6         2,401                 6      400.2x
   8         6,561                 3     2187.0x
  10        14,641                10     1464.1x
  12        28,561                18     1586.7x
  14        50,625                17     2977.9x
  16        83,521                16     5220.1x

Wall-clock time: brute_force_mlcs vs dominant_point_mlcs (K=4, alphabet='ACGT')
   n     brute force    dominant point   speedup
------------------------------------------------
  16        0.2476s          0.0002s   1085.0x
  20        0.5498s          0.0004s   1478.3x
  24        1.1125s          0.0005s   2229.5x
  28        2.3789s          0.0029s    829.5x
  32        4.7224s          0.0036s   1309.0x
  36        8.1986s          0.0259s    316.4x

progressive_mlcs solution quality vs the true optimum, 300 random K=3 instances
25/300 instances (8.3%) had progressive_mlcs strictly below the true MLCS length; mean gap 0.083 characters.
```

**The grid-cells-vs-dominant-points table is the direct evidence for the
whole optimization.** `brute_force_mlcs` always visits exactly
`prod(n_i + 1)` states by construction -- 83,521 of them at n=16, K=4.
`dominant_point_mlcs` visited only 16 dominant points *total, across every
level*, for that same instance: over 5000x fewer states, for the identical
answer. The reduction isn't monotonic in n (compare n=8's 2187x to n=10's
1464x) because how many points survive the dominance filter depends on the
random instance's actual match structure, not just its size -- but the
trend is unmistakable as n grows.

**Wall-clock time confirms it's not just an accounting trick.**
`brute_force_mlcs`'s O(n^K) growth is visible directly: n=16 to n=36 (a
2.25x length increase) is roughly a 33x time increase, in the neighborhood
of `2.25^4 ≈ 25.6x` predicted by `n^4` -- Python-level constant-factor
noise aside, the shape matches. `dominant_point_mlcs` stays under 3
hundredths of a second across the same range, because its cost tracks `D`
(dominant points), which grew far more slowly than `n^4` on these
instances.

**`progressive_mlcs` gets it wrong on a real fraction of instances**, not
just the hand-picked example above -- 8.3% of 300 random K=3 instances hit
a strict gap, with a mean cost of ~0.08 characters per instance. Small in
absolute terms at this scale, but it demonstrates the failure mode is not a
one-off curiosity: any tie in an early pairwise LCS is a coin flip the
algorithm has no way to know is significant until it's already too late to
undo.

## Run it

```bash
cd "challenges/Algorithmic Challenges/Multi-Sequence Alignment (Generalized LCS for K Strings)"

uv run python mlcs.py         # exact + heuristic on one 4-sequence instance
uv run python benchmark.py    # state-space size, wall-clock time, heuristic quality

uv run --with pytest pytest -q   # 138 tests
```

## Where this is used

**Bioinformatics: multiple sequence alignment (MSA).** Comparing 3+ DNA,
RNA, or protein sequences to find conserved regions is the canonical MLCS
application -- the dominant-point method here traces directly back to
Hunt & Szymanski's and Hakata & Imai's work in that context, and real
tools (MUSCLE, MAFFT, Clustal Omega) all still make the same exact-vs-
heuristic tradeoff this challenge demonstrates, typically favoring
progressive-alignment-family heuristics at genome scale precisely because
exact MLCS is impractical past a handful of long sequences.

**Version control and diff/merge tools.** A 3-way merge (comparing a
common ancestor against two diverged branches) is a 3-sequence common-
subsequence problem; tools that need to reconcile more than two divergent
copies of a file face the true K-sequence version of the problem this
challenge solves exactly.

**Plagiarism and code-clone detection across many sources.** Finding
content shared across more than two documents/files generalizes pairwise
diffing the same way this challenge generalizes 2-string LCS.

**What's still state of the art beyond this.** Production MLCS solvers
(the `FAST_LCS` family and its successors, e.g. the "Path Recorder
Algorithm", Bioinformatics 2020) improve on the plain dominant-point method
with faster incremental skyline maintenance and specialized data
structures for very large numbers of dominant points -- genuine further
engineering on top of the same state-space-reduction idea implemented
here, out of scope for this challenge but the natural next step.

## Sources

- [Hunt, J.W. & Szymanski, T.G., "A Fast Algorithm for Computing Longest Common Subsequences," *Communications of the ACM* 20(5):350-353, 1977](https://doi.org/10.1145/359581.359603) -- the origin of the dominant-point/match-point idea for 2 strings.
- [Hakata, K. & Imai, H., "The Longest Common Subsequence Problem for Small Alphabet Size Between Many Strings," *ISAAC* 1992, LNCS 650](https://doi.org/10.1007/3-540-56279-6_99) -- the generalization to K strings this implementation follows.
- [Wang, Q., Korkin, D. & Shang, Y., "A Fast Multiple Longest Common Subsequence (MLCS) Algorithm," *IEEE Transactions on Knowledge and Data Engineering* 23(3):321-334, 2011 (conference version: IJCAI 2009)](https://www.ijcai.org/Proceedings/09/Papers/250.pdf) -- `FAST_LCS`, the modern dominant-point MLCS algorithm this implementation is in the same family as.
- [Wang, Q. et al., "A path recorder algorithm for multiple longest common subsequences (MLCS) problems," *Bioinformatics* 36(10):3035-3042, 2020](https://doi.org/10.1093/bioinformatics/btaa134) -- a current (2020) state-of-the-art refinement, noted above as the next step beyond this implementation's scope.
