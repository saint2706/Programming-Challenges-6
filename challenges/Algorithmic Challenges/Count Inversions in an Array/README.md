# Count Inversions in an Array

**Category:** Algorithmic Challenges
**Difficulty:** B (brief: "compare merge-sort counting vs Fenwick tree approach")

**Status:** Implemented (Python)

An inversion is a pair `i < j` with `a[i] > a[j]` — a pair the array has in the
wrong order. The count is the standard "divide and conquer beats the obvious
O(n²)" exercise, and it means considerably more than the exercise suggests. It
is *exactly*:

- the minimum number of **adjacent transpositions** that sort the array (so,
  exactly the number of swaps bubble sort performs);
- the **Kendall tau distance** from the array to its sorted order;
- the statistic whose distribution over Sₙ is the **Gaussian binomial**
  `[n]_q! = ∏(1 + q + … + q^{i-1})`, the Mahonian numbers.

All three are asserted in the test suite rather than claimed here.

## The comparison the brief asks for, answered honestly

Merge sort and the Fenwick tree are both O(n log n), and **in Python they are
the same speed**, because both spend their O(n log n) *in the interpreter* and
the interpreter is the entire cost. At n = 10⁶:

```
mergesort  2.75s      fenwick  3.65s
```

The asymptotics are identical and the constants are within 30%. Fenwick is
slightly slower here purely because its inner loop is bit arithmetic in Python
while merge sort's is list slicing in C. That is the whole finding, and it is
not very interesting — so this directory adds two methods that move the work
out of the interpreter, and the comparison between *those* is where the real
result is.

| Method              | Complexity     | n = 10⁵   | n = 10⁶  | Notes                              |
| ------------------- | -------------- | --------- | -------- | ---------------------------------- |
| `count_brute`       | O(n²)          | —         | —        | The definition; the test oracle    |
| `count_insort`      | O(n²) memmove  | 0.835     | 84.6     | **Fastest below n ≈ 20 000**       |
| `count_mergesort`   | O(n log n)     | 0.171     | 2.75     | Iterative — no recursion limit     |
| `count_fenwick`     | O(n log n)     | 0.196     | 3.65     | Also gives per-element counts      |
| `count_numpy`       | O(n log² n)    | 0.132     | **1.11** | Vectorised merge sort              |
| `count_numpy_radix` | **O(n log n)** | **0.074** | 1.46     | Vectorised radix; best asymptotics |

## `count_numpy`: batching a merge sort level with one binary search

A bottom-up merge sort at width `w` views the array as a `(blocks, 2w)` matrix
of *independent* merges. Each block needs "how many left-half elements exceed
this right-half element" — a binary search. But `np.searchsorted` searches one
sorted haystack, not a batch of them.

The trick is to make the batch into one haystack. Add `block_index · (n+1)` to
every value first. Values are dense ranks in `[0, n]`, so block `b` occupies
`[b(n+1), b(n+1)+n]` — disjoint from every other block, and *increasing* in
`b`. The concatenated left halves are therefore globally sorted, and a single
binary search over the whole array answers every block simultaneously.
Subtracting `b·w` converts each global position back to a within-block count.

Two searches per level do all the work, and the second one is free information:

```
le[j] = #{left ≤ right[j]}    → inversions are w − le[j],  right[j] merges to j + le[j]
lt[i] = #{right < left[i]}    →                            left[i]  merges to i + lt[i]
```

So the level's merge is a **scatter**, not a second sort. Ties are handled
entirely by the two `side` arguments (`"right"` for `le`, `"left"` for `lt`),
which makes the merge stable and the destination indices collision-free with no
special case.

Cost: O(n log w) per level, O(n log² n) total — one log *worse* than textbook
merge sort, and 2.5× faster in practice.

## `count_numpy_radix`: removing the extra log, and losing anyway

Turn the recursion inside out. Split by **value** instead of by index —
MSB-first on the dense ranks — and the question "how many earlier elements are
larger" becomes "how many 1-bits precede this 0-bit", which is a **cumsum
rather than a binary search**. O(n) per bit, O(n log n) overall.

The identity it rests on: every inverted pair `(i, j)` has a unique highest bit
at which `rank[i]` and `rank[j]` differ, and at that bit `rank[i]` has a 1 and
`rank[j]` a 0. So

```
inversions = Σ over bits b of
             #{(i,j) : i < j, ranks agree above b, r[i] has 1, r[j] has 0}
```

and the three conditions become "same group, earlier position, one before
zero" once the array is kept MSD-radix-partitioned. Equal ranks never differ at
any bit, so ties contribute exactly zero — the correct answer for strict
inversions, with no special case anywhere. The invariant carried across bits is
that positions within a group stay in original index order: true initially (the
permutation is the identity), and preserved because the partition by bit is
stable within each group.

**And it is not the fastest at scale.** That is the most interesting result
here:

```
      n     MiB     radix     numpy     winner
  10000     0.1    0.0061    0.0073      radix
 100000     0.8    0.0791    0.0802      radix
 300000     2.3    0.2938    0.3691      radix
 500000     3.8    0.6807    0.4083      numpy
1000000     7.6    1.5047    1.1326      numpy
2000000    15.3    3.2683    2.1223      numpy
```

The radix method scatters across the whole array once per bit. Below ~300k the
working set fits in last-level cache and its better complexity shows; above
that, memory bandwidth decides the race and `searchsorted`'s far better
locality wins by ~1.4×, despite doing asymptotically more work. Nothing about
the algorithms changes at the crossover — only where the array lives.

**Asymptotics rank algorithms; the memory hierarchy ranks implementations.**
`method="auto"` uses the measured crossover, not the theoretical one.

## The other surprise: the quadratic method wins for a long time

`count_insort` walks right to left keeping the suffix sorted with
`bisect.insort`. That is O(n²), but the quadratic part is `memmove`, which
moves gigabytes per second, while a Python-level merge step manages about one
element per microsecond:

```
    n      insort   mergesort     fenwick      winner
  100     0.00002     0.00007     0.00008      insort
 3000     0.00155     0.00338     0.00657      insort
10000     0.01139     0.01350     0.01466      insort
30000     0.08321     0.04424     0.04998   mergesort
```

`method="auto"` uses n = 3000 as the threshold rather than the ~20 000 measured
crossover, to stay comfortably clear of the quadratic term on adversarial
sizes.

## What else is in here

| Function                                   | What it gives you                                                                             |
| ------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `count_smaller_to_right`                   | Per element: how many later elements are smaller. Sums to the total                           |
| `count_greater_to_left`                    | The same pairs, attributed to the later index                                                 |
| `count_significant_inversions(a, f)`       | Pairs with `a[i] > f·a[j]`. Uses `bisect`, not a two-pointer, so negative factors are correct |
| `kendall_tau_distance`                     | Discordant pairs between two rankings; a genuine metric                                       |
| `kendall_tau_b`                            | Tie-corrected rank correlation, Knight's O(n log n) form                                      |
| `inversion_table` / `from_inversion_table` | Knuth's inversion table and its inverse — the bijection                                       |
| `inversion_polynomial(n)`                  | The Mahonian numbers `[n]_q!`                                                                 |
| `Fenwick`                                  | Standalone BIT with an O(n) bulk constructor                                                  |

### `inversion_table` is indexed by value, not position

This is the distinction that makes it a bijection rather than merely a
statistic. Knuth's table has `b[v]` = how many values *greater than v* precede
v, so `0 ≤ b[v] ≤ n−1−v` **independently for each v**. The table therefore
ranges over a product of ranges of sizes `n, n−1, …, 1` — which has exactly
`n!` elements — and `from_inversion_table` shows the map onto it is invertible
by inserting values largest-first at index `b[v]`, where the placement is
forced.

That bijection is also *why* the generating function factors: each coordinate
of the Lehmer code is independent, so `Σ_{π ∈ Sₙ} q^{inv(π)} = ∏(1 + q + … +
q^{i−1}) = [n]_q!`. The test suite verifies this by enumerating all of Sₙ for
n ≤ 7 and matching the coefficients, and separately checks the row values
against OEIS [A008302](https://oeis.org/A008302).

## Edge cases the tests pin down

- **NaN is rejected, with an explanation.** Comparisons against NaN are all
  false, so the elements are not totally ordered and the inversion count is not
  a property of the data — it becomes a property of the algorithm. Every method
  raises rather than returning a plausible-looking number. `validate=False`
  opts out.
- **`-0.0` ties with `0.0`**, in both orders, because `-0.0 == 0.0`.
- **Infinities are fine**; they are ordinary extremes of the total order.
- **Unhashable elements work.** Rank compression is sort-based (`sorted(range(n),
  key=…)`) and needs only `<`, so lists group correctly despite not being
  hashable. Equality is inferred as "neither is less than the other", which is
  the right definition for a total order.
- **Huge integers, `Fraction`s and mixed numeric types work** — they produce an
  object dtype, so the vectorised methods detect that and fall back.
- **Nothing is mutated**, including by `count_significant_inversions`.
- **`return_sorted=True` returns the original elements**, not the keyed ones,
  so it composes with `key=` the way `sorted` does.
- **`reverse=True`** counts pairs already in ascending order, and
  `inv(a) + inv_desc(a) = C(n,2) − ties` holds for every array.
- **No recursion limit.** The merge sort is bottom-up; tested at n = 2 × 10⁶.
- The total exceeds 2³¹ at n = 200 000 reversed (2 × 10¹⁰ inversions); every
  method returns a Python `int`, so nothing silently wraps.

## Where this is actually used

The inversion count has an unusually short route to production, because it is
the same number as **Kendall's tau**, and rank correlation is a working
statistical tool rather than an exercise.

**Evaluating rankings.** Search relevance, recommender systems and
learning-to-rank models are judged by how far a predicted ordering sits from a
reference one, and tau is one of the standard metrics for it, alongside NDCG and
Spearman. `kendall_tau_b` is the tie-corrected form `scipy.stats.kendalltau`
returns — which matters because real relevance labels are mostly ties. A
five-point judgement scale over a thousand documents is almost entirely tied
pairs, and the uncorrected coefficient reports nonsense on it.

**Nonparametric statistics.** Tau is the default when Pearson's assumptions
fail: ordinal outcomes, monotone-but-not-linear relationships, heavy tails. It
appears in A/B tests on ordinal metrics, in copula estimation in finance, and
anywhere the data are ranks to begin with.

**Rank aggregation.** The Kemeny–Young consensus ranking is *defined* as the
ordering minimizing total Kendall tau to a set of input rankings — the basis of
meta-search, of combining several judges or models, and of a well-studied voting
rule.

**Measuring sortedness, to decide what to do next.** "How far from sorted is
this?" drives adaptive sorting (TimSort's run detection pays off precisely when
inversions are few), query planning over data suspected to be nearly ordered,
and drift detection in a stream that ought to be monotone. Out-of-order arrivals
in a time-series ingest pipeline *are* inversions, and `count_smaller_to_right`
names which records are the offenders rather than only how many there are.

**The Fenwick tree outlives the problem it was brought in for.** Point update,
prefix sum, O(log n), one flat array and no node objects: leaderboards ("what
rank is this score"), running aggregates over time buckets, inventory and
order-book depth, and any order-statistic query on a changing set. It reappears
in [the Josephus solver](../Josephus%20Problem%20Solver/) for a completely
unrelated question.

**And the vectorization trick generalizes.** Offsetting each block so that a
batch of independent sorted arrays becomes one globally sorted array — turning
n/w separate binary searches into a single `np.searchsorted` — works for any
per-segment lookup you want to run without a Python loop: bucketed joins,
per-group interpolation, segment-wise rank assignment.

## Files

| File                 | What it is                                                                                      |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| `inversions.py`      | Six counters, per-element views, Kendall tau, Lehmer code, `Fenwick`, CLI                       |
| `test_inversions.py` | 252 tests: exhaustive over Sₙ (n ≤ 7) and all 2¹² binary arrays, plus identities and edge cases |
| `benchmark.py`       | Scaling, both crossovers, input-shape sensitivity, empirical complexity check                   |

## Running it

```bash
uv run --with numpy python inversions.py --demo
uv run --with numpy python inversions.py --verify
uv run --with numpy python inversions.py 3 1 4 1 5 --detail

uv run --with pytest --with numpy pytest -q            # all 252
uv run --with pytest --with numpy pytest -q -m "not slow"
uv run --with pytest --with numpy --with scipy pytest -q   # also cross-checks tau-b
uv run --with numpy python benchmark.py --quick
```

numpy is optional; without it `method="auto"` falls back to `mergesort`.

## Sources

- Knuth, *TAOCP* Vol. 3 §5.1.1 — inversions, the inversion table, and the Mahonian distribution
- [Knight, "A Computer Method for Calculating Kendall's Tau with Ungrouped Data"](https://doi.org/10.1080/01621459.1966.10480879) (1966) — the O(n log n) tie-corrected tau
- [OEIS A008302](https://oeis.org/A008302) — the Mahonian triangle
- [Chan & Pătrașcu, "Counting Inversions, Offline Orthogonal Range Counting, and Related Problems"](https://doi.org/10.1137/1.9781611973075.15) (SODA 2010) — the O(n √(log n)) bound, noted but not implemented
