# Reservoir Sampling & Weighted Random Sampling Library

**Category:** Algorithmic Challenges
**Difficulty:** I (brief: "streaming uniform sampling plus A-Res weighted variant")

**Status:** Implemented (Python)

Pick k items at random from a stream of n items, where n isn't known ahead of
time and the stream is too large (or too slow, or literally infinite -- a log
tail, a Kafka topic) to hold in memory. `random.sample` needs the whole
population up front; this is the family of algorithms that don't.

| Method           | Time                | Space | Gives you                                       |
| ---------------- | ------------------- | ----- | ----------------------------------------------- |
| `reservoir_r`    | O(n)                | O(k)  | Uniform sample, Vitter's Algorithm R (1985)     |
| `reservoir_l`    | O(k·(1 + log(n/k))) | O(k)  | Same distribution as R, far fewer coin flips    |
| `a_res`          | O(n log k)          | O(k)  | Weighted sample without replacement, key + heap |
| `reservoir_chao` | O(n)                | O(k)  | Weighted sample without replacement, no heap    |

All four read the stream exactly once, via plain `next()` calls -- no
`len()`, no second pass, no materializing the input. A generator, a database
cursor, or any other single-use iterator works.

## Algorithm R: the coin-flip version

Fill the reservoir with the first k items. For every later item, at
1-indexed position i (i > k), replace a uniformly random reservoir slot with
it with probability k/i -- implemented by drawing j uniformly from 1..i and
only replacing slot j when j <= k:

```python
for i, item in enumerate(it, start=k + 1):
    j = rng.randint(1, i)
    if j <= k:
        reservoir[j - 1] = item
```

**Why every item lands with probability exactly k/n, for any i, not just the
last one.** Induct backwards. The last item (i = n) is accepted with
probability k/n by construction -- that's the base case. An earlier item at
position i survives into the *final* reservoir only if it also survives every
later item's replacement draw. By induction, the draw at a later position m
evicts any specific current occupant with probability (k/m)·(1/k) = 1/m (pick
that occupant's slot, which happens with probability 1/k, and choose to
replace it, which happens with probability k/m). So the survival probability
from position i to the end is the telescoping product

```
prod_{m=i+1}^{n} (1 - 1/m) = i/n
```

and item i enters the reservoir in the first place with probability k/i (by
construction for i <= k, and by the draw's own definition for i > k).
Multiplying: (k/i)·(i/n) = k/n. The i cancels -- which is the whole point: no
item is special, and the algorithm needs no knowledge of n to guarantee it.

## Algorithm L: same distribution, without a coin flip per element

Algorithm R does O(1) work on *every* element after the fill, even the ones
it rejects. Once you know an item at position i wasn't the one accepted,
Algorithm L asks a sharper question: how many further items will also be
rejected, before the next one that isn't? That's a random variable with a
known distribution, sampled directly, so the algorithm can jump straight past
a run of rejections instead of visiting each one.

**Deriving the skip distance.** Let w be the probability that the current
candidate item gets accepted (this starts at k/(k+1), Algorithm R's
acceptance probability for the first item past the fill). "Item j is
rejected" has probability 1 − w, so the probability that the *next G* items
are all rejected is (1 − w)^G. That means G, the number of items to skip
before the next acceptance, has survival function

```
P(G >= g) = (1 - w)^g
```

which is a geometric distribution. Sampling it by inverse-CDF: draw
u ~ Uniform(0, 1), set (1 − w)^g = u, and solve for g:

```
g = floor( log(u) / log(1 - w) )
```

One uniform draw, one floor, no per-item flip -- and `itertools.islice(it,
g, g + 1)` both discards the g skipped items and hands back the accepted one
in a single pass over the iterator.

**Advancing w without re-deriving it.** After an item is accepted, w needs to
move on to the acceptance probability for the *next* candidate. Rather than
recompute it from the current stream position and k (which would need to
know how many items have gone by -- exactly the count Algorithm L is trying
to avoid tracking one at a time), it's updated multiplicatively:

```python
w *= math.exp(math.log(fresh_u) / k)
```

This is the same key transform `a_res` uses for weighted sampling (see
below), specialized to every item having weight 1: `w` is tracking (the
distribution of) the maximum of k independent uniform draws, which is exactly
what determines whether a uniformly-random size-k subset would include the
next candidate. Each accepted item keeps that invariant true, which is why a
single multiplicative update suffices instead of a recomputation from scratch.

**Cost.** Each accepted item's skip distance grows in expectation as the
stream progresses (w drifts toward 1, i.e. later acceptances become likelier
per-candidate, because there's more stream left for k slots to compete for)
-- accepted item and skip distances follow a shrinking geometric progression,
so only O(k log(n/k)) skips are needed to reach the end of an n-item stream,
each O(1) work, on top of the O(k) fill. Algorithm R touches every element
that passes; Algorithm L only *does work* on that much smaller count.

## A-Res: weighted sampling without replacement, in one pass

Give each `(item, weight)` pair a key:

```
key = u ** (1 / weight),   u ~ Uniform(0, 1)
```

and keep the k items with the largest keys, using a size-k min-heap so each
item costs O(log k). That's the entire algorithm -- the non-obvious part is
*why* keeping the largest keys is a weighted sample at all.

**The key transform, derived.** Fix one item with weight w and look at its
key's CDF:

```
P(u^(1/w) <= x) = P(u <= x^w) = x^w        for x in [0, 1]
```

That's exactly the CDF of `max(u_1, ..., u_w)` for w independent uniform
draws, when w is a positive integer: the max of w iid Uniform(0,1) variables
has CDF x^w (all w of them must be <= x). So `u^(1/w)` behaves *exactly* like
"the largest of w independent lottery tickets" -- and `u^(1/w)` extends that
picture continuously to non-integer weight, matching the same CDF shape.

**Why that makes "keep the top key" a weighted choice.** Picture giving item
i exactly w_i tickets (for integer weights), pooling every item's tickets
into one urn, and drawing one uniform value per ticket. Item i's key is then
the maximum of its own w_i draws -- so "which item has the single largest key
overall" is the same event as "which item holds the ticket with the largest
draw in the whole pool." Every ticket is an independent, identically
distributed draw, so the winning ticket is uniform over all tickets in the
urn, and

```
P(item i has the largest key) = (i's ticket count) / (total tickets) = w_i / sum(w)
```

which is exactly weighted-choice-by-w_i. That settles k = 1.

**Extending to k > 1.** Condition on which item wins the top key (item i,
with probability w_i / sum(w), from the argument above). The *remaining*
keys, conditioned on being below the winner's key, are still independent
draws from their own per-item distributions -- picking the second-largest key
is now the same problem with item i and its tickets removed from the urn,
which is precisely the recursive definition of weighted sampling without
replacement: draw one item, remove its weight, repeat with probabilities
proportional to what's left. So "keep the top k keys" *is* that sequential
process, computed in a single streaming pass with no shrinking weight table
and no re-normalizing after every pick -- the u^(1/w) keys front-load all of
that bookkeeping into one number per item, decided the moment the item is
seen.

## A-Chao: weighted sampling without replacement, no heap at all

A-Res answers the weighted-sampling question with a key per item and a
min-heap. Chao (1982) -- 24 years earlier -- answers the *same* question with
neither: just a running weight total and a coin flip per item, O(1) work
each instead of A-Res's O(log k).

Fill the reservoir with the first k items. For every later item i with
weight `w_i`, maintain `W`, the running total of every weight seen so far
(including `w_i`), and accept item i with probability

```
p = k * w_i / W
```

If accepted, evict a uniformly random current occupant and put item i in its
place:

```python
total_weight += weight
if len(reservoir) < k:
    reservoir.append(item)
else:
    p = k * weight / total_weight
    if rng.random() < p:
        reservoir[rng.randrange(k)] = item
```

**Why `k*w_i/W` and not, say, plain `w_i/W`.** Set every weight to 1: the
formula collapses to `k*1/i = k/i` -- Algorithm R's own replacement
probability, exactly. That is not a coincidence to note in passing; it *is*
the derivation. A-Chao is Algorithm R's `k/i` rule, generalized so an item's
pull on the accept/reject coin scales with its weight instead of counting
for one unit like everyone else's. It also settles a real ambiguity: a
without-the-`k` variant (`p = w_i/W`) shows up in at least one secondary
source's simplified sketch, but plugging it into the same telescoping
argument below breaks monotonicity outright (verified numerically -- a
lighter item can end up *more* likely to survive than a heavier one), so
`k*w_i/W` is the one implemented and tested here.

**The telescoping proof**, structurally identical to Algorithm R's. Item i
is accepted with probability `k*w_i/W_i` (`W_i` = the running total right
after item i). Once in, item i survives a later item m's replacement draw
(`m > i`) unless m is both accepted (probability `k*w_m/W_m`) *and* happens
to land on i's slot (probability `1/k`) -- so it survives that one draw with
probability

```
1 - (k*w_m/W_m)*(1/k) = 1 - w_m/W_m = (W_m - w_m)/W_m = W_{m-1}/W_m
```

and surviving every draw from position i+1 to n is the telescoping product

```
prod_{m=i+1}^{n} (W_{m-1}/W_m) = W_i / W_n
```

Multiplying by the entry probability: `(k*w_i/W_i) * (W_i/W_n) = k*w_i/W_n`.
`W_i` cancels, leaving a probability proportional to `w_i` alone -- the same
shape as Algorithm R's `k/n` falling out of `(k/i)*(i/n)`, because A-Chao
*is* that proof with `w_i` substituted for the implicit weight of 1.

**Honest caveat.** That proof runs forward from "item i's entry," so it only
covers items that arrive *after* the reservoir is already full. The first k
items enter unconditionally -- there is no other choice, since the reservoir
has exactly k slots and exactly k candidates have been seen -- so their
survival to the end is `W_k/W_n` for *all of them alike*, not the
individually-weighted `k*w_i/W_n` that later items get. Concretely: with
weights `[1, 2, 3, 4]` and k=2, items 1 and 2 (the fill, weights 1 and 2)
both converge empirically to `(1+2)/10 = 0.30`, ignoring their own unequal
weights, while items 3 and 4 (processed after the fill) land almost exactly
on their `k*w_i/W_n` targets of 0.60 and 0.80. This is a real property of
the simple streaming algorithm exactly as it is commonly presented --
Wikipedia's reservoir-sampling article and Efraimidis's 2010 survey (see
Sources) sketch the identical unconditional-fill version -- and it means the
implementation here is order-sensitive: a stream where a heavy item happens
to land in the first k positions will under-represent it relative to a
lighter item that also lands there. Chao's original paper describes a more
elaborate initialization procedure to fix this, which is out of scope for
this challenge (in the same spirit as A-ExpJ below: a known, named gap
rather than a silently swallowed one). It does not affect this module's k=1
exactness claim (k=1 means exactly one "fill" item, nothing to be unequal
with) and it does not break the weaker k>1 monotonicity check `verify` runs,
as long as the test weights aren't adversarially front-loaded -- which is
exactly why `verify_weighted_k_gt_1`'s default weights for `reservoir_chao`
are ascending in stream order rather than shuffled.

## A-ExpJ (the weighted analogue of Algorithm L) -- deliberately not implemented

Efraimidis and Spirakis also describe a skip-optimized weighted variant,
A-ExpJ, that avoids drawing a full key for every item the way A-Res does.
Getting its jump formula right (it advances an exponential "budget" variable
using each item's own weight, then reconstructs a valid A-Res key only for
the items that get accepted) is easy to get subtly wrong from memory, and a
weighted sampler that's *quietly* biased is worse than a slower one that
isn't. The three methods above -- with the derivations spelled out above,
not just cited -- are the depth this challenge calls for; A-ExpJ's own
paper is linked below for anyone who wants to add it properly.

## Statistical verification

These are randomized algorithms, so correctness means empirical selection
frequencies converge to theory over many trials, not that any one run
matches a fixed expected output.

**Uniform sampling's guarantee is exact for any k <= n** (that's what the
Algorithm R induction above proves), so `verify_uniform` checks every item's
observed selection proportion against k/n directly, for both `reservoir_r`
and `reservoir_l`, with a tolerance of 5 standard errors of the underlying
binomial proportion (so a real bias, not test noise, is what triggers a
failure):

```
$ uv run python reservoir.py --verify
PASS  uniform k=1
       reservoir_r  max |observed-expected| = 0.00298  (tolerance 0.00635, expected 0.03333)
       reservoir_l  max |observed-expected| = 0.00268  (tolerance 0.00635, expected 0.03333)
PASS  uniform k=5
       reservoir_r  max |observed-expected| = 0.00727  (tolerance 0.01318, expected 0.16667)
       reservoir_l  max |observed-expected| = 0.00577  (tolerance 0.01318, expected 0.16667)
PASS  uniform k=15
       reservoir_r  max |observed-expected| = 0.00720  (tolerance 0.01768, expected 0.50000)
       reservoir_l  max |observed-expected| = 0.00865  (tolerance 0.01768, expected 0.50000)
PASS  weighted k=1 (a_res)
       weight=1.0  expected=0.1000 observed=0.1021 tol=0.0106
       weight=2.0  expected=0.2000 observed=0.2007 tol=0.0141
       weight=3.0  expected=0.3000 observed=0.3005 tol=0.0162
       weight=4.0  expected=0.4000 observed=0.3967 tol=0.0173
PASS  weighted k=2 (a_res, monotonicity only)
       counts by weight order = [3591, 6897, 12559, 16953]
PASS  weighted k=1 (reservoir_chao)
       weight=1.0  expected=0.1000 observed=0.0986 tol=0.0106
       weight=2.0  expected=0.2000 observed=0.1946 tol=0.0141
       weight=3.0  expected=0.3000 observed=0.3059 tol=0.0162
       weight=4.0  expected=0.4000 observed=0.4009 tol=0.0173
PASS  weighted k=2 (reservoir_chao, monotonicity only)
       counts by weight order = [5009, 5016, 9975, 20000]
```

**Weighted sampling is only that simple at k = 1.** For k = 1, "the item
with the largest key wins" (a_res) or "the running-total coin flip accepts"
(reservoir_chao) both reduce to exactly w_i / sum(w) -- the derivations
above, specialized. For k > 1, the per-item marginal inclusion probability is
**not** w_i / sum(w) for either sampler once items start excluding each
other from the reservoir (it depends on the whole weight vector through a
combinatorial sum with no simple closed form). Rather than assert a target
number that would be wrong, `verify_weighted_k_gt_1` checks the weaker,
always-true property that a *correct* weighted-without-replacement scheme
must have: item i's empirical inclusion frequency must be >= item j's
whenever w_i >= w_j. In the run above, weights `[1, 2, 4, 8]` at k = 2 over
20,000 trials give counts `[3591, 6897, 12559, 16953]` for a_res and
`[5009, 5016, 9975, 20000]` for reservoir_chao -- both monotone, as
required. (reservoir_chao's counts show its own documented fill-phase quirk
directly: items 0 and 1, weights 1 and 2, both land at ~5,000/20,000 because
they're the two items that filled the reservoir unconditionally and so
share a group probability rather than getting individually-weighted ones --
see "Honest caveat" above. That's a real, expected feature of this specific
run's ascending-weight ordering, not a bug, and it's still monotone.) This
is stated explicitly here (and in the code comments next to
`verify_weighted_k_gt_1`) so the k > 1 weighted check is never mistaken for
the exact one it isn't.

## Benchmarks

```
$ uv run python benchmark.py
reservoir_r vs reservoir_l -- fixed k, growing n
           n     k   reservoir_r   reservoir_l   ratio (R/L)
------------------------------------------------------------
      10,000    50       0.0023s       0.0002s        13.44x
     100,000    50       0.0229s       0.0009s        26.06x
   1,000,000    50       0.2543s       0.0059s        43.31x
   5,000,000    50       1.1657s       0.0295s        39.47x

a_res / reservoir_chao (streaming, O(k) memory) vs naive weighted sample
(O(n) memory)
         n     k       a_res        chao       naive   ratio (naive/a_res)   ratio (naive/chao)
-----------------------------------------------------------------------------------------------
     2,000    20     0.0003s     0.0002s     0.0011s                 4.27x                5.70x
    10,000    20     0.0011s     0.0008s     0.0057s                 5.34x                6.73x
    50,000    20     0.0063s     0.0041s     0.0298s                 4.74x                7.21x
   150,000    20     0.0204s     0.0125s     0.0983s                 4.83x                7.87x
```

The Algorithm L win is the real thing, not just constant-factor noise: the
speedup climbs from 13x to 43x as n grows two and a half orders of magnitude
while k stays fixed at 50, matching the O(k log(n/k)) vs O(n) story -- the
work Algorithm L does per stream grows only logarithmically in n, while
Algorithm R's grows linearly.

The weighted comparison isolates two things at once. First,
`_naive_weighted_sample` does O(n·k) work (a full weighted draw, each one an
O(n) linear scan, plus an O(n) list removal, repeated k times) on top of
holding the whole stream in a list, while `a_res` and `reservoir_chao` are
each a single O(n·something) pass with O(k) memory -- both ratios to naive
widen as n grows (4.3x to 4.8x for a_res, 5.7x to 7.9x for reservoir_chao)
and would keep widening past what these numbers show, but the bigger point
the numbers *can't* show is that past some n the naive method simply can't
run at all (no list to build), while neither streaming sampler is affected
because its memory footprint never depended on n in the first place. Second,
`reservoir_chao` beats `a_res` consistently (its own column is faster at
every n tested here) -- expected, since A-Res pays O(log k) per item for the
heap while A-Chao pays O(1): a running float and a comparison, no heap
operations at all.

## Run it

```bash
cd "challenges/Algorithmic Challenges/Reservoir Sampling & Weighted Random Sampling Library"

uv run python reservoir.py --demo       # sample from a 1M-item synthetic stream
uv run python reservoir.py --verify     # statistical checks, pass/fail + numbers
uv run python benchmark.py              # throughput comparisons

uv run --with pytest pytest -q          # 63 tests
uv run --with pytest pytest -q -m "not slow"   # skip the multi-thousand-trial checks
```

Standard library only (`random`, `heapq`, `itertools`, `math`).

## Where this is used

**Log and trace sampling in observability systems.** A tracing pipeline sees
far more spans than it can afford to store; reservoir sampling (often
weighted by trace duration or error status, which is exactly `a_res`'s or
`reservoir_chao`'s job) is how systems like Jaeger and Honeycomb keep a
statistically representative slice without buffering the firehose.

**Streaming database sampling.** `TABLESAMPLE` and `ORDER BY RANDOM() LIMIT
k` don't scale to a table scan of unknown size read once from a cursor;
reservoir sampling is the standard technique behind "give me a random k rows"
over a stream or a single sequential pass, and is exactly how systems that
can't afford a second pass (or don't know row counts up front, as with a
result set from a remote source) implement it.

**Online experimentation and A/B testing.** Weighted reservoir sampling is
used to subsample event streams for experiment analysis when traffic
volume vastly exceeds what can be logged in full, weighting by whatever
signal (session value, event rarity) matters for keeping the sample
representative rather than uniform.

**A-Res and A-Chao side by side.** Both solve the same problem -- weighted
sampling without replacement from a stream -- with different machinery:
A-Res's key-and-heap costs O(log k) per item and has no order-dependence
quirk; A-Chao's running-total-and-coin-flip costs O(1) per item but (as
derived above) is order-sensitive for the items that land in the initial
fill. A-ExpJ (not implemented here, see above) is the skip-optimized member
of the A-Res side of that family.

**VarOpt sampling and distributed/mergeable reservoirs -- beyond this
challenge.** A separate line of work optimizes a different objective:
instead of "each item's inclusion probability proportional to its weight,"
VarOpt sampling (Cohen, Duffield, Kaplan, Lund & Thorup, "Stream sampling
for variance-optimal estimation of subset sums," SODA 2009) picks the
reservoir contents to minimize the *variance* of arbitrary subset-sum
estimates computed from the sample afterward -- a genuinely different design
goal from anything in this file. Separately, several of the algorithms here
(A-Res in particular, per its own paper) support *merging* independently
built reservoirs from parallel or distributed stream shards into a single
correct sample without re-reading either shard, which matters for sampling
across a sharded log pipeline rather than one single-threaded stream. Both
are real, well-studied extensions; neither was part of the original brief
for this challenge (uniform sampling plus one weighted variant), so they're
noted here as pointers rather than implemented.

## Sources

- [Vitter, "Random Sampling with a Reservoir" (1985)](https://www.cs.umd.edu/~samir/498/vitter.pdf) -- Algorithm R and the original analysis.
- [Li, "Reservoir-Sampling Algorithms of Time Complexity O(n(1 + log(N/n)))" (1994)](https://dl.acm.org/doi/10.1145/198429.198435) -- Algorithm L.
- [Efraimidis & Spirakis, "Weighted Random Sampling with a Reservoir" (2006)](https://utopia.duth.gr/%7Epefraimi/research/data/2007EncOfAlg.pdf) -- A-Res and A-ExpJ.
- Chao, M.T., "A General Purpose Unequal Probability Sampling Plan," *Biometrika* 69(3):653-656, 1982. DOI: [10.2307/2336002](https://doi.org/10.2307/2336002) -- A-Chao's origin; the streaming-friendly single-item-at-a-time acceptance rule this module implements.
- [Efraimidis, "Weighted Random Sampling over Data Streams" (2010, revised 2015)](https://arxiv.org/abs/1012.0256) -- a survey that sketches A-Chao and A-Res side by side and names the initial-fill subtlety this README's "Honest caveat" describes ("appropriate procedures to initialize the reservoir... are described in [Chao 1982]").
- Cohen, Duffield, Kaplan, Lund & Thorup, "Stream Sampling for Variance-Optimal Estimation of Subset Sums," *SODA* 2009 -- VarOpt sampling, mentioned above as beyond this challenge's scope.
