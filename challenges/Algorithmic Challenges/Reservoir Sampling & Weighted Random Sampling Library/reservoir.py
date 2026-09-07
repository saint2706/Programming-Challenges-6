"""Reservoir sampling: pick k items from a stream of unknown length n.

The whole family solves one problem -- select a random sample of size k from a
stream you can only read once, without knowing n in advance and without
buffering more than O(k) items. That rules out "collect everything, then
`random.sample`": for a stream too large to hold in memory (log lines, a
database cursor, a socket) it is the only option at all.

    reservoir_r     O(n)                  uniform, Vitter's Algorithm R (1985)
    reservoir_l     O(k*(1 + log(n/k)))   uniform, Li's Algorithm L (1994)
    a_res           O(n log k)            weighted, Efraimidis-Spirakis A-Res (2006)
    reservoir_chao  O(n)                  weighted, Chao's A-Chao (1982)

**reservoir_r** is the textbook version: keep the first k items, then for the
i-th item (i > k) replace a uniformly random reservoir slot with probability
k/i. One coin flip per element, and the proof that every item ends up with
marginal probability exactly k/n is a short induction (see the README).

**reservoir_l** produces the *same distribution* as Algorithm R -- same k/n
marginal for every item -- but stops flipping a coin for every element. Once
you know an item was *not* selected, the number of further elements that will
also not be selected follows a computable distribution, so the algorithm
samples that skip-distance directly and jumps straight to the next real
candidate. The expected number of replacements is O(k log(n/k)), so this is
the version to reach for when k is small and n is huge: it touches the same
n elements (a stream has to be read once regardless) but only *does work* on
O(k log(n/k)) of them instead of doing a coin flip on every single one.

**a_res** is the weighted generalization: each item gets a key u^(1/w) for
u ~ Uniform(0,1) and weight w, and the k items with the largest keys survive.
Larger weight pushes the key distribution up (closer to 1), which is exactly
what makes "keep the top-k keys" equivalent to weighted sampling without
replacement -- the derivation is worth reading in the README rather than
taking on faith, since it is the non-obvious part of the whole module.

**reservoir_chao** is a second, older weighted sampler (Chao 1982, predating
A-Res by 24 years) that reaches the same goal by a completely different route:
no keys, no heap, just a running weight total. Item i is accepted with
probability k*w_i/W_i (W_i = weight seen so far, including i) and, if
accepted, evicts a uniformly random current occupant. Setting every weight to
1 turns k*w_i/W_i into k/i -- Algorithm R's own acceptance rule -- so
reservoir_r is literally reservoir_chao's unweighted special case. See the
function docstring for the telescoping proof and an honestly-documented
caveat about the first k items.

Every sampler takes a true iterator: none of them call ``len()`` on the input
or read it twice, and none of them materialize the whole stream. They accept
a ``random.Random`` (or an int seed) for reproducibility; the default is a
fresh, truly random generator.

    uv run python reservoir.py --demo
    uv run python reservoir.py --verify
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import math
import random
from collections.abc import Iterable, Iterator
from typing import TypeVar

__all__ = [
    "a_res",
    "reservoir_chao",
    "reservoir_l",
    "reservoir_r",
    "verify",
    "verify_uniform",
    "verify_weighted_k1",
]

T = TypeVar("T")

_MISSING = object()


def _rng(seed: random.Random | int | None) -> random.Random:
    """Normalize the ``rng``/``seed`` argument every sampler accepts."""
    if isinstance(seed, random.Random):
        return seed
    return random.Random(seed)


def _check_k(k: int) -> None:
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")


# ---------------------------------------------------------------------------
# 1. Algorithm R -- O(n), one coin flip per element past the first k
# ---------------------------------------------------------------------------


def reservoir_r(
    stream: Iterable[T], k: int, *, rng: random.Random | int | None = None
) -> list[T]:
    """Uniform sample of size ``min(k, n)`` from a stream of unknown length.

    Vitter's Algorithm R: fill the reservoir with the first k items, then for
    each later item at 1-indexed position ``i`` (``i > k``), replace a
    uniformly random reservoir slot with it with probability ``k/i`` -- drawn
    by picking ``j`` uniformly from ``1..i`` and replacing slot ``j`` only
    when ``j <= k``.

    Why every item ends up with marginal probability exactly k/n: induct
    backwards from the last item. The last item lands in the reservoir with
    probability k/n by construction. An earlier item, at position i, is in
    the *final* reservoir iff it survives every later item's replacement
    draw -- and by induction each later draw at position m evicts any given
    occupant with probability (k/m)*(1/k) = 1/m, so survival probability is
    the product of (1 - 1/m) for m from i+1 to n, which telescopes to i/n.
    Combined with "item i enters the reservoir with probability k/i" (true
    for i <= k by construction, and by the draw's own definition for i > k),
    the total is (k/i)*(i/n) = k/n for every i. No item is special.

    Only ``next()`` is ever called on ``stream`` -- no ``len()``, no second
    pass -- so a generator or any other single-use iterator works.
    """
    _check_k(k)
    r = _rng(rng)
    it = iter(stream)
    reservoir = list(itertools.islice(it, k))
    if k == 0 or len(reservoir) < k:
        return reservoir  # the whole stream fit; nothing left to replace with
    for i, item in enumerate(it, start=k + 1):
        j = r.randint(1, i)  # uniform in [1, i]
        if j <= k:
            reservoir[j - 1] = item
    return reservoir


# ---------------------------------------------------------------------------
# 2. Algorithm L -- skip the coin flips, jump straight to the next swap
# ---------------------------------------------------------------------------


def reservoir_l(
    stream: Iterable[T], k: int, *, rng: random.Random | int | None = None
) -> list[T]:
    """Same distribution as :func:`reservoir_r`, without a per-element coin flip.

    After the reservoir fills, Algorithm R accepts item i (i > k) with
    probability k/i. Algorithm L instead samples *how many items to skip*
    before the next acceptance, in closed form, and jumps there directly.

    Derivation of the skip distance. Maintain ``w``, the probability that the
    *current* candidate item is accepted (initialized to k/(k+1), the
    acceptance probability of the first item past the fill). Given "item j is
    not accepted" has probability ``1 - w``, the probability that the next G
    items are all rejected is ``(1-w)^G`` -- so G (the skip count, i.e. the
    number of items to reject before the next acceptance) has the discrete
    tail ``P(G >= g) = (1-w)^g``, which is a geometric distribution. Sampling
    it by inversion: draw ``u ~ Uniform(0,1)``, set ``(1-w)^g = u`` and solve,

        g = floor( log(u) / log(1-w) )

    which needs only one uniform draw and no per-item flip. The reason ``w``
    itself can be advanced without re-deriving it from i and k: an item
    accepted at this step is, like every prior accepted item, keeping the
    property that a uniformly random subset of size k is what's kept -- so
    the same maximum-of-k-uniforms identity that generates ``w`` in the first
    place also updates it, via ``w <- w * exp(log(u')/k)`` for a fresh
    ``u'``. (This is the same key transform :func:`a_res` uses for the
    weighted case, specialized to every item having weight 1 -- see the
    README for the full argument.)

    Expected cost: O(k * (1 + log(n/k))) vs. Algorithm R's O(n), because each
    accepted item's skip distance grows geometrically as the stream
    progresses, so only O(k log(n/k)) skips (each O(1)) are needed to reach
    the end, on top of the O(k) fill.
    """
    _check_k(k)
    r = _rng(rng)
    it = iter(stream)
    reservoir = list(itertools.islice(it, k))
    if k == 0 or len(reservoir) < k:
        return reservoir

    w = math.exp(math.log(r.random()) / k)
    while True:
        skip = math.floor(math.log(r.random()) / math.log(1.0 - w))
        # Discard `skip` items, then take the next one -- one pass, no peeking.
        item = next(itertools.islice(it, skip, skip + 1), _MISSING)
        if item is _MISSING:
            break
        reservoir[r.randrange(k)] = item
        w *= math.exp(math.log(r.random()) / k)
    return reservoir


# ---------------------------------------------------------------------------
# 3. A-Res -- weighted sampling without replacement, via a key transform
# ---------------------------------------------------------------------------


def a_res(
    stream: Iterable[tuple[T, float]],
    k: int,
    *,
    rng: random.Random | int | None = None,
) -> list[T]:
    """Weighted sample of size ``min(k, n)``, no replacement, streamed.

    Efraimidis-Spirakis A-Res: each ``(item, weight)`` gets a key
    ``u**(1/weight)`` for ``u ~ Uniform(0, 1)``, and the k items with the
    largest keys are kept, tracked with a size-k min-heap keyed on the key so
    each item costs O(log k) instead of O(k).

    Why the key transform makes "keep the top-k keys" a weighted sample. Fix
    one item with weight w and look at its key's CDF:

        P(u^(1/w) <= x) = P(u <= x^w) = x^w   for x in [0, 1]

    which is exactly the CDF of ``max(u_1, ..., u_w)`` for w iid uniforms
    when w is a positive integer -- i.e. this key is distributed exactly like
    the *largest* of w independent uniform draws for that item. Sampling one
    key per item and keeping the largest overall is then equivalent to
    pooling w_i "tickets" per item i and asking which ticket-holder drew the
    single largest uniform value across the whole pool: since every ticket is
    an iid uniform draw, that holder is uniform over all tickets, so
    ``P(item i has the max key) = w_i / sum(w)`` -- weighted sampling,
    exactly. (The transform ``u**(1/w)`` generalizes this "max of w draws"
    picture continuously to non-integer weight, matching the same CDF.)

    That handles k=1. For general k, repeat the argument on order statistics:
    conditioning on the top key belonging to item i (probability
    w_i / sum(w)), the *remaining* keys are still iid draws from their
    respective per-item distributions restricted below the winning key, which
    is the same problem one item smaller and one weight lighter -- so keeping
    the top k keys is precisely sequential weighted sampling without
    replacement, one draw at a time, with each draw's probability
    proportional to the weight of what remains. That sequential process is
    the textbook definition of weighted sampling without replacement, and
    A-Res computes its result in one pass with no reservoir shrinking or
    weight bookkeeping needed at all -- the u**(1/w) keys front-load all of
    it into one number per item.

    Streamed one ``(item, weight)`` pair at a time; weight must be positive.
    """
    _check_k(k)
    r = _rng(rng)
    if k == 0:
        return []
    heap: list[tuple[float, int, T]] = []  # (key, tiebreak, item); min-heap on key
    tiebreak = itertools.count()
    for item, weight in stream:
        if weight <= 0:
            raise ValueError(f"a_res requires positive weights, got {weight!r}")
        key = r.random() ** (1.0 / weight)
        if len(heap) < k:
            heapq.heappush(heap, (key, next(tiebreak), item))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, next(tiebreak), item))
    return [item for _, _, item in heap]


# ---------------------------------------------------------------------------
# 4. A-Chao -- weighted sampling without replacement, via a running total
# ---------------------------------------------------------------------------


def reservoir_chao(
    stream: Iterable[tuple[T, float]],
    k: int,
    *,
    rng: random.Random | int | None = None,
) -> list[T]:
    """Weighted sample of size ``min(k, n)``, no replacement, streamed.

    Chao (1982): fill the reservoir with the first k items, then for each
    later item i with weight w_i, maintain the running total W (sum of every
    weight seen so far, including i) and accept item i with probability
    ``k * w_i / W`` -- if accepted, evict a uniformly random current occupant
    and put item i in its place. No keys, no heap, no sort: just one running
    float and a coin flip per item, O(1) work each, O(n) total.

    Why k*w_i/W is the right acceptance probability -- and not, say, plain
    w_i/W. Set every weight to 1: the formula becomes k*1/i = k/i, which is
    exactly :func:`reservoir_r`'s own replacement probability for item i.
    That is not a coincidence to wave away -- it is the derivation. A-Chao
    *is* Algorithm R's k/i rule, generalized so that an item's pull on the
    accept/reject coin scales with its weight instead of counting for
    exactly one unit like everyone else's.

    The telescoping proof (for any item i past the initial fill) mirrors
    :func:`reservoir_r`'s exactly. Item i is accepted with probability
    k*w_i/W_i (W_i = the running total right after item i). Once in, it
    survives item m's replacement draw (m > i) unless m is both accepted
    (probability k*w_m/W_m) and happens to land on i's slot (probability
    1/k) -- so it survives that draw with probability

        1 - (k*w_m/W_m)*(1/k) = 1 - w_m/W_m = (W_m - w_m)/W_m = W_{m-1}/W_m,

    and surviving every draw from i+1 to n is the telescoping product

        prod_{m=i+1}^{n} (W_{m-1}/W_m) = W_i/W_n.

    Multiplying by the entry probability: (k*w_i/W_i)*(W_i/W_n) = k*w_i/W_n.
    The running total W_i cancels, leaving a probability proportional to w_i
    alone -- weighted sampling, exactly the same shape as Algorithm R's k/n
    falling out of (k/i)*(i/n).

    Honest caveat: that proof runs from "item i's entry" forward, so it only
    covers items that arrive *after* the reservoir is already full. The
    first k items enter unconditionally (there is no other choice -- the
    reservoir has exactly k slots and exactly k candidates so far), so their
    survival to the end is W_k/W_n for *all of them alike*, not the
    individually-weighted k*w_i/W_n the later items get. Concretely: with
    weights [1, 2, 3, 4] and k=2, items 1 and 2 (the fill) both converge to
    P = (1+2)/10 = 0.30 regardless of being unequal weights, while items 3
    and 4 (processed after the fill) land almost exactly on their k*w_i/W_n
    targets of 0.60 and 0.80. This is a real property of the simple
    streaming algorithm as commonly presented (Wikipedia's writeup and
    Efraimidis's 2010 survey sketch it the same way; see README Sources) --
    Chao's original paper describes a more elaborate initialization to fix
    it, which is out of scope here. It does not affect this module's k=1
    exactness claim (with k=1 there is exactly one "fill" item and nothing
    to be unequal with), and it does not break monotonicity for weights that
    aren't pathologically front-loaded, which is what :func:`verify` checks.

    Streamed one ``(item, weight)`` pair at a time; weight must be positive.
    """
    _check_k(k)
    r = _rng(rng)
    if k == 0:
        return []
    reservoir: list[T] = []
    total_weight = 0.0
    for item, weight in stream:
        if weight <= 0:
            raise ValueError(
                f"reservoir_chao requires positive weights, got {weight!r}"
            )
        total_weight += weight
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            p = k * weight / total_weight
            if r.random() < p:
                reservoir[r.randrange(k)] = item
    return reservoir


# ---------------------------------------------------------------------------
# Statistical verification
# ---------------------------------------------------------------------------
#
# These are randomized samplers, so "correct" means empirical selection
# frequencies converge to theory, not that any single run matches an oracle
# output. Tolerance is set from the sampling distribution's own standard
# error (a binomial proportion, since "item i is in the reservoir" is a
# Bernoulli event over trials) rather than a fixed arbitrary percentage, so
# it tightens automatically as `trials` grows and stays honest at any size.


def _proportion_tolerance(p: float, trials: int, sigmas: float = 5.0) -> float:
    """Binomial standard-error band at ``sigmas`` standard deviations.

    5 sigma keeps the false-positive rate on a single check below 1 in a
    million, so a failure here means the sampler is actually biased, not
    that the test got unlucky.
    """
    return sigmas * math.sqrt(max(p * (1 - p), 1e-12) / trials)


def verify_uniform(
    n: int = 30,
    k: int = 5,
    trials: int = 20_000,
    *,
    seed: int = 0,
) -> dict[str, object]:
    """Check reservoir_r and reservoir_l against the exact k/n marginal.

    Uniform reservoir sampling's guarantee is exact, not approximate, for
    *any* k <= n (see :func:`reservoir_r`'s docstring for the induction): so
    this is a real correctness check, not a fuzzy sanity pass. Runs both
    samplers over the same trials and reports per-item observed proportions
    against expected = k/n.
    """
    expected = k / n
    tol = _proportion_tolerance(expected, trials)
    counts = {"reservoir_r": [0] * n, "reservoir_l": [0] * n}
    r = random.Random(seed)
    for _ in range(trials):
        for name, fn in (("reservoir_r", reservoir_r), ("reservoir_l", reservoir_l)):
            sample = fn(range(n), k, rng=r)
            for x in sample:
                counts[name][x] += 1

    results: dict[str, object] = {
        "n": n,
        "k": k,
        "trials": trials,
        "expected": expected,
    }
    ok = True
    for name, c in counts.items():
        proportions = [x / trials for x in c]
        max_dev = max(abs(p - expected) for p in proportions)
        passed = max_dev <= tol
        ok = ok and passed
        results[name] = {
            "max_deviation": max_dev,
            "tolerance": tol,
            "min_proportion": min(proportions),
            "max_proportion": max(proportions),
            "passed": passed,
        }
    results["passed"] = ok
    return results


def verify_weighted_k1(
    weights: list[float] | None = None,
    trials: int = 20_000,
    *,
    seed: int = 1,
    sampler=a_res,
) -> dict[str, object]:
    """Check ``sampler(k=1)`` against the exact w_i / sum(w) marginal.

    Restricted to k=1 on purpose: for k=1, "the item with the largest key
    wins" (a_res) or "the running-total coin flip accepts" (reservoir_chao)
    both reduce to exactly weighted-choice-by-w_i (the README derives this
    for a_res as a special case of the general order-statistics argument;
    reservoir_chao's own docstring derives it directly), so it has a simple
    closed-form probability to test against. For k>1 the per-item marginal
    is *not* w_i / sum(w) once items start getting excluded by each other --
    see :func:`verify_weighted_k_gt_1` for what's checked instead, and why.
    Defaults to :func:`a_res`; pass ``sampler=reservoir_chao`` to check the
    other weighted sampler with the identical methodology.
    """
    if weights is None:
        weights = [1.0, 2.0, 3.0, 4.0]
    total = sum(weights)
    expected = [w / total for w in weights]
    counts = [0] * len(weights)
    r = random.Random(seed)
    for _ in range(trials):
        items = list(enumerate(weights))
        (winner,) = sampler(items, 1, rng=r)
        counts[winner] += 1

    ok = True
    per_item = []
    for i, (c, p) in enumerate(zip(counts, expected)):
        observed = c / trials
        tol = _proportion_tolerance(p, trials)
        passed = abs(observed - p) <= tol
        ok = ok and passed
        per_item.append(
            {
                "weight": weights[i],
                "expected": p,
                "observed": observed,
                "tolerance": tol,
                "passed": passed,
            }
        )
    return {"trials": trials, "weights": weights, "per_item": per_item, "passed": ok}


def verify_weighted_k_gt_1(
    weights: list[float] | None = None,
    k: int = 2,
    trials: int = 20_000,
    *,
    seed: int = 2,
    sampler=a_res,
) -> dict[str, object]:
    """Weaker check for k>1: higher weight must mean higher inclusion frequency.

    Weighted-without-replacement inclusion probability for k>1 has no simple
    closed form in general (it depends on the whole weight vector through a
    combinatorial sum over subsets), so this does not assert a target
    number. What it does assert -- monotonicity: item i's empirical
    inclusion frequency must be >= item j's whenever w_i >= w_j -- is
    implied by *any* correct weighted-without-replacement scheme, so a
    violation here is still a real bug signal even without an exact target.
    Defaults to :func:`a_res`; pass ``sampler=reservoir_chao`` to run the same
    check against the other weighted sampler. (reservoir_chao's own docstring
    notes a real wrinkle: items that arrive during the initial k-item fill
    share a group probability rather than an individually-weighted one, so
    the default ``weights`` here are chosen small-to-large in stream order,
    which keeps that wrinkle from producing a monotonicity violation --
    see the docstring for a worked case where a badly-ordered stream can.)
    """
    if weights is None:
        weights = [1.0, 2.0, 4.0, 8.0]
    counts = [0] * len(weights)
    r = random.Random(seed)
    for _ in range(trials):
        items = list(enumerate(weights))
        for winner in sampler(items, k, rng=r):
            counts[winner] += 1

    # Slack in absolute trial count, not a proportion: a noisy tie between
    # two adjacent counts is expected; a large reversal is not.
    slack = _proportion_tolerance(k / len(weights), trials) * trials
    order = sorted(range(len(weights)), key=lambda i: weights[i])
    monotone = all(
        counts[order[i]] <= counts[order[i + 1]] + slack for i in range(len(order) - 1)
    )
    return {
        "trials": trials,
        "weights": weights,
        "k": k,
        "counts": counts,
        "passed": monotone,
    }


def verify(*, verbose: bool = True) -> bool:
    """Run all statistical checks; return whether every one passed."""
    checks = [
        ("uniform k=1", verify_uniform(n=30, k=1)),
        ("uniform k=5", verify_uniform(n=30, k=5)),
        ("uniform k=15", verify_uniform(n=30, k=15)),
        ("weighted k=1 (a_res)", verify_weighted_k1()),
        ("weighted k=2 (a_res, monotonicity only)", verify_weighted_k_gt_1(k=2)),
        (
            "weighted k=1 (reservoir_chao)",
            verify_weighted_k1(sampler=reservoir_chao, seed=3),
        ),
        (
            "weighted k=2 (reservoir_chao, monotonicity only)",
            verify_weighted_k_gt_1(k=2, sampler=reservoir_chao, seed=4),
        ),
    ]
    ok = True
    for name, result in checks:
        passed = result["passed"]
        ok = ok and passed
        if verbose:
            print(f"{'PASS' if passed else 'FAIL'}  {name}")
            if name.startswith("uniform"):
                for method in ("reservoir_r", "reservoir_l"):
                    m = result[method]
                    print(
                        f"       {method:<12} max |observed-expected| = "
                        f"{m['max_deviation']:.5f}  (tolerance {m['tolerance']:.5f}, "
                        f"expected {result['expected']:.5f})"
                    )
            elif name.startswith("weighted k=1"):
                for item in result["per_item"]:
                    print(
                        f"       weight={item['weight']:<4} expected={item['expected']:.4f} "
                        f"observed={item['observed']:.4f} tol={item['tolerance']:.4f}"
                    )
            elif "monotonicity" in name:
                print(f"       counts by weight order = {result['counts']}")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _synthetic_stream(n: int) -> Iterator[int]:
    """A generator, not a list -- proof that samplers never need `len()`."""
    return (i for i in range(n))


def _demo() -> None:
    n, k = 1_000_000, 8
    print(f"streaming {n:,} integers, k={k}\n")

    r = random.Random(42)
    print("reservoir_r:", reservoir_r(_synthetic_stream(n), k, rng=r))

    r = random.Random(42)
    print("reservoir_l:", reservoir_l(_synthetic_stream(n), k, rng=r))

    r = random.Random(42)
    weighted = ((f"item-{i}", (i % 97) + 1) for i in range(n))
    print("a_res (weighted, heavier items favored):", a_res(weighted, k, rng=r))

    r = random.Random(42)
    weighted = ((f"item-{i}", (i % 97) + 1) for i in range(n))
    print(
        "reservoir_chao (weighted, heavier items favored):",
        reservoir_chao(weighted, k, rng=r),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="reservoir",
        description=(
            "Reservoir sampling: uniform (Algorithm R/L) and weighted (A-Res, A-Chao)."
        ),
    )
    ap.add_argument(
        "--demo", action="store_true", help="sample from a synthetic stream"
    )
    ap.add_argument(
        "--verify", action="store_true", help="run statistical checks, print pass/fail"
    )
    args = ap.parse_args(argv)

    if args.verify:
        ok = verify()
        return 0 if ok else 1

    _demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
