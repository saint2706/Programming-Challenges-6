"""Shapley-value attribution: each channel's average marginal contribution to a coalition's results.

The coalition game: the *value* of a set S of channels is the number of conversions the logistic
presence model expects if only the channels in S were ever present (every other channel's touches
are deleted from every journey, converted or not). The Shapley value of a channel averages its
marginal gain ``v(S + c) - v(S)`` over every order in which channels could be added, so a channel
that always appears next to a strong one is neither rewarded for being there nor ignored. With the
presence model the values add up exactly (efficiency): ``sum(phi) = v(all) - v(none)``, the
conversions the model attributes to marketing rather than to the baseline.

It is exact over all 2^K sets, so ``K <= 12``; the values are only as good as the logistic model,
which sees which channels a customer met but not their order, timing, or why they were targeted.
A channel with a negative coefficient gets no credit (the shares are of the positive part).
"""

from __future__ import annotations

from math import factorial

import numpy as np

from attribution.journeys import AttributionError, Journeys
from attribution.logistic import PresenceModel, fit_presence, normalise

MAX_CHANNELS = 12


def popcount(n: int) -> np.ndarray:
    """Number of set bits for every integer 0 .. 2^n - 1."""
    out = np.zeros(1 << n, dtype=np.int64)
    for b in range(n):
        out[1 << b : 1 << (b + 1)] = out[: 1 << b] + 1
    return out


def coalition_values(model: PresenceModel) -> np.ndarray:
    """v[S] for every set S as a bitmask: expected conversions with only the channels in S present."""
    k = model.beta.size - 1
    bits = model.bits  # (M, K)
    subsets = ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1).astype(
        float
    )  # (2^K, K)
    linear = model.beta[0] + bits @ (subsets * model.beta[1:]).T  # (M, 2^K)
    return model.total @ (1.0 / (1.0 + np.exp(-linear)))


def shapley_values(v: np.ndarray, n: int) -> np.ndarray:
    """Exact Shapley values of a game given as the array of coalition values."""
    size = popcount(n)
    weight = np.array(
        [factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)]
    )
    idx = np.arange(1 << n)
    phi = np.zeros(n)
    for c in range(n):
        without = idx[(idx >> c) & 1 == 0]
        phi[c] = float(
            np.sum(weight[size[without]] * (v[without | (1 << c)] - v[without]))
        )
    return phi


def credit(
    j: Journeys, weights: np.ndarray | None = None, prior: float = 0.0
) -> np.ndarray:
    """Conversions credited to each channel by its Shapley value in the logistic game."""
    del prior  # kept so the call signature matches the older, rate-based game
    if j.k > MAX_CHANNELS:
        raise AttributionError(
            f"Shapley needs at most {MAX_CHANNELS} channels, got {j.k}; group the rest"
        )
    w = np.ones(j.n_converted) if weights is None else weights
    phi = shapley_values(coalition_values(fit_presence(j, weights)), j.k)
    return normalise(phi, float(w @ j.value))
