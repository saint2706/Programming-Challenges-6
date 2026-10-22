"""Markov-chain attribution: how much does conversion fall if a channel is taken away?

Journeys are modelled as a first-order Markov chain over START, the channels, and the two absorbing
states CONVERSION and NULL (lost). The chance of reaching CONVERSION from START is computed by
solving the absorbing chain. A channel's *removal effect* is the relative drop in that chance when
every visit to the channel is redirected to NULL, as if the channel had never existed. Credit is the
removal effects scaled to add up to the conversions actually observed.

This answers a counterfactual in the model, not in the world: if people who saw the channel would
have converted anyway, the chain still gives the channel credit for the paths it sits on.
"""

from __future__ import annotations

import numpy as np

from attribution.journeys import Journeys


def transition_counts(j: Journeys, weights: np.ndarray | None = None) -> np.ndarray:
    """Counts out of (START, channels) into (channels, CONVERSION, NULL): shape (K + 1, K + 2)."""
    k = j.k
    w = np.ones(j.n_converted) if weights is None else weights
    counts = np.zeros((k + 1, k + 2))
    starts, ends = j.offsets[:-1], j.offsets[1:] - 1
    np.add.at(counts[k], j.channel[starts], w)  # START -> first channel
    inside = np.flatnonzero(
        np.diff(j.path_of_touch) == 0
    )  # touch i and i+1 are in one journey
    path = j.path_of_touch[inside]
    np.add.at(counts, (j.channel[inside], j.channel[inside + 1]), w[path])
    np.add.at(counts[:, k], j.channel[ends], w * j.value)  # last channel -> CONVERSION
    counts[:k, :k] += j.null_trans
    counts[k, :k] += j.null_start
    np.add.at(counts[:, k + 1], np.arange(k), j.null_end)  # last channel -> NULL
    return counts


def conversion_probability(counts: np.ndarray, removed: int | None = None) -> float:
    """P(START reaches CONVERSION), optionally with one channel's visits redirected to NULL."""
    k = counts.shape[0] - 1
    c = counts.copy()
    if removed is not None:
        c[removed, :] = 0.0
        c[removed, k + 1] = 1.0
    row = c.sum(axis=1)
    p = np.divide(c, row[:, None], out=np.zeros_like(c), where=row[:, None] > 0)
    p[row == 0, k + 1] = 1.0  # a state nobody was seen leaving is a dead end
    # channels first (START is never a target), then START from their values:
    #   h_channel = P_cc h_channel + P_c,conv        h_start = P_s,c . h_channel + P_s,conv
    h = np.linalg.solve(np.eye(k) - p[:k, :k], p[:k, k])
    return float(p[k, :k] @ h + p[k, k])


def credit(j: Journeys, weights: np.ndarray | None = None) -> np.ndarray:
    """Conversions credited to each channel by its removal effect."""
    counts = transition_counts(j, weights)
    base = conversion_probability(counts)
    total = float((np.ones(j.n_converted) if weights is None else weights) @ j.value)
    if base <= 0:
        return np.zeros(j.k)
    effect = np.array(
        [1.0 - conversion_probability(counts, c) / base for c in range(j.k)]
    )
    effect = np.clip(effect, 0.0, None)
    s = effect.sum()
    return effect / s * total if s > 0 else np.zeros(j.k)
