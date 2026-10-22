"""Logistic-regression attribution: what does being exposed to a channel add to the chance of converting?

Every journey, converted or not, is described by which channels it touched. A logistic regression
of "converted" on those presence indicators (fitted by Newton's method on the distinct channel
sets, weighted by how many journeys have each) gives each channel a coefficient. A channel's
credit is the conversions that would vanish if it were taken off the journeys that had it: for every
set of channels, journeys x (chance with the set - chance without the channel), summed over the
sets that include it, scaled to the conversions observed.

It sees only *presence*, not order or timing, and it cannot tell a channel that causes conversion
from one that is shown to people who were going to convert anyway: the model has no idea who was
targeted. Where that matters (retargeting) its credit is inflated just like the rule-based models'.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attribution.journeys import AttributionError, Journeys, mask_of_paths

MAX_CHANNELS = 20


def _design(masks: np.ndarray, k: int) -> np.ndarray:
    bits = (masks[:, None] >> np.arange(k)) & 1
    return np.hstack([np.ones((masks.size, 1)), bits.astype(float)])


def fit(
    x: np.ndarray, conv: np.ndarray, total: np.ndarray, ridge: float = 1e-3
) -> np.ndarray:
    """Weighted logistic regression on grouped data: ``conv`` of ``total`` journeys converted per row."""
    beta = np.zeros(x.shape[1])
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(x @ beta)))
        grad = x.T @ (conv - total * p) - penalty @ beta
        info = (x * (total * p * (1 - p))[:, None]).T @ x + penalty
        step = np.linalg.solve(info, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


@dataclass
class PresenceModel:
    masks: np.ndarray  # distinct channel sets (bitmasks) seen on journeys
    conv: np.ndarray  # converted journeys per set
    total: np.ndarray  # all journeys per set
    beta: np.ndarray  # intercept, then one coefficient per channel

    @property
    def bits(self) -> np.ndarray:
        return _design(self.masks, self.beta.size - 1)[:, 1:]

    def probability(self, bits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.beta[0] + bits @ self.beta[1:])))


def fit_presence(j: Journeys, weights: np.ndarray | None = None) -> PresenceModel:
    """Fit the presence model to every journey (converted journeys weighted by ``weights``)."""
    k = j.k
    if k > MAX_CHANNELS:
        raise AttributionError(
            f"the logistic model needs at most {MAX_CHANNELS} channels, got {k}"
        )
    w = np.ones(j.n_converted) if weights is None else weights
    conv_by: dict[int, float] = {}
    for m, wi in zip(mask_of_paths(j).tolist(), w.tolist(), strict=True):
        conv_by[m] = conv_by.get(m, 0.0) + wi
    total_by = dict(conv_by)
    for m, c in j.null_sets.items():
        total_by[m] = total_by.get(m, 0.0) + c
    masks = np.array(sorted(total_by), dtype=np.int64)
    conv = np.array([conv_by.get(int(m), 0.0) for m in masks])
    total = np.array([total_by[int(m)] for m in masks])
    return PresenceModel(masks, conv, total, fit(_design(masks, k), conv, total))


def credit(j: Journeys, weights: np.ndarray | None = None) -> np.ndarray:
    """Conversions credited to each channel by the logistic model's counterfactual."""
    model = fit_presence(j, weights)
    bits = model.bits
    p = model.probability(bits)
    raw = np.zeros(j.k)
    for c in range(j.k):
        has = bits[:, c] == 1
        without = bits.copy()
        without[:, c] = 0.0
        raw[c] = float(
            np.sum(model.total[has] * (p[has] - model.probability(without)[has]))
        )
    w = np.ones(j.n_converted) if weights is None else weights
    return normalise(raw, float(w @ j.value))


def normalise(raw: np.ndarray, total: float) -> np.ndarray:
    """Scale non-negative credit to the conversions observed; a channel that lowers conversion gets 0."""
    raw = np.clip(raw, 0.0, None)
    s = raw.sum()
    return raw / s * total if s > 0 else np.zeros(raw.size)
