"""The similarity definitions. Each answers: how alike are sale ``i`` and sale ``j``?

Both classes hold a precomputed representation of *every* sale (scaled features, or leaf
memberships) and score by row index, so the caller decides which rows may be compared.

``sim`` returns "bigger = more alike" (negative squared distance for the Euclidean ones, weighted
fraction of shared leaves for the tree one). ``contributions`` splits one pair's similarity over the
features, in the same units and with the same total, so an explanation adds up exactly:

* Euclidean: ``(w_f * dz_f)^2`` per feature, summing to the squared distance.
* Leaf proximity: each shared leaf hands its tree's weight to the features on its decision path,
  summing to the proximity.
"""

from __future__ import annotations

import numpy as np

from comp_finder.trees import HedonicGBM


class WeightedEuclid:
    """Squared distance on standardised features, each scaled by a per-feature weight."""

    higher_is_closer = False

    def __init__(self, z: np.ndarray, weights: np.ndarray | None = None):
        self.weights = (
            np.ones(z.shape[1]) if weights is None else np.asarray(weights, float)
        )
        self.z = z
        self.zw = z * self.weights
        self.sq = (self.zw**2).sum(axis=1)

    def sim(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        d2 = self.sq[q][:, None] + self.sq[p][None, :] - 2.0 * self.zw[q] @ self.zw[p].T
        return -np.maximum(d2, 0.0)

    def contributions(self, q: int, c: int) -> np.ndarray:
        return (self.weights * (self.z[q] - self.z[c])) ** 2

    def total(self, q: int, c: int) -> float:
        """The squared distance, which ``contributions`` sums to."""
        return float(self.contributions(q, c).sum())

    def typical(self, p: np.ndarray, q: int) -> np.ndarray:
        """What each feature would contribute against an average candidate in ``p``."""
        return (self.weights**2) * ((self.z[p] - self.z[q]) ** 2).mean(axis=0)


class LeafProximity:
    """Weighted fraction of boosted trees in which two sales fall in the same leaf."""

    higher_is_closer = True

    def __init__(self, gbm: HedonicGBM, z: np.ndarray):
        self.gbm = gbm
        self.leaf = gbm.leaves(z)
        self.col_w = gbm.column_weight()
        self.leaf_w = self.leaf.multiply(self.col_w[None, :]).tocsr()

    def sim(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (self.leaf_w[q] @ self.leaf[p].T).toarray()

    def contributions(self, q: int, c: int) -> np.ndarray:
        shared = self.leaf[q].multiply(self.leaf[c]).multiply(self.col_w[None, :])
        return np.asarray((shared @ self.gbm.path_share).todense()).ravel()

    def total(self, q: int, c: int) -> float:
        """The proximity, which ``contributions`` sums to (up to trees that never split)."""
        return float(self.sim(np.array([q]), np.array([c]))[0, 0])

    def typical(self, p: np.ndarray, q: int) -> np.ndarray:
        """Mean attribution against candidates in ``p`` (a random pair, for contrast)."""
        rng = np.random.default_rng(0)
        sample = rng.choice(p, size=min(200, len(p)), replace=False)
        return np.mean([self.contributions(q, int(c)) for c in sample], axis=0)


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and scores of the ``k`` highest per row, best first; ``-inf`` entries sort last."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    s = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-s, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(s, order, axis=1)
