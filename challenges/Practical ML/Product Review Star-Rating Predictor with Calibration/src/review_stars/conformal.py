"""Split conformal prediction: sets of stars and intervals with a finite-sample coverage guarantee.

* ``ApsConformal``: adaptive prediction sets (Romano, Sesia and Candes 2020) from any framing's
  5-star probability vector. Stars are taken in order of decreasing probability. Randomized (the
  default) uses one uniform draw per review to break the cumulative-mass tie, which is what makes
  the coverage tight, about ``1 - alpha``, instead of conservative. The deterministic variant
  includes the star at which the mass first crosses the threshold, is also valid, and on five
  ordered stars is far too cautious (a 90% target covers about 99%, with sets over a star larger).
  Randomization is reproducible: the caller supplies the draws.
* ``IntervalConformal``: intervals ``mu +- q * sigma`` for the regression framing, where ``q`` is
  the conformal quantile of the normalized residuals ``|y - mu| / sigma``.

The guarantee needs the calibration and test reviews to be exchangeable. Under a time or domain
shift it does not hold, which is a result worth reporting, not a bug.

With fewer than ``ceil((n + 1) * (1 - alpha))`` calibration reviews the quantile is infinite (all
five stars, an unbounded interval): honest "I cannot say", never a silent undercover.
"""

from __future__ import annotations

import numpy as np

from review_stars.probs import K

EPS = 1e-12


def quantile(scores, alpha: float) -> float:
    """The ``ceil((n + 1) (1 - alpha))``-th smallest score, or ``inf`` if ``n`` is too small."""
    scores = np.asarray(scores, dtype=np.float64)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, not {alpha}")
    if len(scores) == 0:
        raise ValueError("cannot calibrate on an empty set of scores")
    n = len(scores)
    level = (n + 1) * (1.0 - alpha) / n
    if level > 1.0:
        return float("inf")
    return float(np.quantile(scores, level, method="higher"))


def _check_draws(u, n: int) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    if u.shape != (n,):
        raise ValueError(
            f"need one uniform draw per review: expected {n}, got shape {u.shape}"
        )
    return u


def _sorted_cum(P: np.ndarray):
    order = np.argsort(-P, axis=1, kind="stable")
    sorted_p = np.take_along_axis(P, order, axis=1)
    return order, sorted_p, np.cumsum(sorted_p, axis=1)


def fill_empty(mask: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Give every empty set its most probable star (this can only add coverage)."""
    out = mask.copy()
    empty = ~out.any(axis=1)
    out[np.flatnonzero(empty), P[empty].argmax(axis=1)] = True
    return out


def set_stats(mask: np.ndarray, y) -> dict:
    y = np.asarray(y)
    if len(y) == 0:
        return {
            "coverage": float("nan"),
            "mean_size": float("nan"),
            "size_share": [float("nan")] * K,
        }
    size = mask.sum(axis=1)
    return {
        "coverage": float(mask[np.arange(len(y)), y - 1].mean()),
        "mean_size": float(size.mean()),
        "size_share": [float(np.mean(size == s)) for s in range(1, K + 1)],
    }


class ApsConformal:
    def __init__(self, alpha: float = 0.1, randomized: bool = True):
        self.alpha, self.randomized, self.qhat = alpha, randomized, float("nan")

    def fit(self, P: np.ndarray, y, u=None) -> ApsConformal:
        y = np.asarray(y)
        order, sorted_p, cum = _sorted_cum(P)
        rank = np.argmax(order == (y - 1)[:, None], axis=1)
        rows = np.arange(len(y))
        score = cum[rows, rank]  # mass up to and including the true star
        if self.randomized:
            if u is None:
                raise ValueError(
                    "randomized APS needs one uniform draw per calibration review"
                )
            score = score - _check_draws(u, len(y)) * sorted_p[rows, rank]
        self.qhat = quantile(score, self.alpha)
        return self

    def sets(self, P: np.ndarray, u=None, nonempty: bool = True) -> np.ndarray:
        """Boolean ``[n, 5]``: star ``k`` is in the set iff ``cum_k - w * p_k <= qhat``, where
        ``w`` is the review's uniform draw (randomized) or 1 (deterministic: include the crossing
        star)."""
        order, sorted_p, cum = _sorted_cum(P)
        if self.randomized:
            if u is None:
                raise ValueError("randomized APS needs one uniform draw per review")
            w = _check_draws(u, len(P))[:, None]
        else:
            w = 1.0
        keep_sorted = (cum - w * sorted_p) <= self.qhat + EPS
        mask = np.zeros_like(keep_sorted)
        np.put_along_axis(mask, order, keep_sorted, axis=1)
        return fill_empty(mask, P) if nonempty else mask

    def to_arrays(self, prefix: str) -> dict:
        return {
            f"{prefix}alpha": np.array(self.alpha),
            f"{prefix}randomized": np.array(self.randomized),
            f"{prefix}qhat": np.array(self.qhat),
        }

    @classmethod
    def from_arrays(cls, arrays: dict, prefix: str) -> ApsConformal:
        out = cls(float(arrays[f"{prefix}alpha"]), bool(arrays[f"{prefix}randomized"]))
        out.qhat = float(arrays[f"{prefix}qhat"])
        return out


class IntervalConformal:
    def __init__(self, alpha: float = 0.1):
        self.alpha, self.qhat = alpha, float("nan")

    @staticmethod
    def _check_sigma(sigma) -> np.ndarray:
        sigma = np.asarray(sigma, dtype=np.float64)
        if not (np.isfinite(sigma).all() and (sigma > 0).all()):
            raise ValueError("sigma must be finite and positive")
        return sigma

    def fit(self, mu, sigma, y) -> IntervalConformal:
        sigma = self._check_sigma(sigma)
        resid = (
            np.abs(np.asarray(y, dtype=np.float64) - np.asarray(mu, dtype=np.float64))
            / sigma
        )
        self.qhat = quantile(resid, self.alpha)
        return self

    def interval(self, mu, sigma) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma = np.asarray(mu, dtype=np.float64), self._check_sigma(sigma)
        half = self.qhat * sigma
        return mu - half, mu + half

    def to_arrays(self, prefix: str) -> dict:
        return {
            f"{prefix}alpha": np.array(self.alpha),
            f"{prefix}qhat": np.array(self.qhat),
        }

    @classmethod
    def from_arrays(cls, arrays: dict, prefix: str) -> IntervalConformal:
        out = cls(float(arrays[f"{prefix}alpha"]))
        out.qhat = float(arrays[f"{prefix}qhat"])
        return out
