"""One 5-star probability vector per review, whichever framing produced it.

The three framings emit different raw things; this module turns each into ``P[n, 5]`` (star 1..5)
so they can be scored by identical calibration metrics:

* classification: 5 logits, ``softmax(logits / T)``;
* regression: a heteroscedastic Gaussian ``(mu, sigma)``, integrated over ``[k - 0.5, k + 0.5]``
  with the tails assigned to stars 1 and 5, using ``sigma * T``;
* ordinal: a score and 4 increasing thresholds, ``P(y <= k) = sigmoid((theta_k - score) / T)``,
  differenced. A higher score means more stars.

``T`` is the one-parameter temperature each framing has: dividing logits, scaling sigma, scaling
the latent distance to the thresholds. ``Pred`` is one model's raw output; ``Predictions`` is one
model or an ensemble of them (the mean of the members' probability vectors).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, ndtr

from review_stars.config import FRAMINGS

K = 5
STARS = np.arange(1, K + 1)
EDGES = np.array([1.5, 2.5, 3.5, 4.5])


def _check_temperature(T: float) -> float:
    if not (np.isfinite(T) and T > 0):
        raise ValueError(f"temperature must be positive and finite, not {T}")
    return float(T)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def gaussian_probs(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Probability that ``N(mu, sigma)`` rounds to each star; rows sum to exactly 1."""
    mu, sigma = np.asarray(mu, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    if not np.isfinite(mu).all():
        raise ValueError("mu must be finite")
    if not (np.isfinite(sigma).all() and (sigma > 0).all()):
        raise ValueError("sigma must be finite and positive")
    cdf = ndtr((EDGES[None, :] - mu[:, None]) / sigma[:, None])
    cum = np.concatenate([np.zeros((len(mu), 1)), cdf, np.ones((len(mu), 1))], axis=1)
    return np.clip(np.diff(cum, axis=1), 0.0, 1.0)


def ordinal_probs(score: np.ndarray, theta: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Cumulative-link (proportional odds) probabilities; ``theta`` must be increasing."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (K - 1,) or (np.diff(theta) <= 0).any():
        raise ValueError("theta must be 4 strictly increasing thresholds")
    score = np.asarray(score, dtype=np.float64)
    cum = expit((theta[None, :] - score[:, None]) / T)
    full = np.concatenate(
        [np.zeros((len(score), 1)), cum, np.ones((len(score), 1))], axis=1
    )
    return np.clip(np.diff(full, axis=1), 0.0, 1.0)


def expected_star(P: np.ndarray) -> np.ndarray:
    return P @ STARS


def predicted_star(P: np.ndarray) -> np.ndarray:
    """The most probable star (1..5); ties go to the lower star."""
    return P.argmax(axis=1) + 1


@dataclass(frozen=True)
class Pred:
    framing: str
    logits: np.ndarray | None = None  # classification: [n, 5]
    mu: np.ndarray | None = None  # regression: [n]
    sigma: np.ndarray | None = None  # regression: [n]
    score: np.ndarray | None = None  # ordinal: [n]
    theta: np.ndarray | None = None  # ordinal: [4], shared by every row

    def __post_init__(self):
        if self.framing not in FRAMINGS:
            raise ValueError(f"framing must be one of {FRAMINGS}, not {self.framing!r}")
        if self.framing == "classification":
            if (
                self.logits is None
                or self.logits.ndim != 2
                or self.logits.shape[1] != K
            ):
                raise ValueError(f"logits must have shape [n, {K}]")
        elif self.framing == "regression":
            if self.mu is None or self.sigma is None:
                raise ValueError("a regression prediction needs mu and sigma")
            if self.mu.shape != self.sigma.shape or self.mu.ndim != 1:
                raise ValueError("mu and sigma must be 1-d and the same length")
        elif self.score is None or self.theta is None or self.theta.shape != (K - 1,):
            raise ValueError(
                "an ordinal prediction needs a score and 4 thresholds (theta)"
            )

    def __len__(self) -> int:
        field = {
            "classification": self.logits,
            "regression": self.mu,
            "ordinal": self.score,
        }
        return len(field[self.framing])

    def proba(self, T: float = 1.0) -> np.ndarray:
        T = _check_temperature(T)
        if self.framing == "classification":
            return softmax(self.logits.astype(np.float64) / T)
        if self.framing == "regression":
            return gaussian_probs(self.mu, self.sigma * T)
        return ordinal_probs(self.score, self.theta, T)

    def take(self, idx) -> Pred:
        if self.framing == "classification":
            return Pred("classification", logits=self.logits[idx])
        if self.framing == "regression":
            return Pred("regression", mu=self.mu[idx], sigma=self.sigma[idx])
        return Pred("ordinal", score=self.score[idx], theta=self.theta)

    def to_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        names = ("logits", "mu", "sigma", "score", "theta")
        return {
            f"{prefix}{k}": getattr(self, k)
            for k in names
            if getattr(self, k) is not None
        }

    @classmethod
    def from_arrays(cls, framing: str, arrays: dict, prefix: str) -> Pred:
        names = ("logits", "mu", "sigma", "score", "theta")
        return cls(
            framing,
            **{k: arrays[f"{prefix}{k}"] for k in names if f"{prefix}{k}" in arrays},
        )


class Predictions:
    """One model (a single member) or an ensemble: probabilities are the members' mean."""

    def __init__(self, members: list[Pred]):
        if not members:
            raise ValueError("Predictions needs at least one member")
        if len({m.framing for m in members}) != 1:
            raise ValueError("every member must have the same framing")
        if len({len(m) for m in members}) != 1:
            raise ValueError("every member must have the same number of rows")
        self.members = list(members)

    @property
    def framing(self) -> str:
        return self.members[0].framing

    @property
    def is_single(self) -> bool:
        return len(self.members) == 1

    def __len__(self) -> int:
        return len(self.members[0])

    def proba(self, T: float = 1.0) -> np.ndarray:
        return np.mean([m.proba(T) for m in self.members], axis=0)

    def take(self, idx) -> Predictions:
        return Predictions([m.take(idx) for m in self.members])

    def to_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        out = {
            f"{prefix}framing": np.array(self.framing),
            f"{prefix}n": np.array(len(self.members)),
        }
        for i, m in enumerate(self.members):
            out.update(m.to_arrays(f"{prefix}m{i}."))
        return out

    @classmethod
    def from_arrays(cls, arrays: dict, prefix: str) -> Predictions:
        framing = str(arrays[f"{prefix}framing"])
        n = int(arrays[f"{prefix}n"])
        return cls(
            [Pred.from_arrays(framing, arrays, f"{prefix}m{i}.") for i in range(n)]
        )
