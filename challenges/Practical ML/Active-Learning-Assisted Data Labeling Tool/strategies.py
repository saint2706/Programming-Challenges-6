"""Query strategies: ``select(state, b, rng) -> Selection``.

Every strategy returns unique, still-unlabeled pool indices, at most ``b`` of them (fewer only
when the pool runs out, none when it is empty). The uncertainty strategies take a plain
top-``b``; that their batches are redundant is part of what the benchmark measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class State:
    """What a strategy may look at: the pool, the labels so far, and the current head's view."""

    X: np.ndarray  # [n, d] pool embeddings
    labeled: np.ndarray  # pool indices that have a label (any order)
    y: np.ndarray  # [len(labeled)] their class ids
    P: np.ndarray  # [n, n_classes] current head probabilities for the whole pool
    n_classes: int
    C: float  # head regularization, for strategies that fit their own heads

    def unlabeled(self) -> np.ndarray:
        mask = np.ones(len(self.X), dtype=bool)
        mask[self.labeled] = False
        return np.flatnonzero(mask)


@dataclass
class Selection:
    idx: np.ndarray  # pool indices, in suggestion order
    score: (
        np.ndarray
    )  # the strategy's own utility per suggested item (higher = more informative)

    @classmethod
    def empty(cls) -> Selection:
        return cls(np.array([], dtype=np.int64), np.array([], dtype=np.float64))


class Strategy(Protocol):
    name: str

    def select(self, state: State, b: int, rng: np.random.Generator) -> Selection: ...


def top_positions(score: np.ndarray, b: int, rng: np.random.Generator) -> np.ndarray:
    """Positions of the ``b`` largest scores; ties are broken at random, not by pool order."""
    perm = rng.permutation(len(score))
    order = np.argsort(-score[perm], kind="stable")
    return perm[order[:b]]


def margins(P: np.ndarray) -> np.ndarray:
    ps = np.sort(P, axis=1)
    return ps[:, -1] - ps[:, -2]


def least_confidence(P: np.ndarray) -> np.ndarray:
    return 1.0 - P.max(axis=1)


def entropy(P: np.ndarray) -> np.ndarray:
    return -(P * np.log(np.where(P > 0, P, 1.0))).sum(axis=1)


UTILITIES = {
    "least-confidence": least_confidence,
    "margin": lambda P: -margins(P),
    "entropy": entropy,
}


class Random:
    name = "random"

    def select(self, state, b, rng):
        cand = state.unlabeled()
        idx = rng.choice(cand, size=min(b, len(cand)), replace=False)
        return Selection(idx, np.zeros(len(idx)))


class Uncertainty:
    """Top-``b`` by least confidence, smallest margin, or largest predictive entropy."""

    def __init__(self, name: str):
        self.name = name
        self._utility = UTILITIES[name]

    def select(self, state, b, rng):
        cand = state.unlabeled()
        u = self._utility(state.P[cand])
        pos = top_positions(u, min(b, len(cand)), rng)
        return Selection(cand[pos], u[pos])


NAMES = ("random", "least-confidence", "margin", "entropy")


def make(name: str, *, clusters=None, n_clusters: int = 150) -> Strategy:
    if name == "random":
        return Random()
    if name in UTILITIES:
        return Uncertainty(name)
    raise ValueError(f"unknown strategy {name!r}; choose from {NAMES}")
