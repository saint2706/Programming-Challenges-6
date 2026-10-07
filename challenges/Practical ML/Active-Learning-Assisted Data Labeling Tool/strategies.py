"""Query strategies: ``select(state, b, rng) -> Selection``.

Every strategy returns unique, still-unlabeled pool indices, at most ``b`` of them (fewer only
when the pool runs out, none when it is empty). The uncertainty strategies take a plain
top-``b``; that their batches are redundant is part of what the benchmark measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from model import Head
from scipy.cluster.hierarchy import fcluster, linkage


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


def sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = (A * A).sum(axis=1)[:, None] + (B * B).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


class KCenter:
    """Greedy farthest-first (the k-center / core-set heuristic), seeded with the labeled set."""

    name = "k-center"

    def select(self, state, b, rng):
        X, cand = state.X, state.unlabeled()
        take = min(b, len(cand))
        if take == 0:
            return Selection.empty()
        avail = np.zeros(len(X), dtype=bool)
        avail[cand] = True
        picked, score = [], []
        if len(state.labeled):
            mind = sqdist(X, X[state.labeled]).min(axis=1)
        else:  # nothing to be far from yet: start at a random item
            j = int(rng.choice(cand))
            picked.append(j)
            score.append(0.0)
            avail[j] = False
            mind = sqdist(X, X[[j]])[:, 0]
        while len(picked) < take:
            j = int(np.argmax(np.where(avail, mind, -np.inf)))
            picked.append(j)
            score.append(float(mind[j]))
            avail[j] = False
            mind = np.minimum(mind, sqdist(X, X[[j]])[:, 0])
        return Selection(np.array(picked), np.array(score))


def gradient_parts(P: np.ndarray, X: np.ndarray):
    """``A = p - onehot(argmax p)``; the head is linear in the embedding, so ``H = X``.

    The BADGE gradient embedding of item ``i`` is the Kronecker product ``A_i (x) H_i``.
    """
    A = P.copy()
    A[np.arange(len(P)), P.argmax(axis=1)] -= 1.0
    return A, X


def gradient_sqdist(A: np.ndarray, H: np.ndarray, c: int) -> np.ndarray:
    """Squared distance of every gradient embedding to item ``c``'s, via the Kronecker identity.

    ``<a_i (x) h_i, a_c (x) h_c> = <a_i, a_c> <h_i, h_c>``, so the ``n x (K d)`` matrix is
    never built: each call costs ``O(n (K + d))``.
    """
    na, nh = (A * A).sum(axis=1), (H * H).sum(axis=1)
    d = na * nh + na[c] * nh[c] - 2.0 * (A @ A[c]) * (H @ H[c])
    return np.maximum(d, 0.0)


class Badge:
    """k-means++ seeding over gradient embeddings: uncertain *and* diverse (Ash et al. 2020)."""

    name = "badge"

    def select(self, state, b, rng):
        cand = state.unlabeled()
        take = min(b, len(cand))
        if take == 0:
            return Selection.empty()
        A, H = gradient_parts(state.P[cand], state.X[cand])
        norm = (A * A).sum(axis=1) * (H * H).sum(axis=1)
        chosen = [int(np.argmax(norm))]
        score = [float(norm[chosen[0]])]
        mind = gradient_sqdist(A, H, chosen[0])
        while len(chosen) < take:
            w = mind.copy()
            w[chosen] = 0.0
            total = w.sum()
            if total > 0:
                j = int(rng.choice(len(cand), p=w / total))
            else:  # every remaining gradient equals a chosen one: nothing to weight by
                j = int(rng.choice(np.setdiff1d(np.arange(len(cand)), chosen)))
            chosen.append(j)
            score.append(float(mind[j]))
            mind = np.minimum(mind, gradient_sqdist(A, H, j))
        return Selection(cand[np.array(chosen)], np.array(score))


def vote_entropy(votes: np.ndarray) -> np.ndarray:
    return entropy(votes / votes.sum(axis=1, keepdims=True))


class QueryByCommittee:
    """Bootstrap committee of heads; pick the items its members disagree on most."""

    name = "qbc"

    def __init__(self, members: int = 5):
        self.members = members

    def votes(self, state: State, rng: np.random.Generator) -> np.ndarray:
        cand = state.unlabeled()
        Xl, yl = state.X[state.labeled], state.y
        votes = np.zeros((len(cand), state.n_classes))
        for _ in range(self.members):
            boot = rng.integers(0, len(yl), size=len(yl))
            head = Head(state.n_classes, state.C).fit(Xl[boot], yl[boot])
            votes[np.arange(len(cand)), head.predict(state.X[cand])] += 1
        return votes

    def select(self, state, b, rng):
        cand = state.unlabeled()
        if len(cand) == 0:
            return Selection.empty()
        if len(state.labeled) == 0:  # no data to bootstrap: nothing to disagree about
            return Random().select(state, b, rng)
        u = vote_entropy(self.votes(state, rng))
        pos = top_positions(u, min(b, len(cand)), rng)
        return Selection(cand[pos], u[pos])


def cluster_pool(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Ward agglomerative clusters over the pool, ids ``0..k-1``."""
    k = max(1, min(n_clusters, len(X)))
    Z = linkage(np.asarray(X, dtype=np.float64), method="ward")
    return fcluster(Z, t=k, criterion="maxclust") - 1


class ClusterMargin:
    """Cluster-margin (Citovsky et al. 2021): the ``factor * b`` smallest margins, then round-robin
    over clusters so one batch is not ``b`` near-duplicates. Clusters are visited smallest first,
    sized by how many of the candidates they hold."""

    name = "cluster-margin"

    def __init__(self, n_clusters: int = 150, factor: int = 5, clusters=None):
        self.n_clusters, self.factor, self.clusters = n_clusters, factor, clusters

    def select(self, state, b, rng):
        cand = state.unlabeled()
        take = min(b, len(cand))
        if take == 0:
            return Selection.empty()
        if self.clusters is None:
            self.clusters = cluster_pool(state.X, self.n_clusters)
        m = margins(state.P[cand])
        pos = top_positions(-m, min(self.factor * take, len(cand)), rng)
        groups: dict[int, list[int]] = {}
        for p in pos:  # ``pos`` is ordered most uncertain first
            groups.setdefault(int(self.clusters[cand[p]]), []).append(int(p))
        queues = [
            q for _, q in sorted(groups.items(), key=lambda kv: (len(kv[1]), kv[0]))
        ]
        picked: list[int] = []
        while len(picked) < take:
            for q in queues:
                if q and len(picked) < take:
                    picked.append(q.pop(0))
        picked_pos = np.array(picked)
        return Selection(cand[picked_pos], -m[picked_pos])


NAMES = (
    "random",
    "least-confidence",
    "margin",
    "entropy",
    "k-center",
    "badge",
    "qbc",
    "cluster-margin",
)


def make(name: str, *, clusters=None, n_clusters: int = 150) -> Strategy:
    if name == "random":
        return Random()
    if name in UTILITIES:
        return Uncertainty(name)
    if name == "k-center":
        return KCenter()
    if name == "badge":
        return Badge()
    if name == "qbc":
        return QueryByCommittee()
    if name == "cluster-margin":
        return ClusterMargin(n_clusters=n_clusters, clusters=clusters)
    raise ValueError(f"unknown strategy {name!r}; choose from {NAMES}")
