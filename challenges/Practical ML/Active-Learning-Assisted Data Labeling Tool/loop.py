"""The simulated-annotator loop: fit, score, ask for a batch, reveal its gold labels, repeat."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from model import Head
from sklearn.metrics import f1_score
from strategies import State

METRICS = (
    "n",
    "acc",
    "f1",
    "coverage",
    "skew",
    "change",
    "sel_err",
    "pool_err",
    "sel_time",
)


@dataclass
class Problem:
    X: np.ndarray  # pool embeddings
    y: np.ndarray  # pool gold labels: what the simulated annotator reveals
    X_eval: np.ndarray  # scored every round, never used to select
    y_eval: np.ndarray
    n_classes: int
    C: float
    X_val: np.ndarray | None = None  # validation: tuning seeds and the stopping rule
    y_val: np.ndarray | None = None


def initial_labels(n: int, size: int, seed: int) -> np.ndarray:
    """The cold-start set: random, deliberately not stratified (some intents start unseen)."""
    return np.random.default_rng(seed).choice(n, size=size, replace=False)


def run_loop(problem, strategy, seed, budget=1500, b=50, init=77) -> dict:
    X, y, K = problem.X, problem.y, problem.n_classes
    budget = min(budget, len(X) - init)
    labeled = initial_labels(len(X), init, seed)
    rng = np.random.default_rng([seed, 1])
    curve: dict = {key: [] for key in METRICS} | {"picked": []}
    prev = None
    queried = 0
    while True:
        head = Head(K, problem.C).fit(X[labeled], y[labeled])
        P = head.proba(X)
        pred = P.argmax(axis=1)
        ev = head.predict(problem.X_eval)
        counts = np.bincount(y[labeled], minlength=K)
        unl = np.ones(len(X), dtype=bool)
        unl[labeled] = False
        curve["n"].append(len(labeled))
        curve["acc"].append(float((ev == problem.y_eval).mean()))
        curve["f1"].append(
            float(
                f1_score(
                    problem.y_eval,
                    ev,
                    labels=np.arange(K),
                    average="macro",
                    zero_division=0,
                )
            )
        )
        curve["coverage"].append(int((counts > 0).sum()))
        curve["skew"].append(float(counts.max() / counts.sum()))
        curve["change"].append(
            float("nan") if prev is None else float((pred != prev).mean())
        )
        curve["pool_err"].append(
            float((pred[unl] != y[unl]).mean()) if unl.any() else float("nan")
        )
        prev = pred
        if queried >= budget:
            curve["sel_err"].append(float("nan"))
            curve["sel_time"].append(float("nan"))
            return curve
        state = State(X, labeled, y[labeled], P, K, problem.C)
        started = time.perf_counter()
        sel = strategy.select(state, min(b, budget - queried), rng)
        curve["sel_time"].append(time.perf_counter() - started)
        idx = np.asarray(sel.idx, dtype=np.int64)
        if (
            len(idx) == 0
            or np.isin(idx, labeled).any()
            or len(set(idx.tolist())) != len(idx)
        ):
            raise RuntimeError(f"{strategy.name} returned an invalid selection")
        curve["sel_err"].append(float((pred[idx] != y[idx]).mean()))
        curve["picked"].append(idx.tolist())
        labeled = np.concatenate([labeled, idx])
        queried += len(idx)
