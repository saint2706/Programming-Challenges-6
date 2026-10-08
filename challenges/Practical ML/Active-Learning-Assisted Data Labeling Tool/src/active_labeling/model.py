"""The classifier head: multinomial logistic regression on frozen embeddings, always K classes wide."""

from __future__ import annotations

import warnings
from typing import Self

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

C_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
MAX_ITER = 300


class Head:
    """``fit`` on any labeled subset, ``proba`` over all ``n_classes``.

    Intents that have no label yet get probability exactly 0, so a cold start behaves as it
    would in a real project: the model cannot guess a class it has never been shown.
    """

    def __init__(self, n_classes: int, C: float = 10.0):
        self.n_classes, self.C = n_classes, C
        self._model: LogisticRegression | None = None
        self._classes: np.ndarray | None = None

    def fit(self, X, y) -> Self:
        y = np.asarray(y)
        if y.size == 0:
            raise ValueError("cannot fit a head on zero labels")
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError(f"labels must be in [0, {self.n_classes})")
        self._classes = np.unique(y)
        if (
            len(self._classes) == 1
        ):  # sklearn refuses a single class; the answer is trivial
            self._model = None
            return self
        self._model = LogisticRegression(C=self.C, max_iter=MAX_ITER)
        with warnings.catch_warnings():
            # lbfgs can stop at max_iter on the large-C fits; the probabilities are still usable
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._model.fit(X, y)
        return self

    def proba(self, X) -> np.ndarray:
        out = np.zeros((len(X), self.n_classes))
        if self._model is None:
            out[:, self._classes[0]] = 1.0
        else:
            out[:, self._model.classes_] = self._model.predict_proba(X)
        return out

    def predict(self, X) -> np.ndarray:
        return self.proba(X).argmax(axis=1)


def select_C(X, y, X_val, y_val, n_classes, grid=C_GRID, n_labels=500, seed=0) -> float:
    """The ``C`` with the lowest validation log loss for a head trained on a random ``n_labels`` subset.

    Chosen once and then fixed, so the benchmark's strategies all share one head. Ties go to
    the smaller (more regularized) value.
    """
    idx = np.random.default_rng(seed).choice(
        len(X), size=min(n_labels, len(X)), replace=False
    )
    best, best_loss = grid[0], np.inf
    for C in grid:
        P = Head(n_classes, C).fit(X[idx], y[idx]).proba(X_val)
        loss = -np.log(np.clip(P[np.arange(len(y_val)), y_val], 1e-12, 1.0)).mean()
        if loss < best_loss - 1e-9:
            best, best_loss = C, loss
    return best
