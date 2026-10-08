"""Post-hoc calibrators. Each is fit on the calibration split only and then applied unchanged.

* ``none``: the model as trained;
* ``temperature``: one scalar ``T`` per model, framing-appropriate (``Predictions.proba(T)``:
  logits / T for classification, ``sigma * T`` for regression, ``(theta - score) / T`` for
  ordinal), fit by minimizing the calibration NLL. For an ensemble the same ``T`` goes to every
  member;
* ``vector``: per-star scale and bias on the logits of a single classification model;
* ``isotonic``: one-vs-rest isotonic regression of each star's probability, renormalized. A star
  with no positives in the calibration split cannot be learned, so its probability is left as the
  model's (and reported in ``degenerate``) rather than flattened to zero.

All of them serialize to plain arrays, so the Streamlit app never unpickles anything.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from sklearn.isotonic import IsotonicRegression

from review_stars.probs import K, Predictions, softmax

KINDS = ("none", "temperature", "vector", "isotonic")
T_BOUNDS = (0.05, 20.0)
FLOOR = 1e-6  # smallest probability an isotonic map may leave a star
VECTOR_RIDGE = (
    1e-3  # keeps per-star bias finite when a star never occurs in the calibration set
)


def _nll(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(P[np.arange(len(y)), y - 1], 1e-12, None)).mean())


def _need_rows(y) -> None:
    if len(y) == 0:
        raise ValueError("cannot fit a calibrator on an empty calibration split")


class Identity:
    name = "none"

    def fit(self, preds: Predictions, y) -> Identity:
        return self

    def proba(self, preds: Predictions) -> np.ndarray:
        return preds.proba()

    def to_arrays(self, prefix: str) -> dict:
        return {f"{prefix}kind": np.array(self.name)}

    @classmethod
    def from_arrays(cls, arrays: dict, prefix: str) -> Identity:
        return cls()


class Temperature:
    name = "temperature"

    def __init__(self, T: float = 1.0):
        self.T = float(T)

    def fit(self, preds: Predictions, y) -> Temperature:
        y = np.asarray(y)
        _need_rows(y)
        lo, hi = np.log(T_BOUNDS[0]), np.log(T_BOUNDS[1])
        res = minimize_scalar(
            lambda t: _nll(preds.proba(float(np.exp(t))), y),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-5},
        )
        self.T = float(np.exp(res.x))
        return self

    def proba(self, preds: Predictions) -> np.ndarray:
        return preds.proba(self.T)

    def to_arrays(self, prefix: str) -> dict:
        return {f"{prefix}kind": np.array(self.name), f"{prefix}T": np.array(self.T)}

    @classmethod
    def from_arrays(cls, arrays: dict, prefix: str) -> Temperature:
        return cls(float(arrays[f"{prefix}T"]))


class VectorScaling:
    name = "vector"

    def __init__(self, a=None, b=None):
        self.a = np.ones(K) if a is None else np.asarray(a, dtype=np.float64)
        self.b = np.zeros(K) if b is None else np.asarray(b, dtype=np.float64)

    def fit(self, preds: Predictions, y) -> VectorScaling:
        if not (preds.framing == "classification" and preds.is_single):
            raise ValueError("vector scaling needs a single classification model")
        y = np.asarray(y)
        _need_rows(y)
        z = preds.members[0].logits.astype(np.float64)
        onehot = np.eye(K)[y - 1]
        n = len(y)

        def loss(theta):
            a, b = theta[:K], theta[K:]
            s = z * a + b
            lse = logsumexp(s, axis=1)
            value = -(s[np.arange(n), y - 1] - lse).mean()
            value += VECTOR_RIDGE * (((a - 1) ** 2).sum() + (b**2).sum())
            g = (np.exp(s - lse[:, None]) - onehot) / n
            grad = np.concatenate(
                [
                    (g * z).sum(axis=0) + 2 * VECTOR_RIDGE * (a - 1),
                    g.sum(axis=0) + 2 * VECTOR_RIDGE * b,
                ]
            )
            return value, grad

        res = minimize(
            loss, np.concatenate([np.ones(K), np.zeros(K)]), jac=True, method="L-BFGS-B"
        )
        self.a, self.b = res.x[:K], res.x[K:]
        return self

    def proba(self, preds: Predictions) -> np.ndarray:
        if not (preds.framing == "classification" and preds.is_single):
            raise ValueError("vector scaling needs a single classification model")
        return softmax(preds.members[0].logits * self.a + self.b)

    def to_arrays(self, prefix: str) -> dict:
        return {
            f"{prefix}kind": np.array(self.name),
            f"{prefix}a": self.a,
            f"{prefix}b": self.b,
        }

    @classmethod
    def from_arrays(cls, arrays: dict, prefix: str) -> VectorScaling:
        return cls(arrays[f"{prefix}a"], arrays[f"{prefix}b"])


class Isotonic:
    name = "isotonic"

    def __init__(self, maps=None):
        self.maps = (
            maps if maps is not None else [None] * K
        )  # per star: (x, y) knots or None

    @property
    def degenerate(self) -> list[int]:
        """Stars (0-based) that had no positives in the calibration split and were left alone."""
        return [k for k, m in enumerate(self.maps) if m is None]

    def fit(self, preds: Predictions, y) -> Isotonic:
        y = np.asarray(y)
        _need_rows(y)
        P = preds.proba()
        maps = []
        for k in range(K):
            target = (y == k + 1).astype(np.float64)
            if target.sum() == 0:
                maps.append(None)
                continue
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
                P[:, k], target
            )
            maps.append((iso.X_thresholds_.copy(), iso.y_thresholds_.copy()))
        self.maps = maps
        return self

    def proba(self, preds: Predictions) -> np.ndarray:
        P = preds.proba()
        cols = [
            P[:, k] if m is None else np.interp(P[:, k], m[0], m[1])
            for k, m in enumerate(self.maps)
        ]
        Q = np.maximum(np.stack(cols, axis=1), FLOOR)
        return Q / Q.sum(axis=1, keepdims=True)

    def to_arrays(self, prefix: str) -> dict:
        out = {f"{prefix}kind": np.array(self.name)}
        for k, m in enumerate(self.maps):
            out[f"{prefix}x{k}"] = np.zeros(0) if m is None else m[0]
            out[f"{prefix}y{k}"] = np.zeros(0) if m is None else m[1]
        return out

    @classmethod
    def from_arrays(cls, arrays: dict, prefix: str) -> Isotonic:
        maps = []
        for k in range(K):
            x, y = arrays[f"{prefix}x{k}"], arrays[f"{prefix}y{k}"]
            maps.append(None if len(x) == 0 else (x, y))
        return cls(maps)


_CLASSES = {c.name: c for c in (Identity, Temperature, VectorScaling, Isotonic)}


def make(kind: str):
    if kind not in _CLASSES:
        raise ValueError(f"unknown calibrator {kind!r}; choose from {KINDS}")
    return _CLASSES[kind]()


def applicable(kind: str, preds: Predictions) -> bool:
    if kind == "vector":
        return preds.framing == "classification" and preds.is_single
    return kind in KINDS


def fit_all(preds: Predictions, y, kinds=KINDS) -> dict:
    """Every applicable calibrator, fit on ``(preds, y)`` (the calibration split)."""
    return {k: make(k).fit(preds, y) for k in kinds if applicable(k, preds)}


def from_arrays(arrays: dict, prefix: str):
    kind = str(arrays[f"{prefix}kind"])
    return _CLASSES[kind].from_arrays(arrays, prefix)
