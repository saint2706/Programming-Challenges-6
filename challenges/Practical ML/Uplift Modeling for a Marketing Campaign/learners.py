"""Uplift learners on LightGBM, all with a constant known propensity ``e`` (an RCT).

tau(x) = E[Y | x, treated] - E[Y | x, control]. Every learner exposes ``fit(X, t, y)`` and
``predict(X) -> tau_hat`` (a ranking score; for the baselines it is just a score).

``propensity`` is either a constant (a clean RCT) or a fitted model with ``predict(X) -> e(x)``
(see ``propensity.py``): X, TO and DR use it; T, S, response and random never touch it.

* S   one model with ``t`` as a feature; f(x,1) - f(x,0).
* T   separate treated and control outcome models; mu1(x) - mu0(x). The brief's two-model approach.
* X   Kunzel et al. 2019: impute each arm's missing outcome with the other arm's model,
      fit two effect models on the imputed effects, blend them with weight ``e``.
* TO  transformed outcome: E[Y (t - e(x)) / (e(x) (1 - e(x))) | x] = tau(x), so regress it on x. The
      classic class-variable transformation z = tY + (1-t)(1-Y) is only valid at e = 0.5; ours is 0.85.
      It is also the learner most exposed to a wrong e: a constant e under even mild confounding
      adds delta * (mu1/e + mu0/(1-e)) to the target (see ``test_learners.py``).
* DR  doubly robust (Kennedy 2020), cross-fitted nuisance models, regressed pseudo-outcome.
* response  P(Y | x, treated): who converts when contacted, which is not who is persuadable.
* random    the chance baseline.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

DEFAULTS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}


def _clf(params, seed):
    return LGBMClassifier(
        **{**DEFAULTS, **params},
        random_state=seed,
        deterministic=True,
        verbose=-1,
        n_jobs=-1,
    )


def _reg(params, seed):
    return LGBMRegressor(
        **{**DEFAULTS, **params},
        random_state=seed,
        deterministic=True,
        verbose=-1,
        n_jobs=-1,
    )


def _p1(model, X):
    return model.predict_proba(X)[:, 1]


def transformed_outcome(y, t, e):
    """Y (t - e) / (e (1 - e)): its conditional mean given x is tau(x) when ``e`` is the true e(x)."""
    return y * (t - e) / (e * (1 - e))


class Learner:
    name = "?"

    def __init__(self, propensity=0.85, params: dict | None = None, seed: int = 0):
        if not hasattr(propensity, "predict") and not 0.0 < propensity < 1.0:
            raise ValueError(
                f"propensity must be strictly between 0 and 1, got {propensity}"
            )
        self.propensity, self.params, self.seed = propensity, dict(params or {}), seed

    def _e(self, X):
        """e(x) for every row: the fitted model's prediction, or the constant."""
        if hasattr(self.propensity, "predict"):
            return self.propensity.predict(X)
        return np.full(len(X), self.propensity)

    def _check(self, t, y):
        for arm, label in ((1, "treated"), (0, "control")):
            ya = y[t == arm]
            if ya.size == 0:
                raise ValueError(f"no {label} rows to fit on")
            if len(np.unique(ya)) < 2:
                raise ValueError(
                    f"the {label} arm has a single outcome class (too few positives to learn from)"
                )

    def fit(self, X, t, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class SLearner(Learner):
    name = "S"

    def fit(self, X, t, y):
        self._check(t, y)
        self.model_ = _clf(self.params, self.seed).fit(np.column_stack([X, t]), y)
        return self

    def predict(self, X):
        ones, zeros = np.ones((len(X), 1)), np.zeros((len(X), 1))
        return _p1(self.model_, np.hstack([X, ones])) - _p1(
            self.model_, np.hstack([X, zeros])
        )


class TLearner(Learner):
    name = "T"

    def fit(self, X, t, y):
        self._check(t, y)
        m = t == 1
        self.m1_ = _clf(self.params, self.seed).fit(X[m], y[m])
        self.m0_ = _clf(self.params, self.seed).fit(X[~m], y[~m])
        return self

    def predict(self, X):
        return _p1(self.m1_, X) - _p1(self.m0_, X)


class XLearner(Learner):
    name = "X"

    def fit(self, X, t, y):
        self._check(t, y)
        m = t == 1
        self.m1_ = _clf(self.params, self.seed).fit(X[m], y[m])
        self.m0_ = _clf(self.params, self.seed).fit(X[~m], y[~m])
        d1 = y[m] - _p1(self.m0_, X[m])  # treated: observed minus imputed control
        d0 = _p1(self.m1_, X[~m]) - y[~m]  # control: imputed treated minus observed
        self.tau1_ = _reg(self.params, self.seed).fit(X[m], d1)
        self.tau0_ = _reg(self.params, self.seed).fit(X[~m], d0)
        return self

    def predict(self, X):
        e = self._e(X)
        return e * self.tau0_.predict(X) + (1 - e) * self.tau1_.predict(X)


class TransformedOutcome(Learner):
    name = "TO"

    def fit(self, X, t, y):
        self._check(t, y)
        self.model_ = _reg(self.params, self.seed).fit(
            X, transformed_outcome(y, t, self._e(X))
        )
        return self

    def predict(self, X):
        return self.model_.predict(X)


class DRLearner(Learner):
    name = "DR"

    def fit(self, X, t, y):
        self._check(t, y)
        e = self._e(X)
        fold = np.random.default_rng(self.seed).integers(0, 2, len(X))
        phi = np.empty(len(X))
        for k in (0, 1):
            tr, te = fold != k, fold == k
            m1 = _clf(self.params, self.seed).fit(X[tr & (t == 1)], y[tr & (t == 1)])
            m0 = _clf(self.params, self.seed).fit(X[tr & (t == 0)], y[tr & (t == 0)])
            mu1, mu0 = _p1(m1, X[te]), _p1(m0, X[te])
            phi[te] = (
                mu1
                - mu0
                + t[te] * (y[te] - mu1) / e[te]
                - (1 - t[te]) * (y[te] - mu0) / (1 - e[te])
            )
        self.model_ = _reg(self.params, self.seed).fit(X, phi)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class ResponseModel(Learner):
    name = "response"

    def fit(self, X, t, y):
        self._check(t, y)
        m = t == 1
        self.model_ = _clf(self.params, self.seed).fit(X[m], y[m])
        return self

    def predict(self, X):
        return _p1(self.model_, X)


class RandomScorer(Learner):
    name = "random"

    def fit(self, X, t, y):
        return self

    def predict(self, X):
        return np.random.default_rng(self.seed).random(len(X))


LEARNERS = {
    c.name: c for c in (TLearner, SLearner, XLearner, TransformedOutcome, DRLearner)
}
BASELINES = {c.name: c for c in (ResponseModel, RandomScorer)}
