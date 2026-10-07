"""The treatment propensity e(x) = P(treated | x), and a size-aware check that it is flat.

A randomized campaign has a constant propensity, but "randomized" is a claim to test, not
assume. On Criteo the arms look balanced feature by feature (standardized mean differences
under 0.06) yet treatment is still weakly predictable from the 12 features together (held-out
AUC about 0.51, treated share 0.845 to 0.859 across propensity quintiles), and the quintile
with the highest treated share also has by far the highest visit rate. That is enough to move
a plain difference in means by about a quarter, so the evaluation reweights each customer by
1 / P(the arm they were in | x) and the X, transformed-outcome and DR learners use e(x)
instead of a constant. The model is fit on the train split only.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

CLIP = (0.05, 0.95)
PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 1000,
}


class PropensityModel:
    """LightGBM classifier of ``treatment`` on the features; ``predict`` returns clipped e(x)."""

    def __init__(self, params: dict | None = None, seed: int = 0):
        self.params, self.seed = {**PARAMS, **(params or {})}, seed

    def fit(self, X, t):
        self.model_ = LGBMClassifier(
            **self.params,
            random_state=self.seed,
            deterministic=True,
            verbose=-1,
            n_jobs=-1,
        ).fit(X, t)
        return self

    def predict(self, X):
        return np.clip(self.model_.predict_proba(X)[:, 1], *CLIP)


def ipw_weights(t, e):
    """``1 / P(the arm the customer was actually in | x)``."""
    t, e = np.asarray(t), np.asarray(e)
    return np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))


def auc_ci(auc: float, n1: int, n0: int, z: float = 1.96) -> tuple[float, float]:
    """Hanley-McNeil (1982) normal-approximation CI for an AUC with ``n1`` positives, ``n0`` negatives."""
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (
        n1 * n0
    )
    se = float(np.sqrt(var))
    return auc - z * se, auc + z * se


def diagnose(model: PropensityModel, X, t, n_bins: int = 5) -> dict:
    """Held-out check: can treatment be predicted from the features, and by how much does it vary?

    ``auc`` is 0.5 for a clean randomization. ``bins`` split the held-out customers into equal
    groups by predicted propensity and give the *observed* treated share in each.
    """
    t = np.asarray(t)
    e = model.predict(X)
    auc = float(roc_auc_score(t, e))
    lo, hi = auc_ci(auc, int((t == 1).sum()), int((t == 0).sum()))
    bins = []
    for i, idx in enumerate(np.array_split(np.argsort(e, kind="stable"), n_bins), 1):
        bins.append(
            {
                "bin": i,
                "n": len(idx),
                "mean_propensity": float(e[idx].mean()),
                "treated_share": float(t[idx].mean()),
            }
        )
    return {
        "auc": auc,
        "auc_lo": lo,
        "auc_hi": hi,
        "e_min": float(e.min()),
        "e_max": float(e.max()),
        "bins": bins,
    }
