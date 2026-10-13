"""Train and evaluate end to end: load, clean, split, fit the scorecard, benchmark against a GBM."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from credit_scorecard import data, metrics
from credit_scorecard.scorecard import Scorecard, fit_scorecard


@dataclass
class Result:
    scorecard: Scorecard
    train: pl.DataFrame
    test: pl.DataFrame
    metrics: dict = field(default_factory=dict)
    calibration: pl.DataFrame | None = None
    feature_psi: dict[str, float] = field(default_factory=dict)
    risk: dict[str, np.ndarray] = field(
        default_factory=dict
    )  # test risk per model, higher = worse


def train_scorecard(
    seed: int = 42, **kwargs
) -> tuple[Scorecard, pl.DataFrame, pl.DataFrame]:
    df = data.clean(data.load_raw())
    train, test = data.split(df, seed=seed)
    card = fit_scorecard(
        train, data.TARGET, data.FEATURES, specials=data.SPECIAL_VALUES, **kwargs
    )
    return card, train, test


def evaluate(
    card: Scorecard, train: pl.DataFrame, test: pl.DataFrame, seed: int = 42
) -> Result:
    """Scorecard versus an untuned gradient-boosted tree on the raw features.

    Both are scored on the test set only. The scorecard is "higher score = better", so its metrics
    are computed on ``-score`` (higher = more likely bad), the same direction as the GBM.
    """
    y_tr, y_te = train[data.TARGET].to_numpy(), test[data.TARGET].to_numpy()
    s_tr, s_te = card.score(train), card.score(test)
    gbm = HistGradientBoostingClassifier(random_state=seed).fit(
        train.select(data.FEATURES).to_numpy(), y_tr
    )
    p_gbm = gbm.predict_proba(test.select(data.FEATURES).to_numpy())[:, 1]

    def block(y, risk):
        return {
            "auc": metrics.auc(y, risk),
            "gini": metrics.gini(y, risk),
            "ks": metrics.ks(y, risk),
        }

    lo, hi = metrics.bootstrap_auc_interval(y_te, -s_te)
    pd_te = card.probability_bad(s_te)
    out = {
        "n_train": train.height,
        "n_test": test.height,
        "bad_rate_train": float(y_tr.mean()),
        "bad_rate_test": float(y_te.mean()),
        "scorecard_train": block(y_tr, -s_tr),
        "scorecard_test": block(y_te, -s_te),
        "scorecard_test_auc_ci95": [lo, hi],
        "gbm_test": block(y_te, p_gbm),
        "score_psi_train_vs_test": metrics.psi(s_tr, s_te),
        "brier_test": float(np.mean((pd_te - y_te) ** 2)),
        "mean_predicted_pd_test": float(pd_te.mean()),
        "features_used": card.features,
    }
    return Result(
        card,
        train,
        test,
        out,
        metrics.calibration_table(y_te, s_te, pd_te),
        {
            f: metrics.psi(train[f].to_numpy(), test[f].to_numpy())
            for f in card.features
        },
        {"Scorecard": -s_te, "Gradient boosting": p_gbm},
    )


def run(seed: int = 42, **kwargs) -> Result:
    card, train, test = train_scorecard(seed, **kwargs)
    return evaluate(card, train, test, seed)
