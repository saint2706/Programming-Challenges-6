"""The monitored model: a LightGBM classifier on the raw features."""

from __future__ import annotations

from itertools import pairwise

import lightgbm as lgb
import numpy as np
import polars as pl

from data import FEATURES


def _x(df: pl.DataFrame) -> np.ndarray:
    return df.select(FEATURES).to_numpy().astype(np.float64)


def fit(train: pl.DataFrame, seed: int = 0) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
    )
    return clf.fit(_x(train), train["label"].to_numpy())


def score(clf: lgb.LGBMClassifier, df: pl.DataFrame) -> np.ndarray:
    """P(label = UP) for every row."""
    return clf.predict_proba(_x(df))[:, 1]


def oof_scores(train: pl.DataFrame, k: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold P(UP) for every training row, from contiguous blocks.

    The score *baseline* must describe what the model says on rows it has not
    seen; scores on its own training rows are far more confident, which would make
    every unseen window look drifted.
    """
    n = train.height
    bounds = np.linspace(0, n, k + 1).astype(int)
    out = np.empty(n)
    for a, b in pairwise(bounds):
        rest = pl.concat([train[:a], train[b:]])
        out[a:b] = score(fit(rest, seed=seed), train[a:b])
    return out
