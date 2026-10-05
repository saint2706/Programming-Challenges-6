"""The monitored model: a LightGBM classifier on the raw features."""

from __future__ import annotations

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
