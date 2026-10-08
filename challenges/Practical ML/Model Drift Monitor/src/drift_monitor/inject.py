"""Controlled synthetic drift, applied from ``start_row`` onward.

Each injector leaves every row before ``start_row`` untouched and changes only
what its name says, so a benchmark knows exactly what drifted and when:

- ``covariate``: one feature shifted by some standard deviations (inputs move)
- ``prior``: the share of UP labels changes (label prior moves)
- ``concept``: a fraction of labels is flipped, features untouched (accuracy
  falls although the inputs look the same)
- ``NoisyClassifier``: noise added to the model's score only (scores move though
  inputs and labels do not)
"""

from __future__ import annotations

import numpy as np
import polars as pl

from drift_monitor.data import NUMERIC


def covariate(
    df: pl.DataFrame, feature: str, sigmas: float, start_row: int
) -> pl.DataFrame:
    if feature not in NUMERIC:
        raise ValueError(f"covariate shift needs a numeric feature, got {feature!r}")
    std = float(df[feature].std() or 1.0)
    shift = (
        pl.when(pl.int_range(pl.len()) >= start_row).then(sigmas * std).otherwise(0.0)
    )
    return df.with_columns((pl.col(feature) + shift).alias(feature))


def prior(df: pl.DataFrame, rate: float, start_row: int, seed: int = 0) -> pl.DataFrame:
    """After ``start_row`` resample rows (with replacement) so the UP rate is ``rate``."""
    if not 0.0 < rate < 1.0:
        raise ValueError(f"rate must be in (0, 1), got {rate}")
    rng = np.random.default_rng(seed)
    tail = df[start_row:]
    y = tail["label"].to_numpy()
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n = tail.height
    n_pos = round(rate * n)
    pick = np.concatenate([rng.choice(pos, n_pos), rng.choice(neg, n - n_pos)])
    rng.shuffle(pick)
    return pl.concat([df[:start_row], tail[pick.tolist()]])


def concept(
    df: pl.DataFrame, strength: float, start_row: int, seed: int = 0
) -> pl.DataFrame:
    """Flip about ``strength`` of the labels after ``start_row``; features are untouched."""
    rng = np.random.default_rng(seed)
    flip = rng.random(df.height) < strength
    flip[:start_row] = False
    y = df["label"].to_numpy()
    return df.with_columns(pl.Series("label", np.where(flip, 1 - y, y)))


class NoisyClassifier:
    """Wraps a classifier; adds Gaussian noise to P(UP) for rows from ``start_row``."""

    def __init__(self, clf, sd: float, start_row: int, seed: int = 0):
        self._clf, self.sd, self.start_row, self.seed = clf, sd, start_row, seed

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        proba = self._clf.predict_proba(x)
        up = proba[:, 1].copy()
        noise = np.random.default_rng(self.seed).normal(0.0, self.sd, len(up))
        noise[: self.start_row] = 0.0
        up = np.clip(up + noise, 0.0, 1.0)
        return np.column_stack([1.0 - up, up])
