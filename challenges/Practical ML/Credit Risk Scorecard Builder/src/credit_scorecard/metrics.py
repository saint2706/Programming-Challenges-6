"""Scorecard validation metrics: discrimination (AUC, Gini, KS), stability (PSI), calibration."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

PSI_EPS = 1e-6  # floor for an empty bucket's share so the log stays finite


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve for "higher score = more likely bad"."""
    return float(roc_auc_score(y, score))


def gini(y: np.ndarray, score: np.ndarray) -> float:
    return 2 * auc(y, score) - 1


def ks(y: np.ndarray, score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: the largest gap between the bad and good cumulative score curves.

    Tied scores are cumulated together, otherwise an integer-points score would report a gap that
    depends on the arbitrary order of ties.
    """
    y = np.asarray(y)
    order = np.argsort(score, kind="stable")
    s, yb = np.asarray(score)[order], y[order]
    boundary = np.r_[s[1:] != s[:-1], True]
    cum_bad = np.cumsum(yb)[boundary] / yb.sum()
    cum_good = np.cumsum(1 - yb)[boundary] / (len(y) - yb.sum())
    return float(np.max(np.abs(cum_bad - cum_good)))


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population stability index of ``actual`` against ``expected``.

    Buckets are the quantiles of ``expected``. Rule of thumb: < 0.1 stable, 0.1-0.25 shifting,
    > 0.25 a different population.
    """
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e = np.bincount(
        np.searchsorted(cuts, expected, side="left"), minlength=len(cuts) + 1
    )
    a = np.bincount(np.searchsorted(cuts, actual, side="left"), minlength=len(cuts) + 1)
    pe = np.maximum(e / e.sum(), PSI_EPS)
    pa = np.maximum(a / a.sum(), PSI_EPS)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def bootstrap_auc_interval(
    y: np.ndarray, score: np.ndarray, *, n: int = 200, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap interval for the AUC (the test set has ~3,000 bads, not infinite)."""
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        out.append(roc_auc_score(y[i], score[i]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def calibration_table(
    y: np.ndarray, score: np.ndarray, predicted_pd: np.ndarray, n_bins: int = 10
) -> pl.DataFrame:
    """Observed bad rate versus the scorecard's predicted PD, per score quantile."""
    cuts = np.unique(np.quantile(score, np.linspace(0, 1, n_bins + 1)[1:-1]))
    band = np.searchsorted(cuts, score, side="left")
    return (
        pl.DataFrame({"band": band, "score": score, "bad": y, "pd": predicted_pd})
        .group_by("band")
        .agg(
            pl.col("score").min().alias("score_min"),
            pl.col("score").max().alias("score_max"),
            pl.len().alias("n"),
            pl.col("bad").mean().alias("observed_bad_rate"),
            pl.col("pd").mean().alias("predicted_pd"),
        )
        .sort("band")
    )
