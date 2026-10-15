"""Metrics for a stockout-risk score: ranking, calibration, a daily worklist and a cost model.

``y`` is the 0/1 outcome and ``p`` the score (a probability for the calibrated models). Plain numpy
arrays throughout, so everything is testable without the dataset.
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

RATIOS = (2, 5, 10, 20)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over equal-count bins (the weighted gap |mean(p) - mean(y)|)."""
    order = np.argsort(p, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            total += len(chunk) * abs(float(p[chunk].mean()) - float(y[chunk].mean()))
    return total / len(p)


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """Equal-count reliability table: mean score against observed rate per bin."""
    order = np.argsort(p, kind="stable")
    rows = []
    for chunk in np.array_split(order, bins):
        if len(chunk):
            rows.append(
                {
                    "mean_p": float(p[chunk].mean()),
                    "rate": float(y[chunk].mean()),
                    "n": len(chunk),
                }
            )
    return rows


def ranking_scores(y: np.ndarray, p: np.ndarray) -> dict:
    """PR-AUC and ROC-AUC only, for scores that are not probabilities (hours, counts, flags).

    A constant score gets PR-AUC = base rate and ROC-AUC = 0.5.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    constant = len(np.unique(p)) == 1
    return {
        "n": len(y),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": 0.5 if constant else float(roc_auc_score(y, p)),
    }


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    """``ranking_scores`` plus log-loss, Brier and ECE, which need ``p`` to be a probability."""
    y = np.asarray(y).astype(int)
    probs = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        **ranking_scores(y, p),
        "log_loss": float(log_loss(y, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probs)),
        "ece": ece(y, probs),
    }


def worklist(
    y: np.ndarray, p: np.ndarray, day: np.ndarray, fractions=(0.01, 0.05, 0.10, 0.20)
):
    """Precision and recall of "check the top share of each day's store-products".

    ``day`` groups the rows; for every day the highest-scored ``fraction`` of its rows is the
    worklist (ties broken by position, which is arbitrary but fixed). Reported are the pooled
    precision and recall over all days, and the lift of precision over the base rate.
    """
    y = np.asarray(y).astype(int)
    order = np.lexsort((-p, day))
    day_sorted = day[order]
    y_sorted = y[order]
    bounds = np.flatnonzero(np.r_[True, day_sorted[1:] != day_sorted[:-1], True])
    out = []
    for frac in fractions:
        hits = picked = 0
        for lo, hi in itertools.pairwise(bounds):
            k = max(1, round(frac * (hi - lo)))
            hits += int(y_sorted[lo : lo + k].sum())
            picked += k
        precision = hits / picked
        out.append(
            {
                "fraction": frac,
                "precision": precision,
                "recall": hits / int(y.sum()),
                "lift": precision / float(y.mean()),
            }
        )
    return out


def cost(y: np.ndarray, p: np.ndarray, threshold: float, ratio: float) -> float:
    """Cost per row of "check when ``p >= threshold``": 1 per check, ``ratio`` per missed stockout.

    A check is assumed to prevent the stockout it finds (an upper bound on the benefit), so a hit
    costs 1 and a miss costs ``ratio``. Never checking costs ``ratio * base_rate``; always checking
    costs 1.
    """
    y = np.asarray(y).astype(int)
    flag = p >= threshold
    return float((flag.sum() + ratio * ((y == 1) & ~flag).sum()) / len(y))


def best_threshold(y: np.ndarray, p: np.ndarray, ratio: float) -> float:
    """The threshold minimising ``cost``, scanned over the score's own quantiles."""
    grid = np.unique(np.quantile(p, np.linspace(0, 1, 401)))
    grid = np.append(grid, np.nextafter(grid[-1], np.inf))  # "never check"
    costs = [cost(y, p, t, ratio) for t in grid]
    return float(grid[int(np.argmin(costs))])


def cost_table(y_val, p_val, y_test, p_test, ratios=RATIOS) -> list[dict]:
    """For each cost ratio: the threshold picked on validation, and what it costs on test.

    ``bayes`` is ``1 / ratio``, the threshold that is optimal when ``p`` is a calibrated probability;
    the gap between it and the validation-picked one shows how far the score is from calibrated.
    """
    base = float(np.mean(y_test))
    rows = []
    for ratio in ratios:
        thr = best_threshold(y_val, p_val, ratio)
        spent = cost(y_test, p_test, thr, ratio)
        rows.append(
            {
                "ratio": ratio,
                "threshold": thr,
                "bayes": 1 / ratio,
                "cost": spent,
                "never": ratio * base,
                "always": 1.0,
                "saving_vs_best_trivial": 1 - spent / min(ratio * base, 1.0),
                "checked_share": float((p_test >= thr).mean()),
            }
        )
    return rows


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index of ``actual`` against ``expected`` on the expected deciles."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def group_scores(y, p, groups, min_rows: int = 500) -> list[dict]:
    """PR-AUC and base rate per group value, for groups with at least ``min_rows`` rows."""
    out = []
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() >= min_rows and 0 < y[mask].sum() < mask.sum():
            out.append(
                {
                    "group": int(g),
                    "n": int(mask.sum()),
                    "base_rate": float(y[mask].mean()),
                    "pr_auc": float(average_precision_score(y[mask], p[mask])),
                }
            )
    return sorted(out, key=lambda r: r["group"])
