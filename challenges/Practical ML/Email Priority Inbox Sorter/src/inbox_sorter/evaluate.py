"""Metrics: ranking quality, calibration, per-day inbox ranking and ablation.

The headline numbers are *inbox* metrics: for each (mailbox, day) with a real
mix of acted and ignored mail, how well does the model put the acted mail first,
against the same day in random order. Days with no acted mail, only acted mail,
or fewer than five messages say nothing about ranking and are excluded.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from inbox_sorter.features import LOCAL_OFFSET
from inbox_sorter.models import fit_lgbm

MIN_DAY_MESSAGES = 5
NDCG_K = 5
TOP_FRACTION = 0.2


def pr_auc(y, s) -> float:
    """Average precision; NaN when there are no positives."""
    y = np.asarray(y)
    return float(average_precision_score(y, s)) if y.sum() > 0 else math.nan


def roc_auc(y, s) -> float:
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return math.nan
    return float(roc_auc_score(y, s))


def _bin_index(p: np.ndarray, bins: int) -> np.ndarray:
    return np.minimum((np.clip(p, 0.0, 1.0) * bins).astype(int), bins - 1)


def reliability(y, p, bins: int = 10) -> list[dict]:
    """Populated equal-width probability bins: size, mean prediction, observed rate."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    idx = _bin_index(p, bins)
    out = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            out.append(
                {
                    "bin": b,
                    "n": int(mask.sum()),
                    "mean_pred": float(p[mask].mean()),
                    "frac_pos": float(y[mask].mean()),
                }
            )
    return out


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: bin-size-weighted |observed rate - mean prediction|."""
    n = len(np.asarray(y))
    return float(
        sum(
            r["n"] / n * abs(r["frac_pos"] - r["mean_pred"])
            for r in reliability(y, p, bins)
        )
    )


def summarize(y, scores) -> dict:
    y = np.asarray(y).astype(int)
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "pos_rate": float(y.mean()) if len(y) else math.nan,
        "pr_auc": pr_auc(y, scores),
        "roc_auc": roc_auc(y, scores),
    }


def _day_metrics(acted: np.ndarray, scores: np.ndarray, rng, k: int) -> dict:
    n = len(acted)
    # ties are broken by a seeded shuffle, so a constant scorer is not given the
    # arrival order for free
    order = np.lexsort((rng.random(n), -scores))
    ranked = acted[order].astype(float)
    pos = ranked.sum()
    discounts = 1.0 / np.log2(np.arange(2, NDCG_K + 2))
    dcg = float((ranked[:NDCG_K] * discounts[: len(ranked[:NDCG_K])]).sum())
    ideal = np.sort(ranked)[::-1][:NDCG_K]
    idcg = float((ideal * discounts[: len(ideal)]).sum())
    top = math.ceil(TOP_FRACTION * n)
    return {
        "n": n,
        "n_acted": int(pos),
        "precision_at_3": float(ranked[:k].sum() / min(k, n)),
        "ndcg_at_5": dcg / idcg,
        "recall_top20": float(ranked[:top].sum() / pos),
    }


def daily_inbox_metrics(
    df: pl.DataFrame, scores, k: int = 3, seed: int = 0
) -> pl.DataFrame:
    """Per eligible (mailbox, local day): precision@k, nDCG@5, recall in the top 20%."""
    rng = np.random.default_rng(seed)
    work = df.select("mailbox", "date", "acted").with_columns(
        pl.Series("score", np.asarray(scores, dtype=float)),
        (pl.col("date") + LOCAL_OFFSET).dt.date().alias("day"),
    )
    rows = []
    for (mailbox, day), grp in work.group_by(["mailbox", "day"], maintain_order=True):
        acted = grp["acted"].to_numpy().astype(bool)
        n_acted = int(acted.sum())
        if len(acted) < MIN_DAY_MESSAGES or n_acted == 0 or n_acted == len(acted):
            continue
        rows.append(
            {
                "mailbox": mailbox,
                "day": day,
                **_day_metrics(acted, grp["score"].to_numpy(), rng, k),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "mailbox": pl.Utf8,
            "day": pl.Date,
            "n": pl.Int64,
            "n_acted": pl.Int64,
            "precision_at_3": pl.Float64,
            "ndcg_at_5": pl.Float64,
            "recall_top20": pl.Float64,
        },
    )


def bootstrap_ci(
    values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """``(mean, lo, hi)`` with a percentile bootstrap over the given values."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def ablation(
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    groups: dict[str, list[str]],
    seed: int = 0,
) -> dict[str, float]:
    """Test PR-AUC lost when each metadata feature group is removed and LightGBM retrained.

    Positive = the group mattered. Every retrain early-stops on the same
    validation split, so the comparison is like for like.
    """
    all_cols = [c for cols in groups.values() for c in cols]

    def test_ap(cols: list[str]) -> float:
        clf = fit_lgbm(
            train.select(cols).to_numpy().astype(float),
            train["acted"].to_numpy().astype(int),
            val.select(cols).to_numpy().astype(float),
            val["acted"].to_numpy().astype(int),
            seed,
        )
        proba = clf.predict_proba(test.select(cols).to_numpy().astype(float))[:, 1]
        return pr_auc(test["acted"].to_numpy(), proba)

    full = test_ap(all_cols)
    return {
        name: full - test_ap([c for c in all_cols if c not in set(cols)])
        for name, cols in groups.items()
    }
