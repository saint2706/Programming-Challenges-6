"""Ranking metrics and (later in this module) the tune-on-val / report-on-test run."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scorers import zscore


def ndcg_at_k(rels: Sequence[int], k: int) -> float:
    """Binary-relevance nDCG@k for a list already in ranked order.

    The ideal ranking puts every relevant item in the list first, so a query
    with more than ``k`` relevant items can still reach 1.0.
    """
    top = list(rels[:k])
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(top))
    n_ideal = min(int(sum(rels)), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(rels: Sequence[int]) -> float:
    for i, r in enumerate(rels):
        if r:
            return 1 / (i + 1)
    return 0.0


def precision_at_k(rels: Sequence[int], k: int) -> float:
    """Hits in the top ``k`` divided by ``k`` (a short list counts as misses)."""
    return sum(rels[:k]) / k


def average_precision_at_k(rels: Sequence[int], k: int) -> float:
    hits = 0
    total = 0.0
    for i, r in enumerate(rels[:k]):
        if r:
            hits += 1
            total += hits / (i + 1)
    n_relevant = min(int(sum(rels)), k)
    return total / n_relevant if n_relevant else 0.0


def rank_metrics(scores: np.ndarray, relevant: np.ndarray) -> dict[str, float]:
    """Metrics for one query. Ties break by index (stable), never at random."""
    order = np.argsort(-scores, kind="stable")
    rels = relevant[order].astype(int).tolist()
    return {
        "ndcg@10": ndcg_at_k(rels, 10),
        "mrr": mrr(rels),
        "p@10": precision_at_k(rels, 10),
        "map@50": average_precision_at_k(rels, 50),
    }


def rank_metrics_matrix(
    scores: np.ndarray, relevant: np.ndarray
) -> dict[str, np.ndarray]:
    """``rank_metrics`` for every row of a ``[queries, docs]`` score matrix."""
    rows = [rank_metrics(s, r) for s, r in zip(scores, relevant, strict=True)]
    return {
        key: np.array([row[key] for row in rows])
        for key in ("ndcg@10", "mrr", "p@10", "map@50")
    }


def bootstrap_ci(
    values: np.ndarray, n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean over queries."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def random_scores(shape: tuple[int, int], seed: int) -> np.ndarray:
    """The chance baseline every real scorer has to beat."""
    return np.random.default_rng(seed).random(shape)


def summarise(
    scores: np.ndarray, relevant: np.ndarray, query_categories: Sequence[str], seed: int
) -> dict:
    """Mean and 95% bootstrap interval for each metric, plus nDCG@10 per query category."""
    per_query = rank_metrics_matrix(scores, relevant)
    out: dict = {}
    for key, values in per_query.items():
        lo, hi = bootstrap_ci(values, seed=seed)
        out[key] = {"mean": float(values.mean()), "lo": lo, "hi": hi}
    cats = np.array(query_categories)
    out["per_category_ndcg@10"] = {
        c: float(per_query["ndcg@10"][cats == c].mean())
        for c in sorted(set(query_categories))
    }
    return out


def tune_fusion_weight(
    dense: np.ndarray, sparse: np.ndarray, relevant: np.ndarray, grid: Sequence[float]
) -> float:
    """Weight on the dense part that maximises mean nDCG@10 (smallest on ties).

    Run on *validation* matrices only; the chosen weight is then frozen for test.
    """
    zd, zs = zscore(dense), zscore(sparse)
    best_w, best = float(grid[0]), -1.0
    for w in grid:
        ndcg = rank_metrics_matrix(w * zd + (1 - w) * zs, relevant)["ndcg@10"].mean()
        if ndcg > best + 1e-12:
            best_w, best = float(w), float(ndcg)
    return best_w
