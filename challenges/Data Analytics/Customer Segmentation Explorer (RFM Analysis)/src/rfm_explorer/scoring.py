"""Turning raw recency/frequency/monetary values into scores and named segments.

Three ways to do it, because they disagree and the disagreement is the interesting part:

* ``quintile``: each dimension is cut into five equal-population groups. Tied values always get
  the same score, so a column where 40% of customers have frequency 1 gets uneven groups.
  (The common ``qcut(rank(method="first"))`` recipe splits ties by row order, which hands two
  identical customers different scores; this does not.)
* ``fixed``: business-style thresholds that do not move when the customer base changes.
* ``kmeans``: clusters on log-scaled, standardized R/F/M. No 1 to 5 grid, so no named segments;
  clusters are ranked by their centroid and described in words.

The two scored methods map ``(R, round((F + M) / 2))`` through ``SEGMENT_GRID``, a 5x5 table in the
spirit of the well-known Putler grid. It is a labelling convention, not a result: edit it freely.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

Method = Literal["quintile", "fixed", "kmeans"]
METHODS: tuple[Method, ...] = ("quintile", "fixed", "kmeans")
BINS = 5

# Upper edges of scores 1..4 (the fifth score is everything above the last edge).
# Recency is lower-is-better, so its edges are the upper bounds of scores 5, 4, 3, 2.
DEFAULT_EDGES: dict[str, list[float]] = {
    "recency_days": [30, 90, 180, 270],
    "frequency": [1, 2, 4, 9],
    "monetary": [200, 500, 1000, 2500],
}

# Rows: R score 5 (bought very recently) down to 1. Columns: FM score 1 to 5.
_GRID = {
    5: ["New Customers", "Potential Loyalist", "Loyal", "Champions", "Champions"],
    4: ["Promising", "Potential Loyalist", "Loyal", "Loyal", "Champions"],
    3: ["Promising", "Need Attention", "Need Attention", "Loyal", "Loyal"],
    2: ["About To Sleep", "About To Sleep", "Need Attention", "At Risk", "At Risk"],
    1: ["Lost", "Hibernating", "Hibernating", "At Risk", "Can't Lose Them"],
}
SEGMENT_GRID: dict[tuple[int, int], str] = {
    (r, fm + 1): name for r, row in _GRID.items() for fm, name in enumerate(row)
}
SEGMENT_ORDER = [
    "Champions",
    "Loyal",
    "Potential Loyalist",
    "New Customers",
    "Promising",
    "Need Attention",
    "About To Sleep",
    "At Risk",
    "Can't Lose Them",
    "Hibernating",
    "Lost",
]


def quantile_scores(
    values, higher_is_better: bool = True, bins: int = BINS
) -> np.ndarray:
    """Scores 1..bins by population share; equal values always get equal scores."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return np.array([], dtype=int)
    share = rankdata(v if higher_is_better else -v, method="average") / v.size
    return np.clip(np.ceil(share * bins), 1, bins).astype(int)


def fixed_scores(
    values, edges: list[float], higher_is_better: bool = True
) -> np.ndarray:
    """Scores 1..len(edges)+1 from ascending ``edges`` (a value equal to an edge scores below it)."""
    if list(edges) != sorted(edges):
        raise ValueError(f"edges must be ascending, got {edges}")
    v = np.asarray(values, dtype=float)
    score = np.searchsorted(np.asarray(edges, dtype=float), v, side="left") + 1
    return (len(edges) + 2 - score if not higher_is_better else score).astype(int)


def fm_score(f: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The combined frequency/monetary score, rounding half up so a 2 and a 3 become 3."""
    return (f + m + 1) // 2


def _features(rfm: pl.DataFrame) -> np.ndarray:
    """log1p(R), log1p(F), log1p(max(M, 0)), standardized; negative net spend is clipped to 0."""
    raw = np.column_stack(
        [
            np.log1p(rfm["recency_days"].to_numpy()),
            np.log1p(rfm["frequency"].to_numpy()),
            np.log1p(np.clip(rfm["monetary"].to_numpy(), 0, None)),
        ]
    )
    std = raw.std(axis=0)
    return (raw - raw.mean(axis=0)) / np.where(std == 0, 1.0, std)


def silhouette_by_k(
    rfm: pl.DataFrame, ks=range(2, 9), seed: int = 0, sample: int = 2000
) -> dict[int, float]:
    """Mean silhouette per k on at most ``sample`` customers (deterministic)."""
    x = _features(rfm)
    return {
        k: float(
            silhouette_score(
                x,
                KMeans(k, n_init=10, random_state=seed).fit_predict(x),
                sample_size=min(sample, len(x)),
                random_state=seed,
            )
        )
        for k in ks
    }


def _describe(z: np.ndarray) -> str:
    """Words for one centroid's (log-recency, log-frequency, log-monetary) z-scores."""
    r, f, m = z
    return "·".join(
        [
            "recent" if r < -0.5 else "lapsed" if r > 0.5 else "mid-recency",
            "frequent" if f > 0.5 else "rare" if f < -0.5 else "mid-frequency",
            "high-spend" if m > 0.5 else "low-spend" if m < -0.5 else "mid-spend",
        ]
    )


def kmeans_segments(
    rfm: pl.DataFrame, k: int = 5, seed: int = 0
) -> tuple[list[str], np.ndarray]:
    """``(names, priority)`` per customer; cluster 1 has the best centroid (recent, frequent, rich)."""
    if not 2 <= k <= len(rfm):
        raise ValueError(
            f"k must be between 2 and the number of customers ({len(rfm)}), got {k}"
        )
    x = _features(rfm)
    model = KMeans(k, n_init=10, random_state=seed).fit(x)
    quality = (
        -model.cluster_centers_[:, 0]
        + model.cluster_centers_[:, 1]
        + model.cluster_centers_[:, 2]
    )
    order = np.argsort(-quality, kind="stable")  # best first
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(1, k + 1)
    labels = {c: f"K{rank[c]} {_describe(model.cluster_centers_[c])}" for c in range(k)}
    return [labels[c] for c in model.labels_], (k + 1 - rank[model.labels_]).astype(
        float
    )


def score_customers(
    rfm: pl.DataFrame,
    method: Method = "quintile",
    *,
    k: int = 5,
    edges: dict[str, list[float]] | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """``rfm`` plus ``r``, ``f``, ``m`` scores, a ``segment`` and a numeric ``priority`` (higher is better).

    For ``kmeans`` the r/f/m columns are still the quintile scores, shown for reference; the
    segment and priority come from the clusters.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    if rfm.is_empty():
        return rfm.with_columns(
            r=pl.lit(0, pl.Int64),
            f=pl.lit(0, pl.Int64),
            m=pl.lit(0, pl.Int64),
            segment=pl.lit("", pl.Utf8),
            priority=pl.lit(0.0),
        )
    recency, frequency, monetary = (
        rfm[c].to_numpy() for c in ("recency_days", "frequency", "monetary")
    )
    if method == "fixed":
        e = {**DEFAULT_EDGES, **(edges or {})}
        r = fixed_scores(recency, e["recency_days"], higher_is_better=False)
        f = fixed_scores(frequency, e["frequency"])
        m = fixed_scores(monetary, e["monetary"])
    else:
        r = quantile_scores(recency, higher_is_better=False)
        f = quantile_scores(frequency)
        m = quantile_scores(monetary)
    if method == "kmeans":
        names, priority = kmeans_segments(rfm, k, seed)
        segment = names
    else:
        segment = [SEGMENT_GRID[(int(a), int(b))] for a, b in zip(r, fm_score(f, m))]
        priority = (r + f + m).astype(float)
    return rfm.with_columns(
        r=pl.Series(r),
        f=pl.Series(f),
        m=pl.Series(m),
        segment=pl.Series(segment),
        priority=pl.Series(priority),
    )


def summarize(scored: pl.DataFrame) -> pl.DataFrame:
    """Per segment: customers, share of customers, share of money, and mean R/F/M."""
    total_money = scored["monetary"].sum()
    return (
        scored.group_by("segment")
        .agg(
            customers=pl.len(),
            monetary=pl.col("monetary").sum(),
            mean_recency=pl.col("recency_days").mean(),
            mean_frequency=pl.col("frequency").mean(),
            mean_monetary=pl.col("monetary").mean(),
            priority=pl.col("priority").mean(),
        )
        .with_columns(
            customer_share=pl.col("customers") / pl.col("customers").sum(),
            money_share=pl.col("monetary") / total_money,
        )
        .sort("priority", descending=True)
    )
