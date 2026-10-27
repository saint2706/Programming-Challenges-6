"""Pairwise comparisons after an omnibus test, and the multiple-comparison corrections."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (controls the false discovery rate)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adjusted = np.empty(m)
    running = 1.0
    for back, idx in enumerate(order):
        rank = m - back
        running = min(running, p[idx] * m / rank)
        adjusted[idx] = running
    return adjusted.tolist()


def games_howell(
    groups: Sequence[np.ndarray], names: Sequence[str], alpha: float = 0.05
) -> list[dict]:
    """Games-Howell comparisons: unequal variances and sizes, studentised range per pair."""
    k = len(groups)
    means = [float(np.mean(g)) for g in groups]
    variances = [float(np.var(g, ddof=1)) for g in groups]
    sizes = [len(g) for g in groups]
    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            vi, vj = variances[i] / sizes[i], variances[j] / sizes[j]
            se2 = vi + vj
            df = se2**2 / (vi**2 / (sizes[i] - 1) + vj**2 / (sizes[j] - 1))
            diff = means[i] - means[j]
            half_width_unit = math.sqrt(se2 / 2)
            q_crit = stats.studentized_range.ppf(1 - alpha, k, df)
            rows.append(
                {
                    "a": names[i],
                    "b": names[j],
                    "diff": diff,
                    "ci_low": diff - q_crit * half_width_unit,
                    "ci_high": diff + q_crit * half_width_unit,
                    "p": float(
                        stats.studentized_range.sf(abs(diff) / half_width_unit, k, df)
                    ),
                    "df": df,
                }
            )
    return rows


def dunn_holm(groups: Sequence[np.ndarray], names: Sequence[str]) -> list[dict]:
    """Dunn's test on the pooled ranks with the tie correction, Holm-adjusted across pairs."""
    k = len(groups)
    sizes = np.array([len(g) for g in groups])
    pooled = np.concatenate(groups)
    n_total = len(pooled)
    ranks = stats.rankdata(pooled)
    _, counts = np.unique(pooled, return_counts=True)
    ties = float(np.sum(counts**3 - counts))
    sigma2 = n_total * (n_total + 1) / 12 - ties / (12 * (n_total - 1))
    mean_ranks = []
    start = 0
    for size in sizes:
        mean_ranks.append(float(ranks[start : start + size].mean()))
        start += size
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    raw = []
    for i, j in pairs:
        z = (mean_ranks[i] - mean_ranks[j]) / math.sqrt(
            sigma2 * (1 / sizes[i] + 1 / sizes[j])
        )
        raw.append(2 * stats.norm.sf(abs(z)))
    adjusted = holm(raw)
    return [
        {"a": names[i], "b": names[j], "p_raw": p, "p": q}
        for (i, j), p, q in zip(pairs, raw, adjusted, strict=True)
    ]
