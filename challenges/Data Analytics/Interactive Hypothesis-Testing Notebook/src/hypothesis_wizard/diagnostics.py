"""Descriptive and assumption diagnostics. They are shown to the user; they do not pick the test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class GroupSummary:
    name: str
    n: int
    mean: float
    sd: float
    median: float
    skew: float
    excess_kurtosis: float
    outlier_share: float
    shapiro_p: float


def summarise(values: np.ndarray, name: str) -> GroupSummary:
    """Tukey-fence outlier share (beyond 1.5 IQR) and Shapiro-Wilk p (3 to 5000 values)."""
    x = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    outside = np.mean((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr))
    shapiro = float(stats.shapiro(x).pvalue) if 3 <= len(x) <= 5000 else float("nan")
    return GroupSummary(
        name=name,
        n=len(x),
        mean=float(np.mean(x)),
        sd=float(np.std(x, ddof=1)),
        median=float(np.median(x)),
        skew=float(stats.skew(x, bias=False)) if len(x) > 2 else float("nan"),
        excess_kurtosis=float(stats.kurtosis(x, bias=False))
        if len(x) > 3
        else float("nan"),
        outlier_share=float(outside),
        shapiro_p=shapiro,
    )


def heavy_tails(summaries: list[GroupSummary]) -> bool:
    """True when any group is skewed (|skew| > 1) or has more than 5% Tukey outliers."""
    return any(
        (abs(s.skew) > 1 if s.skew == s.skew else False) or s.outlier_share > 0.05
        for s in summaries
    )


def levene_brown_forsythe(groups: list[np.ndarray]) -> float:
    """Brown-Forsythe variance test (Levene with the median): p-value, shown only."""
    return float(stats.levene(*groups, center="median").pvalue)


def sd_ratio(groups: list[np.ndarray]) -> float:
    sds = [np.std(g, ddof=1) for g in groups]
    return float(max(sds) / min(sds))
