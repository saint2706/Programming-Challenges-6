"""Thresholds that decide what is worth saying. Every one is a field so a test or user can move it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Evidence: Benjamini-Hochberg false discovery rate across all tests of one kind.
    fdr: float = 0.01
    # Effect size: a pattern must also be big enough to matter.
    min_trend_change: float = 0.10  # fitted trend moves by >= 10% of its starting level
    seasonal_p: float = (
        0.01  # permutation p-value that cycles repeat the same pattern ...
    )
    seasonal_min_rho: float = (
        0.3  # ... and the mean correlation between cycles is at least this
    )
    min_seasonal_swing: float = 0.10  # peak-to-trough >= 10% of the average level
    min_shift: float = 0.10  # a level shift of >= 10% of the series mean ...
    min_shift_sigmas: float = 3.0  # ... and >= 3 noise standard deviations
    anomaly_z: float = (
        4.5  # robust z-score (median/MAD) of a seasonally adjusted residual
    )
    min_anomaly_deviation: float = (
        0.20  # an outlier must also be >= 20% off its expected value
    )
    min_epsilon_squared: float = 0.06  # Kruskal-Wallis effect size: a medium difference
    min_abs_rho: float = 0.5  # Spearman correlation worth mentioning
    top_share: float = 0.40  # one group holding >= 40% of a total ...
    pareto_share: float = 0.60  # ... or the top 20% of >= 50 entities holding >= 60%
    min_group_rows: int = 20  # groups smaller than this are left out of comparisons
    min_skew: float = 2.0
    missing_share: float = 0.05
    max_findings: int = 12  # headline findings in the report (data notes come on top)
    max_per_section: int = 3
    pelt_penalty_factor: float = 3.0  # penalty = factor * sigma^2 * ln(n)
