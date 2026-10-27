"""Effect sizes, their intervals, and the smallest effect a design can detect.

Intervals for Hedges' g and the paired dz use the usual large-sample approximations (Hedges and
Olkin 1985; Lakens 2013); they are stated as approximate in the reports.
"""

from __future__ import annotations

import math

import numpy as np
from statsmodels.stats.power import TTestIndPower, TTestPower


def hedges_g(x: np.ndarray, y: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Standardised mean difference (x minus y) with Hedges' small-sample correction."""
    n1, n2 = len(x), len(y)
    df = n1 + n2 - 2
    pooled = math.sqrt(
        ((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / df
    )
    g = (np.mean(x) - np.mean(y)) / pooled * (1 - 3 / (4 * df - 1))
    se = math.sqrt((n1 + n2) / (n1 * n2) + g * g / (2 * (n1 + n2)))
    return float(g), (float(g - 1.96 * se), float(g + 1.96 * se))


def dz(x: np.ndarray, y: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Mean of the paired differences over their standard deviation, with an approximate CI."""
    d = np.asarray(x) - np.asarray(y)
    n = len(d)
    value = float(np.mean(d) / np.std(d, ddof=1))
    se = math.sqrt(1 / n + value * value / (2 * n))
    return value, (value - 1.96 * se, value + 1.96 * se)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) minus P(X < Y), ties counted as half: the rank-based effect used with Mann-Whitney."""
    from scipy.stats import rankdata

    n1, n2 = len(x), len(y)
    ranks = rankdata(np.concatenate([x, y]))
    u_x = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    return float(2 * u_x / (n1 * n2) - 1)


def rank_biserial_signed(x: np.ndarray, y: np.ndarray) -> float:
    """Signed rank-biserial correlation for paired differences (zeros dropped)."""
    from scipy.stats import rankdata

    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    ranks = rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    return float((w_plus - w_minus) / (w_plus + w_minus))


def cramers_v(chi2: float, n: int, rows: int, cols: int) -> float:
    return math.sqrt(chi2 / (n * (min(rows, cols) - 1)))


def epsilon_squared_kruskal(h: float, n: int) -> float:
    """Epsilon-squared for the Kruskal-Wallis H statistic (Tomczak and Tomczak 2014)."""
    return float(h * (n + 1) / (n * n - 1))


def mde_two_sample(n1: int, n2: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest standardised difference detected with the given power (equal-variance t test)."""
    return float(
        TTestIndPower().solve_power(nobs1=n1, ratio=n2 / n1, alpha=alpha, power=power)
    )


def mde_paired(n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    return float(TTestPower().solve_power(nobs=n, alpha=alpha, power=power))


def size_label(d: float) -> str:
    """Cohen's conventional words for a standardised effect (a rule of thumb, not a verdict)."""
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"
