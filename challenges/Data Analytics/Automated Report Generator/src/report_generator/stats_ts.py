"""Time-series statistics the findings rest on: trend test, decomposition, change points.

* ``mann_kendall``: the Mann-Kendall trend test with the Hamed-Rao (1998) variance correction, which
  keeps serially correlated series (nearly all business series) from looking significant by luck.
  Slope and interval are Theil-Sen, so one outlier cannot tilt them.
* ``stl``: STL decomposition (robust), with Wang-Smith-Hyndman strength measures.
* ``pelt``: the PELT algorithm for changes in the mean (L2 cost, BIC-style penalty), plus the O(n^2)
  dynamic program it must agree with, used by the tests as an oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, pairwise

import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import STL


@dataclass(frozen=True)
class MannKendall:
    s: float
    var_s: float
    z: float
    p: float  # the larger of p_hr and p_bootstrap: a trend must survive both
    p_hr: float  # Hamed-Rao corrected Mann-Kendall
    p_bootstrap: (
        float | None
    )  # circular block bootstrap of the Theil-Sen slope (None if p_hr >= 0.05)
    tau: float
    slope: float  # Theil-Sen, per step
    intercept: float
    slope_low: float
    slope_high: float


def bootstrap_slope_p(
    y: np.ndarray, slope: float, draws: int = 1999, seed: int = 0
) -> float:
    """p-value for a trend that respects serial dependence, by a circular block bootstrap.

    The line is removed, the residuals are resampled in blocks long enough to keep their
    autocorrelation (about twice its time scale), and the Theil-Sen slope of each resample says what
    a slope looks like when there is no trend. Hamed-Rao's correction alone still lets about one
    smooth (AR(1), phi = 0.85) series in seven look trending at p < 0.01; this brings it to about one
    in fourteen at the same power, and the report requires both tests.
    """
    n = y.size
    t = np.arange(n, dtype=float)
    resid = y - slope * t
    resid = resid - np.median(resid)
    d = resid - resid.mean()
    r1 = float(np.clip((d[:-1] @ d[1:]) / max(d @ d, 1e-12), 0.0, 0.95))
    block = int(min(n // 3, max(3, round(2 * (1 + r1) / (1 - r1)))))
    n_blocks = -(-n // block)
    i, j = np.triu_indices(n, 1)
    rng = np.random.default_rng(seed)
    exceed = done = 0
    while done < draws:
        m = min(100, draws - done)
        starts = rng.integers(0, n, (m, n_blocks))
        idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(m, -1)[:, :n]
        r = resid[idx]
        slopes = np.median((r[:, j] - r[:, i]) / (j - i), axis=1)
        exceed += int(np.sum(np.abs(slopes) >= abs(slope) - 1e-12))
        done += m
    return (exceed + 1) / (draws + 1)


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x - x.mean()
    denom = float(x @ x)
    return np.array([float(x[: x.size - k] @ x[k:]) / denom for k in range(nlags + 1)])


def _tie_variance(x: np.ndarray) -> float:
    n = x.size
    _, counts = np.unique(x, return_counts=True)
    ties = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    return (n * (n - 1) * (2 * n + 5) - ties) / 18.0


def mann_kendall(y: np.ndarray, alpha: float = 0.05) -> MannKendall:
    """Two-sided Mann-Kendall test for a monotone trend against time steps 0..n-1."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 8:
        raise ValueError("need at least 8 points for a trend test")
    diff = np.sign(y[None, :] - y[:, None])
    s = float(np.triu(diff, 1).sum())
    var_s = _tie_variance(y)
    t = np.arange(n, dtype=float)
    slope, intercept, lo, hi = stats.theilslopes(y, t, alpha=1 - alpha)
    # Hamed-Rao: autocorrelations of the ranks of the detrended series, significant lags only
    ranks = stats.rankdata(y - slope * (t + 1))
    acf = _acf(ranks, n - 1)
    bound = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    sni = 0.0
    for k in range(1, n):
        if abs(acf[k]) > bound:
            sni += (n - k) * (n - k - 1) * (n - k - 2) * acf[k]
    var_s *= 1 + 2 / (n * (n - 1) * (n - 2)) * sni
    var_s = max(var_s, 1e-12)
    z = 0.0 if s == 0 else (s - np.sign(s)) / np.sqrt(var_s)
    p_hr = float(2 * stats.norm.sf(abs(z)))
    p_boot = bootstrap_slope_p(y, slope) if p_hr < 0.05 else None
    p = p_hr if p_boot is None else max(p_hr, p_boot)
    tau = s / (0.5 * n * (n - 1))
    return MannKendall(
        s, var_s, float(z), p, p_hr, p_boot, tau, slope, intercept, lo, hi
    )


@dataclass(frozen=True)
class Decomposition:
    trend: np.ndarray
    seasonal: np.ndarray
    resid: np.ndarray
    seasonal_strength: float
    trend_strength: float
    period: int


def stl_kwargs(n: int, period: int) -> dict:
    """STL with a fixed (periodic) seasonal pattern, like R's ``s.window = "periodic"``.

    With the default seasonal window each phase is smoothed over only a handful of points; with
    three or four cycles that interpolates the noise (a fit with half its residuals exactly zero was
    seen). Robust reweighting is used only from five cycles up: with three points per phase there
    is no telling which one is the outlier, and the fit lets a collapsed week into the season.
    """
    return {"seasonal": 2 * n + 1, "seasonal_deg": 0, "robust": n // period >= 5}


def stl(y: np.ndarray, period: int) -> Decomposition:
    """STL (LOESS) decomposition and Hyndman's strength of seasonality and trend.

    ``F_s = max(0, 1 - Var(R) / Var(S + R))``: 0 means no seasonal pattern, 1 means the
    remainder is negligible next to it. ``F_t`` is the same with the trend in place of the season.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 2 * period:
        raise ValueError("need at least two full periods")
    res = STL(y, period=period, **stl_kwargs(y.size, period)).fit()
    seasonal, trend, resid = (
        np.asarray(a) for a in (res.seasonal, res.trend, res.resid)
    )
    fs = max(0.0, 1 - np.var(resid) / np.var(seasonal + resid))
    ft = max(0.0, 1 - np.var(resid) / np.var(trend + resid))
    return Decomposition(trend, seasonal, resid, float(fs), float(ft), period)


def robust_sigma(y: np.ndarray) -> float:
    """Noise level from first differences (immune to level shifts and trend), scaled to a sd."""
    d = np.diff(np.asarray(y, dtype=float))
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def _cost_table(y: np.ndarray):
    c1 = np.concatenate([[0.0], np.cumsum(y)])
    c2 = np.concatenate([[0.0], np.cumsum(y * y)])

    def cost(
        a: int, b: int
    ) -> float:  # sum of squared deviations from the mean of y[a:b]
        m = b - a
        total = c1[b] - c1[a]
        return float(c2[b] - c2[a] - total * total / m)

    return cost


def pelt(y: np.ndarray, penalty: float, min_size: int = 3) -> list[int]:
    """Optimal change points in the mean by PELT; returns the start index of each new segment."""
    y = np.asarray(y, dtype=float)
    n = y.size
    cost = _cost_table(y)
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    candidates = [0]
    for t in range(min_size, n + 1):
        options = [
            (best[s] + cost(s, t) + penalty, s) for s in candidates if t - s >= min_size
        ]
        if not options:
            continue
        best[t], last[t] = min(options)
        # prune candidates that can never be optimal again (K = 0 for the L2 cost)
        candidates = [s for s in candidates if best[s] + cost(s, t) <= best[t]]
        if t - min_size + 1 >= min_size:
            candidates.append(t - min_size + 1)
    cuts, t = [], n
    while t > 0:
        s = int(last[t])
        if s > 0:
            cuts.append(s)
        t = s
    return sorted(cuts)


def pelt_exact(y: np.ndarray, penalty: float, min_size: int = 3) -> list[int]:
    """The same optimum by plain O(n^2) dynamic programming (the oracle for ``pelt``)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    cost = _cost_table(y)
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    for t in range(min_size, n + 1):
        for s in range(t - min_size + 1):
            if np.isfinite(best[s]):
                value = best[s] + cost(s, t) + penalty
                if value < best[t]:
                    best[t], last[t] = value, s
    cuts, t = [], n
    while t > 0:
        s = int(last[t])
        if s > 0:
            cuts.append(s)
        t = s
    return sorted(cuts)


def segment_sse(y: np.ndarray, cuts: list[int]) -> float:
    cost = _cost_table(np.asarray(y, dtype=float))
    edges = [0, *cuts, len(y)]
    return sum(cost(a, b) for a, b in pairwise(edges))


def block_size(period: int) -> int:
    """How many neighbouring buckets are averaged before cycles are compared (52 weeks -> 4)."""
    return 4 if period == 52 else 1


@dataclass(frozen=True)
class Repeatability:
    rho: float  # mean Spearman correlation between the cycles' patterns of change
    p: float  # permutation p-value for that mean
    cycles: int


def repeatability(
    y: np.ndarray, period: int, permutations: int = 999, seed: int = 0
) -> Repeatability | None:
    """Does the same within-cycle pattern show up in every cycle?

    The series is cut into whole cycles and first-differenced *within* each cycle (so a drifting
    level, however wandering, cannot fake a pattern); the cycles' patterns of change are
    rank-correlated pairwise. A real seasonal pattern repeats, noise does not. The p-value shuffles
    each cycle's changes at random and asks how often the mean pairwise correlation is as high: it
    needs no assumption about the size of the noise or the number of cycles, and a stricter
    standard applies automatically when there are few cycles to compare.
    """
    y = np.asarray(y, dtype=float)
    block = block_size(period)
    if (
        block > 1
    ):  # long cycles: average neighbouring buckets (52 weeks -> 13 blocks) to lift the signal
        y = y[: y.size // block * block].reshape(-1, block).mean(axis=1)
        period //= block
    cycles = y.size // period
    if cycles < 2:
        return None
    d = np.diff(
        y[: cycles * period].reshape(cycles, period), axis=1
    )  # changes within each cycle
    ranks = stats.rankdata(d, axis=1)
    pairs = list(combinations(range(cycles), 2))

    def mean_corr(r: np.ndarray) -> np.ndarray:  # r: (..., cycles, steps) -> (...,)
        z = r - r.mean(axis=-1, keepdims=True)
        z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
        return np.mean(
            [np.sum(z[..., i, :] * z[..., j, :], axis=-1) for i, j in pairs], axis=0
        )

    observed = float(mean_corr(ranks))
    rng = np.random.default_rng(seed)
    shuffled = rng.permuted(
        np.broadcast_to(ranks, (permutations, *ranks.shape)).copy(), axis=2
    )
    null = mean_corr(shuffled)
    p = (1 + int(np.sum(null >= observed - 1e-12))) / (permutations + 1)
    return Repeatability(observed, float(p), cycles)


def periodic_decompose(y: np.ndarray, period: int) -> Decomposition:
    """Fixed seasonal pattern around a robust straight trend, for two or three cycles of history.

    STL needs several cycles to tell a season from noise; with two it fits the noise. Here the
    pattern is the cycle-average of the Theil-Sen-detrended values, lightly smoothed round the cycle
    (5 points for a long period), and the trend is the Theil-Sen line.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    t = np.arange(n, dtype=float)
    slope, intercept, _, _ = stats.theilslopes(y, t)
    trend = intercept + slope * t
    detr = y - trend
    prof = np.array([detr[k::period].mean() for k in range(period)])
    if period >= 24:
        pad = np.concatenate([prof[-2:], prof, prof[:2]])
        prof = np.convolve(pad, np.ones(5) / 5, mode="valid")
    prof = prof - prof.mean()
    seasonal = np.tile(prof, n // period + 1)[:n]
    resid = y - trend - seasonal
    fs = max(0.0, 1 - np.var(resid) / np.var(seasonal + resid))
    ft = max(0.0, 1 - np.var(resid) / np.var(trend - trend.mean() + resid))
    return Decomposition(trend, seasonal, resid, float(fs), float(ft), period)
