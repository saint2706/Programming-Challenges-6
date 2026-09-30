"""Synthetic series with *planted* anomalies, so ground truth is exact.

The real NAB data has labelled windows, but a window is a fuzzy region around
an anomaly. Here each scenario also says exactly which samples were tampered
with, which is what makes the failure-mode demonstrations checkable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scenario:
    name: str
    description: str
    values: np.ndarray
    truth: np.ndarray  # boolean, True where the sample was tampered with
    period: int | None  # true seasonal period (None: not seasonal)


def base_series(
    n: int,
    period: int | None = 24,
    *,
    amplitude: float = 10.0,
    noise_sd: float = 1.0,
    trend_per_step: float = 0.0,
    level: float = 50.0,
    rng: np.random.Generator,
) -> np.ndarray:
    """level + linear trend + sinusoidal seasonality (period `period`) + N(0, noise_sd)."""
    t = np.arange(n)
    x = level + trend_per_step * t + rng.normal(0.0, noise_sd, n)
    if period:
        x += amplitude * np.sin(2 * np.pi * t / period)
    return x


def inject_spikes(
    x: np.ndarray,
    positions: np.ndarray,
    size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add +/- `size` at each position (random sign). Returns the truth mask."""
    signs = rng.choice([-1.0, 1.0], len(positions))
    x[positions] += signs * size
    truth = np.zeros(len(x), dtype=bool)
    truth[positions] = True
    return truth


def inject_level_shift(
    x: np.ndarray, start: int, length: int, offset: float
) -> np.ndarray:
    x[start : start + length] += offset
    truth = np.zeros(len(x), dtype=bool)
    truth[start : start + length] = True
    return truth


def inject_flatline(x: np.ndarray, start: int, length: int, level: float) -> np.ndarray:
    """A stuck sensor: the seasonal swing disappears and the value sits at `level`
    (NAB's art_daily_flatmiddle). Every value stays inside the series' normal range."""
    x[start : start + length] = level
    truth = np.zeros(len(x), dtype=bool)
    truth[start : start + length] = True
    return truth


def _spread_positions(
    n: int, count: int, rng: np.random.Generator, margin: int
) -> np.ndarray:
    """`count` distinct positions, well separated, away from both ends."""
    slots = np.linspace(margin, n - margin, count, dtype=int)
    jitter = rng.integers(-3, 4, count)
    return np.clip(slots + jitter, margin, n - margin)


def make_scenarios(seed: int = 0, n: int = 1200, period: int = 24) -> list[Scenario]:
    """The scenario battery used by the benchmark (all deterministic in `seed`)."""
    rng = np.random.default_rng(seed)
    out: list[Scenario] = []

    # 1. Spikes 6 noise-sigmas high on a seasonal swing of +/-10: a spike at the
    #    seasonal midpoint is well inside the series' overall range.
    x = base_series(n, period, rng=rng)
    truth = inject_spikes(x, _spread_positions(n, 10, rng, 100), 6.0, rng)
    out.append(
        Scenario(
            "seasonal spikes",
            "10 spikes of 6 sigma on a +/-10 seasonal swing",
            x,
            truth,
            period,
        )
    )

    # 2. The same, plus a trend that moves the mean by ~40 over the series.
    x = base_series(n, period, trend_per_step=40.0 / n, rng=rng)
    truth = inject_spikes(x, _spread_positions(n, 10, rng, 100), 6.0, rng)
    out.append(
        Scenario(
            "trend + seasonal spikes",
            "10 spikes of 6 sigma on seasonality plus a +40 drift",
            x,
            truth,
            period,
        )
    )

    # 3. Level shift lasting 2.5 cycles.
    x = base_series(n, period, rng=rng)
    start = int(rng.integers(n // 3, n // 2))
    truth = inject_level_shift(x, start, int(2.5 * period), 8.0)
    out.append(
        Scenario(
            "level shift",
            f"+8 offset for {int(2.5 * period)} samples",
            x,
            truth,
            period,
        )
    )

    # 4. Stuck sensor: three cycles of flat line at the series mean.
    x = base_series(n, period, rng=rng)
    start = int(rng.integers(n // 3, n // 2))
    truth = inject_flatline(x, start, 3 * period, 50.0)
    out.append(
        Scenario(
            "flatline (seasonal break)",
            f"seasonal swing replaced by a flat line for {3 * period} samples",
            x,
            truth,
            period,
        )
    )

    # 5. Non-seasonal, 10% of the sample is a cluster of large outliers: the
    #    outliers inflate the standard deviation enough to hide themselves.
    m = 100
    x = rng.normal(0.0, 1.0, m)
    positions = rng.choice(m, 10, replace=False)
    truth = np.zeros(m, dtype=bool)
    x[positions] = rng.normal(8.0, 0.3, 10)
    truth[positions] = True
    out.append(
        Scenario(
            "masking (10% outliers)",
            "100 points N(0,1) with 10 outliers near +8, no seasonality",
            x,
            truth,
            None,
        )
    )
    return out


def clean_noise(n: int, seed: int) -> np.ndarray:
    """i.i.d. N(0, 1): by construction it contains no anomalies."""
    return np.random.default_rng(seed).normal(0.0, 1.0, n)
