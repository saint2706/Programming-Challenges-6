"""Alert thresholds calibrated from the baseline's own window-to-window noise.

A fixed rule such as "PSI > 0.2" ignores that every statistic's noise depends on
the window size and the column. Instead: draw many same-size windows from the
**reference** split (never the live one), score each against the training
baseline, and alert above the (1 - alpha) quantile, so about ``alpha`` of
no-drift windows raise a false alarm.

The null windows are sampled i.i.d. with replacement. That ignores the serial
correlation of a real time series, so real consecutive windows can be noisier
than the null; the pipeline therefore also reports the false-alarm rate actually
seen on real windows.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from stats import Baseline

WINDOW = 336
ALPHA = 0.01
RULE_OF_THUMB = {"psi": 0.2}


def null_distribution(
    base: Baseline,
    reference: Mapping[str, np.ndarray],
    n_windows: int = 500,
    window: int = WINDOW,
    seed: int = 0,
) -> dict[tuple[str, str], np.ndarray]:
    """``(column, statistic) -> statistic values over no-drift windows``."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(reference.values())))
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _ in range(n_windows):
        idx = rng.integers(0, n, window)  # the same rows for every column
        for column, values in reference.items():
            for stat, value in base.window_stats(
                column, np.asarray(values)[idx]
            ).items():
                out[(column, stat)].append(value)
    return {k: np.asarray(v) for k, v in out.items()}


def calibrate(
    null: Mapping[tuple[str, str], np.ndarray], alpha: float = ALPHA
) -> dict[tuple[str, str], float]:
    """The (1 - alpha) quantile of each null distribution."""
    return {key: float(np.quantile(values, 1 - alpha)) for key, values in null.items()}
