"""Alert thresholds calibrated from the baseline's own window-to-window noise.

A fixed rule such as "PSI > 0.2" ignores that every statistic's noise depends on
the window size and the column. Instead: draw many same-size windows from the
**reference** split (never the live one), score each against the training
baseline, and alert above the (1 - alpha) quantile, so about ``alpha`` of
no-drift windows raise a false alarm.

Two ways to draw the null windows:

- ``"iid"``: rows sampled i.i.d. with replacement. Right for stationary,
  independent data, but it has no week-to-week variation of the level, so on a
  real time series (the electricity data) it is far too tight: 28% of
  threshold cells alert on the real reference windows instead of 1%.
- ``"blocks"`` (default): contiguous slices of the reference at random start
  positions, which keep the within-window autocorrelation and the level
  variation between neighbouring weeks. Windows overlap, so the null is only as
  rich as the reference is long.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from drift_monitor.stats import Baseline

WINDOW = 336
ALPHA = 0.01
RULE_OF_THUMB = {"psi": 0.2}


def null_distribution(
    base: Baseline,
    reference: Mapping[str, np.ndarray],
    n_windows: int = 500,
    window: int = WINDOW,
    seed: int = 0,
    mode: str = "blocks",
) -> dict[tuple[str, str], np.ndarray]:
    """``(column, statistic) -> statistic values over no-drift windows``."""
    if mode not in ("iid", "blocks"):
        raise ValueError(f"mode must be 'iid' or 'blocks', got {mode!r}")
    rng = np.random.default_rng(seed)
    n = len(next(iter(reference.values())))
    if mode == "blocks" and n < window:
        raise ValueError(f"reference has {n} rows, fewer than one window of {window}")
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _ in range(n_windows):
        if mode == "iid":
            idx = rng.integers(0, n, window)  # the same rows for every column
        else:
            idx = np.arange(window) + rng.integers(0, n - window + 1)
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
