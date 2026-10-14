"""Error metrics for price estimates, with bootstrap intervals over subjects."""

from __future__ import annotations

import numpy as np


def signed_error(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """(estimate - sale price) / sale price."""
    return (pred - actual) / actual


def summarize(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """MdAPE, share within 10% and 20%, and the median signed error (bias)."""
    err = signed_error(pred, actual)
    ape = np.abs(err)
    return {
        "n": len(err),
        "mdape": float(np.median(ape)),
        "within10": float((ape <= 0.10).mean()),
        "within20": float((ape <= 0.20).mean()),
        "bias": float(np.median(err)),
    }


def _stat(ape: np.ndarray, name: str) -> np.ndarray:
    """Statistic of each bootstrap row of absolute errors (``ape`` is ``(B, n)`` or ``(n,)``)."""
    if name == "mdape":
        return np.median(ape, axis=-1)
    if name == "within10":
        return (ape <= 0.10).mean(axis=-1)
    if name == "within20":
        return (ape <= 0.20).mean(axis=-1)
    raise ValueError(name)


STATS = ("mdape", "within10", "within20")


def boot_indices(n: int, reps: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(reps, n))


def interval(
    pred: np.ndarray, actual: np.ndarray, idx: np.ndarray, level: float = 0.95
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap interval of each statistic, resampling subjects."""
    ape = np.abs(signed_error(pred, actual))[idx]
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {
        s: tuple(float(v) for v in np.quantile(_stat(ape, s), [lo, hi])) for s in STATS
    }


def paired_difference(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    actual: np.ndarray,
    idx: np.ndarray,
    level: float = 0.95,
) -> dict[str, tuple[float, float, float]]:
    """A minus B for each statistic on the same resampled subjects: (point, low, high).

    Negative MdAPE difference means A is more accurate. Resampling the same subjects for both keeps
    the pairing, so the interval reflects how often A beats B and not how hard the subjects are.
    """
    a = np.abs(signed_error(pred_a, actual))
    b = np.abs(signed_error(pred_b, actual))
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    out = {}
    for s in STATS:
        boot = _stat(a[idx], s) - _stat(b[idx], s)
        point = float(_stat(a, s) - _stat(b, s))
        out[s] = (point, *(float(v) for v in np.quantile(boot, [lo, hi])))
    return out
