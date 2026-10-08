"""Scores for 5-star probability vectors: accuracy and star error, proper scoring rules, and the
calibration estimators (equal-mass ECE, smooth ECE, class-wise ECE, reliability and coverage).

Everything takes ``P [n, 5]`` (rows sum to 1) and ``y [n]`` (stars 1..5). The public functions
validate their inputs; ``bundle`` validates once and calls the raw functions, because the cluster
bootstrap evaluates it thousands of times.

Smooth ECE (Blasiok and Nakkiran, "Smooth ECE: Principled Reliability Diagrams via Kernel
Smoothing", 2023) replaces the arbitrary bins of ECE with a kernel: the residuals ``y - f`` are
smoothed with a Gaussian of bandwidth ``sigma`` reflected at 0 and 1, the integral of the absolute
smoothed residual density is ``smECE_sigma``, and the reported value is the fixed point
``smECE_sigma = sigma``. Here the reflection is exact: the residual mass on a fine grid is
mirrored into a periodic signal and smoothed in the Fourier domain, which sums every reflection
image of the kernel.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.stats import norm

from review_stars.probs import STARS, K

BUNDLE_KEYS = (
    "acc",
    "mae_argmax",
    "mae_expected",
    "rmse_expected",
    "qwk",
    "nll",
    "brier",
    "rps",
    "ece",
    "smece",
    "cw_ece",
)
PROB_FLOOR = (
    1e-12  # NLL clips probabilities here so a hard zero is a large loss, not infinity
)
GRID = 2048  # cells for the smooth-ECE signal


def _validate(P, y):
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y)
    if P.ndim != 2 or P.shape[1] != K:
        raise ValueError(f"probabilities must have shape [n, {K}], not {P.shape}")
    if len(y) != len(P):
        raise ValueError("probabilities and labels must have the same length")
    if len(P) and np.abs(P.sum(axis=1) - 1.0).max() > 1e-6:
        raise ValueError("probability rows must sum to 1")
    if len(y) and (y.min() < 1 or y.max() > K or not np.all(y == np.round(y))):
        raise ValueError(f"star labels must be whole numbers between 1 and {K}")
    return P, y.astype(np.int64)


def _checked(fn):
    @functools.wraps(fn)
    def wrapper(P, y, *args, **kwargs):
        P, y = _validate(P, y)
        return fn(P, y, *args, **kwargs)

    wrapper.raw = fn
    return wrapper


# ---------------------------------------------------------------- accuracy and star error


@_checked
def top_label(P, y):
    """``(confidence, correct)``: the top probability and whether the top star is the true one."""
    pred = P.argmax(axis=1) + 1
    return P.max(axis=1), pred == y


@_checked
def accuracy(P, y):
    return float(np.mean(P.argmax(axis=1) + 1 == y)) if len(y) else float("nan")


@_checked
def mae_argmax(P, y):
    return float(np.abs(P.argmax(axis=1) + 1 - y).mean()) if len(y) else float("nan")


@_checked
def mae_expected(P, y):
    return float(np.abs(P @ STARS - y).mean()) if len(y) else float("nan")


@_checked
def rmse_expected(P, y):
    return float(np.sqrt(((P @ STARS - y) ** 2).mean())) if len(y) else float("nan")


@_checked
def qwk(P, y):
    """Quadratic weighted kappa of the top star; NaN when it is undefined (one star everywhere)."""
    if not len(y):
        return float("nan")
    pred = P.argmax(axis=1)
    O = (
        np.bincount((y - 1) * K + pred, minlength=K * K)
        .reshape(K, K)
        .astype(np.float64)
    )
    E = np.outer(O.sum(axis=1), O.sum(axis=0)) / len(y)
    i = np.arange(K)
    W = (i[:, None] - i[None, :]) ** 2 / (K - 1) ** 2
    den = float((W * E).sum())
    return float("nan") if den == 0 else 1.0 - float((W * O).sum()) / den


# ---------------------------------------------------------------- proper scoring rules


@_checked
def nll(P, y):
    if not len(y):
        return float("nan")
    return float(-np.log(np.clip(P[np.arange(len(y)), y - 1], PROB_FLOOR, None)).mean())


@_checked
def brier(P, y):
    if not len(y):
        return float("nan")
    onehot = np.eye(K)[y - 1]
    return float(((P - onehot) ** 2).sum(axis=1).mean())


@_checked
def rps(P, y):
    """Ranked probability score: the Brier score of the cumulative distribution (ordinal-aware)."""
    if not len(y):
        return float("nan")
    cum_p = np.cumsum(P, axis=1)[:, :-1]
    cum_y = np.cumsum(np.eye(K)[y - 1], axis=1)[:, :-1]
    return float(((cum_p - cum_y) ** 2).sum(axis=1).mean() / (K - 1))


# ---------------------------------------------------------------- calibration estimators


def _equal_mass_bins(conf, correct, bins):
    n = len(conf)
    order = np.argsort(conf, kind="stable")
    c = np.asarray(conf, dtype=np.float64)[order]
    a = np.asarray(correct, dtype=np.float64)[order]
    edges = np.linspace(0, n, min(bins, n) + 1).astype(np.int64)
    # Equal confidences must share a bin: splitting a tie group by row order would let an arbitrary
    # ordering decide the answer (isotonic calibration outputs are piecewise constant, so full of
    # ties). Snap each inner edge back to the first member of its tie group; groups can merge.
    snapped = np.searchsorted(c, c[edges[1:-1]], side="left")
    edges = np.unique(np.concatenate([[0], snapped, [n]]))
    starts = edges[:-1]
    return np.diff(edges), np.add.reduceat(c, starts), np.add.reduceat(a, starts)


def ece_equal_mass(conf, correct, bins: int = 15) -> float:
    """Expected calibration error with bins holding (nearly) equal numbers of reviews."""
    if len(conf) == 0:
        return float("nan")
    _, sum_c, sum_a = _equal_mass_bins(conf, correct, bins)
    return float(np.abs(sum_a - sum_c).sum() / len(conf))


def reliability_curve(conf, correct, bins: int = 15) -> dict:
    """Per equal-mass bin: mean confidence, accuracy and count (what a reliability diagram plots)."""
    if len(conf) == 0:
        return {
            "conf": np.zeros(0),
            "acc": np.zeros(0),
            "n": np.zeros(0, dtype=np.int64),
        }
    counts, sum_c, sum_a = _equal_mass_bins(conf, correct, bins)
    return {"conf": sum_c / counts, "acc": sum_a / counts, "n": counts}


def fixed_bin_counts(conf, correct, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Reviews and correct reviews in ``bins`` equal-width confidence bins over [0, 1].

    Unlike equal-mass bins these have the same edges in every bootstrap replicate, so replicates
    can be compared bin by bin (reliability bands), and the counts are the confidence histogram
    (how sharp the model is)."""
    idx = np.minimum(
        (np.clip(np.asarray(conf, dtype=np.float64), 0.0, 1.0) * bins).astype(np.int64),
        bins - 1,
    )
    n = np.bincount(idx, minlength=bins)
    k = np.bincount(idx, weights=np.asarray(correct, dtype=np.float64), minlength=bins)
    return n, k


@_checked
def top_label_ece(P, y, bins: int = 15) -> float:
    conf, correct = top_label.raw(P, y)
    return ece_equal_mass(conf, correct, bins)


@_checked
def classwise_ece(P, y, bins: int = 15) -> float:
    """Mean over stars of the ECE of that star's probability against whether it was the truth."""
    if not len(y):
        return float("nan")
    return float(np.mean([ece_equal_mass(P[:, k], y == k + 1, bins) for k in range(K)]))


def _smooth_signal(conf, correct, grid: int):
    f = np.clip(np.asarray(conf, dtype=np.float64), 0.0, 1.0)
    residual = np.asarray(correct, dtype=np.float64) - f
    cells = np.minimum((f * grid).astype(np.int64), grid - 1)
    mass = np.bincount(cells, weights=residual, minlength=grid)
    mirrored = np.concatenate(
        [mass, mass[::-1]]
    )  # reflection at 0 and 1 as a periodic signal
    return len(f), np.fft.rfft(mirrored), np.fft.rfftfreq(2 * grid)


def _smece_sigma(n: int, spectrum, freq, sigma: float, grid: int) -> float:
    response = np.exp(
        -2.0 * np.pi**2 * (sigma * grid * freq) ** 2
    )  # Gaussian, std sigma * grid cells
    smoothed = np.fft.irfft(spectrum * response, n=2 * grid)[:grid]
    return float(np.abs(smoothed).sum() / n)


def smooth_ece_at(conf, correct, sigma: float, grid: int = GRID) -> float:
    """``smECE_sigma``: smooth ECE at a fixed bandwidth."""
    if len(conf) == 0:
        return float("nan")
    n, spectrum, freq = _smooth_signal(conf, correct, grid)
    return _smece_sigma(n, spectrum, freq, sigma, grid)


def smooth_ece(conf, correct, grid: int = GRID, sigma_min: float = 1e-3) -> float:
    """Smooth ECE at its own fixed-point bandwidth (``smECE_sigma = sigma``), found by bisection."""
    if len(conf) == 0:
        return float("nan")
    n, spectrum, freq = _smooth_signal(conf, correct, grid)
    lo, hi = sigma_min, 1.0
    if _smece_sigma(n, spectrum, freq, lo, grid) <= lo:
        return _smece_sigma(n, spectrum, freq, lo, grid)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _smece_sigma(n, spectrum, freq, mid, grid) > mid:
            lo = mid
        else:
            hi = mid
    return _smece_sigma(n, spectrum, freq, 0.5 * (lo + hi), grid)


# ---------------------------------------------------------------- coverage


@_checked
def coverage_curve(P, y, levels) -> list[dict]:
    """For each nominal level ``c``: the smallest set of stars with probability mass at least ``c``,
    its empirical coverage and mean size. A calibrated model covers at least ``c`` (discreteness
    makes it overshoot); an overconfident one falls short."""
    order = np.argsort(-P, axis=1, kind="stable")
    cum = np.cumsum(np.take_along_axis(P, order, axis=1), axis=1)
    rank_true = np.argmax(order == (y - 1)[:, None], axis=1)
    out = []
    for c in levels:
        size = np.minimum((cum < c - 1e-12).sum(axis=1) + 1, K)
        out.append(
            {
                "level": float(c),
                "coverage": float(np.mean(rank_true < size))
                if len(y)
                else float("nan"),
                "mean_size": float(size.mean()) if len(y) else float("nan"),
            }
        )
    return out


def interval_coverage(mu, sigma, y, levels) -> list[dict]:
    """Central ``level`` intervals ``mu +- z sigma`` of a Gaussian: empirical coverage and width."""
    mu, sigma, y = (np.asarray(a, dtype=np.float64) for a in (mu, sigma, y))
    out = []
    for c in levels:
        half = norm.ppf(0.5 + c / 2.0) * sigma
        out.append(
            {
                "level": float(c),
                "coverage": float(np.mean(np.abs(y - mu) <= half)),
                "width": float(np.mean(2.0 * half)),
            }
        )
    return out


# ---------------------------------------------------------------- everything the bootstrap needs


def bundle(P, y, bins: int = 15) -> dict[str, float]:
    P, y = _validate(P, y)
    conf, correct = top_label.raw(P, y)
    return {
        "acc": accuracy.raw(P, y),
        "mae_argmax": mae_argmax.raw(P, y),
        "mae_expected": mae_expected.raw(P, y),
        "rmse_expected": rmse_expected.raw(P, y),
        "qwk": qwk.raw(P, y),
        "nll": nll.raw(P, y),
        "brier": brier.raw(P, y),
        "rps": rps.raw(P, y),
        "ece": ece_equal_mass(conf, correct, bins),
        "smece": smooth_ece(conf, correct),
        "cw_ece": classwise_ece.raw(P, y, bins),
    }
