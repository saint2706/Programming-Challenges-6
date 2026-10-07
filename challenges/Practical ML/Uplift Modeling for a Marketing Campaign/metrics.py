"""Uplift metrics on a randomized trial.

A *ranking* of customers by predicted uplift is judged by comparing, inside the top-k,
the outcome rate of treated and control customers. Two curves (``x`` = share targeted):

* uplift curve ``u(k) = (Yt/Nt - Yc/Nc) * n_k / N``: incremental outcomes per customer of
  the *whole population* if the top-k were contacted; ``u(1)`` is the ATE.
* Qini curve ``q(k) = (Yt - Yc * Nt/Nc) / N`` (Radcliffe): incremental gain with the
  control group rescaled to the treated group's size; ``q(1) = ATE * Nt / N``.

Both coefficients below are areas *above the random-targeting line* (the straight line to
the curve's end point), so they are 0 in expectation for a useless ranking.

A ranking is sorted once (``rank``); the bootstrap then reuses the order and varies only
integer row weights (a multinomial resample is exactly a weight vector), so a 200-draw
paired bootstrap costs 200 cumulative sums per scorer, not 200 sorts. Cutoffs where either
arm is still empty have no estimate and are set to 0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ranked:
    order: np.ndarray  # permutation of the original rows, best score first
    t: np.ndarray  # treatment in ranked order
    y: np.ndarray  # outcome in ranked order
    ends: np.ndarray  # index of the last row of each run of tied scores
    arm: np.ndarray | None = None  # per-row arm weights (e.g. IPW) in ranked order


def rank(score, t, y, arm_weights=None) -> Ranked:
    """Sort once. ``arm_weights`` (e.g. ``1 / P(own arm | x)``) correct the treated-vs-control
    comparison when treatment was not assigned with a constant probability; they weight each
    arm's outcome rate, while the share of customers targeted always counts customers."""
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="stable")
    s = score[order]
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    return Ranked(
        order,
        np.asarray(t, dtype=np.float64)[order],
        np.asarray(y, dtype=np.float64)[order],
        ends,
        None
        if arm_weights is None
        else np.asarray(arm_weights, dtype=np.float64)[order],
    )


def _weights(r: Ranked, w):
    """``(count, arm)``: bootstrap multiplicities (customers) and the weights of the arm rates."""
    count = np.ones(len(r.t)) if w is None else np.asarray(w, dtype=np.float64)[r.order]
    return count, count if r.arm is None else count * r.arm


def curve(r: Ranked, w=None):
    """``(x, q, u)``, each starting at 0: share targeted, Qini gain, incremental outcomes."""
    count, arm = _weights(r, w)
    wt, wc = arm * r.t, arm * (1.0 - r.t)
    nt, nc = np.cumsum(wt)[r.ends], np.cumsum(wc)[r.ends]
    yt, yc = np.cumsum(wt * r.y)[r.ends], np.cumsum(wc * r.y)[r.ends]
    n_k = np.cumsum(count)[r.ends]
    n = count.sum()
    both = (nt > 0) & (nc > 0)
    safe_nt, safe_nc = np.where(both, nt, 1.0), np.where(both, nc, 1.0)
    q = np.where(both, yt - yc * safe_nt / safe_nc, 0.0)
    u = np.where(both, (yt / safe_nt - yc / safe_nc) * n_k, 0.0)
    return np.r_[0.0, n_k / n], np.r_[0.0, q / n], np.r_[0.0, u / n]


def qini_coefficient(r: Ranked, w=None) -> float:
    x, q, _ = curve(r, w)
    return float(np.trapezoid(q - x * q[-1], x))


def auuc(r: Ranked, w=None) -> float:
    x, _, u = curve(r, w)
    return float(np.trapezoid(u - x * u[-1], x))


def ate(r: Ranked, w=None) -> float:
    return float(curve(r, w)[2][-1])


def uplift_at(r: Ranked, frac: float, w=None) -> float:
    """Observed treated-minus-control outcome rate inside the top ``frac`` of the ranking."""
    count, arm = _weights(r, w)
    cut = int(np.searchsorted(np.cumsum(count), frac * count.sum()))
    sl = slice(0, cut + 1)
    wt, wc = arm[sl] * r.t[sl], arm[sl] * (1.0 - r.t[sl])
    if wt.sum() == 0 or wc.sum() == 0:
        return 0.0
    return float((wt * r.y[sl]).sum() / wt.sum() - (wc * r.y[sl]).sum() / wc.sum())


def summarize(r: Ranked, w=None, fracs=(0.1, 0.2, 0.3, 0.5)) -> dict[str, float]:
    x, q, u = curve(r, w)
    out = {
        "qini": float(np.trapezoid(q - x * q[-1], x)),
        "auuc": float(np.trapezoid(u - x * u[-1], x)),
        "ate": float(u[-1]),
    }
    for f in fracs:
        pct = round(f * 100)
        out[f"uplift@{pct}"] = uplift_at(r, f, w)
        out[f"incremental@{pct}"] = float(np.interp(f, x, u))
    return out


@dataclass(frozen=True)
class Boot:
    point: dict[str, dict[str, float]]  # scorer -> stat -> estimate on the full sample
    draws: dict[str, dict[str, np.ndarray]]  # scorer -> stat -> bootstrap draws


def bootstrap(
    ranked: dict[str, Ranked],
    n_boot: int = 200,
    seed: int = 0,
    stat: Callable = summarize,
) -> Boot:
    """Paired bootstrap: every scorer sees the same resampled rows in each draw."""
    n = len(next(iter(ranked.values())).t)
    rng = np.random.default_rng(seed)
    point = {k: stat(r) for k, r in ranked.items()}
    draws = {k: {s: np.empty(n_boot) for s in v} for k, v in point.items()}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        for k, r in ranked.items():
            for s, v in stat(r, w).items():
                draws[k][s][b] = v
    return Boot(point, draws)


def _ci(est: float, d: np.ndarray, level: float) -> dict[str, float]:
    lo, hi = np.quantile(d, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"est": float(est), "lo": float(lo), "hi": float(hi)}


def interval(boot: Boot, name: str, stat: str, level: float = 0.95) -> dict[str, float]:
    return _ci(boot.point[name][stat], boot.draws[name][stat], level)


def diff_interval(
    boot: Boot, a: str, b: str, stat: str, level: float = 0.95
) -> dict[str, float]:
    """Paired difference ``a - b`` of one statistic."""
    return _ci(
        boot.point[a][stat] - boot.point[b][stat],
        boot.draws[a][stat] - boot.draws[b][stat],
        level,
    )


def decile_calibration(score, t, y, n_bins: int = 10, arm_weights=None) -> list[dict]:
    """Per predicted-uplift bin (1 = highest): observed uplift (difference in means) and its SE.

    With ``arm_weights`` (IPW) each arm's rate is a weighted mean and the SE is the usual
    linearization ``sqrt(sum w^2 (y - p)^2) / sum w``, which equals ``sqrt(p (1 - p) / n)`` at unit weights.
    """
    score, t, y = (np.asarray(a) for a in (score, t, y))
    w = np.ones(len(y)) if arm_weights is None else np.asarray(arm_weights, float)
    rows = []
    order = np.argsort(-score, kind="stable")

    def rate(ix):
        if len(ix) == 0:
            return float("nan"), 0.0
        wm, ym = w[ix], y[ix]
        p = float((wm * ym).sum() / wm.sum())
        return p, float((wm**2 * (ym - p) ** 2).sum() / wm.sum() ** 2)

    for i, idx in enumerate(np.array_split(order, n_bins), 1):
        (p1, v1), (p0, v0) = rate(idx[t[idx] == 1]), rate(idx[t[idx] == 0])
        rows.append(
            {
                "decile": i,
                "n": len(idx),
                "mean_pred": float(score[idx].mean()),
                "observed": p1 - p0,
                "se": float(np.sqrt(v1 + v0)),
            }
        )
    return rows
