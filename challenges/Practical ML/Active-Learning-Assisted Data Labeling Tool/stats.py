"""Learning-curve statistics: ALC, labels-to-target, bootstrap CIs, the stopping rule."""

from __future__ import annotations

import numpy as np

THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05)
KS = (2, 3)


def as_float(x) -> np.ndarray:
    """``None`` (how NaN survives JSON) becomes NaN."""
    return np.array(x, dtype=np.float64)


def alc(n, acc) -> float:
    """Area under the learning curve over the label range, normalized: mean accuracy per label spent."""
    n, acc = as_float(n), as_float(acc)
    return float(np.trapezoid(acc, n) / (n[-1] - n[0]))


def labels_to_target(n, acc, target: float) -> float:
    """Labels needed to first reach ``target`` accuracy (linear between rounds); NaN if never."""
    n, acc = as_float(n), as_float(acc)
    hit = np.flatnonzero(acc >= target)
    if len(hit) == 0:
        return float("nan")
    i = int(hit[0])
    if i == 0:
        return float(n[0])
    a0, a1 = acc[i - 1], acc[i]
    return float(n[i - 1] + (target - a0) / (a1 - a0) * (n[i] - n[i - 1]))


def mean_ci(values, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Bootstrap CI of the mean. Pass paired differences to get a paired interval."""
    v = as_float(values)
    if len(v) == 0:
        nan = float("nan")
        return {"est": nan, "lo": nan, "hi": nan, "n": 0}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"est": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": len(v)}


def stop_round(change, threshold: float, k: int):
    """First round whose last ``k`` prediction-change fractions are all below ``threshold``.

    ``change[r]`` is the share of pool predictions that differ between rounds ``r - 1`` and
    ``r`` (Bloodgood & Vijay-Shanker 2009); ``change[0]`` is NaN. Returns ``None`` if never.
    """
    c = as_float(change)
    for r in range(k, len(c)):
        if (c[r - k + 1 : r + 1] < threshold).all():
            return r
    return None


def stop_outcome(curve: dict, threshold: float, k: int) -> dict:
    """Where the rule would stop on one curve and the accuracy that leaves on the table."""
    n, acc = as_float(curve["n"]), as_float(curve["acc"])
    r = stop_round(curve["change"], threshold, k)
    if r is None:
        return {"fired": False, "n": float(n[-1]), "gap": 0.0}
    return {"fired": True, "n": float(n[r]), "gap": float(acc[-1] - acc[r])}


def pick_stop_rule(curves, thresholds=THRESHOLDS, ks=KS, tol: float = 0.01) -> dict:
    """Earliest-stopping (threshold, k) whose mean accuracy gap to the budget end is <= ``tol``.

    Meant for curves scored on validation data from tuning seeds, so the benchmark seeds never
    influence it. If nothing is within tolerance the strictest rule is returned.
    """
    best = None
    for thr in thresholds:
        for k in ks:
            outs = [stop_outcome(c, thr, k) for c in curves]
            cand = {
                "threshold": thr,
                "k": k,
                "labels": float(np.mean([o["n"] for o in outs])),
                "acc_gap": float(np.mean([o["gap"] for o in outs])),
                "tol": tol,
            }
            if cand["acc_gap"] <= tol and (
                best is None or cand["labels"] < best["labels"]
            ):
                best = cand
    if best is None:
        thr, k = min(thresholds), max(ks)
        outs = [stop_outcome(c, thr, k) for c in curves]
        best = {
            "threshold": thr,
            "k": k,
            "labels": float(np.mean([o["n"] for o in outs])),
            "acc_gap": float(np.mean([o["gap"] for o in outs])),
            "tol": tol,
        }
    return best
