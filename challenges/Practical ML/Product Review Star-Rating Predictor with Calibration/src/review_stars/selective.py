"""Selective prediction: abstain on low-confidence reviews and see what that buys.

A risk-coverage curve keeps the most confident fraction ``coverage`` of reviews and reports the
average loss (error rate, or error in stars) on them. Its area (AURC) is the single-number
summary, lower is better. Confidences from isotonic calibration are piecewise constant, so every
quantity here treats tied confidences as one group (the expected value over their order) and
cannot depend on row order.

What calibration buys is not a better ranking (a monotone recalibration barely moves it) but a
confidence that *means* something: "keep reviews the model is 80% sure of" gives about 80%
accuracy only if the confidence is calibrated, with no labeled data needed to set the threshold.
``transfer`` shows the other practical route: choose the threshold on labeled calibration data and
report what it delivers elsewhere.
"""

from __future__ import annotations

import numpy as np


def risk_coverage(conf, loss) -> dict:
    """Coverage and risk at each distinct confidence level, and the tie-averaged AURC."""
    conf, loss = np.asarray(conf, dtype=np.float64), np.asarray(loss, dtype=np.float64)
    n = len(conf)
    if n == 0:
        return {"coverage": np.zeros(0), "risk": np.zeros(0), "aurc": float("nan")}
    order = np.argsort(-conf, kind="stable")
    c, ell = conf[order], loss[order]
    ends = np.flatnonzero(np.r_[c[1:] != c[:-1], True])
    starts = np.r_[0, ends[:-1] + 1]
    cum = np.cumsum(ell)
    group_sum = cum[ends] - np.r_[0.0, cum[ends[:-1]]]
    sizes = ends - starts + 1
    per_review = np.repeat(
        group_sum / sizes, sizes
    )  # random order inside a tie: its mean
    aurc = float(np.mean(np.cumsum(per_review) / np.arange(1, n + 1)))
    return {"coverage": (ends + 1) / n, "risk": cum[ends] / (ends + 1), "aurc": aurc}


def at_threshold(conf, loss, tau: float) -> dict:
    """Keep reviews with confidence >= ``tau``: the coverage and the risk on those."""
    conf, loss = np.asarray(conf, dtype=np.float64), np.asarray(loss, dtype=np.float64)
    kept = conf >= tau
    return {
        "tau": float(tau),
        "coverage": float(kept.mean()) if len(conf) else float("nan"),
        "risk": float(loss[kept].mean()) if kept.any() else float("nan"),
    }


def choose_threshold(conf, loss, target_risk: float) -> float:
    """The lowest confidence threshold (most coverage) whose kept reviews have risk <= target;
    ``inf`` (abstain on everything) if none does."""
    rc = risk_coverage(conf, loss)
    conf = np.asarray(conf, dtype=np.float64)
    if len(rc["risk"]) == 0:
        return float("inf")
    levels = np.unique(conf)[::-1]  # distinct confidences, descending: aligned with rc
    ok = np.flatnonzero(rc["risk"] <= target_risk + 1e-12)
    return float(levels[ok.max()]) if len(ok) else float("inf")


def transfer(conf_cal, loss_cal, conf_test, loss_test, targets) -> list[dict]:
    """For each risk target: the threshold picked on calibration data and its effect on both."""
    out = []
    for target in targets:
        tau = choose_threshold(conf_cal, loss_cal, target)
        out.append(
            {
                "target_risk": float(target),
                "tau": tau,
                "cal": at_threshold(conf_cal, loss_cal, tau),
                "test": at_threshold(conf_test, loss_test, tau),
            }
        )
    return out


def _downsample(rc: dict, n_points: int) -> dict:
    k = len(rc["coverage"])
    keep = (
        np.unique(np.linspace(0, k - 1, min(n_points, k)).round().astype(int))
        if k
        else []
    )
    return {
        "coverage": [float(rc["coverage"][i]) for i in keep],
        "risk": [float(rc["risk"][i]) for i in keep],
    }


def summarize(P, y, n_points: int = 50) -> dict:
    """AURC and a bounded risk-coverage curve for the 0/1 error and for the error in stars."""
    P, y = np.asarray(P, dtype=np.float64), np.asarray(y)
    conf, pred = P.max(axis=1), P.argmax(axis=1) + 1
    out = {}
    for name, loss in (
        ("error", (pred != y).astype(float)),
        ("mae", np.abs(pred - y).astype(float)),
    ):
        rc = risk_coverage(conf, loss)
        out[name] = {"aurc": rc["aurc"], "curve": _downsample(rc, n_points)}
    return out
