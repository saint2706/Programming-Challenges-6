"""Metrics and the operating-point rules for a fraud score.

``y`` is 0/1 (1 = fraud), ``s`` a score where higher means more likely fraud (a probability for the
calibrated models), ``amount`` the transaction amount. A transaction is *flagged* when a rule says
so; every rule here returns a boolean mask or a threshold with the convention "flag when
``s >= threshold``". A threshold of ``None`` means "flag nothing" (or "cannot be reached").
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

PRECISIONS = (0.5, 0.8, 0.9, 0.95)
RECALLS = (0.6, 0.7, 0.8, 0.9)
REVIEW_COSTS = (1.0, 5.0, 25.0, 100.0)
CAPACITIES = (1, 3, 10)  # reviews per hour
BINS = (0.0, 1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0 + 1e-9)


def ranking_scores(y: np.ndarray, s: np.ndarray) -> dict:
    """PR-AUC (average precision) and ROC-AUC, for any score."""
    y = np.asarray(y).astype(int)
    return {
        "n": len(y),
        "frauds": int(y.sum()),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
    }


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    """``ranking_scores`` plus log-loss and Brier, which need ``p`` to be a probability."""
    y = np.asarray(y).astype(int)
    probs = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return {
        **ranking_scores(y, p),
        "log_loss": float(log_loss(y, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probs)),
    }


def reliability(y: np.ndarray, p: np.ndarray, edges=BINS) -> list[dict]:
    """Mean score against observed fraud rate in fixed probability bins (rare events need log-ish bins)."""
    rows = []
    for lo, hi in itertools.pairwise(edges):
        mask = (p >= lo) & (p < hi)
        if mask.any():
            rows.append(
                {
                    "lo": lo,
                    "hi": min(hi, 1.0),
                    "n": int(mask.sum()),
                    "mean_p": float(p[mask].mean()),
                    "rate": float(y[mask].mean()),
                    "frauds": int(y[mask].sum()),
                }
            )
    return rows


def confusion(y: np.ndarray, flag: np.ndarray) -> dict:
    """Counts, precision, recall and alert rate of a flagging decision."""
    y = np.asarray(y).astype(bool)
    flag = np.asarray(flag).astype(bool)
    tp, fp = int((y & flag).sum()), int((~y & flag).sum())
    fn, tn = int((y & ~flag).sum()), int((~y & ~flag).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "flagged": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "alerts_per_10k": 1e4 * (tp + fp) / len(y),
    }


def curve(y: np.ndarray, s: np.ndarray):
    """Thresholds (descending) and the flagged count and true positives at each, ties collapsed."""
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], np.asarray(y)[order].astype(int)
    ends = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    return ss[ends], ends + 1, np.cumsum(yy)[ends], order, ends


def threshold_for_precision(
    y: np.ndarray, s: np.ndarray, target: float
) -> float | None:
    """The lowest threshold whose precision on ``(y, s)`` is at least ``target`` (most recall).

    Precision at the very top of a ranking is a handful of rows and swings wildly, so on a small
    window this picks the deepest point that still clears the bar, not the first.
    """
    thr, flagged, tp, _, _ = curve(y, s)
    ok = np.flatnonzero(tp / flagged >= target)
    return float(thr[ok[-1]]) if len(ok) else None


def threshold_for_recall(y: np.ndarray, s: np.ndarray, target: float) -> float | None:
    """The highest threshold whose recall on ``(y, s)`` is at least ``target`` (most precision)."""
    thr, _, tp, _, _ = curve(y, s)
    total = int(np.asarray(y).sum())
    if total == 0:
        return None
    reach = np.flatnonzero(tp / total >= target)
    return float(thr[reach[0]]) if len(reach) else None


def policy_cost(
    y: np.ndarray,
    amount: np.ndarray,
    flag: np.ndarray,
    review_cost: float,
    friction: float = 0.0,
) -> dict:
    """What a flagging decision costs: missed fraud amounts + one review per flag (+ friction per false alarm).

    A flagged fraud is assumed stopped in full, so its amount is not lost (an upper bound on the
    benefit; a real review stops only some of it).
    """
    y = np.asarray(y).astype(bool)
    flag = np.asarray(flag).astype(bool)
    missed = float(np.asarray(amount)[y & ~flag].sum())
    reviews = review_cost * int(flag.sum())
    annoyance = friction * int((~y & flag).sum())
    return {
        "total": missed + reviews + annoyance,
        "missed_loss": missed,
        "review_cost": reviews,
        "friction_cost": annoyance,
        "caught_amount": float(np.asarray(amount)[y & flag].sum()),
    }


def trivial_costs(
    y: np.ndarray, amount: np.ndarray, review_cost: float, friction: float = 0.0
) -> dict:
    """Flag nothing, flag everything, and the oracle that flags exactly the frauds worth a review.

    The oracle skips a fraud smaller than the review cost: looking at it costs more than losing it.
    """
    y = np.asarray(y).astype(bool)
    worth = y & (np.asarray(amount) > review_cost)
    return {
        "flag_nothing": policy_cost(
            y, amount, np.zeros(len(y), bool), review_cost, friction
        )["total"],
        "flag_everything": policy_cost(
            y, amount, np.ones(len(y), bool), review_cost, friction
        )["total"],
        "oracle": policy_cost(y, amount, worth, review_cost, friction)["total"],
    }


def best_cost_threshold(
    y: np.ndarray,
    amount: np.ndarray,
    s: np.ndarray,
    review_cost: float,
    friction: float = 0.0,
) -> float | None:
    """The threshold minimising ``policy_cost`` exactly (every distinct score tried); None = flag nothing."""
    y = np.asarray(y).astype(int)
    thr, flagged, tp, order, ends = curve(y, s)
    fraud_amount = np.where(y == 1, amount, 0.0)
    caught = np.cumsum(fraud_amount[order])[ends]
    cost = (
        fraud_amount.sum() - caught + review_cost * flagged + friction * (flagged - tp)
    )
    best = int(np.argmin(cost))
    return float(thr[best]) if cost[best] < fraud_amount.sum() else None


def amount_aware_flag(
    p: np.ndarray, amount: np.ndarray, review_cost: float, friction: float = 0.0
) -> np.ndarray:
    """Flag when the expected loss avoided beats the cost of looking: ``p * amount > c + (1 - p) * f``.

    This is the cost-optimal rule when ``p`` is a calibrated probability: a large amount deserves a
    review at a small ``p``, a tiny one does not at a large ``p``. A static threshold cannot do that.
    """
    return (
        np.asarray(p) * np.asarray(amount)
        > review_cost + (1 - np.asarray(p)) * friction
    )


def capacity_flag(score: np.ndarray, hour: np.ndarray, per_hour: int) -> np.ndarray:
    """Flag the ``per_hour`` highest-scored transactions of every hour (analysts have a fixed number of reviews)."""
    order = np.lexsort((-np.asarray(score), np.asarray(hour)))
    hours = np.asarray(hour)[order]
    starts = np.flatnonzero(np.r_[True, hours[1:] != hours[:-1]])
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    flag = np.zeros(len(order), dtype=bool)
    flag[order[rank < per_hour]] = True
    return flag


def pr_points(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The precision-recall curve reduced to the points that define it, ready to plot.

    ``precision_recall_curve`` returns a point per distinct score, tens of thousands of them, and
    nearly all sit where recall is already high. Thinning them evenly throws away the few dozen
    points near the top, where the curve lives. Average precision depends only on the points where
    recall changes (and the precision just before each change), so those are the ones kept.
    """
    precision, recall, _ = precision_recall_curve(y, s)
    keep = np.zeros(len(recall), dtype=bool)
    step = np.flatnonzero(np.diff(recall) != 0)
    keep[step] = keep[step + 1] = True
    keep[[0, -1]] = True
    return recall[keep], precision[keep]


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    top = np.argsort(-s, kind="stable")[:k]
    return float(np.asarray(y)[top].mean())


def recall_at_precision(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """Best recall over all thresholds whose precision is at least ``target`` (0 when none is)."""
    _, flagged, tp, _, _ = curve(y, s)
    ok = tp / flagged >= target
    return float(tp[ok].max() / max(int(np.sum(y)), 1)) if ok.any() else 0.0


def bootstrap_ap(
    y: np.ndarray, s: np.ndarray, n_boot: int = 300, seed: int = 0
) -> tuple[float, float]:
    """95% interval for PR-AUC from resampling frauds and legitimate rows separately."""
    rng = np.random.default_rng(seed)
    pos, neg = s[y == 1], s[y == 0]
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    stats = [
        average_precision_score(
            labels,
            np.r_[
                pos[rng.integers(0, len(pos), len(pos))],
                neg[rng.integers(0, len(neg), len(neg))],
            ],
        )
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_decision(
    y: np.ndarray, flag: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict[str, tuple[float | None, float | None]]:
    """95% intervals for the precision and recall of a *fixed* decision, in closed form.

    Resampling the frauds with replacement makes the caught count binomial in (frauds, recall), and
    resampling the legitimate rows makes the false-alarm count binomial in (legitimate, false-alarm
    rate); both are drawn directly instead of resampling 90,000 rows a thousand times.
    """
    rng = np.random.default_rng(seed)
    c = confusion(y, flag)
    pos, neg = c["tp"] + c["fn"], c["fp"] + c["tn"]
    tp = rng.binomial(pos, c["tp"] / pos, n_boot)
    fp = rng.binomial(neg, c["fp"] / neg, n_boot)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
    recall = tp / pos
    out = {}
    for name, arr in (("precision", precision), ("recall", recall)):
        if np.isnan(arr).all():
            out[name] = (None, None)
        else:
            lo, hi = np.nanpercentile(arr, [2.5, 97.5])
            out[name] = (float(lo), float(hi))
    return out


def bootstrap_cost(
    y: np.ndarray,
    amount: np.ndarray,
    flag: np.ndarray,
    review_cost: float,
    friction: float = 0.0,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% interval for ``policy_cost`` of a fixed decision (frauds resampled as (amount, flagged) pairs)."""
    rng = np.random.default_rng(seed)
    y, flag, amount = (
        np.asarray(y).astype(bool),
        np.asarray(flag).astype(bool),
        np.asarray(amount),
    )
    fa, ff = amount[y], flag[y]
    neg, fp = int((~y).sum()), int((~y & flag).sum())
    totals = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(fa), len(fa))
        false_alarms = rng.binomial(neg, fp / neg)
        flagged_frauds = ff[pick].sum()
        totals[i] = (
            fa[pick][~ff[pick]].sum()
            + review_cost * (flagged_frauds + false_alarms)
            + friction * false_alarms
        )
    lo, hi = np.percentile(totals, [2.5, 97.5])
    return float(lo), float(hi)


def threshold_stability(
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    target: float,
    n_boot: int = 300,
    seed: int = 0,
) -> dict:
    """How much the threshold for a precision target moves if the validation window had been drawn again.

    Each bootstrap resample of the validation window gives a threshold; applying it to the test
    window gives the precision and recall the team would then have got. The spread is the price of
    choosing an operating point on ~90 frauds.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y_val == 1), np.flatnonzero(y_val == 0)
    srt = np.sort(s_test)
    test_pos = int(np.sum(y_test))
    order = np.argsort(s_test)
    ranked_y = np.asarray(y_test)[order]
    tail_tp = np.cumsum(ranked_y[::-1])[::-1]  # frauds at or above each position
    rows = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        thr = threshold_for_precision(y_val[idx], s_val[idx], target)
        if thr is None:
            continue
        first = int(np.searchsorted(srt, thr, side="left"))
        flagged = len(srt) - first
        tp = int(tail_tp[first]) if first < len(srt) else 0
        rows.append((thr, tp / flagged if flagged else np.nan, tp / test_pos, flagged))
    arr = np.array(rows, dtype=float)
    if len(arr) == 0:
        return {"target": target, "resamples": 0}
    out = {"target": target, "resamples": len(arr)}
    for i, name in enumerate(
        ("threshold", "test_precision", "test_recall", "test_flagged")
    ):
        col = arr[:, i]
        lo, med, hi = np.nanpercentile(col, [5, 50, 95])
        out[name] = {"p5": float(lo), "median": float(med), "p95": float(hi)}
    return out


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index of ``actual`` against ``expected`` on the expected deciles."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
