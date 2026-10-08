"""What happens to a calibrated model when the reviews change: slices and the recalibration budget.

The recalibration-budget curve answers the practical question after a shift: how many freshly
labeled reviews from the new domain does it take to put a model's confidence right again? Only
the one-parameter temperature is refit (that is all a handful of labels can support), from ``n``
reviews drawn without replacement from a labeled pool, ``draws`` times for each ``n``; the refit
model is scored on a separate, larger test set from the same new domain.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from review_stars import calibrate, metrics
from review_stars.probs import Predictions


def _spread(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}


def recalibration_curve(
    pool: Predictions,
    y_pool,
    test: Predictions,
    y_test,
    ns,
    draws: int,
    seed: int,
    bins: int = 15,
) -> dict:
    y_pool, y_test = np.asarray(y_pool), np.asarray(y_test)
    P0 = test.proba()
    full_T = calibrate.Temperature().fit(pool, y_pool).T
    P_full = test.proba(full_T)
    rows, skipped = [], []
    for n in ns:
        if n > len(y_pool):
            skipped.append(int(n))
            continue
        Ts, eces, nlls = [], [], []
        for d in range(draws):
            idx = np.random.default_rng((seed, int(n), d)).choice(
                len(y_pool), int(n), replace=False
            )
            T = calibrate.Temperature().fit(pool.take(idx), y_pool[idx]).T
            P = test.proba(T)
            Ts.append(T)
            eces.append(metrics.top_label_ece(P, y_test, bins))
            nlls.append(metrics.nll(P, y_test))
        rows.append(
            {"n": int(n), "T": _spread(Ts), "ece": _spread(eces), "nll": _spread(nlls)}
        )
    return {
        "rows": rows,
        "skipped": skipped,
        "draws": int(draws),
        "uncalibrated": {
            "ece": metrics.top_label_ece(P0, y_test, bins),
            "nll": metrics.nll(P0, y_test),
        },
        "full_pool": {
            "T": float(full_T),
            "ece": metrics.top_label_ece(P_full, y_test, bins),
            "nll": metrics.nll(P_full, y_test),
        },
    }


def slice_masks(df: pl.DataFrame, seen_products: set, n_tok) -> dict[str, np.ndarray]:
    """Boolean masks over the rows of ``df`` (columns ``parent_asin`` and ``verified``)."""
    n_tok = np.asarray(n_tok)
    seen = df["parent_asin"].is_in(list(seen_products)).to_numpy()
    verified = df["verified"].to_numpy().astype(bool)
    return {
        "all": np.ones(len(df), dtype=bool),
        "unseen_product": ~seen,
        "seen_product": seen,
        "verified": verified,
        "unverified": ~verified,
        "tokens<=64": n_tok <= 64,
        "tokens 65-128": (n_tok > 64) & (n_tok <= 128),
        "tokens>128": n_tok > 128,
    }


def slice_table(P: np.ndarray, y, masks: dict, bins: int = 15, min_n: int = 50) -> dict:
    """Headline metrics per slice; a slice with fewer than ``min_n`` reviews is reported, not scored."""
    y = np.asarray(y)
    out = {}
    for name, m in masks.items():
        n = int(m.sum())
        if n < min_n:
            out[name] = {"n": n, "skipped": f"fewer than {min_n} reviews"}
            continue
        Pm, ym = P[m], y[m]
        out[name] = {
            "n": n,
            "acc": metrics.accuracy(Pm, ym),
            "nll": metrics.nll(Pm, ym),
            "ece": metrics.top_label_ece(Pm, ym, bins),
            "mean_conf": float(Pm.max(axis=1).mean()),
        }
    return out
