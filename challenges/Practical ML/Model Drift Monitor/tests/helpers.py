"""Test code shared by several test modules (moved out of test files that used to import each other)."""

import numpy as np
import polars as pl
from drift_monitor import model
from drift_monitor.data import NUMERIC
from drift_monitor.monitor import build_baseline, columns_of
from drift_monitor.sequential import calibrate_sequential
from drift_monitor.thresholds import calibrate, null_distribution

W = 100


def drifting(n=6000, seed=0, step_at=3600):
    """Stationary, then nswprice shifts by +1.0 from row ``step_at``."""
    rng = np.random.default_rng(seed)
    cols = {"day": rng.integers(1, 8, n)}
    for name in NUMERIC:
        cols[name] = rng.random(n)
    cols["nswprice"][step_at:] += 1.0
    p = 1 / (1 + np.exp(-(6 * (np.clip(cols["nswprice"], 0, 1) - 0.5))))
    cols["label"] = (rng.random(n) < p).astype(int)
    return pl.DataFrame(cols)


def make_fitted():
    train, ref = synth(2500, 0), synth(1500, 1)
    clf = model.fit(train, seed=0)
    base = build_baseline(clf, train, seed=0)
    thr = calibrate(
        null_distribution(base, columns_of(clf, ref), n_windows=300, window=W, seed=0)
    )
    err = (model.score(clf, ref) > 0.5) != ref["label"].to_numpy()
    seq = {
        "score": calibrate_sequential(model.score(clf, ref), window=W, reps=4, seed=0),
        "error": calibrate_sequential(err.astype(float), window=W, reps=4, seed=0),
    }
    return clf, base, thr, seq, ref


def synth(n, seed, shift_feature=None, shift=0.0, from_row=0):
    rng = np.random.default_rng(seed)
    cols = {"day": rng.integers(1, 8, n)}
    for name in NUMERIC:
        cols[name] = rng.random(n)
    if shift_feature:
        cols[shift_feature] = cols[shift_feature].copy()
        cols[shift_feature][from_row:] += shift
    p = 1 / (1 + np.exp(-(6 * (cols["nswprice"] - 0.5))))
    cols["label"] = (rng.random(n) < p).astype(int)
    return pl.DataFrame(cols)
