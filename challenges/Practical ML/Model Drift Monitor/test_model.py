import numpy as np
import polars as pl

from data import FEATURES, NUMERIC
from model import fit, score


def _task(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    cols = {"day": rng.integers(1, 8, n)}
    for name in NUMERIC:
        cols[name] = rng.random(n)
    p = 1 / (1 + np.exp(-(6 * (cols["nswprice"] - 0.5))))
    cols["label"] = (rng.random(n) < p).astype(int)
    return pl.DataFrame(cols)


def test_model_beats_the_majority_rate_on_a_planted_signal():
    df = _task()
    train, test = df[:1000], df[1000:]
    clf = fit(train, seed=0)
    pred = (score(clf, test) > 0.5).astype(int)
    acc = (pred == test["label"].to_numpy()).mean()
    majority = max(test["label"].mean(), 1 - test["label"].mean())
    assert acc > majority + 0.05


def test_score_is_a_probability_one_per_row_and_deterministic():
    df = _task(400)
    a = score(fit(df, seed=1), df)
    b = score(fit(df, seed=1), df)
    assert a.shape == (400,) and (a >= 0).all() and (a <= 1).all()
    np.testing.assert_allclose(a, b)


def test_model_uses_exactly_the_feature_columns():
    clf = fit(_task(300), seed=0)
    assert clf.n_features_in_ == len(FEATURES)
