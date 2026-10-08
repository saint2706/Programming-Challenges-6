import numpy as np
import polars as pl
from drift_monitor.data import FEATURES, NUMERIC
from drift_monitor.model import fit, score


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


def test_out_of_fold_scores_are_honest_not_in_sample():
    from drift_monitor.model import oof_scores
    from sklearn.metrics import log_loss

    df = _task(1200)
    clf = fit(df, seed=0)
    in_sample = score(clf, df)
    oof = oof_scores(df, k=4, seed=0)
    y = df["label"].to_numpy()
    assert oof.shape == (1200,) and (oof >= 0).all() and (oof <= 1).all()
    assert log_loss(y, oof) > log_loss(
        y, in_sample
    )  # unseen rows are scored less confidently
    np.testing.assert_allclose(oof, oof_scores(df, k=4, seed=0))


def test_oof_folds_are_contiguous_blocks_so_no_row_scores_itself():
    from drift_monitor.model import oof_scores

    df = _task(400)
    oof = oof_scores(df, k=4, seed=0)
    # a model trained on rows 0-299 scores rows 300-399 identically to oof
    clf = fit(df[:300], seed=0)
    np.testing.assert_allclose(oof[300:], score(clf, df[300:]), atol=1e-9)
