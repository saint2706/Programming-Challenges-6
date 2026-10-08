from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from helpers import T0, synthetic
from inbox_sorter.models import (
    MODEL_NAMES,
    Calibrator,
    TextFeatures,
    fit_models,
    time_split,
)
from sklearn.metrics import average_precision_score


def test_time_split_is_ordered_and_disjoint_per_mailbox():
    df = synthetic(200)
    train, val, test = time_split(df)
    assert train.height + val.height + test.height == df.height
    ids = [set(x["message_id"]) for x in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    for box in ("m1", "m2"):
        tr, va, te = (
            x.filter(pl.col("mailbox") == box)["date"] for x in (train, val, test)
        )
        assert tr.max() < va.min() and va.max() < te.min()
        assert tr.len() == 140 and va.len() == 30 and te.len() == 30


def test_time_split_never_splits_identical_timestamps_across_sets():
    rows = [
        {
            "message_id": str(i),
            "mailbox": "m",
            "date": T0 + timedelta(days=i // 10),
            "acted": False,
        }
        for i in range(100)
    ]
    df = pl.DataFrame(rows, schema_overrides={"date": pl.Datetime("us")})
    train, val, test = time_split(df)
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()


def test_text_features_fit_only_on_train_text():
    tf = TextFeatures(n_components=3, seed=0).fit(
        [
            "alpha beta gamma",
            "alpha beta delta",
            "beta gamma delta",
            "alpha gamma delta",
        ]
        * 3
    )
    assert "alpha" in tf.vocabulary
    assert "zzzonlyintest" not in tf.vocabulary
    # words unseen at fit time are ignored at transform time, not an error
    out = tf.svd(["zzzonlyintest alpha"])
    assert out.shape == (1, tf.n_components)


@pytest.fixture(scope="module")
def fitted():
    df = synthetic()
    train, val, test = time_split(df)
    return fit_models(train, val, seed=0), test


def test_every_model_beats_random_on_a_planted_signal(fitted):
    models, test = fitted
    y = test["acted"].to_numpy()
    base = y.mean()
    ap = {n: average_precision_score(y, models.predict(n, test)) for n in MODEL_NAMES}
    assert ap["random"] < base + 0.08
    # to_me is a weak heuristic baseline that only has to run
    for name in ("tfidf_lr", "lgbm_meta", "lgbm_meta_text"):
        assert ap[name] > ap["random"] + 0.05, (name, ap)
    assert (
        ap["lgbm_meta_text"] >= ap["lgbm_meta"] - 0.02
    )  # text signal is there to be found


def test_models_are_deterministic_for_a_seed():
    df = synthetic(400)
    train, val, test = time_split(df)
    a = fit_models(train, val, seed=5).predict("lgbm_meta_text", test)
    b = fit_models(train, val, seed=5).predict("lgbm_meta_text", test)
    np.testing.assert_allclose(a, b)
    r1 = fit_models(train, val, seed=5).predict("random", test)
    r2 = fit_models(train, val, seed=6).predict("random", test)
    assert not np.allclose(r1, r2)


def test_predict_returns_one_score_per_row_for_every_model(fitted):
    models, test = fitted
    for name in MODEL_NAMES:
        assert models.predict(name, test).shape == (test.height,)


def test_a_validation_set_with_one_class_still_trains_without_early_stopping():
    df = synthetic(300)
    train, val, test = time_split(df)
    val = val.with_columns(pl.lit(False).alias("acted"))
    scores = fit_models(train, val, seed=0).predict("lgbm_meta", test)
    assert np.isfinite(scores).all()


def test_calibrator_is_bounded_and_monotone():
    rng = np.random.default_rng(1)
    s = rng.random(500)
    y = rng.random(500) < s**2
    cal = Calibrator().fit(s, y)
    grid = np.linspace(-0.5, 1.5, 200)
    out = cal.predict(grid)
    assert out.min() >= 0 and out.max() <= 1
    assert np.all(np.diff(out) >= -1e-12)
