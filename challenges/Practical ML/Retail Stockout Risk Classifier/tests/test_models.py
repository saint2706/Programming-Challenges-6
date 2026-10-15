from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest
from stockout_risk import evaluate, models
from stockout_risk.features import FEATURES, GROUPS


def test_split_is_by_label_day_and_the_windows_do_not_overlap(feat):
    w = models.Windows()
    train, val, test = models.split(feat, w)
    assert train["label_day"].max() < val["label_day"].min()
    assert val["label_day"].max() < test["label_day"].min()
    assert test["label_day"].min() == w.test_start
    assert train["dt"].min() >= feat["dt"].min() + (date(2000, 1, 8) - date(2000, 1, 1))
    # every feature row's day is strictly before its label day, so train never sees val or test days
    assert (train["dt"] < train["label_day"]).all()
    assert train["dt"].max() < val["label_day"].min()


def test_label_is_the_next_days_hours_at_or_above_the_threshold(daily, feat):
    train, _, _ = models.split(feat, hours=4)
    keys = ["store_id", "product_id"]
    nxt = daily.select(
        *keys, pl.col("dt").alias("label_day"), pl.col("oos_window").alias("hours_next")
    )
    joined = train.join(nxt, on=[*keys, "label_day"], how="left")
    assert joined["hours_next"].null_count() == 0
    assert ((joined["hours_next"] >= 4).cast(pl.Int8) == joined["y"]).all()


def test_split_rejects_an_empty_window(feat):
    with pytest.raises(ValueError, match="empty"):
        models.split(feat, models.Windows(test_start=date(2030, 1, 1)))


def test_without_drops_exactly_one_group():
    for group, cols in GROUPS.items():
        kept = models.without(group)
        assert set(FEATURES) - set(kept) == set(cols)


def test_lightgbm_beats_the_base_rate_on_the_synthetic_table(feat):
    train, val, test = models.split(feat)
    booster = models.fit_lgbm(train, val, rounds=150)
    p = models.predict(booster, test)
    sc = evaluate.scores(test["y"].to_numpy(), p)
    assert sc["pr_auc"] > sc["base_rate"] + 0.1 and sc["roc_auc"] > 0.65
    assert booster.best_iteration > 1


def test_logistic_regression_fits_and_predicts_probabilities(feat):
    train, _, test = models.split(feat)
    model, cols = models.fit_logistic(train, max_rows=3000)
    assert not set(cols) & set(GROUPS["static"])
    p = model.predict_proba(models.matrix(test, cols))[:, 1]
    assert (
        p.min() >= 0
        and p.max() <= 1
        and evaluate.scores(test["y"].to_numpy(), p)["roc_auc"] > 0.6
    )


def test_calibrator_is_monotone_bounded_and_survives_json():
    rng = np.random.default_rng(0)
    p = rng.random(20_000)
    y = (rng.random(20_000) < p**2).astype(int)  # an over-confident score
    cal = models.Calibrator.fit(p, y)
    grid = np.linspace(-0.2, 1.2, 50)  # out-of-range inputs clip
    out = cal(grid)
    assert (np.diff(out) >= -1e-12).all() and out.min() >= 0 and out.max() <= 1
    again = models.Calibrator.from_dict(cal.to_dict())
    assert np.allclose(again(grid), out)
    assert evaluate.ece(y, cal(p)) < evaluate.ece(y, p) / 3


def test_calibrator_does_not_turn_a_few_lucky_extremes_into_certainty():
    """The top 64 of 100,000 scores are all positive, but they share a bin with 936 others."""
    rng = np.random.default_rng(1)
    p = rng.random(100_000)
    y = (rng.random(100_000) < p * 0.9).astype(int)
    y[np.argsort(p)[-64:]] = 1
    top = models.Calibrator.fit(p, y)(np.array([p.max()]))[0]
    assert 0.8 < top < 0.99
    # per-row isotonic regression, for contrast, does reach 1.0 here
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    assert iso.predict([p.max()])[0] == 1.0


def test_calibrator_handles_a_small_sample_and_ties():
    p = np.array([0.2] * 30 + [0.8] * 30)
    y = np.array([0] * 25 + [1] * 5 + [1] * 25 + [0] * 5)
    cal = models.Calibrator.fit(p, y, bin_rows=10)
    assert cal(np.array([0.2]))[0] == pytest.approx(5 / 30, abs=0.1)
    assert cal(np.array([0.8]))[0] == pytest.approx(25 / 30, abs=0.1)


def test_model_round_trips_through_text_and_json(feat, tmp_path):
    train, val, test = models.split(feat)
    booster = models.fit_lgbm(train, val, rounds=40)
    models.save_model(tmp_path, booster, {"features": FEATURES, "label_hours": 4})
    loaded, meta = models.load_model(tmp_path)
    assert meta["label_hours"] == 4
    assert np.allclose(models.predict(loaded, test), models.predict(booster, test))
    assert not list(tmp_path.glob("*.pkl")) and not list(tmp_path.glob("*.joblib"))


def test_loading_a_missing_model_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="stockout evaluate"):
        models.load_model(tmp_path)


def test_contributions_add_up_to_the_raw_score(feat):
    train, val, test = models.split(feat)
    booster = models.fit_lgbm(train, val, rounds=30)
    rows = test.head(50)
    contrib = booster.predict(models.matrix(rows), pred_contrib=True)
    logit = contrib.sum(axis=1)
    p = models.predict(booster, rows)
    assert np.allclose(1 / (1 + np.exp(-logit)), p, atol=1e-6)
    assert models.contributions(booster, rows).shape == (50, len(FEATURES))
