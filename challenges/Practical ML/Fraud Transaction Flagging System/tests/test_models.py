from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from fraud_flagging import evaluate, models
from fraud_flagging.features import FEATURES


def test_split_cuts_by_time_and_the_windows_do_not_overlap(frame):
    train, val, test = models.split(frame)
    assert train["Time"].max() < 24 * 3600 <= val["Time"].min()
    assert val["Time"].max() < 36 * 3600 <= test["Time"].min()
    assert train.height + val.height + test.height == frame.height


def test_split_requires_fraud_in_every_window(frame):
    with pytest.raises(ValueError, match="no fraud"):
        models.split(frame, models.Windows(val_hour=24, test_hour=47.999))
    with pytest.raises(ValueError, match="no fraud"):
        models.split(frame.filter(pl.col("y") == 0))


def test_every_strategy_trains_a_model_that_beats_chance(frame):
    train, val, test = models.split(frame)
    y = test["y"].to_numpy()
    for strategy in models.STRATEGIES:
        booster = models.fit_lgbm(train, val, strategy=strategy, rounds=120)
        pr = evaluate.ranking_scores(y, models.predict(booster, test))["pr_auc"]
        assert pr > 5 * y.mean(), strategy


def test_undersampling_keeps_every_fraud_and_a_fixed_ratio_of_legitimate_rows(frame):
    train, _, _ = models.split(frame)
    x, y, _ = models._prepare(train, FEATURES, "undersample", seed=0)
    frauds = int(train["y"].sum())
    assert int(y.sum()) == frauds and len(y) == frauds * (1 + models.UNDERSAMPLE_RATIO)
    x2, _, _ = models._prepare(train, FEATURES, "undersample", seed=0)
    assert np.array_equal(x, x2)  # seeded


def test_weighting_strategies_set_scale_pos_weight(frame):
    train, _, _ = models.split(frame)
    ratio = models.class_ratio(train["y"].to_numpy())
    assert "scale_pos_weight" not in models._prepare(train, FEATURES, "none", 0)[2]
    assert models._prepare(train, FEATURES, "balanced", 0)[2][
        "scale_pos_weight"
    ] == pytest.approx(ratio)
    assert models._prepare(train, FEATURES, "sqrt", 0)[2][
        "scale_pos_weight"
    ] == pytest.approx(ratio**0.5)


def test_unknown_strategy_is_rejected(frame):
    train, val, _ = models.split(frame)
    with pytest.raises(ValueError, match="strategy"):
        models.fit_lgbm(train, val, strategy="smote")


def test_fixed_round_fit_uses_the_same_preparation(frame):
    train, _, test = models.split(frame)
    booster = models.fit_lgbm_fixed(train, strategy="undersample", rounds=30)
    assert booster.num_trees() == 30
    assert (
        evaluate.ranking_scores(test["y"].to_numpy(), models.predict(booster, test))[
            "roc_auc"
        ]
        > 0.9
    )


def test_logistic_picks_c_from_the_grid_and_the_forest_ranks_frauds_above_chance(frame):
    train, val, test = models.split(frame)
    model, c = models.fit_logistic(train, val, grid=(0.01, 1.0))
    assert c in (0.01, 1.0)
    assert model.predict_proba(models.matrix(test))[:, 1].max() <= 1
    forest = models.fit_isolation_forest(train)
    score = models.anomaly_score(forest, test)
    assert score.shape == (test.height,)
    assert evaluate.ranking_scores(test["y"].to_numpy(), score)["roc_auc"] > 0.6


def test_platt_scaling_recovers_probabilities_from_a_distorted_score():
    rng = np.random.default_rng(0)
    true = rng.beta(0.4, 30, 200_000)  # real fraud probabilities, mostly tiny
    y = (rng.random(200_000) < true).astype(int)
    distorted = 1 / (
        1 + np.exp(-(3 * np.log(true / (1 - true)) + 4))
    )  # as undersampling would
    platt = models.Platt.fit(distorted, y)
    fixed = platt(distorted)
    assert abs(fixed.mean() - y.mean()) < 0.0005  # the mean is back at the base rate
    # log-loss (Brier barely moves for events this rare) is as good as the true probabilities
    assert evaluate.scores(y, fixed)["log_loss"] == pytest.approx(
        evaluate.scores(y, true)["log_loss"], rel=0.01
    )
    assert (
        evaluate.scores(y, fixed)["log_loss"]
        < 0.8 * evaluate.scores(y, distorted)["log_loss"]
    )
    assert (
        np.diff(platt(np.sort(distorted))) >= -1e-12
    ).all()  # monotone, so ranking is untouched


def test_platt_survives_json_and_clips_extreme_scores():
    platt = models.Platt(1.2, -2.5)
    again = models.Platt.from_dict(platt.to_dict())
    scores = np.array([0.0, 1e-12, 0.5, 1.0])
    assert (
        np.allclose(again(scores), platt(scores)) and np.isfinite(platt(scores)).all()
    )


def test_model_round_trips_through_text_and_json(frame, tmp_path):
    train, val, test = models.split(frame)
    booster = models.fit_lgbm(train, val, rounds=40)
    models.save_model(tmp_path, booster, {"features": FEATURES})
    loaded, meta = models.load_model(tmp_path)
    assert meta["features"] == FEATURES
    assert np.allclose(models.predict(loaded, test), models.predict(booster, test))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.json", "model.txt"]


def test_loading_a_missing_model_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="fraud evaluate"):
        models.load_model(tmp_path)


def test_contributions_add_up_to_the_raw_score(frame):
    train, val, test = models.split(frame)
    booster = models.fit_lgbm(train, val, rounds=30)
    rows = test.head(40)
    full = booster.predict(models.matrix(rows), pred_contrib=True)
    assert np.allclose(
        1 / (1 + np.exp(-full.sum(axis=1))), models.predict(booster, rows), atol=1e-6
    )
    assert models.contributions(booster, rows).shape == (40, len(FEATURES))
