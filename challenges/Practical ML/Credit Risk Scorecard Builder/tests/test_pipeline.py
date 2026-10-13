"""The scorecard on the real data: the properties a credit-risk reviewer would check."""

from __future__ import annotations

import numpy as np
from credit_scorecard import data


def test_scorecard_discriminates_close_to_a_gradient_boosted_tree(result):
    m = result.metrics
    assert m["scorecard_test"]["auc"] > 0.84
    # Interpretability should cost at most a couple of AUC points here, not ten.
    assert m["gbm_test"]["auc"] - m["scorecard_test"]["auc"] < 0.02
    lo, hi = m["scorecard_test_auc_ci95"]
    assert lo < m["scorecard_test"]["auc"] < hi


def test_no_overfit_between_train_and_test(result):
    m = result.metrics
    assert abs(m["scorecard_train"]["auc"] - m["scorecard_test"]["auc"]) < 0.01
    assert m["score_psi_train_vs_test"] < 0.01
    assert max(result.feature_psi.values()) < 0.01


def test_every_range_bin_sequence_is_monotone_in_bad_rate(result):
    for feature in result.scorecard.features:
        rates = [
            b.bad_rate
            for b in result.scorecard.binnings[feature].bins
            if b.kind == "range"
        ]
        assert rates == sorted(rates) or rates == sorted(rates, reverse=True), feature


def test_sentinel_codes_have_their_own_high_risk_bin(result):
    binning = result.scorecard.binnings["NumberOfTimes90DaysLate"]
    special = next(b for b in binning.bins if b.kind == "special")
    worst_range = max(b.bad_rate for b in binning.bins if b.kind == "range")
    assert special.bad_rate > 0.5 and special.bad_rate > worst_range
    assert (
        min(result.scorecard.points["NumberOfTimes90DaysLate"])
        == result.scorecard.points["NumberOfTimes90DaysLate"][
            binning.bins.index(special)
        ]
    )


def test_missing_income_gets_a_bin_and_is_not_dropped(result):
    binning = result.scorecard.binnings["MonthlyIncome"]
    missing = next(b for b in binning.bins if b.kind == "missing")
    assert missing.count > 10_000


def test_selection_log_explains_every_candidate(result):
    card = result.scorecard
    assert [row["feature"] for row in card.selection] == data.FEATURES
    for row in card.selection:
        assert row["status"] in {"kept", "dropped"}
        assert (row["status"] == "dropped") == bool(row["reason"])
        assert (row["feature"] in card.features) == (row["status"] == "kept")
    assert all(b > 0 for b in card.coefs.values())


def test_integer_points_and_score_range(result):
    card = result.scorecard
    scores = card.score(result.test)
    assert np.issubdtype(scores.dtype, np.integer)
    assert all(isinstance(p, int) for table in card.points.values() for p in table)
    assert 300 < scores.min() < scores.max() < 700


def test_calibrated_overall_and_in_the_risky_half(result):
    m = result.metrics
    assert abs(m["mean_predicted_pd_test"] - m["bad_rate_test"]) < 0.005
    risky = result.calibration.filter(result.calibration["band"] < 5)
    ratio = (risky["predicted_pd"] / risky["observed_bad_rate"]).to_numpy()
    assert np.all((ratio > 0.8) & (ratio < 1.3))


def test_scorecard_is_trained_on_the_training_split_only(result):
    train_rows = result.train.height
    assert train_rows == sum(b.count for b in result.scorecard.binnings["age"].bins)
