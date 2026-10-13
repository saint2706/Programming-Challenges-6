from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from credit_scorecard import metrics
from credit_scorecard.scorecard import Scorecard, fit_scorecard
from helpers import synthetic_frame

FEATURES = ["strong", "weak", "noise", "copy_of_strong"]


@pytest.fixture(scope="module")
def fitted():
    df = synthetic_frame()
    return fit_scorecard(df, "bad", FEATURES), df


def _row_in_bin(card: Scorecard, pick) -> dict[str, float]:
    """One value per feature, taken from the range bin chosen by ``pick(binning) -> bin index``."""
    row = {}
    for f in card.features:
        b = card.binnings[f]
        i = pick(b)
        row[f] = (b.edges[i] - 1e-6) if i < len(b.edges) else b.edges[-1] + 1
    return row


def test_selection_drops_noise_and_the_near_duplicate(fitted):
    card, _ = fitted
    by_feature = {row["feature"]: row for row in card.selection}
    assert card.features == ["strong", "weak"]
    assert (
        by_feature["noise"]["status"] == "dropped"
        and "IV" in by_feature["noise"]["reason"]
    )
    assert by_feature["copy_of_strong"]["status"] == "dropped"
    assert "corr" in by_feature["copy_of_strong"]["reason"]


def test_every_coefficient_is_positive_and_near_one_on_clean_signal(fitted):
    card, _ = fitted
    assert all(b > 0 for b in card.coefs.values())
    assert all(
        0.7 < b < 1.3 for b in card.coefs.values()
    )  # WOE is already a log odds ratio


def test_score_is_the_sum_of_per_feature_points(fitted):
    card, df = fitted
    parts = card.feature_points(df)
    assert np.array_equal(card.score(df), sum(parts.values()))
    assert all(np.issubdtype(p.dtype, np.integer) for p in parts.values())


def test_points_double_the_odds_every_pdo(fitted):
    card, _ = fitted

    def odds(score):
        p = card.probability_bad(score)
        return (1 - p) / p

    assert odds(card.base_score) == pytest.approx(card.base_odds)
    assert odds(card.base_score + card.pdo) == pytest.approx(2 * card.base_odds)
    assert odds(card.base_score - 2 * card.pdo) == pytest.approx(card.base_odds / 4)


def test_integer_points_stay_within_rounding_of_the_exact_score(fitted):
    card, df = fitted
    woe = sum(
        card.coefs[f] * card.binnings[f].transform(df[f].to_numpy())
        for f in card.features
    )
    exact = card.offset + card.factor * (card.intercept + woe)
    assert np.max(np.abs(card.score(df) - exact)) <= 0.5 * len(card.features) + 1e-9


def test_scorecard_ranks_better_than_chance_on_fresh_data(fitted):
    card, _ = fitted
    fresh = synthetic_frame(seed=99)
    assert metrics.auc(fresh["bad"].to_numpy(), -card.score(fresh)) > 0.75


def test_save_and_load_roundtrip(fitted, tmp_path):
    card, df = fitted
    card.save(tmp_path / "card.json")
    again = Scorecard.load(tmp_path / "card.json")
    assert again.points == card.points
    assert np.array_equal(again.score(df), card.score(df))
    assert again.selection == card.selection


def test_reason_codes_rank_features_by_points_lost(fitted):
    card, _ = fitted
    riskiest = _row_in_bin(
        card, lambda b: max(range(len(b.edges) + 1), key=lambda i: b.bins[i].bad_rate)
    )
    reasons = card.reason_codes(riskiest)
    assert {f for f, _ in reasons} == set(card.features)
    assert [lost for _, lost in reasons] == sorted(
        (lost for _, lost in reasons), reverse=True
    )


def test_applicant_in_the_best_bin_everywhere_has_no_reasons(fitted):
    card, _ = fitted
    safest = _row_in_bin(
        card, lambda b: min(range(len(b.edges) + 1), key=lambda i: b.bins[i].bad_rate)
    )
    assert card.reason_codes(safest) == []


def test_missing_input_is_scored_neutrally_not_rejected(fitted):
    card, _ = fitted
    frame = pl.DataFrame(
        {f: [None] for f in card.features},
        schema={f: pl.Float64 for f in card.features},
    )
    score = card.score(frame)
    # WOE 0 everywhere leaves only the intercept: offset + factor * intercept, within rounding.
    neutral = card.offset + card.factor * card.intercept
    assert score[0] == pytest.approx(neutral, abs=0.5 * len(card.features))
    assert math.isfinite(float(card.probability_bad(score)[0]))
