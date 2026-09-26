from __future__ import annotations

import base64

import explain
import numpy as np
import pytest


@pytest.fixture(scope="module")
def background(ames_split):
    return ames_split["X_train"].sample(100, random_state=1)


@pytest.fixture(scope="module")
def house(ames_split):
    return ames_split["X_test"].iloc[[0]]


def test_aggregate_by_original_column_sums_grouped_values():
    groups = ["A", "A", "B", "C", "C", "C"]
    values = np.array([1.0, 2.0, 5.0, -1.0, -1.0, 1.0])
    result = explain.aggregate_by_original_column(values, groups)
    assert result == {"A": 3.0, "B": 5.0, "C": -1.0}


def test_ridge_shap_contributions_sum_to_prediction_minus_base_in_log_space(
    trained_models, house, background
):
    explanation = explain.explain_ridge(trained_models, house, background)
    total = sum(explanation.contributions_log.values()) + explanation.base_value_log
    assert total == pytest.approx(explanation.prediction_log, abs=1e-4)


def test_lightgbm_shap_contributions_sum_to_prediction_minus_base_in_log_space(
    trained_models, house
):
    explanation = explain.explain_lightgbm(trained_models, house)
    total = sum(explanation.contributions_log.values()) + explanation.base_value_log
    assert total == pytest.approx(explanation.prediction_log, abs=1e-4)


def test_ridge_explanation_covers_every_original_column_exactly_once(
    trained_models, house, background
):
    explanation = explain.explain_ridge(trained_models, house, background)
    expected_columns = set(trained_models.numeric_cols) | set(
        trained_models.categorical_cols
    )
    assert set(explanation.contributions_log.keys()) == expected_columns


def test_lightgbm_explanation_covers_every_original_column_exactly_once(
    trained_models, house
):
    explanation = explain.explain_lightgbm(trained_models, house)
    expected_columns = set(trained_models.numeric_cols) | set(
        trained_models.categorical_cols
    )
    assert set(explanation.contributions_log.keys()) == expected_columns


def test_dollar_delta_is_zero_for_a_zero_shap_value():
    assert explain.dollar_delta(log_pred=12.0, shap_value_log=0.0) == pytest.approx(0.0)


def test_dollar_delta_sign_matches_shap_value_sign():
    log_pred = 12.0
    positive = explain.dollar_delta(log_pred, shap_value_log=0.1)
    negative = explain.dollar_delta(log_pred, shap_value_log=-0.1)
    assert positive > 0
    assert negative < 0


def test_top_contributions_dollars_are_sorted_by_absolute_log_magnitude(
    trained_models, house
):
    explanation = explain.explain_lightgbm(trained_models, house)
    magnitudes = [
        abs(explanation.contributions_log[name])
        for name, _ in explanation.top_contributions_dollars
    ]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_compute_global_importance_returns_both_models_with_no_nan(
    trained_models, ames_split
):
    sample = ames_split["X_test"].sample(20, random_state=3)
    importance = explain.compute_global_importance(trained_models, sample)
    assert set(importance.keys()) == {"ridge", "lightgbm"}
    for scores in importance.values():
        assert all(np.isfinite(v) for v in scores.values())
        assert all(v >= 0 for v in scores.values())  # mean |shap|, never negative


def test_top_combined_features_returns_requested_count(trained_models, ames_split):
    sample = ames_split["X_test"].sample(20, random_state=3)
    importance = explain.compute_global_importance(trained_models, sample)
    top = explain.top_combined_features(importance, 5)
    assert len(top) == 5
    assert len(set(top)) == 5


def test_render_global_importance_chart_returns_valid_base64_png(
    trained_models, ames_split
):
    sample = ames_split["X_test"].sample(20, random_state=3)
    importance = explain.compute_global_importance(trained_models, sample)
    png_b64 = explain.render_global_importance_chart(importance)
    raw = base64.b64decode(png_b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_contribution_chart_returns_valid_base64_png(trained_models, house):
    explanation = explain.explain_lightgbm(trained_models, house)
    png_b64 = explain.render_contribution_chart(explanation)
    raw = base64.b64decode(png_b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
