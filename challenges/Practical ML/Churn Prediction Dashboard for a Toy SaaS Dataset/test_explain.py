from __future__ import annotations

import explain
import matplotlib.pyplot as plt
import model
import numpy as np
import pytest


@pytest.fixture(scope="module")
def background(churn_split):
    return churn_split["X_train"].sample(100, random_state=1)


@pytest.fixture(scope="module")
def customer(churn_split):
    return churn_split["X_test"].iloc[[0]]


def test_aggregate_by_original_column_sums_grouped_values():
    groups = ["A", "A", "B", "C", "C", "C"]
    values = np.array([1.0, 2.0, 5.0, -1.0, -1.0, 1.0])
    result = explain.aggregate_by_original_column(values, groups)
    assert result == {"A": 3.0, "B": 5.0, "C": -1.0}


def test_logreg_shap_contributions_sum_to_prediction_minus_base_in_logit_space(
    trained_models, customer, background
):
    explanation = explain.explain_logreg(trained_models, customer, background)
    total = sum(explanation.contributions_logit.values()) + explanation.base_logit
    assert total == pytest.approx(explanation.prediction_logit, abs=1e-4)


def test_lightgbm_shap_contributions_sum_to_prediction_minus_base_in_logit_space(
    trained_models, customer
):
    explanation = explain.explain_lightgbm(trained_models, customer)
    total = sum(explanation.contributions_logit.values()) + explanation.base_logit
    assert total == pytest.approx(explanation.prediction_logit, abs=1e-4)


def test_logreg_explanation_covers_every_original_column_exactly_once(
    trained_models, customer, background
):
    explanation = explain.explain_logreg(trained_models, customer, background)
    expected_columns = set(trained_models.numeric_cols) | set(
        trained_models.categorical_cols
    )
    assert set(explanation.contributions_logit.keys()) == expected_columns


def test_lightgbm_explanation_covers_every_original_column_exactly_once(
    trained_models, customer
):
    explanation = explain.explain_lightgbm(trained_models, customer)
    expected_columns = set(trained_models.numeric_cols) | set(
        trained_models.categorical_cols
    )
    assert set(explanation.contributions_logit.keys()) == expected_columns


def test_probability_delta_is_zero_for_a_zero_shap_value():
    assert explain.probability_delta(
        logit_pred=0.5, shap_value_logit=0.0
    ) == pytest.approx(0.0)


def test_probability_delta_sign_matches_shap_value_sign():
    logit_pred = 0.5
    positive = explain.probability_delta(logit_pred, shap_value_logit=0.3)
    negative = explain.probability_delta(logit_pred, shap_value_logit=-0.3)
    assert positive > 0
    assert negative < 0


def test_top_contributions_probability_are_sorted_by_absolute_logit_magnitude(
    trained_models, customer
):
    explanation = explain.explain_lightgbm(trained_models, customer)
    magnitudes = [
        abs(explanation.contributions_logit[name])
        for name, _ in explanation.top_contributions_probability
    ]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_compute_global_importance_returns_both_models_with_no_nan(
    trained_models, churn_split
):
    sample = churn_split["X_test"].sample(20, random_state=3)
    importance = explain.compute_global_importance(trained_models, sample)
    assert set(importance.keys()) == {"logreg", "lightgbm"}
    for scores in importance.values():
        assert all(np.isfinite(v) for v in scores.values())
        assert all(v >= 0 for v in scores.values())  # mean |shap|, never negative


def test_top_combined_features_returns_requested_count(trained_models, churn_split):
    sample = churn_split["X_test"].sample(20, random_state=3)
    importance = explain.compute_global_importance(trained_models, sample)
    top = explain.top_combined_features(importance, 5)
    assert len(top) == 5
    assert len(set(top)) == 5


def test_render_global_importance_chart_returns_a_figure(trained_models, churn_split):
    sample = churn_split["X_test"].sample(20, random_state=3)
    importance = explain.compute_global_importance(trained_models, sample)
    fig = explain.render_global_importance_chart(importance)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_render_contribution_chart_returns_a_figure(trained_models, customer):
    explanation = explain.explain_lightgbm(trained_models, customer)
    fig = explain.render_contribution_chart(explanation)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_render_roc_and_pr_curve_charts_return_figures(trained_models):
    curve = model.compute_curves(trained_models.y_test, trained_models.logreg_proba)
    metrics = model.metrics_at_threshold(
        trained_models.y_test, trained_models.logreg_proba, 0.5
    )
    roc_fig = explain.render_roc_curve_chart(curve, metrics)
    pr_fig = explain.render_pr_curve_chart(curve, metrics)
    assert isinstance(roc_fig, plt.Figure)
    assert isinstance(pr_fig, plt.Figure)
    plt.close(roc_fig)
    plt.close(pr_fig)
