from __future__ import annotations

import model
import numpy as np
import pandas as pd

# -- Category schema: same "hard part" as the House Price Predictor --------


def test_category_schema_assigns_consistent_codes_regardless_of_row_order():
    X_train = pd.DataFrame({"Contract": ["A", "B", "C", "A", "B"]})
    schema = model.build_category_schema(X_train, ["Contract"])

    row_b_first = pd.DataFrame({"Contract": ["B"]})
    row_c_then_b = pd.concat(
        [pd.DataFrame({"Contract": ["C"]}), pd.DataFrame({"Contract": ["B"]})],
        ignore_index=True,
    )

    coded_1 = model.apply_category_schema(row_b_first, schema)[
        "Contract"
    ].cat.codes.iloc[0]
    coded_2 = model.apply_category_schema(row_c_then_b, schema)[
        "Contract"
    ].cat.codes.iloc[1]
    assert coded_1 == coded_2


def test_category_schema_maps_unseen_category_to_missing_not_a_wrong_code():
    X_train = pd.DataFrame({"Contract": ["A", "B", "C"]})
    schema = model.build_category_schema(X_train, ["Contract"])
    unseen = pd.DataFrame({"Contract": ["NeverSeenBefore"]})
    result = model.apply_category_schema(unseen, schema)
    assert pd.isna(result["Contract"].iloc[0])


# -- Logistic Regression pipeline sanity ------------------------------------


def test_logreg_pipeline_handles_missing_and_unseen_categorical_without_crashing():
    numeric_cols, categorical_cols = ["MonthlyCharges"], ["Contract"]
    X_train = pd.DataFrame(
        {"MonthlyCharges": [20.0, 50.0, np.nan], "Contract": ["A", "B", "A"]}
    )
    y_train = pd.Series([0, 1, 0])
    pipeline = model.build_logreg_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    unseen = pd.DataFrame({"MonthlyCharges": [np.nan], "Contract": ["NeverSeenBefore"]})
    proba = pipeline.predict_proba(unseen)
    assert proba.shape == (1, 2)
    assert np.isfinite(proba).all()


# -- Threshold metrics: exact arithmetic on a known small example ----------


def test_metrics_at_threshold_matches_hand_computed_confusion_matrix():
    y_true = pd.Series([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.4, 0.7, 0.9])
    metrics = model.metrics_at_threshold(y_true, y_proba, threshold=0.5)
    # >= 0.5: predictions are [0, 1, 0, 1, 1] -> tp=2 (idx3,4), fp=1 (idx1),
    # fn=1 (idx2), tn=1 (idx0).
    assert (metrics.tp, metrics.fp, metrics.tn, metrics.fn) == (2, 1, 1, 1)
    assert metrics.accuracy == 3 / 5
    assert metrics.precision == 2 / 3
    assert metrics.recall == 2 / 3
    assert metrics.f1 == 2 / 3


def test_metrics_at_threshold_extremes_are_all_positive_or_all_negative():
    y_true = pd.Series([0, 1, 1, 0])
    y_proba = np.array([0.2, 0.4, 0.6, 0.8])
    all_negative = model.metrics_at_threshold(y_true, y_proba, threshold=1.01)
    assert (all_negative.tp, all_negative.fp) == (0, 0)
    all_positive = model.metrics_at_threshold(y_true, y_proba, threshold=0.0)
    assert (all_positive.tn, all_positive.fn) == (0, 0)


# -- Curve metrics + threshold selection ------------------------------------


def test_compute_curves_perfect_separation_gives_auc_one():
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    curve = model.compute_curves(y_true, y_proba)
    assert curve.roc_auc == 1.0
    assert curve.pr_auc == 1.0


def test_compute_curves_random_guessing_gives_roc_auc_near_half():
    rng = np.random.default_rng(0)
    y_true = pd.Series(rng.integers(0, 2, size=2000))
    y_proba = rng.uniform(size=2000)
    curve = model.compute_curves(y_true, y_proba)
    assert abs(curve.roc_auc - 0.5) < 0.05


def test_best_f1_threshold_finds_the_gap_between_two_well_separated_clusters():
    y_true = pd.Series([0] * 5 + [1] * 5)
    y_proba = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.75, 0.8, 0.85, 0.9, 0.95])
    threshold = model.best_f1_threshold(y_true, y_proba)
    # Perfectly separated -- any threshold strictly between the clusters
    # achieves F1 = 1.0; the scan only ever considers observed
    # probabilities, so the returned threshold must be one of the
    # positive-cluster values (>= 0.75), each of which achieves F1 = 1.0.
    assert model.metrics_at_threshold(y_true, y_proba, threshold).f1 == 1.0


# -- Real-data training: both models must actually be useful classifiers --


def test_both_models_beat_a_reasonable_roc_auc_floor_on_real_data(trained_models):
    # A floor, not the actually-achieved score -- tight enough to catch a
    # real regression (e.g. accidentally training on a leaked target or a
    # broken preprocessing pipeline), loose enough not to be flaky.
    logreg_curve = model.compute_curves(
        trained_models.y_test, trained_models.logreg_proba
    )
    lgbm_curve = model.compute_curves(
        trained_models.y_test, trained_models.lightgbm_proba
    )
    assert logreg_curve.roc_auc > 0.75
    assert lgbm_curve.roc_auc > 0.75


def test_predict_proba_both_returns_probabilities_in_unit_interval(trained_models):
    customer = trained_models.X_test.iloc[[0]]
    probs = model.predict_proba_both(trained_models, customer)
    for name, proba in probs.items():
        assert 0.0 <= proba[0] <= 1.0, f"{name} probability out of range"


def test_predict_proba_both_is_deterministic_for_the_same_customer(trained_models):
    customer = trained_models.X_test.iloc[[0]]
    first = model.predict_proba_both(trained_models, customer)
    second = model.predict_proba_both(trained_models, customer)
    for name in first:
        assert first[name][0] == second[name][0]


def test_best_f1_threshold_on_real_data_beats_naive_half_on_recall_or_matches_f1(
    trained_models,
):
    # This dataset is imbalanced (~26.5% churn); the F1-optimal threshold
    # should never score *worse* on F1 than the naive 0.5 default -- if it
    # did, `best_f1_threshold` would be broken, not just "different."
    naive = model.metrics_at_threshold(
        trained_models.y_test, trained_models.logreg_proba, 0.5
    )
    best_threshold = model.best_f1_threshold(
        trained_models.y_test, trained_models.logreg_proba
    )
    tuned = model.metrics_at_threshold(
        trained_models.y_test, trained_models.logreg_proba, best_threshold
    )
    assert tuned.f1 >= naive.f1
