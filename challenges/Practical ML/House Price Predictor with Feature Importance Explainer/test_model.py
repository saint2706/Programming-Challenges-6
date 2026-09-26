from __future__ import annotations

import model
import numpy as np
import pandas as pd

# -- Category schema: the "hard part" (see model.py's module docstring) ----


def test_category_schema_assigns_consistent_codes_regardless_of_row_order():
    X_train = pd.DataFrame({"Neighborhood": ["A", "B", "C", "A", "B"]})
    schema = model.build_category_schema(X_train, ["Neighborhood"])

    # Two single-row frames presenting the SAME value, but where a naive
    # `.astype("category")` on just that one row would each build a
    # *different* category list (and therefore a different code) purely
    # from the order categories happen to appear.
    row_b_first = pd.DataFrame({"Neighborhood": ["B"]})
    row_c_then_b = pd.concat(
        [pd.DataFrame({"Neighborhood": ["C"]}), pd.DataFrame({"Neighborhood": ["B"]})],
        ignore_index=True,
    )

    coded_1 = model.apply_category_schema(row_b_first, schema)[
        "Neighborhood"
    ].cat.codes.iloc[0]
    coded_2 = model.apply_category_schema(row_c_then_b, schema)[
        "Neighborhood"
    ].cat.codes.iloc[1]
    assert coded_1 == coded_2  # both refer to "B" -> must be the same code


def test_category_schema_maps_unseen_category_to_missing_not_a_wrong_code():
    X_train = pd.DataFrame({"Neighborhood": ["A", "B", "C"]})
    schema = model.build_category_schema(X_train, ["Neighborhood"])
    unseen = pd.DataFrame({"Neighborhood": ["NeverSeenBefore"]})
    result = model.apply_category_schema(unseen, schema)
    assert pd.isna(result["Neighborhood"].iloc[0])


# -- Ridge pipeline sanity ---------------------------------------------------


def test_ridge_pipeline_handles_missing_and_unseen_categorical_without_crashing():
    numeric_cols, categorical_cols = ["LotArea"], ["Neighborhood"]
    X_train = pd.DataFrame(
        {"LotArea": [1000.0, 2000.0, np.nan], "Neighborhood": ["A", "B", "A"]}
    )
    y_train = pd.Series([11.5, 12.0, 11.6])
    pipeline = model.build_ridge_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    unseen = pd.DataFrame({"LotArea": [np.nan], "Neighborhood": ["NeverSeenBefore"]})
    prediction = pipeline.predict(unseen)
    assert np.isfinite(prediction).all()


# -- Real-data training: both models must actually be good models ----------


def test_both_models_beat_a_reasonable_r2_floor_on_real_data(trained_models):
    # A floor, not the actual achieved score (~0.90 for both as measured) --
    # tight enough to catch a real regression (e.g. accidentally training
    # on the untransformed target, or a broken preprocessing pipeline),
    # loose enough not to be a flaky assertion on model internals.
    assert trained_models.ridge_metrics.r2_log > 0.7
    assert trained_models.lightgbm_metrics.r2_log > 0.7


def test_predict_both_returns_positive_dollar_prices_in_plausible_range(
    ames_split, trained_models
):
    house = ames_split["X_test"].iloc[[0]]
    true_price = float(np.expm1(ames_split["y_test_log"].iloc[0]))
    prices = model.predict_both(trained_models, house)

    for name, price in prices.items():
        assert price > 0, f"{name} predicted a non-positive price"
        # Coarse sanity bound -- catches gross bugs (e.g. a forgotten
        # expm1) without asserting tight accuracy on any single house.
        assert true_price / 3 < price < true_price * 3, (
            f"{name} price {price} implausible vs {true_price}"
        )


def test_predict_both_is_deterministic_for_the_same_house(ames_split, trained_models):
    house = ames_split["X_test"].iloc[[0]]
    first = model.predict_both(trained_models, house)
    second = model.predict_both(trained_models, house)
    assert first == second


def test_default_house_row_has_no_missing_values_and_all_training_columns(ames_split):
    row = model.default_house_row(
        ames_split["X_train"],
        ames_split["numeric_cols"],
        ames_split["categorical_cols"],
    )
    assert list(row.columns) == list(ames_split["X_train"].columns)
    assert row.isna().sum().sum() == 0


def test_default_house_row_predicts_a_plausible_price(ames_split, trained_models):
    row = model.default_house_row(
        ames_split["X_train"],
        ames_split["numeric_cols"],
        ames_split["categorical_cols"],
    )
    prices = model.predict_both(trained_models, row)
    mean_price = float(np.expm1(ames_split["y_train_log"]).mean())
    for price in prices.values():
        assert mean_price / 3 < price < mean_price * 3
