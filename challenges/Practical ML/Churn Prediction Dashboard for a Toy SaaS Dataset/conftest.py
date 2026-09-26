from __future__ import annotations

import data
import model
import pytest


@pytest.fixture(scope="session")
def churn_split():
    """Real Telco Customer Churn data (from cache after the first fetch),
    cleaned and split once and shared across the whole test session --
    training two real models per test file would otherwise dominate the
    suite's runtime."""
    df = data.clean_features(data.load_telco_churn())
    numeric_cols, categorical_cols = data.numeric_and_categorical_columns(df)
    X_train, X_test, y_train, y_test = data.train_test_split_frame(df)
    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@pytest.fixture(scope="session")
def trained_models(churn_split):
    return model.train_models(
        churn_split["X_train"],
        churn_split["y_train"],
        churn_split["X_test"],
        churn_split["y_test"],
        churn_split["numeric_cols"],
        churn_split["categorical_cols"],
    )
