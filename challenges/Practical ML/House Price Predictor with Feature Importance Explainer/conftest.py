from __future__ import annotations

import data
import model
import pytest


@pytest.fixture(scope="session")
def ames_split():
    """Real Ames Housing data (from cache after the first fetch), split
    once and shared across the whole test session -- training two real
    models per test file would otherwise dominate the suite's runtime."""
    df = data.clean_features(data.load_ames_housing())
    numeric_cols, categorical_cols = data.numeric_and_categorical_columns(df)
    X_train, X_test, y_train_log, y_test_log = data.train_test_split_frame(df)
    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train_log": y_train_log,
        "y_test_log": y_test_log,
    }


@pytest.fixture(scope="session")
def trained_models(ames_split):
    return model.train_models(
        ames_split["X_train"],
        ames_split["y_train_log"],
        ames_split["X_test"],
        ames_split["y_test_log"],
        ames_split["numeric_cols"],
        ames_split["categorical_cols"],
    )
