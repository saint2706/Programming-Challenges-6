"""Model training + prediction: a Ridge (linear) baseline and a LightGBM
(tree) model, trained on the same log-price target for an honest
head-to-head comparison.

The hard part: LightGBM's native pandas-categorical support encodes each
category as an integer *at fit time*, keyed to the column's `pd.Categorical`
category order. A naive `.astype("category")` on a single new row builds a
fresh category list from just the values present in that one row -- which
can assign a *different* integer code to the very same string value than
training used, so the model would silently split on the wrong branch with
no error and no warning. `build_category_schema` freezes the training-time
category order for every categorical column into a `pd.CategoricalDtype`,
and `apply_category_schema` re-applies that exact schema to any inference
frame, so a category's code is identical whether it's row 800 of training
or a single `predict()` call later. An unseen category value maps to
`NaN` (SHAP/LightGBM's native missing-value code), never a wrong one.
Covered by `test_model.py::test_category_schema_is_order_independent`.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

CategorySchema = dict[str, pd.CategoricalDtype]


@dataclass(frozen=True)
class Metrics:
    rmse_log: float
    mae_log: float
    r2_log: float
    rmse_dollars: float
    mae_dollars: float
    r2_dollars: float

    def as_dict(self) -> dict[str, float]:
        return {
            "rmse_log": self.rmse_log,
            "mae_log": self.mae_log,
            "r2_log": self.r2_log,
            "rmse_dollars": self.rmse_dollars,
            "mae_dollars": self.mae_dollars,
            "r2_dollars": self.r2_dollars,
        }


@dataclass
class TrainedModels:
    ridge: Pipeline
    lightgbm: lgb.LGBMRegressor
    numeric_cols: list[str]
    categorical_cols: list[str]
    category_schema: CategorySchema
    ridge_metrics: Metrics
    lightgbm_metrics: Metrics


def build_category_schema(
    X: pd.DataFrame, categorical_cols: list[str]
) -> CategorySchema:
    return {
        col: pd.CategoricalDtype(categories=sorted(X[col].astype(str).unique()))
        for col in categorical_cols
    }


def apply_category_schema(X: pd.DataFrame, schema: CategorySchema) -> pd.DataFrame:
    X = X.copy()
    for col, dtype in schema.items():
        if col in X.columns:
            # Both `Series.astype(CategoricalDtype)` and
            # `pd.Categorical(values, categories=...)` map an
            # out-of-categories value to NaN today, but pandas has
            # deprecated *constructing* a Categorical from any
            # non-null value outside its categories at all -- a future
            # version raises instead. So unseen values are masked to
            # None explicitly, first, with plain non-categorical
            # `Series.where`, and only in-schema values ever reach the
            # Categorical constructor.
            values = X[col].astype(str)
            values = values.where(values.isin(dtype.categories), other=None)
            X[col] = pd.Categorical(values, categories=dtype.categories)
    return X


def build_ridge_pipeline(
    numeric_cols: list[str], categorical_cols: list[str], alpha: float = 10.0
) -> Pipeline:
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("ridge", Ridge(alpha=alpha))])


def _log_and_dollar_metrics(y_true_log: pd.Series, y_pred_log: np.ndarray) -> Metrics:
    y_true_dollars = np.expm1(y_true_log)
    y_pred_dollars = np.expm1(y_pred_log)
    return Metrics(
        rmse_log=float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        mae_log=float(mean_absolute_error(y_true_log, y_pred_log)),
        r2_log=float(r2_score(y_true_log, y_pred_log)),
        rmse_dollars=float(np.sqrt(mean_squared_error(y_true_dollars, y_pred_dollars))),
        mae_dollars=float(mean_absolute_error(y_true_dollars, y_pred_dollars)),
        r2_dollars=float(r2_score(y_true_dollars, y_pred_dollars)),
    )


def train_models(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    X_test: pd.DataFrame,
    y_test_log: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> TrainedModels:
    ridge = build_ridge_pipeline(numeric_cols, categorical_cols)
    ridge.fit(X_train, y_train_log)
    ridge_metrics = _log_and_dollar_metrics(y_test_log, ridge.predict(X_test))

    schema = build_category_schema(X_train, categorical_cols)
    X_train_lgb = apply_category_schema(X_train, schema)
    X_test_lgb = apply_category_schema(X_test, schema)
    lgbm = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    lgbm.fit(X_train_lgb, y_train_log, categorical_feature=categorical_cols)
    lightgbm_metrics = _log_and_dollar_metrics(y_test_log, lgbm.predict(X_test_lgb))

    return TrainedModels(
        ridge=ridge,
        lightgbm=lgbm,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        category_schema=schema,
        ridge_metrics=ridge_metrics,
        lightgbm_metrics=lightgbm_metrics,
    )


def default_house_row(
    X_train: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> pd.DataFrame:
    """One synthetic 'typical' house: the training median for every
    numeric column, the training mode for every categorical column. The
    API starts from this and overrides only the handful of fields the
    user actually edits."""
    row: dict[str, object] = {}
    for col in numeric_cols:
        row[col] = X_train[col].median()
    for col in categorical_cols:
        row[col] = X_train[col].mode(dropna=True).iat[0]
    return pd.DataFrame([row])[list(X_train.columns)]


def predict_both(models: TrainedModels, house: pd.DataFrame) -> dict[str, float]:
    """Dollar-scale prediction from both models for a single-row house
    dataframe shaped like the training features."""
    ridge_pred_log = float(models.ridge.predict(house)[0])
    house_lgb = apply_category_schema(house, models.category_schema)
    lgbm_pred_log = float(models.lightgbm.predict(house_lgb)[0])
    return {
        "ridge": float(np.expm1(ridge_pred_log)),
        "lightgbm": float(np.expm1(lgbm_pred_log)),
    }
