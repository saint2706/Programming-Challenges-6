"""Model training + threshold-dependent scoring: a Logistic Regression
baseline and a LightGBM classifier, trained once and both exposing
`predict_proba` so the dashboard can turn one slider into every
threshold-dependent metric without retraining anything.

The hard part -- carried over near-identically from this repo's House
Price Predictor challenge, same shape, different transform: LightGBM's
native pandas-categorical support encodes each category as an integer
*at fit time*, keyed to that column's `pd.Categorical` category order. A
naive `.astype("category")` on a single new row builds a fresh category
list from just the values present in that one row, which can assign a
*different* code to the same string value than training used -- the
model would silently split on the wrong branch, no error, no warning.
`build_category_schema` freezes the training-time category order per
column; `apply_category_schema` re-applies that exact schema to any
inference frame, mapping an unseen value to `NaN` (LightGBM's native
missing-value code) rather than a wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

CategorySchema = dict[str, pd.CategoricalDtype]


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True)
class CurveMetrics:
    roc_auc: float
    pr_auc: float
    fpr: np.ndarray
    tpr: np.ndarray
    precision_curve: np.ndarray
    recall_curve: np.ndarray


@dataclass
class TrainedModels:
    logreg: Pipeline
    lightgbm: lgb.LGBMClassifier
    numeric_cols: list[str]
    categorical_cols: list[str]
    category_schema: CategorySchema
    X_test: pd.DataFrame
    y_test: pd.Series
    logreg_proba: np.ndarray
    lightgbm_proba: np.ndarray


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
            # A future pandas version raises (rather than warns) when a
            # Categorical is constructed from a value outside its
            # declared categories, so unseen values are masked to None
            # with plain (non-categorical) Series.where first, and only
            # in-schema values ever reach the Categorical constructor.
            values = X[col].astype(str)
            values = values.where(values.isin(dtype.categories), other=None)
            X[col] = pd.Categorical(values, categories=dtype.categories)
    return X


def build_logreg_pipeline(
    numeric_cols: list[str], categorical_cols: list[str]
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
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("logreg", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> TrainedModels:
    logreg = build_logreg_pipeline(numeric_cols, categorical_cols)
    logreg.fit(X_train, y_train)
    logreg_proba = logreg.predict_proba(X_test)[:, 1]

    schema = build_category_schema(X_train, categorical_cols)
    X_train_lgb = apply_category_schema(X_train, schema)
    X_test_lgb = apply_category_schema(X_test, schema)
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    lgbm.fit(X_train_lgb, y_train, categorical_feature=categorical_cols)
    lightgbm_proba = lgbm.predict_proba(X_test_lgb)[:, 1]

    return TrainedModels(
        logreg=logreg,
        lightgbm=lgbm,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        category_schema=schema,
        X_test=X_test,
        y_test=y_test,
        logreg_proba=logreg_proba,
        lightgbm_proba=lightgbm_proba,
    )


def predict_proba_both(models: TrainedModels, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Churn probability from both models for an arbitrary feature frame
    (used by the dashboard's per-customer drill-down, not just the
    cached test-set predictions)."""
    X_lgb = apply_category_schema(X, models.category_schema)
    return {
        "logreg": models.logreg.predict_proba(X)[:, 1],
        "lightgbm": models.lightgbm.predict_proba(X_lgb)[:, 1],
    }


def metrics_at_threshold(
    y_true: pd.Series | np.ndarray, y_proba: np.ndarray, threshold: float
) -> ThresholdMetrics:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return ThresholdMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )


def compute_curves(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> CurveMetrics:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    return CurveMetrics(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        fpr=fpr,
        tpr=tpr,
        precision_curve=precision_curve,
        recall_curve=recall_curve,
    )


def best_f1_threshold(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> float:
    """The threshold (scanned over the probabilities actually produced,
    not a fixed grid) that maximizes F1 on this data -- used as the
    dashboard's *starting* slider position instead of a naive 0.5, since
    a naive 0.5 threshold on a classifier trained on a ~26.5%-positive
    class systematically under-flags churners (see README for the
    measured gap). The user can still drag the slider anywhere; this
    only picks where it starts."""
    candidates = np.unique(y_proba)
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        f1 = metrics_at_threshold(y_true, y_proba, float(threshold)).f1
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(threshold)
    return best_threshold
