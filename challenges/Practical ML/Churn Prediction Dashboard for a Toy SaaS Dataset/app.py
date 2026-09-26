"""Streamlit dashboard: train a Logistic Regression baseline and a
LightGBM classifier once on the Telco Customer Churn data (used here as a
stand-in for a toy SaaS churn scenario -- see `data.py`), then let the
user drag one slider to see every threshold-dependent metric, curve
point, and per-customer flag update live -- no retraining involved,
since `model.metrics_at_threshold` recomputes everything from the
already-cached test-set probabilities.
"""

from __future__ import annotations

import data
import explain
import model
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

MODEL_LABELS = {"logreg": "Logistic Regression", "lightgbm": "LightGBM"}
GLOBAL_IMPORTANCE_SAMPLE_SIZE = 300
EXPLANATION_BACKGROUND_SIZE = 100


@st.cache_resource
def load_trained_models() -> model.TrainedModels:
    df = data.clean_features(data.load_telco_churn())
    numeric_cols, categorical_cols = data.numeric_and_categorical_columns(df)
    X_train, X_test, y_train, y_test = data.train_test_split_frame(df)
    return model.train_models(
        X_train, y_train, X_test, y_test, numeric_cols, categorical_cols
    )


@st.cache_resource
def compute_global_importance(_trained: model.TrainedModels) -> dict:
    sample = _trained.X_test.sample(
        n=min(GLOBAL_IMPORTANCE_SAMPLE_SIZE, len(_trained.X_test)), random_state=0
    )
    return explain.compute_global_importance(_trained, sample)


trained = load_trained_models()
proba_by_model = {"logreg": trained.logreg_proba, "lightgbm": trained.lightgbm_proba}
curves_by_model = {
    key: model.compute_curves(trained.y_test, proba)
    for key, proba in proba_by_model.items()
}
default_threshold_by_model = {
    key: model.best_f1_threshold(trained.y_test, proba)
    for key, proba in proba_by_model.items()
}

st.title("Churn Prediction Dashboard for a Toy SaaS Dataset")
st.caption(
    "Telco Customer Churn (OpenML data_id=42178, used here as a stand-in "
    "for a toy SaaS churn scenario -- see README) -- ~26.5% of customers "
    "in this data churned."
)

model_label = st.sidebar.selectbox("Model", list(MODEL_LABELS.values()))
model_key = next(key for key, label in MODEL_LABELS.items() if label == model_label)
proba = proba_by_model[model_key]
curve = curves_by_model[model_key]
default_threshold = default_threshold_by_model[model_key]

threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(default_threshold),
    step=0.01,
)
st.sidebar.caption(
    f"Starts at {default_threshold:.2f}, the F1-optimal threshold for "
    f"{model_label} on the held-out test set -- not a naive 0.50, which "
    "under-flags churners on this imbalanced (~26.5% positive) dataset. "
    "Drag it anywhere."
)

metrics = model.metrics_at_threshold(trained.y_test, proba, threshold)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics.accuracy:.1%}")
col2.metric("Precision", f"{metrics.precision:.1%}")
col3.metric("Recall", f"{metrics.recall:.1%}")
col4.metric("F1 score", f"{metrics.f1:.3f}")
st.caption(
    f"ROC-AUC: {curve.roc_auc:.3f} | PR-AUC: {curve.pr_auc:.3f} "
    "(both threshold-independent -- they describe the model's ranking "
    "quality, not this specific cutoff)"
)

left, right = st.columns([1, 2])
with left:
    st.subheader("Confusion matrix")
    confusion_df = pd.DataFrame(
        [[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]],
        index=["Actual: stayed", "Actual: churned"],
        columns=["Predicted: stayed", "Predicted: churned"],
    )
    st.dataframe(confusion_df)
with right:
    curve_col1, curve_col2 = st.columns(2)
    curve_col1.pyplot(explain.render_roc_curve_chart(curve, metrics))
    curve_col2.pyplot(explain.render_pr_curve_chart(curve, metrics))

st.subheader("Global feature importance (SHAP, logit scale)")
importance = compute_global_importance(trained)
st.pyplot(explain.render_global_importance_chart(importance))

st.subheader("Per-customer drill-down")
row_index = st.number_input(
    "Test-set row index",
    min_value=0,
    max_value=len(trained.X_test) - 1,
    value=0,
    step=1,
)
customer = trained.X_test.iloc[[row_index]]
customer_proba = float(proba[row_index])
actual_label = "Churned" if int(trained.y_test.iloc[row_index]) == 1 else "Stayed"
flagged_label = "Yes" if customer_proba >= threshold else "No"

info_col1, info_col2, info_col3 = st.columns(3)
info_col1.metric("Actual outcome", actual_label)
info_col2.metric("Predicted churn probability", f"{customer_proba:.1%}")
info_col3.metric("Flagged at current threshold", flagged_label)

background = trained.X_test.sample(
    n=min(EXPLANATION_BACKGROUND_SIZE, len(trained.X_test)), random_state=0
)
if model_key == "logreg":
    explanation = explain.explain_logreg(trained, customer, background)
else:
    explanation = explain.explain_lightgbm(trained, customer)
st.pyplot(explain.render_contribution_chart(explanation))
