"""SHAP-based explainability for both models, at a comparable grain and
in a comparable *space*.

The hard part, same shape as this repo's House Price Predictor challenge
but with logits standing in for log-prices: both models' raw outputs are
model-internal (Logistic Regression's linear decision function, LightGBM's
raw tree-ensemble score), and SHAP's additivity guarantee
(`sum(shap_values) + base_value == raw_output`) holds in that *logit*
(log-odds) space, not after converting to a probability with the sigmoid
(sigmoid doesn't distribute over a sum, exactly like `expm1` didn't for
the house-price challenge's log-target model). For display, each
feature's probability-point figure is instead an honest, independently
well-defined counterfactual -- "how much would the predicted probability
drop if exactly this feature's logit contribution were removed, holding
every other feature's fixed" (`sigmoid(logit) - sigmoid(logit - shap_i)`)
-- rather than a fabricated linear rescale of the logit-space number.
These per-feature probability deltas do *not* sum exactly to
`prediction - base` in probability space (that decomposition doesn't
exist there for a logit-space-additive model); only the logit-space
values carry SHAP's exact guarantee, and that's what the correctness
tests check.

A second hard part, also carried over from the House Price Predictor:
Logistic Regression's coefficients live in the *transformed* feature
space -- scaled numerics plus one-hot-expanded categoricals, one output
column per category value -- while LightGBM's `TreeExplainer` values
already live in the model's native per-column space. `_transformed_feature_groups`
uses the *lengths* of `OneHotEncoder.categories_` (not string-parsing of
generated feature names, which breaks the moment one column name is a
prefix of another) to know exactly which transformed columns came from
which original column, and `aggregate_by_original_column` sums a linear
model's per-one-hot-column SHAP values back down to the same grain
LightGBM already reports at.

A third, smaller hard part: `shap.TreeExplainer` on an `LGBMClassifier`
has changed its output shape across shap versions (a plain 2D array in
some versions, a `[class_0, class_1]` list or a 3D array in others) --
`_positive_class_shap_values` normalizes all of these to one 2D
(n_samples, n_features) array of the positive (churn) class's
contribution, so the rest of this module doesn't have to care which
shap version is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from model import CurveMetrics, ThresholdMetrics, TrainedModels, apply_category_schema

TOP_N = 8


@dataclass
class Explanation:
    model_name: str
    base_logit: float
    prediction_logit: float
    base_probability: float
    prediction_probability: float
    contributions_logit: dict[str, float]  # every original column, logit-space, signed
    top_contributions_probability: list[tuple[str, float]] = field(default_factory=list)


def _to_scalar(value: object) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _positive_class_shap_values(shap_values: object) -> np.ndarray:
    """Normalize shap.TreeExplainer's binary-classifier output -- a plain
    2D array, a [class_0, class_1] list, or a 3D (n, features, classes)
    array depending on the installed shap version -- to one 2D
    (n_samples, n_features) array of the positive class's contribution."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _transformed_feature_groups(models: TrainedModels) -> list[str]:
    preprocessor = models.logreg.named_steps["preprocess"]
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    groups = list(models.numeric_cols)
    for col, categories in zip(models.categorical_cols, onehot.categories_):
        groups.extend([col] * len(categories))
    return groups


def aggregate_by_original_column(
    shap_row: np.ndarray, groups: list[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for group, value in zip(groups, shap_row):
        out[group] = out.get(group, 0.0) + float(value)
    return out


def _logreg_transform(models: TrainedModels, X: pd.DataFrame) -> np.ndarray:
    preprocessor = models.logreg.named_steps["preprocess"]
    transformed = preprocessor.transform(X)
    return (
        transformed.toarray()
        if hasattr(transformed, "toarray")
        else np.asarray(transformed)
    )


def probability_delta(logit_pred: float, shap_value_logit: float) -> float:
    """The single-feature counterfactual probability-point impact of
    `shap_value_logit` -- see module docstring for why this, not a
    linear rescale."""
    return _sigmoid(logit_pred) - _sigmoid(logit_pred - shap_value_logit)


def _top_n_probability(
    contributions_logit: dict[str, float], logit_pred: float, n: int = TOP_N
) -> list[tuple[str, float]]:
    ranked = sorted(contributions_logit.items(), key=lambda kv: -abs(kv[1]))[:n]
    return [(name, probability_delta(logit_pred, value)) for name, value in ranked]


def explain_logreg(
    models: TrainedModels, customer: pd.DataFrame, background: pd.DataFrame
) -> Explanation:
    background_t = _logreg_transform(models, background)
    customer_t = _logreg_transform(models, customer)
    explainer = shap.LinearExplainer(models.logreg.named_steps["logreg"], background_t)
    shap_row = np.asarray(explainer.shap_values(customer_t))[0]
    groups = _transformed_feature_groups(models)
    contributions_logit = aggregate_by_original_column(shap_row, groups)

    base_logit = _to_scalar(explainer.expected_value)
    pred_logit = float(models.logreg.decision_function(customer)[0])
    return Explanation(
        model_name="logreg",
        base_logit=base_logit,
        prediction_logit=pred_logit,
        base_probability=_sigmoid(base_logit),
        prediction_probability=_sigmoid(pred_logit),
        contributions_logit=contributions_logit,
        top_contributions_probability=_top_n_probability(
            contributions_logit, pred_logit
        ),
    )


def explain_lightgbm(models: TrainedModels, customer: pd.DataFrame) -> Explanation:
    customer_lgb = apply_category_schema(customer, models.category_schema)
    explainer = shap.TreeExplainer(models.lightgbm)
    shap_row = _positive_class_shap_values(explainer.shap_values(customer_lgb))[0]
    contributions_logit = {
        col: float(value) for col, value in zip(customer_lgb.columns, shap_row)
    }

    base_logit = _to_scalar(explainer.expected_value)
    pred_logit = float(models.lightgbm.predict(customer_lgb, raw_score=True)[0])
    return Explanation(
        model_name="lightgbm",
        base_logit=base_logit,
        prediction_logit=pred_logit,
        base_probability=_sigmoid(base_logit),
        prediction_probability=_sigmoid(pred_logit),
        contributions_logit=contributions_logit,
        top_contributions_probability=_top_n_probability(
            contributions_logit, pred_logit
        ),
    )


def compute_global_importance(
    models: TrainedModels, X_sample: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Mean |SHAP value| per original column, across a sample of rows,
    for both models -- logit-space, since that's the space where the
    magnitudes are directly comparable between the two models."""
    background_t = _logreg_transform(models, X_sample)
    logreg_explainer = shap.LinearExplainer(
        models.logreg.named_steps["logreg"], background_t
    )
    logreg_shap = np.asarray(logreg_explainer.shap_values(background_t))
    groups = _transformed_feature_groups(models)
    logreg_importance: dict[str, float] = {}
    for row in logreg_shap:
        for name, value in aggregate_by_original_column(row, groups).items():
            logreg_importance[name] = logreg_importance.get(name, 0.0) + abs(value)
    logreg_importance = {k: v / len(logreg_shap) for k, v in logreg_importance.items()}

    X_sample_lgb = apply_category_schema(X_sample, models.category_schema)
    lgbm_explainer = shap.TreeExplainer(models.lightgbm)
    lgbm_shap = _positive_class_shap_values(lgbm_explainer.shap_values(X_sample_lgb))
    lightgbm_importance = dict(
        zip(X_sample_lgb.columns, np.abs(lgbm_shap).mean(axis=0).tolist())
    )

    return {"logreg": logreg_importance, "lightgbm": lightgbm_importance}


def top_combined_features(
    importance_by_model: dict[str, dict[str, float]], n: int
) -> list[str]:
    """Rank features by normalized combined importance across both
    models (each model's scores divided by its own max first, so one
    model's larger raw SHAP magnitudes don't drown out the other's)."""
    combined: dict[str, float] = {}
    for scores in importance_by_model.values():
        peak = max(scores.values()) if scores else 1.0
        for name, value in scores.items():
            combined[name] = combined.get(name, 0.0) + (value / peak if peak else 0.0)
    return [name for name, _ in sorted(combined.items(), key=lambda kv: -kv[1])[:n]]


def render_global_importance_chart(
    importance_by_model: dict[str, dict[str, float]], top_n: int = 12
) -> plt.Figure:
    features = top_combined_features(importance_by_model, top_n)
    logreg_vals = [importance_by_model["logreg"].get(f, 0.0) for f in features]
    lgbm_vals = [importance_by_model["lightgbm"].get(f, 0.0) for f in features]

    y = np.arange(len(features))
    height = 0.38
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(features) + 1))
    ax.barh(
        y + height / 2,
        logreg_vals,
        height=height,
        label="Logistic Regression",
        color="#4c72b0",
    )
    ax.barh(y - height / 2, lgbm_vals, height=height, label="LightGBM", color="#dd8452")
    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (logit scale)")
    ax.set_title("Global feature importance")
    ax.legend()
    fig.tight_layout()
    return fig


def render_roc_curve_chart(
    curve: CurveMetrics, current: ThresholdMetrics
) -> plt.Figure:
    denom_fpr = current.fp + current.tn
    denom_tpr = current.tp + current.fn
    current_fpr = current.fp / denom_fpr if denom_fpr else 0.0
    current_tpr = current.tp / denom_tpr if denom_tpr else 0.0

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        curve.fpr, curve.tpr, color="#4c72b0", label=f"ROC (AUC={curve.roc_auc:.3f})"
    )
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    ax.scatter(
        [current_fpr],
        [current_tpr],
        color="#c62828",
        zorder=5,
        label="current threshold",
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def render_pr_curve_chart(curve: CurveMetrics, current: ThresholdMetrics) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        curve.recall_curve,
        curve.precision_curve,
        color="#4c72b0",
        label=f"PR (AUC={curve.pr_auc:.3f})",
    )
    ax.scatter(
        [current.recall],
        [current.precision],
        color="#c62828",
        zorder=5,
        label="current threshold",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def render_contribution_chart(
    explanation: Explanation, top_n: int = TOP_N
) -> plt.Figure:
    items = explanation.top_contributions_probability[:top_n][::-1]
    names = [name for name, _ in items]
    values = [value for _, value in items]
    colors = [
        "#c62828" if v >= 0 else "#2e7d32" for v in values
    ]  # red = pushes toward churn

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(items) + 1.2))
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Approx. change in churn probability")
    ax.set_title(
        f"{explanation.model_name}: base {explanation.base_probability:.0%} "
        f"-> predicted {explanation.prediction_probability:.0%}"
    )
    fig.tight_layout()
    return fig
