"""SHAP-based explainability for both models, at a comparable grain.

The hard part: Ridge's coefficients live in the *transformed* feature
space -- scaled numerics plus one-hot-expanded categoricals, one output
column per category value -- while LightGBM's `TreeExplainer` values
already live in the model's native per-column space (one value per
original Ames column, categories handled natively, no expansion).
Displaying Ridge's raw per-one-hot-column SHAP values next to
LightGBM's per-column ones isn't comparable: a 25-value column like
`Neighborhood` would show 25 tiny slivers next to LightGBM's single
`Neighborhood: +$5,100`. `_transformed_feature_groups` uses the
*lengths* of `OneHotEncoder.categories_` -- not string-parsing of
generated feature names, which breaks the moment one column name is a
prefix of another (e.g. `Bsmt` vs `BsmtQual`) -- to know exactly which
transformed columns came from which original column, and
`aggregate_by_original_column` sums a linear model's per-one-hot-column
SHAP values back down to one number per original column: the same
grain LightGBM already reports at.

A second, smaller hard part: both models were trained on `log1p(price)`,
and SHAP's additivity guarantee (`sum(shap_values) + base_value ==
model_output`) holds *in that log space*, not after converting to
dollars (`expm1` doesn't distribute over a sum). The correctness test in
`test_explain.py` checks the real invariant in log space. For display,
each feature's dollar figure is instead an honest, independently
well-defined counterfactual -- "how much would the dollar prediction
drop if exactly this feature's log-space contribution were removed,
holding every other feature's contribution fixed"
(`expm1(log_pred) - expm1(log_pred - shap_i)`) -- rather than a fabricated
linear rescale of the log-space number. These per-feature dollar deltas
do *not* sum exactly to `prediction - base` in dollars (that quantity
isn't decomposable in dollar space at all for a log-target model); only
the log-space values carry that exact guarantee.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from model import TrainedModels, apply_category_schema

TOP_N = 8


@dataclass
class Explanation:
    model_name: str
    base_value_log: float
    prediction_log: float
    base_value_dollars: float
    prediction_dollars: float
    contributions_log: dict[str, float]  # every original column, log-space, signed
    top_contributions_dollars: list[tuple[str, float]] = field(default_factory=list)


def _to_scalar(value: object) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _transformed_feature_groups(models: TrainedModels) -> list[str]:
    preprocessor = models.ridge.named_steps["preprocess"]
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


def _ridge_transform(models: TrainedModels, X: pd.DataFrame) -> np.ndarray:
    preprocessor = models.ridge.named_steps["preprocess"]
    transformed = preprocessor.transform(X)
    return (
        transformed.toarray()
        if hasattr(transformed, "toarray")
        else np.asarray(transformed)
    )


def dollar_delta(log_pred: float, shap_value_log: float) -> float:
    """The single-feature counterfactual dollar impact of `shap_value_log`
    -- see module docstring for why this, not a linear rescale."""
    return float(np.expm1(log_pred) - np.expm1(log_pred - shap_value_log))


def _top_n_dollars(
    contributions_log: dict[str, float], log_pred: float, n: int = TOP_N
) -> list[tuple[str, float]]:
    ranked = sorted(contributions_log.items(), key=lambda kv: -abs(kv[1]))[:n]
    return [(name, dollar_delta(log_pred, value)) for name, value in ranked]


def explain_ridge(
    models: TrainedModels, house: pd.DataFrame, background: pd.DataFrame
) -> Explanation:
    background_t = _ridge_transform(models, background)
    house_t = _ridge_transform(models, house)
    explainer = shap.LinearExplainer(models.ridge.named_steps["ridge"], background_t)
    shap_row = np.asarray(explainer.shap_values(house_t))[0]
    groups = _transformed_feature_groups(models)
    contributions_log = aggregate_by_original_column(shap_row, groups)

    base_log = _to_scalar(explainer.expected_value)
    pred_log = float(models.ridge.predict(house)[0])
    return Explanation(
        model_name="ridge",
        base_value_log=base_log,
        prediction_log=pred_log,
        base_value_dollars=float(np.expm1(base_log)),
        prediction_dollars=float(np.expm1(pred_log)),
        contributions_log=contributions_log,
        top_contributions_dollars=_top_n_dollars(contributions_log, pred_log),
    )


def explain_lightgbm(models: TrainedModels, house: pd.DataFrame) -> Explanation:
    house_lgb = apply_category_schema(house, models.category_schema)
    explainer = shap.TreeExplainer(models.lightgbm)
    shap_row = np.asarray(explainer.shap_values(house_lgb))[0]
    contributions_log = {
        col: float(value) for col, value in zip(house_lgb.columns, shap_row)
    }

    base_log = _to_scalar(explainer.expected_value)
    pred_log = float(models.lightgbm.predict(house_lgb)[0])
    return Explanation(
        model_name="lightgbm",
        base_value_log=base_log,
        prediction_log=pred_log,
        base_value_dollars=float(np.expm1(base_log)),
        prediction_dollars=float(np.expm1(pred_log)),
        contributions_log=contributions_log,
        top_contributions_dollars=_top_n_dollars(contributions_log, pred_log),
    )


def compute_global_importance(
    models: TrainedModels, X_sample: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Mean |SHAP value| per original column, across a sample of rows,
    for both models -- used both for the global chart and to pick which
    features the API's form lets you edit."""
    background_t = _ridge_transform(models, X_sample)
    ridge_explainer = shap.LinearExplainer(
        models.ridge.named_steps["ridge"], background_t
    )
    ridge_shap = np.asarray(ridge_explainer.shap_values(background_t))
    groups = _transformed_feature_groups(models)
    ridge_importance: dict[str, float] = {}
    for row in ridge_shap:
        for name, value in aggregate_by_original_column(row, groups).items():
            ridge_importance[name] = ridge_importance.get(name, 0.0) + abs(value)
    ridge_importance = {k: v / len(ridge_shap) for k, v in ridge_importance.items()}

    X_sample_lgb = apply_category_schema(X_sample, models.category_schema)
    lgbm_explainer = shap.TreeExplainer(models.lightgbm)
    lgbm_shap = np.asarray(lgbm_explainer.shap_values(X_sample_lgb))
    lightgbm_importance = dict(
        zip(X_sample_lgb.columns, np.abs(lgbm_shap).mean(axis=0).tolist())
    )

    return {"ridge": ridge_importance, "lightgbm": lightgbm_importance}


def top_combined_features(
    importance_by_model: dict[str, dict[str, float]], n: int
) -> list[str]:
    """Rank features by normalized combined importance across both
    models (each model's scores divided by its own max first, so
    LightGBM's larger raw SHAP magnitudes don't drown out Ridge's)."""
    combined: dict[str, float] = {}
    for scores in importance_by_model.values():
        peak = max(scores.values()) if scores else 1.0
        for name, value in scores.items():
            combined[name] = combined.get(name, 0.0) + (value / peak if peak else 0.0)
    return [name for name, _ in sorted(combined.items(), key=lambda kv: -kv[1])[:n]]


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_global_importance_chart(
    importance_by_model: dict[str, dict[str, float]], top_n: int = 12
) -> str:
    features = top_combined_features(importance_by_model, top_n)
    ridge_vals = [importance_by_model["ridge"].get(f, 0.0) for f in features]
    lgbm_vals = [importance_by_model["lightgbm"].get(f, 0.0) for f in features]

    y = np.arange(len(features))
    height = 0.38
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(features) + 1))
    ax.barh(y + height / 2, ridge_vals, height=height, label="Ridge", color="#4c72b0")
    ax.barh(y - height / 2, lgbm_vals, height=height, label="LightGBM", color="#dd8452")
    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (log-price scale)")
    ax.set_title("Global feature importance")
    ax.legend()
    fig.tight_layout()
    return _fig_to_base64(fig)


def render_contribution_chart(explanation: Explanation, top_n: int = TOP_N) -> str:
    items = explanation.top_contributions_dollars[:top_n][::-1]
    names = [name for name, _ in items]
    values = [value for _, value in items]
    colors = ["#2e7d32" if v >= 0 else "#c62828" for v in values]

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(items) + 1.2))
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Approx. dollar impact of this feature")
    ax.set_title(
        f"{explanation.model_name}: base ${explanation.base_value_dollars:,.0f} "
        f"-> predicted ${explanation.prediction_dollars:,.0f}"
    )
    fig.tight_layout()
    return _fig_to_base64(fig)
