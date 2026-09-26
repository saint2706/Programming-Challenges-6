"""House Price Predictor with Feature Importance Explainer.

Trains a Ridge (linear) baseline and a LightGBM (tree) model on the Ames
Housing dataset at startup, then serves a small FastAPI+HTMX dashboard:
edit the handful of features SHAP itself ranked most important, get both
models' dollar predictions, and a per-model breakdown of which features
pushed the price up or down and by how much.

Run:
    uv run --with fastapi --with "uvicorn[standard]" --with scikit-learn \\
        --with lightgbm --with shap --with pandas --with numpy \\
        --with matplotlib --with python-multipart python app.py

Pages are plain f-string HTML fragments (no template directory, no
static file directory) per this repo's convention. HTMX swaps in the
`/predict` result without a full page reload.
"""

from __future__ import annotations

import html
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
import uvicorn
from data import (
    clean_features,
    load_ames_housing,
    numeric_and_categorical_columns,
    train_test_split_frame,
)
from explain import (
    compute_global_importance,
    explain_lightgbm,
    explain_ridge,
    render_contribution_chart,
    render_global_importance_chart,
    top_combined_features,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from model import default_house_row, predict_both, train_models

EDITABLE_FEATURE_COUNT = 12
BACKGROUND_SAMPLE_SIZE = (
    100  # shap.LinearExplainer's default masker caps at 100 samples
)

STATE: dict[str, object] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    df = clean_features(load_ames_housing())
    numeric_cols, categorical_cols = numeric_and_categorical_columns(df)
    X_train, X_test, y_train_log, y_test_log = train_test_split_frame(df)
    models = train_models(
        X_train, y_train_log, X_test, y_test_log, numeric_cols, categorical_cols
    )

    importance_sample = X_test.sample(min(100, len(X_test)), random_state=7)
    importance = compute_global_importance(models, importance_sample)
    editable_features = top_combined_features(importance, EDITABLE_FEATURE_COUNT)

    STATE.update(
        models=models,
        X_train=X_train,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        editable_features=editable_features,
        default_row=default_house_row(X_train, numeric_cols, categorical_cols),
        background=X_train.sample(
            min(BACKGROUND_SAMPLE_SIZE, len(X_train)), random_state=11
        ),
        global_chart=render_global_importance_chart(importance),
        category_options={
            col: sorted(X_train[col].astype(str).unique()) for col in categorical_cols
        },
    )
    yield
    STATE.clear()


app = FastAPI(title="House Price Predictor", lifespan=lifespan)


CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
header { margin-bottom: 1.5rem; }
header a.brand { font-weight: 700; font-size: 1.25rem; text-decoration: none; }
.card { border: 1px solid #8884; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0; }
form .row { margin: 0.6rem 0; display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
label { display: flex; flex-direction: column; font-size: 0.85rem; gap: 0.25rem; min-width: 180px; }
input, select, button { font-size: 1rem; padding: 0.4rem 0.6rem; }
button { cursor: pointer; }
img { max-width: 100%; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0; }
th, td { border: 1px solid #8884; padding: 0.35rem 0.6rem; text-align: left; }
.muted { color: #666; font-size: 0.85rem; }
.pred-row { display: flex; gap: 2rem; flex-wrap: wrap; }
.pred-box { font-size: 1.4rem; font-weight: 700; }
</style>
"""


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script src="https://unpkg.com/htmx.org@2.0.3/dist/htmx.min.js"
        integrity="sha384-0895/pl2MU10Hqc6jd4RvrthNlDiE9U1tWmX7WRESftEDRosgxNsQG/Ze9YMRzHq"
        crossorigin="anonymous"></script>
<style>{CSS}
</head>
<body>
<header><a class="brand" href="/">House Price Predictor</a></header>
<main>{body}</main>
</body>
</html>""")


def _is_integer_column(col: str) -> bool:
    X_train: pd.DataFrame = STATE["X_train"]  # type: ignore[assignment]
    return pd.api.types.is_integer_dtype(X_train[col].dtype)


def _field_html(col: str) -> str:
    default_row: pd.DataFrame = STATE["default_row"]  # type: ignore[assignment]
    categorical_cols: list[str] = STATE["categorical_cols"]  # type: ignore[assignment]
    category_options: dict[str, list[str]] = STATE["category_options"]  # type: ignore[assignment]
    current = default_row.iloc[0][col]

    if col in categorical_cols:
        options = "".join(
            f'<option value="{html.escape(opt)}"{" selected" if opt == current else ""}>{html.escape(opt)}</option>'
            for opt in category_options[col]
        )
        return f'<label>{html.escape(col)}<select name="{html.escape(col)}">{options}</select></label>'

    step = "1" if _is_integer_column(col) else "any"
    value = int(current) if _is_integer_column(col) else float(current)
    return (
        f"<label>{html.escape(col)}"
        f'<input type="number" step="{step}" name="{html.escape(col)}" value="{value}"></label>'
    )


def render_dashboard() -> HTMLResponse:
    editable_features: list[str] = STATE["editable_features"]  # type: ignore[assignment]
    fields = "".join(
        f'<div class="row">{_field_html(col)}</div>' for col in editable_features
    )
    body = f"""
<div class="card">
  <h2>Predict a house's price</h2>
  <p class="muted">These {len(editable_features)} fields are the ones SHAP ranked as the
  most important across both models. Every other one of the dataset's ~79 features is set
  to its training-set median (numeric) or most common value (categorical) automatically.</p>
  <form hx-post="/predict" hx-target="#result" hx-swap="innerHTML">
    {fields}
    <button type="submit">Predict</button>
  </form>
  <div id="result"></div>
</div>

<div class="card">
  <h2>Global feature importance</h2>
  <p class="muted">Mean |SHAP value| across a held-out sample, both models side by side.</p>
  <img alt="Global feature importance chart" src="data:image/png;base64,{STATE["global_chart"]}">
</div>
"""
    return page("House Price Predictor", body)


def _contribution_table(top_contributions_dollars: list[tuple[str, float]]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td>"
        f'<td style="color:{"#2e7d32" if value >= 0 else "#c62828"}">'
        f"{'+' if value >= 0 else ''}${value:,.0f}</td></tr>"
        for name, value in top_contributions_dollars
    )
    return f"<table><thead><tr><th>Feature</th><th>Approx. dollar impact</th></tr></thead><tbody>{rows}</tbody></table>"


def _parse_editable_fields(form: dict) -> pd.DataFrame:
    default_row: pd.DataFrame = STATE["default_row"]  # type: ignore[assignment]
    categorical_cols: list[str] = STATE["categorical_cols"]  # type: ignore[assignment]
    editable_features: list[str] = STATE["editable_features"]  # type: ignore[assignment]
    category_options: dict[str, list[str]] = STATE["category_options"]  # type: ignore[assignment]

    house = default_row.copy()
    for col in editable_features:
        if col not in form:
            continue
        raw = str(form[col])
        if col in categorical_cols:
            if raw not in category_options[col]:
                raise HTTPException(422, f"Unknown value {raw!r} for {col!r}")
            house.loc[0, col] = raw
        else:
            try:
                house.loc[0, col] = float(raw)
            except ValueError as exc:
                raise HTTPException(
                    422, f"Invalid numeric value {raw!r} for {col!r}"
                ) from exc
    return house


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return render_dashboard()


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request) -> HTMLResponse:
    form = dict(await request.form())
    house = _parse_editable_fields(form)

    models = STATE["models"]
    background: pd.DataFrame = STATE["background"]  # type: ignore[assignment]
    prices = predict_both(models, house)  # type: ignore[arg-type]
    ridge_explanation = explain_ridge(models, house, background)  # type: ignore[arg-type]
    lgbm_explanation = explain_lightgbm(models, house)  # type: ignore[arg-type]

    body = f"""
<div class="pred-row">
  <div class="pred-box">Ridge: ${prices["ridge"]:,.0f}</div>
  <div class="pred-box">LightGBM: ${prices["lightgbm"]:,.0f}</div>
</div>
<div class="pred-row">
  <div>
    <h3>Ridge contribution breakdown</h3>
    <img alt="Ridge contribution chart" src="data:image/png;base64,{render_contribution_chart(ridge_explanation)}">
    {_contribution_table(ridge_explanation.top_contributions_dollars)}
  </div>
  <div>
    <h3>LightGBM contribution breakdown</h3>
    <img alt="LightGBM contribution chart" src="data:image/png;base64,{render_contribution_chart(lgbm_explanation)}">
    {_contribution_table(lgbm_explanation.top_contributions_dollars)}
  </div>
</div>
"""
    return HTMLResponse(body)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "models_loaded": "models" in STATE}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)
