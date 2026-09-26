# House Price Predictor with Feature Importance Explainer

**Category:** Practical ML
**Difficulty:** B (brief: "Linear/tree model plus a clear feature-contribution breakdown.")

**Status:** Implemented (Python)

Trains a Ridge (linear) baseline and a LightGBM (tree) model on the
[Ames Housing dataset](https://www.openml.org/search?type=data&id=42165)
(the classic Kaggle "House Prices: Advanced Regression Techniques" data —
1,460 houses, ~79 features, real 2006–2010 Iowa sale prices), then serves
a FastAPI+HTMX dashboard where you edit the handful of features SHAP
itself ranked as most important, and get both models' predicted price
plus a signed, per-feature dollar breakdown of what pushed it up or down.

## The hard parts

**Reconciling two models' explanations onto one comparable scale.**
Ridge's coefficients live in the *transformed* feature space — scaled
numeric columns plus one-hot-expanded categoricals, one output column
per category value — while LightGBM's `TreeExplainer` values already
live in the model's native per-column space (categories handled
directly, no expansion). Showing Ridge's raw per-one-hot-column SHAP
values next to LightGBM's per-column ones isn't comparable: a 25-value
column like `Neighborhood` would show 25 tiny slivers next to
LightGBM's single `Neighborhood: +$5,100`. `explain._transformed_feature_groups`
uses the *lengths* of `OneHotEncoder.categories_` — not string-parsing
of generated feature names, which breaks the moment one column name is
a prefix of another (`Bsmt` vs. `BsmtQual`) — to know exactly which
transformed columns came from which original column, and
`aggregate_by_original_column` sums a linear model's per-one-hot-column
SHAP values back down to the same grain LightGBM already reports at.
Covered by `test_explain.py::test_ridge_explanation_covers_every_original_column_exactly_once`.

**A log-target model's SHAP values are only exactly additive in log
space.** Both models are trained on `log1p(SalePrice)` (the target is
right-skewed; this is standard practice for this dataset), and SHAP's
guarantee — `sum(shap_values) + base_value == model_output` — holds
*there*, not after converting to dollars (`expm1` doesn't distribute
over a sum). `test_explain.py` checks the real invariant in log space
for both explainers. For display, each feature's dollar figure is
instead an independently well-defined counterfactual — "how much would
the dollar prediction drop if exactly this feature's log-space
contribution were removed, holding every other feature's fixed"
(`expm1(log_pred) - expm1(log_pred - shap_i)`) — rather than a fabricated
linear rescale. These dollar deltas don't sum exactly to
`prediction - base` in dollars (that decomposition doesn't exist in
dollar space for a log-target model); only the log-space values carry
that exact guarantee.

**LightGBM's pandas-categorical codes must match between training and
inference.** LightGBM's native categorical support encodes each
category to an integer *at fit time*, keyed to that column's
`pd.Categorical` category order. A naive `.astype("category")` on a
single new row builds a fresh category list from just the values
present in that one row, which can assign a *different* code to the
same string value than training used — the model would silently split
on the wrong branch, no error, no warning. `model.build_category_schema`
freezes the training-time category order per column; `apply_category_schema`
re-applies that exact schema to any inference frame, mapping an unseen
value to `NaN` (LightGBM's native missing-value code) rather than a
wrong one. Also had to work around a pandas 3.x deprecation:
constructing a `Categorical` from a value outside its declared
categories now warns (a future version raises), so unseen values are
masked to `None` with plain `Series.where` *before* the `Categorical`
constructor ever sees them. Covered by
`test_model.py::test_category_schema_assigns_consistent_codes_regardless_of_row_order`
and `test_category_schema_maps_unseen_category_to_missing_not_a_wrong_code`.

## Design

- **`data.py`** — fetches Ames Housing from OpenML (`data_id=42165`)
  once, caches it to `data/ames_housing.csv` (gitignored — regenerated
  from the network on first run rather than vendored), and resolves
  every categorical `NaN` to an explicit sentinel: `"None"` for the
  ~15 columns the Ames data dictionary documents as "this house
  doesn't have the feature" (no pool, no fence, no basement…), vs.
  `"Missing"` for a genuinely missing value elsewhere (e.g. one row's
  `Electrical`) — collapsing the two into one fill value would corrupt
  exactly the columns most likely to swing a prediction. The target is
  `log1p`-transformed here, once, so every downstream consumer
  (training, tests, the API) works from the same transform.
- **`model.py`** — builds the Ridge pipeline (median-impute +
  `StandardScaler` for numerics, most-frequent-impute + one-hot for
  categoricals) and trains the LightGBM regressor directly on
  `pd.Categorical`-typed columns (native missing-value + categorical
  handling, no imputation needed). `predict_both` returns both models'
  dollar-scale prediction for one house. `default_house_row` builds a
  synthetic "typical" house (training median/mode for every column) that
  the API starts from and overrides only the fields you actually edit.
- **`explain.py`** — wraps `shap.LinearExplainer` (Ridge) and
  `shap.TreeExplainer` (LightGBM) behind one interface, aggregated to
  the same per-original-column grain (see "hard parts" above), plus
  matplotlib rendering of a global importance chart (both models, side
  by side) and a per-prediction signed contribution chart — both
  returned as base64 PNGs so the HTMX page can embed them inline with
  no static-file directory, per this repo's convention. A custom chart
  was used instead of SHAP's own `waterfall`/`force` plots, which
  assume the model's raw output units are the ones worth showing; here
  the dollar-converted counterfactual deltas are the more honest and
  readable choice.
- **`app.py`** — trains both models once at startup (`lifespan`), then
  computes each model's global SHAP importance on a held-out sample and
  uses the combined top 12 features (normalized per model, so
  LightGBM's larger raw SHAP magnitudes don't drown out Ridge's) to
  decide which ~12 of the ~79 columns the dashboard's form actually
  exposes — every other column is silently defaulted, and the UI says
  so. `POST /predict` validates each submitted value (a numeric field
  that doesn't parse as a float, or a categorical value that isn't one
  of that column's known training values, both return `422`, not
  `500`) before ever building a house row or rendering anything.

## What it deliberately doesn't do

- No hyperparameter tuning or AutoML — LightGBM's parameters (500
  trees, depth via `num_leaves=31`, `learning_rate=0.05`) are reasonable
  defaults for a dataset this size, not a tuned-to-the-limit result.
- No time-series or market-trend modeling — every sale in this dataset
  is treated as interchangeable regardless of sale date; Ames Housing's
  `YrSold`/`MoSold` columns are just two more numeric features, not a
  trend signal.
- No multi-user support, no persistence of past predictions — a single
  in-memory pair of trained models per process, same as every other
  self-contained challenge in this repo.
- Doesn't require filling in all ~79 features to get a prediction — the
  ~65 columns not exposed by the form are always set to the training
  median (numeric) or mode (categorical), which is stated in the UI
  copy rather than hidden.
- Ridge's dollar contribution figures are approximate single-feature
  counterfactuals, not an exact decomposition of the dollar prediction
  (see "hard parts" — no such exact decomposition exists for a
  log-target model). The log-space values, which SHAP does guarantee
  are exactly additive, are what the correctness tests check.

## Usage

```bash
cd "challenges/Practical ML/House Price Predictor with Feature Importance Explainer"

uv run --with fastapi --with "uvicorn[standard]" --with scikit-learn \
    --with lightgbm --with shap --with pandas --with numpy \
    --with matplotlib --with python-multipart python app.py
# -> http://127.0.0.1:8010  (first run fetches Ames Housing from OpenML,
#    a few MB; cached afterward)

uv run --with fastapi --with "uvicorn[standard]" --with scikit-learn \
    --with lightgbm --with shap --with pandas --with numpy \
    --with matplotlib --with python-multipart --with httpx2 --with pytest \
    pytest -q   # 39 tests
```

Open `http://127.0.0.1:8010`. The dashboard shows a form for the 12
features SHAP ranked most important (e.g. `OverallQual`, `GrLivArea`,
`Neighborhood`, `GarageCars`, `YearBuilt`…), pre-filled with typical
values; edit any subset and hit **Predict** to see both models'
dollar price, plus a bar-chart-and-table breakdown of which features
pushed each model's number up (green) or down (red), and by how much.
The global feature-importance chart (both models compared) is always
visible below the form. `GET /health` reports
`{"status": "ok", "models_loaded": true}`.

**Measured on this run** (held-out 20% test split, 292 houses):

| Model    | RMSE (log) | MAE (log) | R² (log) | RMSE ($) | MAE ($) | R² ($) |
| -------- | ---------- | --------- | -------- | -------- | ------- | ------ |
| Ridge    | 0.136      | 0.095     | 0.900    | $25,112  | $16,435 | 0.918  |
| LightGBM | 0.131      | 0.085     | 0.909    | $24,409  | $15,016 | 0.922  |

LightGBM edges out Ridge on every metric here, as expected for a tree
model on a dataset with this many categorical interactions — but Ridge
is close, which is itself a useful finding: most of Ames Housing's
price signal is close to linear once you have the right features.

## Tests

39 pytest cases across four files.

- `test_data.py` (12) — cache-hit vs. cache-miss vs. `force_refresh`
  fetch behavior (network call mocked), numeric/categorical column
  splitting (including on a frame with no target column, the inference
  case), the `NONE_MEANS_ABSENT` vs. ordinary-missing sentinel fill, and
  `log1p` target transform + split reproducibility.
- `test_model.py` (8) — the category-schema order-independence and
  unseen-category-to-`NaN` correctness tests described above, the Ridge
  pipeline surviving missing/unseen categorical input without crashing,
  both real-data-trained models clearing an R² floor (0.7 — well below
  the ~0.90 actually measured, so this only fires on a genuine
  regression), predicted prices landing in a plausible range and being
  deterministic, and the default "typical house" row predicting a
  plausible price.
- `test_explain.py` (12) — the SHAP-sum-equals-prediction-minus-base
  invariant in log space for both explainers, both explanations
  covering every original column exactly once (no double-counting, no
  gaps), the dollar-delta helper's sign/zero behavior, global importance
  scores being finite and non-negative, and both PNG-rendering functions
  producing valid PNG bytes.
- `test_app.py` (7) — every editable feature rendering a form field,
  the happy-path prediction showing both models' numbers, missing form
  fields falling back to defaults instead of erroring, a garbage numeric
  value and an unknown/adversarial categorical value both returning
  `422` (not `500` or a silent wrong prediction), and the health check.

The suite runs warning-free: `pytest.ini` filters one remaining warning
that `shap`'s own colors module emits at import time against newer
matplotlib (`Colormap.set_bad`/`set_over`/`set_under` deprecation) — an
unfixed upstream shap/matplotlib version interaction that nothing in
this test suite triggers or can work around.
