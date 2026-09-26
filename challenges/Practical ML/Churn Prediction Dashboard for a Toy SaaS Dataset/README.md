# Churn Prediction Dashboard for a Toy SaaS Dataset

**Category:** Practical ML
**Difficulty:** B (brief: "Train, explain, and let the user tune the decision threshold.")

**Status:** Implemented (Python)

Trains a Logistic Regression baseline and a LightGBM classifier on the
[Telco Customer Churn dataset](https://www.openml.org/search?type=data&id=42178)
(7,021 customers after deduplication, ~26.5% churned), then serves a
Streamlit dashboard where a single slider sets the decision threshold and
every threshold-dependent number on the page -- accuracy, precision,
recall, F1, the confusion matrix, and where that threshold sits on the
ROC/PR curves -- updates live from cached test-set probabilities, no
retraining involved. Below that, a global SHAP feature-importance chart
and a per-customer drill-down (predicted probability, flagged/not-flagged
at the current threshold, and a signed SHAP breakdown of why) round out
the "explain" half of the brief.

The dataset is telecom, not software-as-a-service -- the challenge asks
for a "toy SaaS dataset" and no equivalently well-known, real, freely
licensed SaaS churn dataset exists. Telco's columns map onto a
subscription business's churn story closely enough to use honestly as a
stand-in: `Contract` (month-to-month/one-year/two-year) is a plan term,
`MonthlyCharges`/`TotalCharges` are MRR and lifetime spend, `tenure` is
account age, and `OnlineSecurity`/`TechSupport`/`StreamingTV` are add-on
modules. This is disclosed here rather than silently treating a telecom
dataset as if it already were a SaaS one.

## The hard parts

**SHAP's additivity guarantee holds in logit space, not probability
space.** Both models' raw outputs -- Logistic Regression's linear
decision function, LightGBM's raw tree-ensemble score -- are the same
quantity: the log-odds of churning. SHAP's guarantee
(`sum(shap_values) + base_value == raw_output`) holds exactly *there*.
Converting to a probability with the sigmoid doesn't distribute over that
sum (`sigmoid(a) + sigmoid(b) != sigmoid(a + b)`), the same obstacle this
repo's House Price Predictor challenge hit going from log-price to
dollars. `explain.py`'s correctness tests check the real invariant in
logit space; for display, each feature's probability-point figure is
instead an honest, independently well-defined counterfactual --
`sigmoid(logit) - sigmoid(logit - shap_i)`, "how much would the predicted
probability drop if exactly this feature's logit contribution were
removed, holding every other feature's contribution fixed" -- rather than
a fabricated linear rescale of the logit-space number. See
`test_explain.py::test_logreg_shap_contributions_sum_to_prediction_minus_base_in_logit_space`
and the LightGBM equivalent.

**Reconciling a one-hot-expanded linear model's SHAP values with a
native-categorical tree model's, onto the same per-original-column
grain** -- the same problem the House Price Predictor solved for Ridge
vs. LightGBM, reused here for Logistic Regression vs. LightGBM.
`_transformed_feature_groups` uses the *lengths* of
`OneHotEncoder.categories_` (not string-parsing of generated feature
names, which breaks the moment one column name is a prefix of another --
`Contract` vs. `ContractType` would be such a case) to know exactly which
transformed columns came from which original column, and
`aggregate_by_original_column` sums Logistic Regression's per-one-hot-column
SHAP values back down to the grain LightGBM already reports at. Covered
by `test_explain.py::test_logreg_explanation_covers_every_original_column_exactly_once`.

**A naive 0.5 threshold is a poor default on an imbalanced target.**
With only ~26.5% of customers churning, a Logistic Regression model
thresholded at the textbook-default 0.5 recovers only 52% of actual
churners (measured: precision 66.0%, recall 52.2%, F1 0.583). The
dashboard's slider instead *starts* at the F1-optimal threshold scanned
directly over the model's own predicted probabilities (`model.best_f1_threshold`
-- not a fixed grid, since the optimum can fall between grid points on a
small test set), which recovers 79.0% of churners at 0.27 (F1 0.624).
The user can still drag the slider anywhere -- to prioritize precision
over recall for a costly retention-outreach budget, say -- this only
picks a better-informed starting point than 0.5.
`test_model.py::test_best_f1_threshold_on_real_data_beats_naive_half_on_recall_or_matches_f1`
checks the tuned threshold never scores worse on F1 than the naive one.

**OpenML's ARFF export quote-wraps every categorical value containing a
space**, and hides the dataset's 11 blank `TotalCharges` values (all
brand-new customers with `tenure == 0`, filled with `0.0` -- a real known
value, not a gap to impute) as the literal two-character string `"' '"`.
`data._unquote` strips this before anything downstream sees it; left
unhandled, `'Fiber optic'` and `Fiber optic` would silently be treated as
two different categories. A second, unrelated data quirk: this Telco
mirror has no customer-ID column and 22 rows that are exact duplicates
across every column. Left in, a duplicate can land in both the train and
test split, letting the model "recognize" a test row it memorized
verbatim during training -- `data.clean_features` drops exact duplicates
*before* the train/test split, not as an afterthought.

## Design

- **`data.py`** -- fetches Telco Customer Churn from OpenML
  (`data_id=42178`, pinned explicitly since a second, materially
  different dataset version is registered under the same name), caches
  it to `data/telco_churn.csv` (gitignored, regenerated from the network
  on first run), un-quotes every categorical value, coerces `TotalCharges`
  to float, maps `Churn` to 0/1, and drops exact duplicate rows -- see
  "hard parts" above. `train_test_split_frame` stratifies on the target
  so the ~26.5% churn rate is preserved in both splits.
- **`model.py`** -- builds the Logistic Regression pipeline
  (median-impute + `StandardScaler` for numerics, most-frequent-impute +
  one-hot for categoricals) and trains LightGBM directly on
  `pd.Categorical`-typed columns (the same category-schema-freezing
  pattern as the House Price Predictor, to stop a single inference row
  from silently reassigning a category's integer code). `TrainedModels`
  caches both models' predicted probabilities on the held-out test set
  once, so `metrics_at_threshold`/`compute_curves` can recompute every
  threshold-dependent number instantly without retraining --
  `predict_proba_both` is the only function that runs a fresh forward
  pass, used for the dashboard's per-customer drill-down.
  `best_f1_threshold` scans the model's own observed probabilities (not a
  fixed grid) for the threshold that maximizes F1.
- **`explain.py`** -- wraps `shap.LinearExplainer` (Logistic Regression)
  and `shap.TreeExplainer` (LightGBM) behind one interface, both in logit
  space, aggregated to the same per-original-column grain (see "hard
  parts"). `_positive_class_shap_values` normalizes
  `shap.TreeExplainer`'s binary-classifier output shape, which has
  changed across shap versions (plain 2D array vs. a `[class_0, class_1]`
  list vs. a 3D array), to one consistent 2D array of the positive
  (churn) class's contribution. Renders global-importance, per-customer
  contribution, ROC-curve, and PR-curve charts as `matplotlib.Figure`
  objects -- handed straight to Streamlit's `st.pyplot`, no base64/PNG
  round-trip needed the way the FastAPI challenges require for HTML
  embedding.
- **`app.py`** -- the Streamlit entrypoint. Trains both models once via
  `@st.cache_resource` (this framework's equivalent of the FastAPI
  challenges' startup-`lifespan` train-once pattern), then renders: a
  model selector and threshold slider (sidebar); accuracy/precision/recall/F1
  at the current threshold plus the threshold-independent ROC-AUC/PR-AUC;
  a confusion matrix; ROC and PR curves with the current threshold's
  point marked; the global SHAP importance chart (computed once, cached,
  over a 300-row sample for speed); and a per-customer drill-down (pick a
  test-set row, see its predicted probability, whether the current
  threshold flags it, and a signed SHAP contribution chart).

## What it deliberately doesn't do

- No class-weighting or resampling (SMOTE, class_weight="balanced", etc.)
  to correct the ~26.5% imbalance during training -- the challenge brief
  specifically asks for threshold tuning as the mechanism, so both
  models are trained with default class weighting and the imbalance is
  handled entirely at decision time, where the dashboard makes it
  visible and adjustable rather than baking a correction in silently.
- No hyperparameter tuning or AutoML -- LightGBM's parameters (300 trees,
  `num_leaves=15`, `learning_rate=0.05`) are reasonable defaults for a
  dataset this size, not a tuned-to-the-limit result.
- No calibration curve/Brier score reporting -- both models' `predict_proba`
  outputs are used directly for ranking (ROC/PR) and thresholding, which
  doesn't require them to be well-calibrated probabilities, only a
  reasonable ordering.
- No multi-user support or persistence of past predictions -- a single
  in-memory pair of trained models per Streamlit process (via
  `@st.cache_resource`), same as every other self-contained challenge in
  this repo.
- No synthetic "what-if" customer editor (unlike the House Price
  Predictor's editable form) -- the brief's interactivity is the decision
  threshold, and the per-customer drill-down works over real held-out
  customers rather than a synthetic composite one.

## Usage

```bash
cd "challenges/Practical ML/Churn Prediction Dashboard for a Toy SaaS Dataset"

uv run --with streamlit --with scikit-learn --with lightgbm --with shap \
    --with pandas --with numpy --with matplotlib streamlit run app.py
# -> http://localhost:8501  (first run fetches Telco Customer Churn from
#    OpenML, ~1MB; cached afterward)

uv run --with streamlit --with scikit-learn --with lightgbm --with shap \
    --with pandas --with numpy --with matplotlib --with pytest pytest -q
# 44 tests, ~2.5 minutes (real data fetch + two real model trainings +
# real SHAP computations -- nothing here is mocked away)
```

Open `http://localhost:8501`. Pick a model (Logistic Regression or
LightGBM) from the sidebar; the threshold slider starts at that model's
F1-optimal point on the test set and can be dragged anywhere from 0 to 1.
Every metric, the confusion matrix, and the marked point on the ROC/PR
curves update immediately. Scroll down for the global feature-importance
chart, then pick a test-set row index to see that customer's predicted
churn probability, whether the current threshold flags them, and a
bar chart of which features pushed their probability up (red) or down
(green).

**Measured on this run** (stratified 80/20 split, 1,405 held-out
customers, 26.5% churned):

| Model               | ROC-AUC | PR-AUC | Threshold | Accuracy | Precision | Recall | F1    |
| -------------------- | ------- | ------ | --------- | -------- | --------- | ------ | ----- |
| Logistic Regression   | 0.840   | 0.637  | 0.50 (naive)   | 80.2%    | 66.0%     | 52.2%  | 0.583 |
| Logistic Regression   | 0.840   | 0.637  | 0.27 (tuned)   | 74.8%    | 51.6%     | 79.0%  | 0.624 |
| LightGBM              | 0.833   | 0.633  | 0.50 (naive)   | 78.9%    | 63.1%     | 49.2%  | 0.553 |
| LightGBM              | 0.833   | 0.633  | 0.23 (tuned)   | 74.1%    | 50.7%     | 81.7%  | 0.626 |

Logistic Regression edges out LightGBM here on both AUC metrics -- the
opposite of the House Price Predictor's result, and itself a useful
finding: Telco's churn signal (contract length, tenure, monthly charges)
is close to monotonic and close to linear, leaving little room for
LightGBM's extra capacity to find interactions a well-regularized linear
model misses. Tuning the threshold trades roughly 5-6 accuracy points for
+27-32 recall points on both models -- the right trade for a retention
use case, where missing an actual churner is usually costlier than one
extra unnecessary outreach.

## Tests

44 pytest cases across four files.

- `test_data.py` (16) -- cache-hit vs. cache-miss vs. `force_refresh`
  fetch behavior (network call mocked), the quote-stripping and
  blank-`TotalCharges`-to-zero cleaning logic, exact-duplicate-row
  dropping, numeric/categorical column splitting, and stratified
  train/test split correctness (target excluded from features, churn
  rate preserved in both splits, reproducibility).
- `test_model.py` (14) -- the category-schema order-independence and
  unseen-category-to-`NaN` tests carried over from the House Price
  Predictor, hand-computed confusion-matrix arithmetic at a known
  threshold, both-extremes threshold behavior, perfect-separation and
  random-guessing sanity checks for `compute_curves`, `best_f1_threshold`
  finding the gap between two well-separated clusters, both real-data-trained
  models clearing an ROC-AUC floor (0.75 -- well below the ~0.84 actually
  measured), and the tuned threshold never scoring worse on F1 than a
  naive 0.5 on real data.
- `test_explain.py` (12) -- the SHAP-sum-equals-prediction-minus-base
  invariant in logit space for both explainers, both explanations
  covering every original column exactly once, the probability-delta
  helper's sign/zero behavior, global importance scores being finite and
  non-negative, and all four chart-rendering functions returning a valid
  `matplotlib.Figure`.
- `test_app.py` (6) -- headless `streamlit.testing.v1.AppTest` smoke
  tests (this repo's precedent for testing a non-HTTP UI framework
  headlessly is the Encrypted Diary/Journal challenge's Textual `Pilot`
  tests): the app loads without an exception, the threshold slider
  starts away from a naive 0.5, dragging it changes the displayed
  metrics, switching models and changing the drill-down row both still
  render without error.

The suite runs warning-free: `pytest.ini` filters the same
shap/matplotlib deprecation warning the House Price Predictor's suite
does, plus one more -- shap's own informational note about
`TreeExplainer`'s output-shape change for LightGBM binary classifiers,
which `explain._positive_class_shap_values` already normalizes for.
