"""Telco Customer Churn fetch/cache/preprocessing helpers.

Used here as a stand-in for a "toy SaaS" churn dataset: the raw domain is
telecom, not software-as-a-service, but the shape of the problem and every
column it exposes -- contract length, monthly recurring charge, tenure,
paperless billing, add-on services -- maps directly onto a subscription
business's churn story (`Contract` ~ plan term, `MonthlyCharges` ~ MRR per
seat, `OnlineSecurity`/`TechSupport`/`StreamingTV` ~ add-on modules). This
is the closest well-known, real, freely licensed dataset that plays that
role honestly, rather than fabricating a synthetic dataset whose signal
would be an artifact of whatever generator wrote it.

The hard part: OpenML's ARFF export wraps every categorical value that
contains a space in literal single quotes (`'No phone service'`,
`'One year'`, `'Electronic check'`, `'Fiber optic'`...) -- those quote
characters are literally part of the string pandas reads, not formatting.
`_unquote` strips them before anything downstream (models, SHAP, the
dashboard) has any excuse to treat `'Fiber optic'` and `Fiber optic` as
two different categories. The same quoting hides the dataset's 11 blank
`TotalCharges` values as the two-character string `"' '"` (a quoted single
space) rather than an empty string or a proper NaN.

A second hard part: this dataset has no customer-ID column, and 22 rows
are exact duplicates across every column (a documented quirk of this
particular Telco mirror, not a bug introduced here). Left in place, a
duplicate row can land in both the train and test split, letting the
model "predict" a test row it memorized verbatim during training --
silently inflating every test metric. `clean_features` drops exact
duplicates before the split ever happens, which is also why deduplication
has to run *before* `train_test_split_frame`, not as an afterthought.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# OpenML "Telco-Customer-Churn", version 1 -- pinned explicitly by data_id
# (not fetched by name) since a second, materially different version
# (id=45568) is also registered under the same name.
DATA_ID = 42178
BASE_DIR = Path(__file__).parent
CACHE_PATH = BASE_DIR / "data" / "telco_churn.csv"
TARGET = "Churn"


def _unquote(value: object) -> object:
    """Strip OpenML/ARFF's literal single-quote wrapping from a string
    value that contains a space (e.g. "'Fiber optic'" -> "Fiber optic").
    Non-string values, and strings that aren't quoted, pass through
    unchanged."""
    if (
        isinstance(value, str)
        and len(value) >= 2
        and value[0] == "'"
        and value[-1] == "'"
    ):
        return value[1:-1]
    return value


def load_telco_churn(*, force_refresh: bool = False) -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataframe.

    Fetches from OpenML only the first time (or when `force_refresh` is
    set); every call after that reads the local CSV cache, so tests and
    repeated runs never re-hit the network."""
    if not force_refresh and CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH)
    dataset = fetch_openml(data_id=DATA_ID, as_frame=True)
    df = dataset.frame.copy()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def numeric_and_categorical_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split a dataframe's *feature* columns (target excluded, if present)
    into numeric vs. categorical by dtype. Works equally on the full
    training frame or a single inference row missing the target."""
    features = df.drop(columns=[TARGET], errors="ignore")
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    return numeric, categorical


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Un-quote every categorical value, coerce `TotalCharges` to a float
    (the 11 blank values are all `tenure == 0` brand-new customers who
    haven't been billed yet, so they're filled with `0.0` -- a real known
    value, not a gap to impute), map the `Churn` target to 0/1, and drop
    exact duplicate rows (see module docstring) before anything is
    split into train/test."""
    df = df.copy()
    object_cols = [
        c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"
    ]
    for col in object_cols:
        df[col] = df[col].map(_unquote)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    ).fillna(0.0)

    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def train_test_split_frame(
    df: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into features/target, stratifying on the target so the
    ~26.5% churn rate is preserved in both the train and test split
    (an unstratified split on a dataset this imbalanced can otherwise
    swing the test-set churn rate by several points on an unlucky
    random split)."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
