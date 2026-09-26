"""Ames Housing fetch/cache/preprocessing helpers.

The hard part: OpenML serves ~15 categorical columns where `NaN` is not a
missing measurement but a documented "this house doesn't have the
feature at all" (no pool, no alley, no fence, no basement...). Blindly
imputing these with the training mode would invent a fictitious "most
common" pool/fence for a house that has none, corrupting exactly the
columns most likely to swing a prediction. `NONE_MEANS_ABSENT` enumerates
the columns the Ames data dictionary documents this way and fills them
with the literal string ``"None"`` (a real, distinct category) instead.
Every *other* remaining categorical `NaN` (a handful of genuinely missing
values, e.g. one row's `Electrical`) is filled with a separate sentinel,
``"Missing"``, so the two situations are never conflated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

DATA_ID = 42165  # OpenML "house_prices" -- the Kaggle Ames Housing dataset.
BASE_DIR = Path(__file__).parent
CACHE_PATH = BASE_DIR / "data" / "ames_housing.csv"
TARGET = "SalePrice"
ID_COLUMN = "Id"

NONE_MEANS_ABSENT = (
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
)


def load_ames_housing(*, force_refresh: bool = False) -> pd.DataFrame:
    """Load the raw Ames Housing dataframe.

    Fetches from OpenML only the first time (or when `force_refresh` is
    set); every call after that reads the local CSV cache, so tests and
    repeated runs never re-hit the network.
    """
    if not force_refresh and CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH)
    dataset = fetch_openml(data_id=DATA_ID, as_frame=True)
    df = dataset.frame.copy()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def numeric_and_categorical_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split a dataframe's *feature* columns (target/id excluded, if
    present) into numeric vs. categorical by dtype. Works equally on the
    full training frame or a single inference row missing the target."""
    features = df.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    return numeric, categorical


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the row-identifier column and resolve every categorical NaN
    to an explicit sentinel ("None" for a documented absent-feature
    column, "Missing" otherwise) so no categorical column carries NaN
    past this point. Numeric NaNs are left untouched -- the Ridge
    pipeline's imputer and LightGBM's native missing-value handling need
    different treatments, decided downstream in `model.py`."""
    df = df.drop(columns=[ID_COLUMN], errors="ignore").copy()
    _, categorical_cols = numeric_and_categorical_columns(df)
    for col in categorical_cols:
        filler = "None" if col in NONE_MEANS_ABSENT else "Missing"
        df[col] = df[col].astype(object).fillna(filler).astype(str)
    return df


def train_test_split_frame(
    df: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into features/target, log1p-transforming the target once
    here so every downstream consumer (training, tests, the API) agrees
    on the same transform. `SalePrice` is right-skewed; modeling
    log-price is standard practice for this dataset and keeps outlier
    mansions from dominating the loss."""
    X = df.drop(columns=[TARGET])
    y_log = np.log1p(df[TARGET].astype(float))
    return train_test_split(X, y_log, test_size=test_size, random_state=random_state)
