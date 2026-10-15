"""The time split, the models, the calibrator and how a trained model is saved.

Nothing here is pickled: LightGBM goes to its own text format and the calibrator and metadata to
JSON, so loading a model file never executes code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stockout_risk.features import CATEGORICAL, FEATURES, GROUPS, HEAVY_HOURS

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.08,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbose": -1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
}


@dataclass(frozen=True)
class Windows:
    """Label days: ``test_start`` onward is test, the ``val_days`` before it validation, the rest train.

    A row for day ``t`` predicts day ``t + 1``, so the split is on the *label* day and a train
    feature row never sees a day that is in validation or test. ``warmup_days`` drops the first days
    of the data, where the rolling windows are still short.
    """

    test_start: date = date(2024, 6, 26)
    val_days: int = 14
    warmup_days: int = 7


def split(feat: pl.DataFrame, windows: Windows | None = None, hours: int = HEAVY_HOURS):
    """(train, val, test) frames with ``y`` and ``label_day`` added; rows without a label drop out."""
    windows = windows or Windows()
    f = feat.drop_nulls("next_oos").with_columns(
        (pl.col("dt") + pl.duration(days=1)).alias("label_day"),
        (pl.col("next_oos") >= hours).cast(pl.Int8).alias("y"),
    )
    first = f["dt"].min() + timedelta(days=windows.warmup_days)
    val_start = windows.test_start - timedelta(days=windows.val_days)
    train = f.filter((pl.col("dt") >= first) & (pl.col("label_day") < val_start))
    val = f.filter(
        (pl.col("label_day") >= val_start) & (pl.col("label_day") < windows.test_start)
    )
    test = f.filter(pl.col("label_day") >= windows.test_start)
    if min(train.height, val.height, test.height) == 0:
        raise ValueError(
            "a time window is empty; check Windows against the data's date range"
        )
    return train, val, test


def matrix(frame: pl.DataFrame, features: list[str] = FEATURES) -> np.ndarray:
    return frame.select(features).to_numpy().astype(np.float32)


def without(group: str) -> list[str]:
    """All features except one group, for the ablation."""
    drop = set(GROUPS[group])
    return [f for f in FEATURES if f not in drop]


def fit_lgbm(
    train: pl.DataFrame,
    val: pl.DataFrame,
    features: list[str] = FEATURES,
    *,
    rounds: int = 600,
    patience: int = 40,
    params: dict | None = None,
) -> lgb.Booster:
    """LightGBM with early stopping on the validation window's log-loss."""
    cats = [c for c in CATEGORICAL if c in features]
    kw = {"feature_name": features, "categorical_feature": cats, "free_raw_data": False}
    dtrain = lgb.Dataset(matrix(train, features), train["y"].to_numpy(), **kw)
    dval = lgb.Dataset(
        matrix(val, features), val["y"].to_numpy(), reference=dtrain, **kw
    )
    return lgb.train(
        {**LGB_PARAMS, **(params or {})},
        dtrain,
        num_boost_round=rounds,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )


def fit_logistic(train: pl.DataFrame, *, max_rows: int = 1_000_000, seed: int = 0):
    """Standardised logistic regression on the numeric features, weekday one-hot, no id columns.

    Ids are category codes, not quantities, so they stay out; a sample of at most ``max_rows`` rows
    keeps the fit quick (the loss is smooth and the sample is large).
    """
    cols = [f for f in FEATURES if f not in GROUPS["static"]]
    if train.height > max_rows:
        train = train.sample(max_rows, seed=seed)
    dow = cols.index("dow_next")
    rest = [i for i in range(len(cols)) if i != dow]
    pre = ColumnTransformer(
        [
            ("dow", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [dow]),
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                rest,
            ),
        ]
    )
    model = make_pipeline(pre, LogisticRegression(C=1.0, max_iter=400))
    model.fit(matrix(train, cols), train["y"].to_numpy())
    return model, cols


class Calibrator:
    """Binned isotonic regression, stored as its knots (JSON-friendly; applied with ``np.interp``).

    Plain isotonic regression on every row steps to exactly 0 or 1 wherever a few extreme scores
    happen to share one outcome (64 validation rows, all positive, gave a "100%" risk that held for
    82% on the test week). So the rows are first cut into equal-count bins of at least ``bin_rows``
    and the isotonic fit runs on the bin means, weighted by bin size; the tails are then as smooth
    as the data under them.
    """

    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.xs, self.ys = np.asarray(xs, float), np.asarray(ys, float)

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray, bin_rows: int = 1000) -> Calibrator:
        order = np.argsort(p, kind="stable")
        bins = np.array_split(order, max(2, len(p) // bin_rows))
        mean_p = np.array([p[b].mean() for b in bins])
        mean_y = np.array([y[b].mean() for b in bins])
        size = np.array([len(b) for b in bins])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(mean_p, mean_y, sample_weight=size)
        return cls(iso.X_thresholds_, iso.y_thresholds_)

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return np.interp(p, self.xs, self.ys)

    def to_dict(self) -> dict:
        return {"xs": self.xs.tolist(), "ys": self.ys.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> Calibrator:
        return cls(d["xs"], d["ys"])


def save_model(directory: Path, booster: lgb.Booster, meta: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(directory / "model.txt"))
    (directory / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_model(directory: Path) -> tuple[lgb.Booster, dict]:
    meta_path, model_path = directory / "model.json", directory / "model.txt"
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"no trained model in {directory}; run `stockout evaluate` first"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(model_path)), meta


def predict(booster: lgb.Booster, frame: pl.DataFrame, features: list[str] = FEATURES):
    return booster.predict(matrix(frame, features))


def contributions(
    booster: lgb.Booster, frame: pl.DataFrame, features: list[str] = FEATURES
):
    """SHAP-style per-feature contributions in log-odds, shape (rows, features); the bias is dropped."""
    return booster.predict(matrix(frame, features), pred_contrib=True)[:, :-1]
