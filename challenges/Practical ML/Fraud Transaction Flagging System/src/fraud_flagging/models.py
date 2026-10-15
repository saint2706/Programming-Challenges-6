"""The time split, the models, Platt calibration and how a trained model is saved.

Nothing is pickled: LightGBM goes to its own text format, the calibrator and the policy thresholds
to JSON, so loading a model file never executes code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud_flagging.features import FEATURES

STRATEGIES = ("none", "sqrt", "balanced", "undersample")
UNDERSAMPLE_RATIO = 20  # legitimate rows kept per fraud when undersampling
LGB_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
}


@dataclass(frozen=True)
class Windows:
    """Hours from the first transaction: train before ``val_hour``, validate until ``test_hour``."""

    val_hour: float = 24.0
    test_hour: float = 36.0


def split(frame: pl.DataFrame, windows: Windows | None = None):
    """(train, val, test) by ``Time``; every window must hold at least one fraud."""
    w = windows or Windows()
    t = pl.col("Time")
    parts = (
        frame.filter(t < w.val_hour * 3600),
        frame.filter((t >= w.val_hour * 3600) & (t < w.test_hour * 3600)),
        frame.filter(t >= w.test_hour * 3600),
    )
    for name, part in zip(("train", "validation", "test"), parts, strict=True):
        if part.height == 0 or part["y"].sum() == 0:
            raise ValueError(
                f"the {name} window has no fraud; check Windows against the data"
            )
    return parts


def matrix(frame: pl.DataFrame, features: list[str] = FEATURES) -> np.ndarray:
    return frame.select(features).to_numpy().astype(np.float64)


def class_ratio(y: np.ndarray) -> float:
    """Legitimate rows per fraud."""
    return float((len(y) - y.sum()) / max(int(y.sum()), 1))


def _prepare(train: pl.DataFrame, features: list[str], strategy: str, seed: int):
    """Training matrix, labels and LightGBM parameters for one imbalance strategy.

    ``none`` trains on the data as it is; ``sqrt`` and ``balanced`` weight the fraud class by the
    square root of, or the full, legitimate-to-fraud ratio; ``undersample`` keeps every fraud and a
    random ``UNDERSAMPLE_RATIO`` legitimate rows per fraud.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, not {strategy!r}")
    y = train["y"].to_numpy()
    x = matrix(train, features)
    params = {**LGB_PARAMS, "seed": seed}
    if strategy == "sqrt":
        params["scale_pos_weight"] = float(np.sqrt(class_ratio(y)))
    elif strategy == "balanced":
        params["scale_pos_weight"] = class_ratio(y)
    elif strategy == "undersample":
        rng = np.random.default_rng(seed)
        legit = np.flatnonzero(y == 0)
        keep = rng.choice(
            legit, size=min(len(legit), UNDERSAMPLE_RATIO * int(y.sum())), replace=False
        )
        rows = np.sort(np.concatenate([np.flatnonzero(y == 1), keep]))
        x, y = x[rows], y[rows]
    return x, y, params


def fit_lgbm(
    train: pl.DataFrame,
    val: pl.DataFrame,
    features: list[str] = FEATURES,
    *,
    strategy: str = "none",
    rounds: int = 1000,
    patience: int = 100,
    seed: int = 0,
) -> lgb.Booster:
    """LightGBM with one imbalance strategy, early-stopped on the untouched validation PR-AUC."""
    x, y, params = _prepare(train, features, strategy, seed)
    dtrain = lgb.Dataset(x, y, feature_name=features)
    dval = lgb.Dataset(matrix(val, features), val["y"].to_numpy(), reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )


def fit_lgbm_fixed(
    train: pl.DataFrame,
    features: list[str] = FEATURES,
    *,
    strategy: str = "none",
    rounds: int,
    seed: int = 0,
) -> lgb.Booster:
    """The same model for a given number of trees, with no validation window (rolling-origin folds)."""
    x, y, params = _prepare(train, features, strategy, seed)
    params = {**params, "metric": "binary_logloss"}
    return lgb.train(
        params, lgb.Dataset(x, y, feature_name=features), num_boost_round=rounds
    )


def fit_logistic(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    balanced: bool = False,
    grid: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0),
):
    """Standardised logistic regression; the inverse regularisation ``C`` is picked by validation PR-AUC."""
    xtr, ytr = matrix(train), train["y"].to_numpy()
    xva, yva = matrix(val), val["y"].to_numpy()
    best = None
    for c in grid:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c, class_weight="balanced" if balanced else None, max_iter=2000
            ),
        ).fit(xtr, ytr)
        ap = average_precision_score(yva, model.predict_proba(xva)[:, 1])
        if best is None or ap > best[0]:
            best = (ap, c, model)
    return best[2], best[1]


def fit_isolation_forest(train: pl.DataFrame, *, seed: int = 0) -> IsolationForest:
    """An unsupervised baseline: it never sees a label, only how unusual a transaction is."""
    return IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(
        matrix(train)
    )


def anomaly_score(forest: IsolationForest, frame: pl.DataFrame) -> np.ndarray:
    """Higher means more unusual."""
    return -forest.score_samples(matrix(frame))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


class Platt:
    """Platt scaling: ``sigmoid(a * logit(score) + b)``, two numbers that save to JSON.

    Weighting or undersampling the frauds shifts a model's probabilities; this puts them back on the
    scale of the real fraud rate, which the cost rules need. Two parameters is what ~90 validation
    frauds can support.
    """

    def __init__(self, a: float, b: float):
        self.a, self.b = float(a), float(b)

    @classmethod
    def fit(cls, score: np.ndarray, y: np.ndarray) -> Platt:
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(
            _logit(score).reshape(-1, 1), y
        )
        return cls(lr.coef_[0, 0], lr.intercept_[0])

    def __call__(self, score: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.a * _logit(score) + self.b)))

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict) -> Platt:
        return cls(d["a"], d["b"])


def predict(
    booster: lgb.Booster, frame: pl.DataFrame, features: list[str] | None = None
):
    return booster.predict(matrix(frame, features or booster.feature_name()))


def contributions(booster: lgb.Booster, frame: pl.DataFrame) -> np.ndarray:
    """SHAP-style contributions in log-odds, shape (rows, features); the bias column is dropped."""
    return booster.predict(matrix(frame, booster.feature_name()), pred_contrib=True)[
        :, :-1
    ]


def save_model(directory: Path, booster: lgb.Booster, meta: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(directory / "model.txt"))
    (directory / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_model(directory: Path) -> tuple[lgb.Booster, dict]:
    meta_path, model_path = directory / "model.json", directory / "model.txt"
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"no trained model in {directory}; run `fraud evaluate` first"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(model_path)), meta
