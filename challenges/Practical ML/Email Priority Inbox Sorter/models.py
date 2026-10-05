"""Time splits, text features and the five scorers compared in this challenge.

``random`` and ``to_me`` are context baselines; ``tfidf_lr`` sees text only,
``lgbm_meta`` sees the leakage-safe metadata only, ``lgbm_meta_text`` both. Text
vectorizers are fit on the training split alone, and LightGBM early-stops on
validation PR-AUC, so the test split is never used to choose anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from features import FEATURE_COLUMNS

MODEL_NAMES = ["random", "to_me", "tfidf_lr", "lgbm_meta", "lgbm_meta_text"]
SVD_COMPONENTS = 64
BODY_CHARS = 2000
EARLY_STOPPING_ROUNDS = 50
MAX_TREES = 600
FALLBACK_TREES = 200  # when validation has one class, there is nothing to stop on


def time_split(
    df: pl.DataFrame, fracs: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Per mailbox, strictly by time: oldest 70% / next 15% / newest 15%.

    A cut never lands between two messages with the same timestamp, so every
    train message is strictly older than every validation message, and so on.
    """
    parts: list[list[pl.DataFrame]] = [[], [], []]
    for box in df.partition_by("mailbox", maintain_order=True):
        box = box.sort(["date", "message_id"])
        dates = box["date"].to_list()
        n = box.height
        cuts = []
        for frac_end in (fracs[0], fracs[0] + fracs[1]):
            i = round(n * frac_end)
            while 0 < i < n and dates[i] == dates[i - 1]:
                i += 1
            cuts.append(min(i, n))
        bounds = [0, cuts[0], max(cuts[0], cuts[1]), n]
        for k in range(3):
            parts[k].append(box.slice(bounds[k], bounds[k + 1] - bounds[k]))
    return tuple(pl.concat(p) for p in parts)  # type: ignore[return-value]


def _texts(df: pl.DataFrame) -> list[str]:
    return [
        f"{s or ''}\n{(b or '')[:BODY_CHARS]}"
        for s, b in zip(df["subject"].to_list(), df["body"].to_list(), strict=True)
    ]


class TextFeatures:
    """TF-IDF plus a truncated SVD, both fit on the training texts only."""

    def __init__(self, n_components: int = SVD_COMPONENTS, seed: int = 0):
        self.n_components = n_components
        self.seed = seed
        self._tfidf = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            min_df=2,
            max_features=50_000,
            ngram_range=(1, 2),
        )
        self._svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> TextFeatures:
        x = self._tfidf.fit_transform(texts)
        k = max(1, min(self.n_components, x.shape[1] - 1, x.shape[0] - 1))
        self.n_components = k
        self._svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(x)
        return self

    @property
    def vocabulary(self) -> dict[str, int]:
        return self._tfidf.vocabulary_

    def tfidf(self, texts: list[str]) -> sparse.csr_matrix:
        return self._tfidf.transform(texts)

    def svd(self, texts: list[str]) -> np.ndarray:
        assert self._svd is not None, "fit() first"
        return self._svd.transform(self.tfidf(texts))


def svd_columns(n: int) -> list[str]:
    return [f"svd_{i}" for i in range(n)]


def _fit_lgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> lgb.LGBMClassifier:
    pos = max(int(y_train.sum()), 1)
    neg = max(len(y_train) - pos, 1)
    can_stop = 0 < y_val.sum() < len(y_val)
    clf = lgb.LGBMClassifier(
        n_estimators=MAX_TREES if can_stop else FALLBACK_TREES,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
    )
    if can_stop:
        clf.fit(
            x_train,
            y_train,
            eval_X=x_val,
            eval_y=y_val,
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            ],
        )
    else:
        clf.fit(x_train, y_train)
    return clf


@dataclass
class Models:
    text: TextFeatures
    lr: LogisticRegression
    lgbm_meta: lgb.LGBMClassifier
    lgbm_meta_text: lgb.LGBMClassifier
    seed: int

    @property
    def meta_names(self) -> list[str]:
        return list(FEATURE_COLUMNS)

    @property
    def meta_text_names(self) -> list[str]:
        return [*FEATURE_COLUMNS, *svd_columns(self.text.n_components)]

    def matrix(self, df: pl.DataFrame, with_text: bool) -> np.ndarray:
        x = df.select(FEATURE_COLUMNS).to_numpy().astype(np.float64)
        if with_text:
            x = np.hstack([x, self.text.svd(_texts(df))])
        return x

    def predict(self, name: str, df: pl.DataFrame) -> np.ndarray:
        if name == "random":
            return np.random.default_rng(self.seed).random(df.height)
        if name == "to_me":
            return df["owner_in_to"].to_numpy().astype(np.float64)
        if name == "tfidf_lr":
            return self.lr.predict_proba(self.text.tfidf(_texts(df)))[:, 1]
        if name == "lgbm_meta":
            return self.lgbm_meta.predict_proba(self.matrix(df, False))[:, 1]
        if name == "lgbm_meta_text":
            return self.lgbm_meta_text.predict_proba(self.matrix(df, True))[:, 1]
        raise KeyError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def fit_models(train: pl.DataFrame, val: pl.DataFrame, seed: int = 0) -> Models:
    y_train = train["acted"].to_numpy().astype(int)
    y_val = val["acted"].to_numpy().astype(int)

    text = TextFeatures(seed=seed).fit(_texts(train))
    lr = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", random_state=seed
    ).fit(text.tfidf(_texts(train)), y_train)

    shell = Models(text, lr, None, None, seed)  # type: ignore[arg-type]
    meta_train, meta_val = shell.matrix(train, False), shell.matrix(val, False)
    mt_train, mt_val = shell.matrix(train, True), shell.matrix(val, True)
    shell.lgbm_meta = _fit_lgbm(meta_train, y_train, meta_val, y_val, seed)
    shell.lgbm_meta_text = _fit_lgbm(mt_train, y_train, mt_val, y_val, seed)
    return shell


class Calibrator:
    """Isotonic map from raw scores to probabilities, fit on validation only."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, scores, y) -> Calibrator:
        self._iso.fit(np.asarray(scores, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, scores) -> np.ndarray:
        return self._iso.predict(np.asarray(scores, dtype=float))
