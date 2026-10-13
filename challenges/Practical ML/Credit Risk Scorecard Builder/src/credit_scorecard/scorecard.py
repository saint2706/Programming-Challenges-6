"""From WOE-binned features to an integer-points scorecard.

The model is a logistic regression on WOE features that predicts ln(good/bad) odds:

    ln(odds_good) = a + sum_i b_i * woe_i(x_i)

Because each WOE is already a log-odds-ratio, a well-behaved fit has every b_i near 1 and positive.
A negative coefficient means a feature is fighting a correlated one, so it is dropped and the model
refit. Points then follow the industry scaling: a score of ``base_score`` means odds of
``base_odds`` to 1, and every ``pdo`` points double the odds:

    score = offset + factor * ln(odds_good),  factor = pdo / ln 2,  offset = base_score - factor * ln(base_odds)

Each (feature, bin) gets a whole number of points, the intercept's share is spread evenly, so a
borrower's score is a plain sum of table lookups.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from credit_scorecard.binning import Binning, fit_binning


@dataclass
class Scorecard:
    binnings: dict[str, Binning]
    coefs: dict[str, float]
    intercept: float
    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0
    selection: list[dict] = field(default_factory=list)  # one row per candidate feature
    points: dict[str, list[int]] = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.coefs)
        shared = (self.offset + self.factor * self.intercept) / n if n else 0.0
        self.points = {
            f: [
                round(self.factor * b * w + shared)
                for w in self.binnings[f].woe_table()
            ]
            for f, b in self.coefs.items()
        }

    @property
    def features(self) -> list[str]:
        return list(self.coefs)

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)

    # -- scoring -----------------------------------------------------------------------------
    def feature_points(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """Points per borrower, per feature."""
        return {
            f: np.array(self.points[f])[self.binnings[f].index(df[f].to_numpy())]
            for f in self.features
        }

    def score(self, df: pl.DataFrame) -> np.ndarray:
        return np.sum(list(self.feature_points(df).values()), axis=0)

    def probability_bad(self, score: np.ndarray | float) -> np.ndarray | float:
        """Probability of default implied by a score (inverse of the points scaling)."""
        odds_good = self.base_odds * 2.0 ** (
            (np.asarray(score) - self.base_score) / self.pdo
        )
        return 1.0 / (1.0 + odds_good)

    def reason_codes(
        self, row: dict[str, float | None], top: int = 4
    ) -> list[tuple[str, int]]:
        """Adverse-action reasons: the features losing the most points against their best bin."""
        frame = pl.DataFrame(
            {f: [row.get(f)] for f in self.features},
            schema={f: pl.Float64 for f in self.features},
        )
        got = self.feature_points(frame)
        lost = [(f, max(self.points[f]) - int(got[f][0])) for f in self.features]
        return sorted((x for x in lost if x[1] > 0), key=lambda x: -x[1])[:top]

    # -- persistence -------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "binnings": {f: b.to_dict() for f, b in self.binnings.items()},
            "coefs": self.coefs,
            "intercept": self.intercept,
            "base_score": self.base_score,
            "base_odds": self.base_odds,
            "pdo": self.pdo,
            "selection": self.selection,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Scorecard:
        return cls(
            {f: Binning.from_dict(b) for f, b in d["binnings"].items()},
            d["coefs"],
            d["intercept"],
            d["base_score"],
            d["base_odds"],
            d["pdo"],
            d.get("selection", []),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Scorecard:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def fit_scorecard(
    train: pl.DataFrame,
    target: str,
    features: list[str],
    *,
    specials: dict[str, list[float]] | None = None,
    iv_min: float = 0.02,
    corr_max: float = 0.7,
    max_bins: int = 8,
    min_bin_frac: float = 0.02,
    base_score: float = 600.0,
    base_odds: float = 50.0,
    pdo: float = 20.0,
) -> Scorecard:
    """Bin, select (IV, correlation, coefficient sign), fit and scale. ``train`` only: no test data."""
    specials = specials or {}
    y_bad = train[target].to_numpy().astype(int)
    binnings = {
        f: fit_binning(
            f,
            train[f].to_numpy(),
            y_bad,
            specials=specials.get(f, ()),
            max_bins=max_bins,
            min_bin_frac=min_bin_frac,
        )
        for f in features
    }
    status = {
        f: {
            "feature": f,
            "iv": round(binnings[f].iv, 4),
            "status": "kept",
            "reason": "",
        }
        for f in features
    }
    woe = {f: binnings[f].transform(train[f].to_numpy()) for f in features}

    kept = [
        f
        for f in sorted(features, key=lambda f: -binnings[f].iv)
        if binnings[f].iv >= iv_min
    ]
    for f in set(features) - set(kept):
        status[f].update(status="dropped", reason=f"IV {binnings[f].iv:.3f} < {iv_min}")

    selected: list[str] = []
    for f in kept:  # highest IV first, so the stronger of a correlated pair survives
        clash = next(
            (g for g in selected if abs(np.corrcoef(woe[f], woe[g])[0, 1]) > corr_max),
            None,
        )
        if clash:
            status[f].update(
                status="dropped", reason=f"|corr| > {corr_max} with {clash}"
            )
        else:
            selected.append(f)

    y_good = 1 - y_bad
    while True:
        model = LogisticRegression(C=1e4, max_iter=1000).fit(
            np.column_stack([woe[f] for f in selected]), y_good
        )
        coefs = model.coef_[0]
        if coefs.min() > 0:
            break
        worst = selected[int(np.argmin(coefs))]
        status[worst].update(
            status="dropped", reason=f"negative coefficient ({coefs.min():+.2f})"
        )
        selected.remove(worst)
    return Scorecard(
        binnings={f: binnings[f] for f in selected},
        coefs=dict(zip(selected, map(float, coefs), strict=True)),
        intercept=float(model.intercept_[0]),
        base_score=base_score,
        base_odds=base_odds,
        pdo=pdo,
        selection=[status[f] for f in features],
    )
