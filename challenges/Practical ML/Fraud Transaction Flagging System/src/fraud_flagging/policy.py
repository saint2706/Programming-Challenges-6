"""Scoring transactions with a saved model, and turning scores into flag / don't-flag decisions.

A *policy* is the rule that picks the operating point:

* ``precision``: flag everything scored at or above the threshold that reached a precision target
  on the validation window (the most recall available at that precision).
* ``recall``: the same for a recall target (the most precision available at that recall).
* ``cost``: flag a transaction when its expected loss avoided, ``p * amount``, beats the review
  cost. Needs calibrated probabilities, no validation threshold.
* ``capacity``: flag the few highest-ranked transactions of every hour, because analysts can only
  look at so many.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from fraud_flagging import evaluate, features, models

POLICIES = ("precision", "recall", "cost", "capacity")


def score_frame(booster: lgb.Booster, meta: dict, frame: pl.DataFrame) -> pl.DataFrame:
    """``frame`` (raw columns Time, V1..V28, Amount) with ``hour``, ``p_raw`` and calibrated ``p``."""
    built = features.build(frame)
    raw = models.predict(booster, built, meta["features"])
    return built.with_columns(
        pl.Series("p_raw", raw),
        pl.Series("p", models.Platt.from_dict(meta["platt"])(raw)),
    )


def _grid_threshold(meta: dict, kind: str, target: float) -> float:
    table = meta["thresholds"][kind]
    key = str(float(target))
    if key not in table:
        raise ValueError(
            f"{kind} target must be one of {sorted(float(k) for k in table)}, not {target}"
        )
    if table[key] is None:
        raise ValueError(
            f"a {kind} of {target} was not reachable on the validation window"
        )
    return float(table[key])


def decide(
    frame: pl.DataFrame,
    meta: dict,
    policy: str,
    *,
    precision: float = 0.9,
    recall: float = 0.8,
    review_cost: float = 25.0,
    friction: float = 0.0,
    per_hour: int = 3,
) -> np.ndarray:
    """Boolean flags for a scored ``frame`` (needs ``p``, ``Amount`` and ``hour``) under ``policy``."""
    p = frame["p"].to_numpy()
    amount = frame["Amount"].to_numpy()
    if policy == "precision":
        return p >= _grid_threshold(meta, "precision", precision)
    if policy == "recall":
        return p >= _grid_threshold(meta, "recall", recall)
    if policy == "cost":
        if review_cost <= 0 or friction < 0:
            raise ValueError("review_cost must be positive and friction not negative")
        return evaluate.amount_aware_flag(p, amount, review_cost, friction)
    if policy == "capacity":
        if per_hour < 1:
            raise ValueError("per_hour must be at least 1")
        return evaluate.capacity_flag(p * amount, frame["hour"].to_numpy(), per_hour)
    raise ValueError(f"policy must be one of {POLICIES}, not {policy!r}")
