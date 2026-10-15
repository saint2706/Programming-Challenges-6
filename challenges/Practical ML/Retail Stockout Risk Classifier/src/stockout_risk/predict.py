"""Scoring with a saved model: one store-product, or the whole day's worklist, with reasons."""

from __future__ import annotations

from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import polars as pl

from stockout_risk import models

# Plain-language wording for the features a reason can name; anything else prints its column name.
LABELS = {
    "product_heavy_rate": "share of this product's stores heavily out of stock today",
    "store_heavy_rate": "share of this store's products heavily out of stock today",
    "oos_last4": "out-of-stock hours in the last 4 selling hours today",
    "oos_close": "out of stock at the close of today",
    "oos_t": "out-of-stock hours today",
    "oos_mean14": "average out-of-stock hours, 14 days",
    "oos_mean7": "average out-of-stock hours, 7 days",
    "heavy_rate7": "heavy-stockout days in the last 7",
    "days_since_heavy": "days since the last heavy stockout",
    "discount_next": "tomorrow's discount",
    "promo_next": "tomorrow's promotion",
    "dow_next": "tomorrow's weekday",
    "demand16": "estimated full-day demand",
    "first_oos": "hour today's stockout began",
}


def _calibrated(meta: dict, raw: np.ndarray) -> np.ndarray:
    return models.Calibrator.from_dict(meta["calibrator"])(raw)


def reasons(
    booster: lgb.Booster, meta: dict, row: pl.DataFrame, top: int = 5
) -> list[dict]:
    """The features that moved one row's score the most, in log-odds, with their values."""
    cols = meta["features"]
    contrib = models.contributions(booster, row, cols)[0]
    values = row.select(cols).row(0)
    order = np.argsort(-np.abs(contrib))[:top]
    return [
        {
            "feature": cols[i],
            "meaning": LABELS.get(cols[i], cols[i]),
            "value": None if values[i] is None else float(values[i]),
            "log_odds": float(contrib[i]),
        }
        for i in order
    ]


MISSING_TOMORROW = (
    "tomorrow's promotion, discount, holiday and weather are not in the data for this day; the "
    "model never saw them missing in training, so treat the score as unreliable"
)


def tomorrow_known(rows: pl.DataFrame) -> bool:
    """Whether tomorrow's planned covariates exist for every row (false on the data's last day)."""
    return all(
        rows[c].null_count() == 0
        for c in ("promo_next", "discount_next", "holiday_next")
    )


def score_one(
    booster: lgb.Booster,
    meta: dict,
    feat: pl.DataFrame,
    store_id: int,
    product_id: int,
    day: date,
) -> dict:
    """Risk of a ``meta['label_hours']``+ hour stockout on the day after ``day``."""
    row = feat.filter(
        (pl.col("store_id") == store_id)
        & (pl.col("product_id") == product_id)
        & (pl.col("dt") == day)
    )
    if row.height == 0:
        raise LookupError(f"no row for store {store_id}, product {product_id} on {day}")
    raw = models.predict(booster, row, meta["features"])
    return {
        "store_id": store_id,
        "product_id": product_id,
        "date": str(day),
        "predicts_day": str(day + timedelta(days=1)),
        "label_hours": meta["label_hours"],
        "actual_next_oos_hours": None
        if row["next_oos"][0] is None
        else int(row["next_oos"][0]),
        "warning": None if tomorrow_known(row) else MISSING_TOMORROW,
        "risk": float(_calibrated(meta, raw)[0]),
        "risk_raw": float(raw[0]),
        "reasons": reasons(booster, meta, row),
    }


def worklist(
    booster: lgb.Booster, meta: dict, feat: pl.DataFrame, day: date, top: int = 20
) -> pl.DataFrame:
    """The ``top`` highest-risk store-products as of the end of ``day``."""
    rows = feat.filter(pl.col("dt") == day)
    if rows.height == 0:
        raise LookupError(f"no rows for {day}; the data ends on {feat['dt'].max()}")
    raw = models.predict(booster, rows, meta["features"])
    return (
        rows.select(
            "store_id", "product_id", "oos_t", "oos_mean14", "product_heavy_rate"
        )
        .with_columns(pl.Series("risk", _calibrated(meta, raw)))
        .sort("risk", descending=True)
        .head(top)
    )
