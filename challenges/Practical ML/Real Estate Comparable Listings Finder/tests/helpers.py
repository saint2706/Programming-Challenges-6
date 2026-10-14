"""Synthetic King County-shaped sales with a known price function (the real data is in conftest).

``log price = 12 + 0.8 log(living/1800) + 0.12 (grade-7) + 0.05 view - 0.03 km_from_centre + 0.01
months + noise``. Bedrooms, floors, condition, waterfront and the basement flag do not enter the
price, so an explanation that credits them is wrong, and the month trend is a known 0.01 per month.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
from comp_finder import data

START = date(2014, 5, 2)


def synthetic_raw(n: int = 3000, seed: int = 0, resales: int = 20) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    lat = 47.6 + rng.normal(0, 0.06, n)
    long = -122.3 + rng.normal(0, 0.08, n)
    living = np.exp(rng.normal(np.log(1800), 0.35, n)).round()
    lot = np.exp(rng.normal(np.log(7000), 0.5, n)).round()
    grade = rng.integers(5, 12, n)
    view = rng.choice([0, 0, 0, 1, 2, 3], n)
    day = np.sort(rng.integers(0, 390, n))
    km = np.hypot((long + 122.3) * 75.0, (lat - 47.6) * 111.0)
    log_price = (
        12.0
        + 0.8 * np.log(living / 1800)
        + 0.12 * (grade - 7)
        + 0.05 * view
        - 0.03 * km
        + 0.01 * (day / 30.0)
        + rng.normal(0, 0.04, n)
    )
    basement = np.where(rng.random(n) < 0.3, 400, 0)
    df = pl.DataFrame(
        {
            "id": 1_000_000 + np.arange(n),
            "date": [
                (START + timedelta(days=int(d))).strftime("%Y%m%dT000000") for d in day
            ],
            "price": np.exp(log_price).round(),
            "bedrooms": rng.integers(2, 6, n),
            "bathrooms": rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], n),
            "sqft_living": living.astype(int),
            "sqft_lot": lot.astype(int),
            "floors": rng.choice([1.0, 1.5, 2.0], n),
            "waterfront": (rng.random(n) < 0.03).astype(int),
            "view": view,
            "condition": rng.integers(1, 6, n),
            "grade": grade,
            "sqft_above": (living - basement).astype(int),
            "sqft_basement": basement.astype(int),
            "yr_built": rng.integers(1920, 2014, n),
            "yr_renovated": np.where(rng.random(n) < 0.1, 2005, 0),
            "zipcode": [f"98{int(a * 40) % 90 + 10}" for a in lat],
            "lat": lat,
            "long": long,
            "sqft_living15": (living * rng.normal(1, 0.1, n)).astype(int),
            "sqft_lot15": (lot * rng.normal(1, 0.1, n)).astype(int),
        }
    )
    if (
        resales
    ):  # the last sales repeat an earlier property's id and attributes, as resales do
        src = df.head(resales).with_columns(
            pl.col("date").str.slice(0, 8).str.to_date("%Y%m%d").alias("d")
        )
        later = df.tail(resales).select("date")
        df = pl.concat(
            [
                df.head(n - resales),
                src.drop("d", "date")
                .with_columns(later["date"].alias("date"))
                .select(df.columns),
            ]
        )
    return df


def synthetic_sales(n: int = 3000, seed: int = 0, resales: int = 20) -> pl.DataFrame:
    return data.clean(synthetic_raw(n, seed, resales))
