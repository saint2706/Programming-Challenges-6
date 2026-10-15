"""A small synthetic table in the raw FreshRetailNet schema, with stockout dynamics to learn.

A store-product's chance of a heavy stockout day rises with its own propensity, with yesterday's
state, with a shock shared by all stores of the product that day, with a promotion and on weekends.
Out-of-stock hours are the last ``k`` hours of the 16-hour selling window; sales only happen in
stocked hours. The default span ends on the real data's last day so the default ``Windows`` apply.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl


def synthetic_raw(
    n_stores: int = 40,
    n_products: int = 10,
    days: int = 60,
    end: date = date(2024, 7, 2),
    seed: int = 0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_stores * n_products
    store = np.repeat(np.arange(n_stores), n_products)
    product = np.tile(np.arange(n_products), n_stores)
    propensity = rng.normal(0, 0.7, n)
    base_rate = rng.uniform(0.3, 2.0, n)
    shock = rng.normal(0, 1.0, (days, n_products))
    promo = (rng.random((days, n)) < 0.3).astype(int)
    start = end - timedelta(days=days - 1)
    prev_heavy = np.zeros(n)
    rows = []
    for d in range(days):
        day = start + timedelta(days=d)
        weekend = 1.0 if day.weekday() >= 5 else 0.0
        z = (
            -0.9
            + propensity
            + 1.3 * prev_heavy
            + 0.9 * shock[d, product]
            + 0.8 * promo[d]
            + 0.5 * weekend
        )
        heavy = rng.random(n) < 1 / (1 + np.exp(-z))
        k = np.where(heavy, rng.integers(4, 17, n), rng.integers(0, 4, n))
        prev_heavy = heavy.astype(float)
        hours = np.arange(24)
        status = np.zeros((n, 24), dtype=int)
        for i in range(n):
            if k[i] > 0:
                status[i, 22 - k[i] : 22] = 1
        profile = np.exp(-((hours - 11) ** 2) / 30.0)
        sales = rng.poisson(
            base_rate[:, None]
            * profile[None, :]
            * (1 + 0.5 * promo[d][:, None])
            * (1 - status)
        )
        rows.append(
            pl.DataFrame(
                {
                    "city_id": store % 3,
                    "store_id": store,
                    "management_group_id": store % 2,
                    "first_category_id": product % 3,
                    "second_category_id": product % 5,
                    "third_category_id": product,
                    "product_id": product,
                    "dt": str(day),
                    "sale_amount": sales.sum(axis=1).astype(float),
                    "hours_sale": sales.astype(float).tolist(),
                    "stock_hour6_22_cnt": k.astype(np.int32),
                    "hours_stock_status": status.tolist(),
                    "discount": 1.0 - 0.15 * promo[d],
                    "holiday_flag": np.int32(day.weekday() >= 5),
                    "activity_flag": promo[d].astype(np.int32),
                    "precpt": rng.random(n),
                    "avg_temperature": rng.normal(25, 3, n),
                    "avg_humidity": rng.random(n),
                    "avg_wind_level": rng.random(n),
                }
            )
        )
    return pl.concat(rows)


def one_row(status=None, sales=None, **over) -> pl.DataFrame:
    """A single raw row, for hand-checked reductions."""
    status = [0] * 24 if status is None else status
    sales = [0.0] * 24 if sales is None else sales
    row = {
        "city_id": 1,
        "store_id": 1,
        "management_group_id": 1,
        "first_category_id": 1,
        "second_category_id": 2,
        "third_category_id": 3,
        "product_id": 7,
        "dt": "2024-06-01",
        "sale_amount": float(sum(sales)),
        "hours_sale": sales,
        "stock_hour6_22_cnt": int(sum(status[6:22])),
        "hours_stock_status": status,
        "discount": 1.0,
        "holiday_flag": 0,
        "activity_flag": 0,
        "precpt": 0.0,
        "avg_temperature": 20.0,
        "avg_humidity": 0.5,
        "avg_wind_level": 1.0,
    }
    row.update(over)
    schema = {
        "hours_sale": pl.List(pl.Float64),
        "hours_stock_status": pl.List(pl.Int64),
    }
    return pl.DataFrame([row], schema_overrides=schema)
