from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from stockout_risk import features
from stockout_risk.features import FEATURES, GROUPS


def tiny_daily(oos: list[int], **extra) -> pl.DataFrame:
    """One series, one row per day with the given out-of-stock hours (everything else neutral)."""
    n = len(oos)
    return pl.DataFrame(
        {
            "city_id": [1] * n,
            "store_id": [1] * n,
            "mgmt_id": [1] * n,
            "cat1": [1] * n,
            "cat2": [2] * n,
            "cat3": [3] * n,
            "product_id": [7] * n,
            "dt": [date(2024, 6, 3) + timedelta(days=i) for i in range(n)],  # a Monday
            "sale": [1.0] * n,
            "oos_window": oos,
            "oos_open": [0] * n,
            "oos_close": [0] * n,
            "oos_last4": [0] * n,
            "first_oos": [16] * n,
            "sales_in_stock": [1.0] * n,
            "early_share": [0.5] * n,
            "discount": [1.0] * n,
            "holiday": [0] * n,
            "promo": [0] * n,
            "precip": [0.0] * n,
            "temp": [20.0] * n,
            "humidity": [0.5] * n,
            "wind": [1.0] * n,
        }
    ).with_columns(
        pl.col("oos_window").cast(pl.Int8),
        *[
            pl.lit(v).alias(k) if not isinstance(v, list) else pl.Series(k, v)
            for k, v in extra.items()
        ],
    )


def test_history_features_by_hand():
    f = features.build(tiny_daily([0, 0, 6, 0, 8, 2, 0, 10])).sort("dt")
    last = f.row(7, named=True)
    assert last["oos_t"] == 10 and last["oos_lag1"] == 0 and last["oos_lag2"] == 2
    assert last["oos_lag3"] == 8 and last["oos_lag6"] == 0
    assert last["oos_mean3"] == pytest.approx((2 + 0 + 10) / 3)
    assert last["oos_mean7"] == pytest.approx((0 + 6 + 0 + 8 + 2 + 0 + 10) / 7)
    assert last["heavy_rate7"] == pytest.approx(
        3 / 7
    )  # days with 6, 8 and 10 hours (>= 4)
    assert last["any_rate7"] == pytest.approx(4 / 7)
    assert last["days_since_heavy"] == 0 and last["days_since_full"] == 1
    assert last["heavy_rate_all"] == pytest.approx(3 / 8)
    first = f.row(0, named=True)
    assert first["oos_lag1"] is None and first["days_since_heavy"] == features.MAX_GAP


def test_target_and_next_day_covariates_are_shifted_back_one_day():
    d = tiny_daily([0, 5, 2], promo=[0, 1, 0], discount=[1.0, 0.8, 0.9])
    f = features.build(d).sort("dt")
    assert f["next_oos"].to_list() == [5, 2, None]
    assert f["promo_next"].to_list() == [1, 0, None]
    assert f["discount_next"].to_list() == [0.8, 0.9, None]
    assert f["promo_t"].to_list() == [0, 1, 0]
    # 2024-06-03 is a Monday, so the first row predicts Tuesday = 2; Monday's row wraps Sunday to 1.
    assert f["dow_next"].to_list() == [2, 3, 4]
    assert (
        features.build(tiny_daily([0] * 6 + [1])).sort("dt")["dow_next"].to_list()[-1]
        == 1
    )


def test_demand_estimate_corrects_for_censoring():
    # 8 of 16 hours out of stock and 4 units sold in the other 8: half a unit per stocked hour.
    d = tiny_daily([8, 8, 8], sales_in_stock=[4.0] * 3, sale=[4.0] * 3)
    last = features.build(d).sort("dt").row(2, named=True)
    assert last["demand_rate"] == pytest.approx(0.5)
    assert last["demand16"] == pytest.approx(8.0)
    assert last["censor_gap"] == pytest.approx(8.0 - 4.0)


def test_demand_rate_is_missing_when_nothing_was_in_stock():
    f = features.build(tiny_daily([16, 16])).sort("dt")
    assert f["demand_rate"].is_null().all()


def test_feature_groups_cover_every_feature_once():
    flat = [f for g in GROUPS.values() for f in g]
    assert len(flat) == len(set(flat)) == len(FEATURES)


def test_build_output_has_every_feature_and_one_row_per_day(daily, feat):
    assert set(FEATURES) <= set(feat.columns)
    assert feat.height == daily.height
    assert feat.select(pl.col(FEATURES).is_infinite().any()).to_numpy().sum() == 0


def test_cross_series_rates_use_only_that_day():
    d = pl.concat(
        [
            tiny_daily([8, 0]),
            tiny_daily([0, 0]).with_columns(
                pl.lit(2, dtype=pl.Int64).alias("product_id")
            ),
        ]
    )
    f = features.build(d)
    day0 = f.filter(pl.col("dt") == date(2024, 6, 3))
    assert day0["store_heavy_rate"].to_list() == [
        0.5,
        0.5,
    ]  # one of the store's two products heavy
    assert day0.sort("product_id")["product_heavy_rate"].to_list() == [
        0.0,
        1.0,
    ]  # product 2 sorts first


def test_no_leakage_from_the_future(daily, feat):
    """Rewrite every day after ``cut`` with random values: features of days up to ``cut`` stay put.

    The next-day covariates (promo, discount, holiday, weather, ``next_oos``) are allowed to differ
    on the cut day itself, since they describe day ``cut + 1`` by design; everything else must not.
    """
    cut = date(2024, 6, 20)
    rng = np.random.default_rng(3)
    future = pl.col("dt") > cut
    changed = daily.with_columns(
        pl.when(future)
        .then(rng.integers(0, 17, daily.height))
        .otherwise(pl.col("oos_window"))
        .cast(pl.Int8)
        .alias("oos_window"),
        pl.when(future)
        .then(rng.integers(0, 2, daily.height))
        .otherwise(pl.col("oos_close"))
        .cast(pl.Int8)
        .alias("oos_close"),
        pl.when(future)
        .then(rng.random(daily.height) * 9)
        .otherwise(pl.col("sale"))
        .alias("sale"),
        pl.when(future)
        .then(rng.random(daily.height) * 9)
        .otherwise(pl.col("sales_in_stock"))
        .cast(pl.Float32)
        .alias("sales_in_stock"),
        pl.when(future).then(0.5).otherwise(pl.col("promo")).alias("promo"),
        pl.when(future).then(0.1).otherwise(pl.col("discount")).alias("discount"),
    )
    after = features.build(changed)
    allowed = set(GROUPS["weather"]) | {
        "promo_next",
        "discount_next",
        "holiday_next",
        "next_oos",
    }
    check = [f for f in FEATURES if f not in allowed]
    keys = ["store_id", "product_id", "dt"]
    a = feat.filter(pl.col("dt") <= cut).sort(keys).select(check)
    b = after.filter(pl.col("dt") <= cut).sort(keys).select(check)
    assert a.equals(b)


def test_build_matches_polars_grouped_means(daily, feat):
    check = (
        daily.sort(["store_id", "product_id", "dt"])
        .with_columns(
            pl.col("oos_window")
            .cast(pl.Float64)
            .rolling_mean(7, min_samples=1)
            .over(["store_id", "product_id"])
            .alias("m")
        )
        .select("m")
        .to_series()
    )
    got = feat.sort(["store_id", "product_id", "dt"])["oos_mean7"]
    assert np.allclose(got.to_numpy(), check.to_numpy(), atol=1e-5)
