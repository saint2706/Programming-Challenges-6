"""Features for "will this store-product stock out tomorrow?", built as of the end of day ``t``.

One row per store-product-day ``t``. ``next_oos`` is the out-of-stock hours of day ``t + 1`` (the
target is a threshold on it, see ``label``); it is null on the last day, which can still be scored.

Everything here is either history up to and including day ``t`` or a day ``t + 1`` covariate that a
planner would already know: the promotion flag, the discount, the holiday flag and the calendar.
Weather for ``t + 1`` is a forecast, so it sits in its own group and is ablated in the report.

Feature groups (the ablation drops one group at a time):

* ``history``: out-of-stock hours today and 1-6 days ago, 3/7/14-day averages, days since the last
  heavy / fully stocked day, the expanding heavy-day rate.
* ``close``: how the day ended (out of stock at open / close, in the last four hours, when the first
  out-of-stock hour came), which is the state tomorrow starts from.
* ``sales``: sales and the demand estimate that corrects for censoring, see below.
* ``cross``: today's heavy-stockout rate across the store's products and across the product's stores.
* ``promo``: tomorrow's promotion, discount, holiday and weekday, plus today's promotion state.
* ``weather``: tomorrow's weather.
* ``static``: city, management group and category ids.

Censoring: while a product is out of stock it sells nothing, so a day's sales understate demand
exactly on the days that matter. ``demand_rate`` is sales per in-stock hour, and ``demand16`` the
7-day average of it scaled to the 16-hour window. ``censor_gap`` is how far the naive 7-day sales
average sits below that estimate.
"""

from __future__ import annotations

import polars as pl

from stockout_risk.data import SERIES, WINDOW_HOURS

HEAVY_HOURS = (
    4  # a "heavy" stockout day: a quarter of the 16-hour selling window or more
)
MAX_GAP = 99  # days since an event that has not happened yet

GROUPS: dict[str, list[str]] = {
    "history": [
        "oos_t",
        *[f"oos_lag{k}" for k in range(1, 7)],
        "oos_mean3",
        "oos_mean7",
        "oos_mean14",
        "oos_std7",
        "any_rate7",
        "any_rate14",
        "heavy_rate7",
        "heavy_rate14",
        "heavy_rate_all",
        "days_since_heavy",
        "days_since_full",
    ],
    "close": ["oos_open", "oos_close", "oos_last4", "first_oos", "early_share"],
    "sales": [
        "sale_t",
        "sale_mean7",
        "sale_mean14",
        "demand_rate",
        "demand16",
        "censor_gap",
    ],
    "cross": ["store_heavy_rate", "product_heavy_rate"],
    "promo": [
        "promo_next",
        "discount_next",
        "holiday_next",
        "dow_next",
        "promo_t",
        "discount_t",
    ],
    "weather": ["precip_next", "temp_next", "humidity_next", "wind_next"],
    "static": ["city_id", "mgmt_id", "cat1", "cat2"],
}
FEATURES = [f for group in GROUPS.values() for f in group]
CATEGORICAL = ["city_id", "cat1", "cat2"]


def build(daily: pl.DataFrame, heavy: int = HEAVY_HOURS) -> pl.DataFrame:
    """Features and ``next_oos`` per store-product-day (see the module docstring)."""
    d = daily.sort([*SERIES, "dt"])
    day = pl.int_range(pl.len()).over(SERIES)
    heavy_flag = (pl.col("oos_window") >= heavy).cast(pl.Float32)
    any_flag = (pl.col("oos_window") >= 1).cast(pl.Float32)
    in_stock = (WINDOW_HOURS - pl.col("oos_window")).cast(pl.Float32)
    oos = pl.col("oos_window").cast(pl.Float32)

    def roll(expr: pl.Expr, n: int, how: str = "mean") -> pl.Expr:
        rolled = getattr(expr, f"rolling_{how}")(window_size=n, min_samples=1)
        return rolled.over(SERIES)

    def since(event: pl.Expr) -> pl.Expr:
        last = pl.when(event).then(day).otherwise(None).forward_fill().over(SERIES)
        return (day - last).fill_null(MAX_GAP).clip(upper_bound=MAX_GAP)

    d = d.with_columns(
        day.alias("_day"),
        oos.alias("oos_t"),
        heavy_flag.alias("_heavy"),
        any_flag.alias("_any"),
        in_stock.alias("_in_stock"),
        pl.col("sale").alias("sale_t"),
        pl.col("promo").alias("promo_t"),
        pl.col("discount").alias("discount_t"),
    )
    d = d.with_columns(
        *[oos.shift(k).over(SERIES).alias(f"oos_lag{k}") for k in range(1, 7)],
        roll(oos, 3).alias("oos_mean3"),
        roll(oos, 7).alias("oos_mean7"),
        roll(oos, 14).alias("oos_mean14"),
        roll(oos, 7, "std").alias("oos_std7"),
        roll(any_flag, 7).alias("any_rate7"),
        roll(any_flag, 14).alias("any_rate14"),
        roll(heavy_flag, 7).alias("heavy_rate7"),
        roll(heavy_flag, 14).alias("heavy_rate14"),
        (heavy_flag.cum_sum().over(SERIES) / (day + 1)).alias("heavy_rate_all"),
        since(heavy_flag > 0).alias("days_since_heavy"),
        since(oos == 0).alias("days_since_full"),
        roll(pl.col("sale"), 7).alias("sale_mean7"),
        roll(pl.col("sale"), 14).alias("sale_mean14"),
        pl.when(in_stock > 0)
        .then(pl.col("sales_in_stock") / in_stock)
        .otherwise(None)
        .alias("demand_rate"),
        heavy_flag.mean().over(["store_id", "dt"]).alias("store_heavy_rate"),
        heavy_flag.mean().over(["product_id", "dt"]).alias("product_heavy_rate"),
        pl.col("city_id"),
    )
    d = d.with_columns(
        (roll(pl.col("demand_rate"), 7) * WINDOW_HOURS).alias("demand16"),
    ).with_columns(
        (pl.col("demand16") - pl.col("sale_mean7")).alias("censor_gap"),
        pl.col("oos_window").shift(-1).over(SERIES).alias("next_oos"),
        pl.col("promo").shift(-1).over(SERIES).alias("promo_next"),
        pl.col("discount").shift(-1).over(SERIES).alias("discount_next"),
        pl.col("holiday").shift(-1).over(SERIES).alias("holiday_next"),
        ((pl.col("dt").dt.weekday()) % 7 + 1).alias(
            "dow_next"
        ),  # Monday=1 -> Tuesday=2 ... Sunday=7 -> 1
        pl.col("precip").shift(-1).over(SERIES).alias("precip_next"),
        pl.col("temp").shift(-1).over(SERIES).alias("temp_next"),
        pl.col("humidity").shift(-1).over(SERIES).alias("humidity_next"),
        pl.col("wind").shift(-1).over(SERIES).alias("wind_next"),
    )
    keep = [*SERIES, "dt", "_day", "next_oos", *FEATURES]
    return d.select(list(dict.fromkeys(keep)))


def label(next_oos: pl.Series | pl.Expr, hours: int = HEAVY_HOURS):
    """The target: ``hours`` or more out-of-stock hours tomorrow (null stays null)."""
    return next_oos >= hours
