"""Documented, known corruptions to plant in a clean taxi batch.

Each corruption is a pure function `(frame, rng) -> frame` plus the set of checks that
should notice it. Because the corruption is planted, the ground truth is known, so the
evaluation can report a real detection rate instead of an anecdote.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Corruption:
    name: str
    description: str
    expect: frozenset[
        str
    ]  # detected if ANY of these fires at warn or above (info for column_order)
    apply: Callable[[pl.DataFrame, np.random.Generator], pl.DataFrame]
    min_severity: str = "warn"


def _mask(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Exactly round(frac * n) True entries (at least one), in random places."""
    k = min(n, max(1, round(frac * n)))
    m = np.zeros(n, dtype=bool)
    m[rng.choice(n, size=k, replace=False)] = True
    return m


def _where(
    df: pl.DataFrame, mask: np.ndarray, col: str, replacement: pl.Expr | pl.Series
) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.Series(mask)).then(replacement).otherwise(pl.col(col)).alias(col)
    )


def _renumber_ids(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.Series("trip_id", [f"TX-{i:06d}" for i in range(df.height)])
    )


# --- schema -----------------------------------------------------------------
def drop_column(df, rng):
    return df.drop("tip")


def new_column(df, rng):
    return df.with_columns(pl.lit(0.0).alias("surge_fee"))


def rename_column(df, rng):
    return df.rename({"fare": "fare_amount"})


def shuffle_columns(df, rng):
    cols = list(df.columns)
    rng.shuffle(cols)
    return df.select(cols)


def text_in_numeric(frac):
    def apply(df, rng):
        vals = df["distance"].cast(pl.String)
        return df.with_columns(
            pl.when(pl.Series(_mask(df.height, frac, rng)))
            .then(pl.lit("N/A"))
            .otherwise(vals)
            .alias("distance")
        )

    return apply


# --- volume -----------------------------------------------------------------
def empty_batch(df, rng):
    return df.head(0)


def keep_fraction(frac):
    def apply(df, rng):
        return df.filter(pl.Series(_mask(df.height, frac, rng)))

    return apply


def repeat_batch(times):
    def apply(df, rng):
        return _renumber_ids(pl.concat([df] * times))

    return apply


# --- nulls ------------------------------------------------------------------
def null_out(col, frac):
    def apply(df, rng):
        return _where(
            df, _mask(df.height, frac, rng), col, pl.lit(None, dtype=df.schema[col])
        )

    return apply


def all_null(col):
    def apply(df, rng):
        return df.with_columns(pl.lit(None, dtype=df.schema[col]).alias(col))

    return apply


# --- numeric ----------------------------------------------------------------
def scale_column(col, factor):
    def apply(df, rng):
        return df.with_columns((pl.col(col) * factor).alias(col))

    return apply


def negative_values(col, frac):
    def apply(df, rng):
        return _where(df, _mask(df.height, frac, rng), col, -pl.col(col) - 1.0)

    return apply


def unit_error(col, frac, factor):
    def apply(df, rng):
        return _where(df, _mask(df.height, frac, rng), col, pl.col(col) * factor)

    return apply


# --- categorical ------------------------------------------------------------
def flip_category(col, src, dst, frac):
    """Move `frac` of the rows currently equal to `src` over to `dst`."""

    def apply(df, rng):
        is_src = (df[col] == src).to_numpy()
        idx = np.flatnonzero(is_src)
        if idx.size == 0:
            return df
        pick = rng.choice(idx, size=max(1, round(frac * idx.size)), replace=False)
        m = np.zeros(df.height, dtype=bool)
        m[pick] = True
        return _where(df, m, col, pl.lit(dst))

    return apply


def new_category(col, value, frac):
    def apply(df, rng):
        return _where(df, _mask(df.height, frac, rng), col, pl.lit(value))

    return apply


def drop_category(col, value):
    def apply(df, rng):
        return df.filter(pl.col(col) != value)

    return apply


# --- keys and formats -------------------------------------------------------
def duplicate_keys(frac):
    def apply(df, rng):
        m = _mask(df.height, frac, rng)
        donors = df["trip_id"].to_numpy()[rng.integers(0, df.height, size=df.height)]
        return _where(df, m, "trip_id", pl.Series(donors))

    return apply


def lowercase_keys(frac):
    def apply(df, rng):
        return _where(
            df,
            _mask(df.height, frac, rng),
            "trip_id",
            pl.col("trip_id").str.to_lowercase(),
        )

    return apply


# --- time -------------------------------------------------------------------
def epoch_zero(frac):
    def apply(df, rng):
        return _where(
            df, _mask(df.height, frac, rng), "pickup", pl.lit(0).cast(pl.Datetime("us"))
        )

    return apply


def shift_time(days):
    def apply(df, rng):
        return df.with_columns(
            (pl.col("pickup") + pl.duration(days=days)).alias("pickup"),
            (pl.col("dropoff") + pl.duration(days=days)).alias("dropoff"),
        )

    return apply


def future_rows(frac, days):
    def apply(df, rng):
        return _where(
            df,
            _mask(df.height, frac, rng),
            "pickup",
            pl.col("pickup") + pl.duration(days=days),
        )

    return apply


def spread_over_days(days):
    def apply(df, rng):
        offs = pl.Series(
            rng.integers(0, days, size=df.height).astype("int64") * 86_400_000_000
        ).cast(pl.Duration("us"))
        return df.with_columns((pl.col("pickup") - offs).alias("pickup"))

    return apply


def _c(name, description, expect, fn, min_severity="warn"):
    return Corruption(name, description, frozenset(expect), fn, min_severity)


CORRUPTIONS: list[Corruption] = [
    _c(
        "drop_column", "`tip` column disappears", {"schema.missing_column"}, drop_column
    ),
    _c(
        "new_column",
        "unexpected `surge_fee` column appears",
        {"schema.new_column"},
        new_column,
    ),
    _c(
        "rename_column",
        "`fare` renamed to `fare_amount`",
        {"schema.possible_rename"},
        rename_column,
    ),
    _c(
        "shuffle_columns",
        "same columns, different order",
        {"schema.column_order"},
        shuffle_columns,
        "info",
    ),
    _c(
        "text_in_numeric",
        "5% of `distance` becomes the string 'N/A'",
        {"schema.type_change", "values.cast_failures"},
        text_in_numeric(0.05),
    ),
    _c("empty_batch", "header only, zero rows", {"volume.empty"}, empty_batch),
    _c(
        "volume_tenth",
        "only 10% of the rows arrive",
        {"volume.row_count"},
        keep_fraction(0.10),
    ),
    _c(
        "volume_x4",
        "the batch is delivered four times over",
        {"volume.row_count"},
        repeat_batch(4),
    ),
    _c(
        "nulls_fare_15pct",
        "15% of `fare` is null",
        {"nulls.rate_shift"},
        null_out("fare", 0.15),
    ),
    _c("all_null_tip", "`tip` is entirely null", {"nulls.all_null"}, all_null("tip")),
    _c(
        "fare_x1.5",
        "`fare` scaled by 1.5 (currency/unit change)",
        {"drift.numeric"},
        scale_column("fare", 1.5),
    ),
    _c(
        "negative_fares",
        "2% of `fare` is negative",
        {"values.out_of_range"},
        negative_values("fare", 0.02),
    ),
    _c(
        "distance_x1000",
        "2% of `distance` off by 1000x (metres vs km)",
        {"values.out_of_range"},
        unit_error("distance", 0.02, 1000.0),
    ),
    _c(
        "payment_mix",
        "60% of cash trips relabelled credit card",
        {"drift.categorical"},
        flip_category("payment", "cash", "credit card", 0.6),
    ),
    _c(
        "new_payment_type",
        "10% of `payment` becomes an unseen 'crypto'",
        {"values.unseen_categories"},
        new_category("payment", "crypto", 0.10),
    ),
    _c(
        "payment_type_gone",
        "every 'cash' row missing",
        {"values.missing_categories"},
        drop_category("payment", "cash"),
    ),
    _c(
        "duplicate_keys",
        "5% of `trip_id` values duplicate another row",
        {"keys.duplicates"},
        duplicate_keys(0.05),
    ),
    _c(
        "key_format",
        "10% of `trip_id` lowercased",
        {"format.unexpected_shape"},
        lowercase_keys(0.10),
    ),
    _c(
        "epoch_zero_dates",
        "5% of `pickup` is 1970-01-01",
        {"freshness.implausible"},
        epoch_zero(0.05),
    ),
    _c(
        "future_dates",
        "5% of `pickup` is 30 days in the future",
        {"freshness.future"},
        future_rows(0.05, 30),
    ),
    _c(
        "stale_batch",
        "every timestamp is 10 days old",
        {"freshness.stale"},
        shift_time(-10),
    ),
    _c(
        "span_5_days",
        "rows spread over 5 days instead of one",
        {"freshness.span"},
        spread_over_days(5),
    ),
]


def scale_sweep(factors=(1.05, 1.1, 1.25, 1.5, 2.0)) -> list[Corruption]:
    return [
        _c(f"fare_x{f}", f"`fare` x {f}", {"drift.numeric"}, scale_column("fare", f))
        for f in factors
    ]


def null_sweep(fracs=(0.005, 0.01, 0.02, 0.05, 0.15)) -> list[Corruption]:
    return [
        _c(
            f"fare_null_{f:.1%}",
            f"{f:.1%} of `fare` null",
            {"nulls.rate_shift"},
            null_out("fare", f),
        )
        for f in fracs
    ]


def mix_sweep(fracs=(0.1, 0.25, 0.5, 0.75, 1.0)) -> list[Corruption]:
    return [
        _c(
            f"cash_to_card_{f:.0%}",
            f"{f:.0%} of cash trips relabelled credit card",
            {"drift.categorical"},
            flip_category("payment", "cash", "credit card", f),
        )
        for f in fracs
    ]
