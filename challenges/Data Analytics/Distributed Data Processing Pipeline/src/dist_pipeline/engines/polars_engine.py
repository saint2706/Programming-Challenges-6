"""Polars adapter: the expression API on a LazyFrame, collected with the chosen engine.

``collect(engine="streaming")`` is Polars' batch-streaming engine; ``engine="in-memory"`` is the
default in-memory engine. The single-thread baseline is the in-memory engine with
``POLARS_MAX_THREADS=1``, which the worker sets in its environment before Polars is imported.
"""

from __future__ import annotations

import polars as pl
import pyarrow as pa

from dist_pipeline.engines.base import Dataset
from dist_pipeline.workloads import Query

LABEL = "Polars (expression API)"


def trips_frame(ds: Dataset) -> pl.LazyFrame:
    if ds.fmt == "csv":
        return pl.scan_csv(ds.source, try_parse_dates=True)
    return pl.scan_parquet(ds.parquet_glob)


def build(
    name: str,
    trips: pl.LazyFrame,
    zones: pl.LazyFrame | None,
    exact: bool = False,
) -> pl.LazyFrame:
    """The query as a lazy plan. ``exact`` swaps approximate distinct counts for exact ones."""
    t = trips
    if name == "q0_scan_sum":
        return t.select(total_sum=pl.col("total_amount").sum(), trips=pl.len())
    if name == "q1_filter_count":
        return t.filter(
            (pl.col("trip_distance") > 5) & (pl.col("fare_amount") > 20)
        ).select(trips=pl.len())
    if name == "q2_skewed_payment":
        return (
            t.group_by("payment_type")
            .agg(
                trips=pl.len(),
                total_sum=pl.col("total_amount").sum(),
                avg_tip=pl.col("tip_amount").mean(),
            )
            .sort("payment_type")
        )
    if name == "q3_high_card_od":
        return (
            t.group_by("pu_location_id", "do_location_id")
            .agg(trips=pl.len(), fare_sum=pl.col("fare_amount").sum())
            .sort("pu_location_id", "do_location_id")
        )
    if name == "q4_moments_distinct":
        distinct = (
            pl.col("pu_location_id").n_unique()
            if exact
            else pl.col("pu_location_id").approx_n_unique()
        )
        return (
            t.group_by(
                "vendor_id",
                "payment_type",
                pl.col("pickup_ts").dt.hour().alias("pickup_hour"),
            )
            .agg(
                trips=pl.len(),
                fare_mean=pl.col("fare_amount").mean(),
                fare_var=pl.col("fare_amount").var(ddof=1),
                pu_distinct=distinct,
            )
            .sort("vendor_id", "payment_type", "pickup_hour")
        )
    if name == "q5_median_fare":
        return (
            t.group_by("vendor_id", "payment_type")
            .agg(fare_median=pl.col("fare_amount").median())
            .sort("vendor_id", "payment_type")
        )
    if name == "q6_hour_rollup":
        return (
            t.group_by(pl.col("pickup_ts").dt.truncate("1h").alias("pickup_hour_ts"))
            .agg(trips=pl.len(), fare_sum=pl.col("fare_amount").sum())
            .sort("pickup_hour_ts")
        )
    if name in ("q7_zone_join", "q8_topk_window"):
        assert zones is not None
        joined = t.join(
            zones,
            left_on=pl.col("pu_location_id").cast(pl.Int32),
            right_on="location_id",
            how="inner",
        )
        if name == "q7_zone_join":
            return (
                joined.group_by("borough")
                .agg(trips=pl.len(), total_sum=pl.col("total_amount").sum())
                .sort("borough")
            )
        return (
            joined.group_by("borough", "pu_location_id")
            .agg(trips=pl.len())
            .sort(
                ["borough", "trips", "pu_location_id"], descending=[False, True, False]
            )
            .with_columns(rk=pl.col("trips").cum_count().over("borough"))
            .filter(pl.col("rk") <= 3)
            .select("borough", "pu_location_id", "trips", "rk")
            .sort("borough", "rk")
        )
    raise KeyError(name)


def start(ds: Dataset, opts: dict[str, object]) -> None:
    return None


def run(session: None, query: Query, ds: Dataset, opts: dict[str, object]) -> pa.Table:
    zones = ds.zones_lazy() if query.needs_zones else None
    plan = build(query.name, trips_frame(ds), zones)
    return plan.collect(engine=str(opts.get("collect", "streaming"))).to_arrow()


def stop(session: None) -> None:
    return None
