"""The workload suite: nine queries over the normalized yellow taxi table.

Every query is written once as ANSI-style SQL (DuckDB dialect, which the oracle runs). Engines
implement the same query with their own API and must return the same columns, with the same
names, sorted by ``keys``. ``{trips}``, ``{zones}`` and ``{distinct}`` are placeholders the
harness fills in; ``{distinct}`` is the one place an engine may use an approximate function.

Tolerances (used by ``compare.py``):

* counts and ordinal columns must match exactly;
* sums, means, variances and medians must agree to relative ``1e-9`` (floats summed in a different
  order differ in the last bits; this is the documented float rule);
* ``approx`` columns (distinct counts) must agree with the exact count to a documented relative
  error bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Query:
    name: str
    title: str
    purpose: str
    sql: str
    keys: tuple[str, ...]
    exact: tuple[str, ...]
    approx: dict[str, float] = field(default_factory=dict)
    needs_zones: bool = False
    # Engines that cannot express the query in a mergeable partial-aggregate form mark it N/A.
    mergeable: bool = True


QUERIES: tuple[Query, ...] = (
    Query(
        name="q0_scan_sum",
        title="Full scan, one column",
        purpose="Reads one column of every row: the raw read-plus-sum cost with no pushdown.",
        sql="SELECT sum(total_amount) AS total_sum, count(*) AS trips FROM {trips}",
        keys=(),
        exact=("trips", "total_sum"),
    ),
    Query(
        name="q1_filter_count",
        title="Selective filter, count",
        purpose="Two predicates and a count: rewards predicate and column pushdown.",
        sql=(
            "SELECT count(*) AS trips FROM {trips} "
            "WHERE trip_distance > 5 AND fare_amount > 20"
        ),
        keys=(),
        exact=("trips",),
    ),
    Query(
        name="q2_skewed_payment",
        title="Skewed key, few groups",
        purpose=(
            "Groups by payment_type, where two of the codes carry about 99% of rows. A hash "
            "partition on this key leaves most reducers idle: the straggler case."
        ),
        sql=(
            "SELECT payment_type, count(*) AS trips, sum(total_amount) AS total_sum, "
            "avg(tip_amount) AS avg_tip FROM {trips} GROUP BY payment_type ORDER BY payment_type"
        ),
        keys=("payment_type",),
        exact=("trips", "total_sum", "avg_tip"),
    ),
    Query(
        name="q3_high_card_od",
        title="High cardinality, pickup x dropoff",
        purpose="About 60k groups from the zone pair: a large hash-aggregation state.",
        sql=(
            "SELECT pu_location_id, do_location_id, count(*) AS trips, "
            "sum(fare_amount) AS fare_sum FROM {trips} "
            "GROUP BY pu_location_id, do_location_id "
            "ORDER BY pu_location_id, do_location_id"
        ),
        keys=("pu_location_id", "do_location_id"),
        exact=("trips", "fare_sum"),
    ),
    Query(
        name="q4_moments_distinct",
        title="Multi-key moments and approximate distinct",
        purpose=(
            "Mean and sample variance (a Chan-mergeable pair) plus an approximate distinct count "
            "of pickup zones per vendor, payment type and pickup hour."
        ),
        sql=(
            "SELECT vendor_id, payment_type, hour(pickup_ts) AS pickup_hour, count(*) AS trips, "
            "avg(fare_amount) AS fare_mean, var_samp(fare_amount) AS fare_var, "
            "{distinct} AS pu_distinct FROM {trips} "
            "GROUP BY vendor_id, payment_type, hour(pickup_ts) "
            "ORDER BY vendor_id, payment_type, pickup_hour"
        ),
        keys=("vendor_id", "payment_type", "pickup_hour"),
        exact=("trips", "fare_mean", "fare_var"),
        # Relative error bound for approximate distinct counts. Our HLL (2^12 registers) has a
        # standard error of 1.04/sqrt(4096) = 1.6%, so 5% is about 3 sigma.
        approx={"pu_distinct": 0.05},
    ),
    Query(
        name="q5_median_fare",
        title="Exact median per group",
        purpose=(
            "Exact median fare per vendor and payment type. A median is not a fixed-size partial "
            "aggregate, so the mergeable pipeline marks it N/A."
        ),
        sql=(
            "SELECT vendor_id, payment_type, median(fare_amount) AS fare_median "
            "FROM {trips} GROUP BY vendor_id, payment_type ORDER BY vendor_id, payment_type"
        ),
        keys=("vendor_id", "payment_type"),
        exact=("fare_median",),
        mergeable=False,
    ),
    Query(
        name="q6_hour_rollup",
        title="Time-bucketed rollup (hour)",
        purpose="One group per pickup hour across the scale: a time-series rollup.",
        sql=(
            "SELECT date_trunc('hour', pickup_ts) AS pickup_hour_ts, count(*) AS trips, "
            "sum(fare_amount) AS fare_sum FROM {trips} GROUP BY 1 ORDER BY 1"
        ),
        keys=("pickup_hour_ts",),
        exact=("trips", "fare_sum"),
    ),
    Query(
        name="q7_zone_join",
        title="Join to the 265-row zone lookup, group by borough",
        purpose="A small dimension join (broadcast-able) followed by a six-group aggregate.",
        sql=(
            "SELECT z.borough, count(*) AS trips, sum(t.total_amount) AS total_sum "
            "FROM {trips} t JOIN {zones} z ON t.pu_location_id = z.location_id "
            "GROUP BY z.borough ORDER BY z.borough"
        ),
        keys=("borough",),
        exact=("trips", "total_sum"),
        needs_zones=True,
    ),
    Query(
        name="q8_topk_window",
        title="Top-3 pickup zones per borough (window rank)",
        purpose=(
            "Aggregate by borough and zone, then rank within each borough. The rank is a window "
            "over a second aggregate, so it needs a second exchange in a distributed engine."
        ),
        sql=(
            "WITH agg AS (SELECT z.borough, t.pu_location_id, count(*) AS trips "
            "FROM {trips} t JOIN {zones} z ON t.pu_location_id = z.location_id "
            "GROUP BY z.borough, t.pu_location_id), "
            "ranked AS (SELECT borough, pu_location_id, trips, row_number() OVER "
            "(PARTITION BY borough ORDER BY trips DESC, pu_location_id) AS rk FROM agg) "
            "SELECT borough, pu_location_id, trips, rk FROM ranked WHERE rk <= 3 "
            "ORDER BY borough, rk"
        ),
        keys=("borough", "rk"),
        exact=("pu_location_id", "trips"),
        needs_zones=True,
    ),
)

BY_NAME = {q.name: q for q in QUERIES}


def get(name: str) -> Query:
    try:
        return BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"unknown query {name!r}; known: {', '.join(BY_NAME)}") from exc


def render_sql(query: Query, trips: str, zones: str, distinct: str) -> str:
    """Fill the placeholders. ``trips`` and ``zones`` are SQL table expressions."""
    return query.sql.format(trips=trips, zones=zones, distinct=distinct)
