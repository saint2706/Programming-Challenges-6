"""Oracles and result comparison.

Two oracles, both exact:

* ``duckdb_oracle``: the shared SQL with ``count(DISTINCT pu_location_id)``;
* ``polars_oracle``: the Polars plan with ``n_unique`` in the in-memory engine.

Each engine's answer is compared with the DuckDB oracle, and the two oracles are compared with each
other, so a bug in either shows up. Rules (documented in ``workloads.py``):

* row count, key columns and integer columns must match exactly;
* float columns (sums, means, variances, medians) must match to relative 1e-9 (absolute 1e-9
  near zero), since summation order changes the last bits;
* ``approx`` columns (distinct counts) must match the exact count within the query's relative bound;
* timestamps are compared in UTC microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
import pyarrow as pa

from dist_pipeline.engines import duckdb_engine, polars_engine
from dist_pipeline.engines.base import Dataset
from dist_pipeline.workloads import Query

FLOAT_RTOL = 1e-9
FLOAT_ATOL = 1e-9
APPROX_SLACK = 5.0


@dataclass
class Verdict:
    ok: bool
    rows: int
    detail: str = "ok"
    max_rel_error: dict[str, float] = field(default_factory=dict)


def duckdb_oracle(query: Query, ds: Dataset) -> pa.Table:
    con = duckdb_engine.connect(ds)
    try:
        return duckdb_engine.run_sql(con, query, duckdb_engine.EXACT_DISTINCT)
    finally:
        con.close()


def polars_oracle(query: Query, ds: Dataset) -> pa.Table:
    zones = ds.zones_lazy() if query.needs_zones else None
    plan = polars_engine.build(
        query.name, polars_engine.trips_frame(ds), zones, exact=True
    )
    return plan.collect(engine="in-memory").to_arrow()


def _normalise(t: pa.Table | pl.DataFrame) -> pl.DataFrame:
    df = pl.from_arrow(t) if isinstance(t, pa.Table) else t
    exprs = []
    for name, dtype in df.schema.items():
        if isinstance(dtype, pl.Datetime):
            exprs.append(
                pl.col(name)
                .dt.convert_time_zone("UTC")
                .dt.replace_time_zone(None)
                .dt.cast_time_unit("us")
                .alias(name)
            )
    return df.with_columns(exprs) if exprs else df


def _float_close(ac: pl.Series, ec: pl.Series) -> pl.Series:
    """Element-wise: both null, or within the float rule of each other."""
    frame = pl.DataFrame({"a": ac.cast(pl.Float64), "e": ec.cast(pl.Float64)})
    both_null = pl.col("a").is_null() & pl.col("e").is_null()
    near = (pl.col("a") - pl.col("e")).abs() <= pl.max_horizontal(
        pl.lit(FLOAT_ATOL), FLOAT_RTOL * pl.col("e").abs()
    )
    return frame.select((both_null | near).fill_null(False)).to_series()


def compare(
    actual: pa.Table | pl.DataFrame,
    expected: pa.Table | pl.DataFrame,
    query: Query,
) -> Verdict:
    a, e = _normalise(actual), _normalise(expected)
    rows = e.height
    if set(a.columns) != set(e.columns):
        return Verdict(
            False, rows, f"columns {sorted(a.columns)} != {sorted(e.columns)}"
        )
    if a.height != e.height:
        return Verdict(False, rows, f"row count {a.height} != {e.height}")
    a = a.select(e.columns)
    if query.keys:
        a = a.sort(list(query.keys), nulls_last=True)
        e = e.sort(list(query.keys), nulls_last=True)
    max_rel: dict[str, float] = {}
    for col in e.columns:
        ac, ec = a[col], e[col]
        if col in query.keys or col in query.exact:
            if ec.dtype.is_float() or ac.dtype.is_float():
                ok_mask = _float_close(ac, ec)
            else:
                ok_mask = ac.eq_missing(ec.cast(ac.dtype))
            if not ok_mask.all():
                i = int(ok_mask.arg_min())
                return Verdict(
                    False,
                    rows,
                    f"{col}: row {i} got {ac[i]!r}, want {ec[i]!r}",
                    max_rel,
                )
        if col in query.approx:
            bound = query.approx[col]
            # Relative bound, with an absolute slack of APPROX_SLACK distinct values: HyperLogLog
            # is biased on groups of a few dozen values, so a purely relative bound is too tight.
            frame = pl.DataFrame({"a": ac.cast(pl.Float64), "e": ec.cast(pl.Float64)})
            err = (pl.col("a") - pl.col("e")).abs()
            rel = frame.select(
                (err / pl.max_horizontal(pl.col("e"), pl.lit(1.0))).alias("r")
            )["r"]
            over = frame.select(
                (err > pl.max_horizontal(bound * pl.col("e"), pl.lit(APPROX_SLACK)))
                .fill_null(False)
                .alias("o")
            )["o"]
            worst = float(rel.max() or 0.0)
            max_rel[col] = worst
            if over.any():
                i = int(over.arg_max())
                return Verdict(
                    False,
                    rows,
                    f"{col}: row {i} got {ac[i]}, want {ec[i]} (relative error "
                    f"{rel[i]:.4f} > bound {bound} beyond {APPROX_SLACK} distinct values)",
                    max_rel,
                )
    return Verdict(True, rows, "ok", max_rel)
