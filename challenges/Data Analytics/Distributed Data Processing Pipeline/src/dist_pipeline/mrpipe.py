"""A from-scratch map -> hash shuffle -> reduce pipeline with mergeable partial aggregates.

This is the challenge's own "distributed" engine. It is deliberately small: one ``multiprocessing``
pool (``ProcessPoolExecutor``), one Parquet part file per map task, and the data exchanged between
stages as Arrow IPC bytes, so the serialisation and shuffle cost is measured rather than hidden.

Stages for a query:

1. **map** (one task per part file, in parallel): read the part file, apply the filter and derived
   columns (a broadcast join to the zone lookup where the query needs one), and compute a *partial*
   aggregate per group: row count, sums, and the moments (count, mean, M2) of each value column.
   Distinct counts use a HyperLogLog with 2^12 registers; the partial is the register maximum per
   (group, register). Each partial is hash-partitioned on the group key into ``R`` buckets.
2. **shuffle**: the bucket bytes travel from the map workers to the driver and on to the reduce
   tasks. ``ShuffleStats`` records the bytes and the rows per bucket.
3. **reduce** (one task per bucket, in parallel): merge the partials of the keys in that bucket.
   Counts and sums add. Moments merge with Chan et al.'s parallel update, generalised to any number
   of partials: N = sum n_i, mean = sum(n_i * mean_i) / N, M2 = sum(M2_i + n_i (mean_i - mean)^2).
   HyperLogLog registers merge by element-wise maximum. The distinct estimate is the standard
   HyperLogLog estimator with linear counting for small cardinalities.
4. **final** (in the driver, on the merged result, which is small): projection, and for the
   top-k query the per-borough rank.

Not supported: exact medians (q5). A median is not a fixed-size mergeable state; the reduce would
have to ship every value of a group to one task. The engine reports that query as not supported.

Hashing for the HyperLogLog uses Polars' ``hash`` (fixed seed, so it is deterministic across
processes and runs). Its 64-bit output is split into a 12-bit register index and a 52-bit rank.
"""

from __future__ import annotations

import io
import multiprocessing
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

HLL_P = 12
HLL_M = 1 << HLL_P
HLL_RANK_BITS = 64 - HLL_P
HLL_ALPHA = 0.7213 / (1 + 1.079 / HLL_M)


# ---------------------------------------------------------------- query specs


@dataclass(frozen=True)
class Spec:
    columns: tuple[str, ...]
    keys: tuple[str, ...] = ()
    prep: Callable[[pl.DataFrame, pl.DataFrame | None], pl.DataFrame] = lambda df, z: df
    sums: tuple[str, ...] = ()
    moments: tuple[str, ...] = ()
    hll: str | None = None
    needs_zones: bool = False
    project: Callable[[pl.DataFrame], pl.DataFrame] = field(default=lambda df: df)


def _keep(*cols: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    return lambda df: df.select(*cols)


def _filter_q1(df: pl.DataFrame, _z: pl.DataFrame | None) -> pl.DataFrame:
    return df.filter((pl.col("trip_distance") > 5) & (pl.col("fare_amount") > 20))


def _hour(df: pl.DataFrame, _z: pl.DataFrame | None) -> pl.DataFrame:
    return df.with_columns(pickup_hour=pl.col("pickup_ts").dt.hour())


def _hour_ts(df: pl.DataFrame, _z: pl.DataFrame | None) -> pl.DataFrame:
    return df.with_columns(pickup_hour_ts=pl.col("pickup_ts").dt.truncate("1h"))


def _zone_join(df: pl.DataFrame, zones: pl.DataFrame | None) -> pl.DataFrame:
    assert zones is not None, "zone lookup must be broadcast to the map task"
    return df.join(zones, left_on="pu_location_id", right_on="location_id", how="inner")


def _proj_q4(df: pl.DataFrame) -> pl.DataFrame:
    n = pl.col("fare_amount__n")
    return df.select(
        "vendor_id",
        "payment_type",
        "pickup_hour",
        pl.col("trips"),
        pl.col("fare_amount__mean").alias("fare_mean"),
        pl.when(n > 1)
        .then(pl.col("fare_amount__m2") / (n - 1))
        .otherwise(None)
        .alias("fare_var"),
        pl.col("pu_distinct"),
    )


def _rank_q8(df: pl.DataFrame) -> pl.DataFrame:
    ranked = (
        df.sort(["borough", "trips", "pu_location_id"], descending=[False, True, False])
        .with_columns(rk=pl.col("trips").cum_count().over("borough"))
        .filter(pl.col("rk") <= 3)
    )
    return ranked.select("borough", "pu_location_id", "trips", "rk").sort(
        "borough", "rk"
    )


SPECS: dict[str, Spec | None] = {
    "q0_scan_sum": Spec(
        columns=("total_amount",),
        sums=("total_amount",),
        project=lambda df: df.select(
            "trips", pl.col("total_amount_sum").alias("total_sum")
        ),
    ),
    "q1_filter_count": Spec(
        columns=("trip_distance", "fare_amount"),
        prep=_filter_q1,
        project=_keep("trips"),
    ),
    "q2_skewed_payment": Spec(
        columns=("payment_type", "total_amount", "tip_amount"),
        keys=("payment_type",),
        sums=("total_amount",),
        moments=("tip_amount",),
        project=lambda df: df.select(
            "payment_type",
            "trips",
            pl.col("total_amount_sum").alias("total_sum"),
            pl.col("tip_amount__mean").alias("avg_tip"),
        ),
    ),
    "q3_high_card_od": Spec(
        columns=("pu_location_id", "do_location_id", "fare_amount"),
        keys=("pu_location_id", "do_location_id"),
        sums=("fare_amount",),
        project=lambda df: df.select(
            "pu_location_id",
            "do_location_id",
            "trips",
            pl.col("fare_amount_sum").alias("fare_sum"),
        ),
    ),
    "q4_moments_distinct": Spec(
        columns=(
            "vendor_id",
            "payment_type",
            "pickup_ts",
            "fare_amount",
            "pu_location_id",
        ),
        prep=_hour,
        keys=("vendor_id", "payment_type", "pickup_hour"),
        moments=("fare_amount",),
        hll="pu_location_id",
        project=_proj_q4,
    ),
    "q5_median_fare": None,  # not mergeable: see the module docstring
    "q6_hour_rollup": Spec(
        columns=("pickup_ts", "fare_amount"),
        prep=_hour_ts,
        keys=("pickup_hour_ts",),
        sums=("fare_amount",),
        project=lambda df: df.select(
            "pickup_hour_ts", "trips", pl.col("fare_amount_sum").alias("fare_sum")
        ),
    ),
    "q7_zone_join": Spec(
        columns=("pu_location_id", "total_amount"),
        prep=_zone_join,
        needs_zones=True,
        keys=("borough",),
        sums=("total_amount",),
        project=lambda df: df.select(
            "borough", "trips", pl.col("total_amount_sum").alias("total_sum")
        ),
    ),
    "q8_topk_window": Spec(
        columns=("pu_location_id",),
        prep=_zone_join,
        needs_zones=True,
        keys=("borough", "pu_location_id"),
        project=_rank_q8,
    ),
}


# ------------------------------------------------------------------ helpers


def _ipc(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_ipc(buf)
    return buf.getvalue()


def _from_ipc(blob: bytes) -> pl.DataFrame:
    return pl.read_ipc(io.BytesIO(blob))


def _bit_length(expr: pl.Expr) -> pl.Expr:
    """Exact bit length of a value below 2^52: the number of powers of two it reaches."""
    return pl.sum_horizontal(
        [(expr >= (1 << j)).cast(pl.Int8) for j in range(HLL_RANK_BITS)]
    )


def _hll_rows(df: pl.DataFrame, keys: list[str], col: str) -> pl.DataFrame:
    """Per (group, register) maximum rank for one partition."""
    h = pl.col(col).hash().cast(pl.UInt64)
    low = h & pl.lit((1 << HLL_RANK_BITS) - 1, dtype=pl.UInt64)
    rows = df.select(
        *keys,
        idx=(h // pl.lit(1 << HLL_RANK_BITS, dtype=pl.UInt64)).cast(pl.Int32),
        rk=(HLL_RANK_BITS - _bit_length(low) + 1).cast(pl.Int8),
    )
    return rows.group_by([*keys, "idx"]).agg(pl.col("rk").max())


def _hll_estimate(regs: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """HyperLogLog estimate per group from merged (group, register, max rank) rows."""
    stats = regs.group_by(keys).agg(
        z=pl.lit(0.5).pow(pl.col("rk").cast(pl.Float64)).sum(),
        present=pl.len(),
    )
    z = pl.col("z") + (HLL_M - pl.col("present")).cast(pl.Float64)
    raw = HLL_ALPHA * HLL_M * HLL_M / z
    zeros = HLL_M - pl.col("present")
    linear = HLL_M * (pl.lit(float(HLL_M)) / zeros.cast(pl.Float64)).log()
    est = pl.when((raw <= 2.5 * HLL_M) & (zeros > 0)).then(linear).otherwise(raw)
    return stats.select(*keys, pu_distinct=est.round(0).cast(pl.Int64))


def _merge_moments(df: pl.DataFrame, keys: list[str], col: str) -> pl.DataFrame:
    """Chan et al.'s parallel update over any number of partials per group."""
    n, mean, m2 = f"{col}__n", f"{col}__mean", f"{col}__m2"
    totals = df.group_by(keys).agg(
        _N=pl.col(n).sum(),
        _S=(pl.col(n) * pl.col(mean)).sum(),
    )
    joined = df.join(totals, on=keys, how="left", nulls_equal=True)
    tot_mean = (
        pl.when(pl.col("_N") > 0).then(pl.col("_S") / pl.col("_N")).otherwise(0.0)
    )
    joined = joined.with_columns(_mu=tot_mean)
    contrib = pl.col(m2) + pl.col(n) * (pl.col(mean) - pl.col("_mu")) ** 2
    return joined.group_by(keys).agg(
        pl.col(n).sum(),
        pl.col("_mu").first().alias(mean),
        contrib.sum().alias(m2),
    )


# ----------------------------------------------------------------- map stage


def map_task(
    path: str,
    query: str,
    buckets: int,
    zones_ipc: bytes | None,
) -> dict[str, object]:
    """Map one part file: read, filter, partial-aggregate, hash-partition into ``buckets``."""
    spec = SPECS[query]
    assert spec is not None
    t0 = time.perf_counter()
    df = pl.read_parquet(path, columns=list(spec.columns))
    rows_in = df.height
    zones = _from_ipc(zones_ipc) if zones_ipc is not None else None
    df = spec.prep(df, zones)
    keys = list(spec.keys) or ["_k"]
    if not spec.keys:
        df = df.with_columns(_k=pl.lit(0, dtype=pl.Int8))
    exprs: list[pl.Expr] = [pl.len().alias("trips")]
    exprs += [pl.col(c).sum().alias(f"{c}_sum") for c in spec.sums]
    for c in spec.moments:
        exprs += [
            pl.col(c).count().alias(f"{c}__n"),
            pl.col(c).mean().fill_null(0.0).alias(f"{c}__mean"),
            (pl.col(c).var(ddof=0) * pl.col(c).count())
            .fill_null(0.0)
            .alias(f"{c}__m2"),
        ]
    agg = df.group_by(keys).agg(exprs)
    hll = _hll_rows(df, keys, spec.hll) if spec.hll else None
    agg_parts, hll_parts = (
        _split(agg, keys, buckets),
        (_split(hll, keys, buckets) if hll is not None else [None] * buckets),
    )
    blobs = [
        (_ipc(a), _ipc(h) if h is not None else None)
        for a, h in zip(agg_parts, hll_parts, strict=True)
    ]
    shuffle_bytes = sum(len(a) + (len(h) if h else 0) for a, h in blobs)
    return {
        "blobs": blobs,
        "rows_in": rows_in,
        "groups": agg.height,
        "shuffle_bytes": shuffle_bytes,
        "map_s": time.perf_counter() - t0,
    }


def _split(df: pl.DataFrame, keys: list[str], buckets: int) -> list[pl.DataFrame]:
    if buckets == 1:
        return [df]
    bucket = pl.struct(keys).hash() % buckets
    tagged = df.with_columns(_bucket=bucket.cast(pl.Int32))
    return [
        tagged.filter(pl.col("_bucket") == b).drop("_bucket") for b in range(buckets)
    ]


# -------------------------------------------------------------- reduce stage


def reduce_task(
    query: str,
    agg_blobs: list[bytes],
    hll_blobs: list[bytes | None],
) -> dict[str, object]:
    """Merge every map partial that landed in one bucket."""
    spec = SPECS[query]
    assert spec is not None
    t0 = time.perf_counter()
    keys = list(spec.keys) or ["_k"]
    frames = [_from_ipc(b) for b in agg_blobs]
    merged_in = pl.concat(frames, how="vertical_relaxed") if frames else None
    if merged_in is None or merged_in.height == 0:
        return {
            "agg": None,
            "hll": None,
            "reduce_s": time.perf_counter() - t0,
            "groups": 0,
        }
    sums = [("trips", "trips")] + [(f"{c}_sum", f"{c}_sum") for c in spec.sums]
    counts_exprs = [pl.col(src).sum().alias(dst) for src, dst in sums]
    merged = merged_in.group_by(keys).agg(counts_exprs)
    for c in spec.moments:
        mom = _merge_moments(
            merged_in.select(*keys, f"{c}__n", f"{c}__mean", f"{c}__m2"), keys, c
        )
        merged = merged.join(mom, on=keys, how="left", nulls_equal=True)
    hll_out = None
    if spec.hll:
        regs = [_from_ipc(b) for b in hll_blobs if b is not None]
        if regs:
            hll_out = pl.concat(regs).group_by([*keys, "idx"]).agg(pl.col("rk").max())
    return {
        "agg": _ipc(merged),
        "hll": _ipc(hll_out) if hll_out is not None else None,
        "reduce_s": time.perf_counter() - t0,
        "groups": merged.height,
    }


# ------------------------------------------------------------------ driver


@dataclass
class RunStats:
    map_s: float = 0.0
    shuffle_bytes: int = 0
    rows_in: int = 0
    reduce_s: list[float] = field(default_factory=list)
    reduce_groups: list[int] = field(default_factory=list)

    @property
    def straggler_ratio(self) -> float | None:
        """Slowest reduce task over the mean reduce task. 1.0 means perfectly balanced."""
        busy = [t for t in self.reduce_s if t > 0]
        if not busy:
            return None
        return max(busy) / (sum(busy) / len(busy))

    @property
    def group_skew(self) -> float | None:
        """Largest bucket's group count over the mean bucket's group count."""
        nz = [g for g in self.reduce_groups if g > 0]
        if not nz:
            return None
        return max(nz) / (sum(nz) / len(nz))


class MrPipeline:
    """A pool of ``workers`` processes that runs the three stages for one query at a time."""

    def __init__(self, workers: int) -> None:
        self.workers = max(1, workers)
        # Always spawn: Linux's default fork copies a parent that already runs Polars/DuckDB
        # threads, and the children can deadlock. Windows only spawns, so this also keeps the
        # measured behaviour the same on every platform.
        self.pool = ProcessPoolExecutor(
            max_workers=self.workers, mp_context=multiprocessing.get_context("spawn")
        )
        self.last: RunStats = RunStats()

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)

    def run(
        self,
        query: str,
        parts: list[Path],
        zones: pl.DataFrame | None,
        buckets: int | None = None,
    ) -> pl.DataFrame:
        spec = SPECS[query]
        if spec is None:
            raise NotImplementedError(
                f"{query} needs exact per-group values; not mergeable"
            )
        if spec.needs_zones and zones is None:
            raise ValueError(f"{query} needs the zone lookup")
        buckets = buckets or self.workers
        zones_ipc = _ipc(zones) if zones is not None else None
        stats = RunStats()
        t_map = time.perf_counter()
        futures = [
            self.pool.submit(map_task, str(p), query, buckets, zones_ipc) for p in parts
        ]
        per_bucket_agg: list[list[bytes]] = [[] for _ in range(buckets)]
        per_bucket_hll: list[list[bytes | None]] = [[] for _ in range(buckets)]
        for fut in futures:
            res = fut.result()
            stats.shuffle_bytes += res["shuffle_bytes"]
            stats.rows_in += res["rows_in"]
            for b, (agg_blob, hll_blob) in enumerate(res["blobs"]):
                if agg_blob and _nonempty(agg_blob):
                    per_bucket_agg[b].append(agg_blob)
                    per_bucket_hll[b].append(hll_blob)
        red = [
            self.pool.submit(reduce_task, query, per_bucket_agg[b], per_bucket_hll[b])
            for b in range(buckets)
        ]
        outs = []
        for fut in red:
            res = fut.result()
            stats.reduce_s.append(res["reduce_s"])
            stats.reduce_groups.append(res["groups"])
            if res["agg"] is not None:
                outs.append(
                    (
                        _from_ipc(res["agg"]),
                        _from_ipc(res["hll"]) if res["hll"] else None,
                    )
                )
        # Wall-clock for the whole map stage; summing per-task times would count parallel work twice.
        stats.map_s = time.perf_counter() - t_map
        self.last = stats
        return self._finish(spec, outs)

    def _finish(
        self, spec: Spec, outs: list[tuple[pl.DataFrame, pl.DataFrame | None]]
    ) -> pl.DataFrame:
        keys = list(spec.keys) or ["_k"]
        if not outs:
            return pl.DataFrame()
        agg = pl.concat([a for a, _ in outs], how="vertical_relaxed")
        if spec.hll:
            regs = [h for _, h in outs if h is not None]
            est = _hll_estimate(pl.concat(regs), keys)
            agg = agg.join(est, on=keys, how="left", nulls_equal=True)
        if not spec.keys:
            agg = agg.drop("_k")
        return spec.project(agg)


def _nonempty(blob: bytes) -> bool:
    return len(blob) > 0


def partial_bytes(df: pl.DataFrame) -> int:
    """Size of a frame as shuffled (Arrow IPC), for the overhead report."""
    return len(_ipc(df))
