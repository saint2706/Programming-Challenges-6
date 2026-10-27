"""Dask adapter: ``dask.dataframe`` (pandas API) on a ``LocalCluster`` of worker processes.

Each query is the pandas-style equivalent of the shared SQL. Group-bys go through Dask's shuffle
(``split_out=True``), so the cost of the exchange is part of the timing. Distinct counts are exact
(``nunique``): Dask has no approximate distinct aggregate in its group-by, which the report notes.
The zone lookup is small and is broadcast to every partition with ``broadcast=True``.
"""

from __future__ import annotations

import os

import pandas as pd
import pyarrow as pa

from dist_pipeline.engines.base import Dataset
from dist_pipeline.workloads import Query

LABEL = "Dask (DataFrame, LocalCluster)"


class DaskSession:
    def __init__(self, workers: int, threads: int) -> None:
        from dask.distributed import Client, LocalCluster

        self.cluster = LocalCluster(
            n_workers=workers,
            threads_per_worker=threads,
            processes=True,
            dashboard_address=None,
            silence_logs=40,
        )
        self.client = Client(self.cluster)
        self.workers = workers

    def close(self) -> None:
        self.client.close()
        self.cluster.close()


def start(ds: Dataset, opts: dict[str, object]) -> DaskSession:
    threads = int(opts.get("threads") or 2)
    workers = int(opts.get("workers") or max(1, (os.cpu_count() or 2) // threads))
    return DaskSession(workers, threads)


def _read(ds: Dataset):
    import dask.dataframe as dd

    if ds.fmt == "csv":
        return dd.read_csv(ds.source, parse_dates=["pickup_ts", "dropoff_ts"])
    return dd.read_parquet(ds.location.as_posix())


def _grouped(df, keys: list[str], aggs: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """One shuffle group-by: ``aggs`` maps output name to (column, Dask aggregation).

    ``nunique`` is not a string aggregation in Dask's ``agg``, so it runs as its own group-by.
    ``dropna=False`` keeps the NULL group, as SQL does.
    """
    spec: dict[str, list[str]] = {}
    for col, fn in aggs.values():
        if fn != "nunique":
            spec.setdefault(col, [])
            if fn not in spec[col]:
                spec[col].append(fn)
    gb = df.groupby(keys, dropna=False)
    res = gb.agg(spec, split_out=True).compute()
    out = res.index.to_frame(index=False)
    for name, (col, fn) in aggs.items():
        if fn == "nunique":
            nu = gb[col].nunique(split_out=True).compute()
            out[name] = nu.reindex(res.index).to_numpy()
        else:
            out[name] = res[(col, fn)].to_numpy()
    return out


def _zones(ds: Dataset) -> pd.DataFrame:
    # "N/A" is a real borough name in the lookup; pandas' default NA list would turn it into NULL.
    z = pd.read_csv(
        ds.zones_csv,
        usecols=["LocationID", "Borough"],
        keep_default_na=False,
        na_values=[""],
    )
    z = z.rename(columns={"LocationID": "location_id", "Borough": "borough"})
    return z.astype({"location_id": "int32"})


def run(
    session: DaskSession, query: Query, ds: Dataset, opts: dict[str, object]
) -> pa.Table:
    import dask
    import dask.dataframe as dd

    df = _read(ds)
    name = query.name
    if name == "q0_scan_sum":
        s, n = dask.compute(df.total_amount.sum(), df.total_amount.size)
        out = pd.DataFrame({"total_sum": [float(s)], "trips": [int(n)]})
    elif name == "q1_filter_count":
        f = df[(df.trip_distance > 5) & (df.fare_amount > 20)]
        out = pd.DataFrame({"trips": [int(f.total_amount.size.compute())]})
    elif name == "q2_skewed_payment":
        out = _grouped(
            df,
            ["payment_type"],
            {
                "trips": ("total_amount", "size"),
                "total_sum": ("total_amount", "sum"),
                "avg_tip": ("tip_amount", "mean"),
            },
        )
    elif name == "q3_high_card_od":
        out = _grouped(
            df,
            ["pu_location_id", "do_location_id"],
            {
                "trips": ("fare_amount", "size"),
                "fare_sum": ("fare_amount", "sum"),
            },
        )
    elif name == "q4_moments_distinct":
        df = df.assign(pickup_hour=df.pickup_ts.dt.hour)
        out = _grouped(
            df,
            ["vendor_id", "payment_type", "pickup_hour"],
            {
                "trips": ("fare_amount", "size"),
                "fare_mean": ("fare_amount", "mean"),
                "fare_var": ("fare_amount", "var"),
                "pu_distinct": ("pu_location_id", "nunique"),
            },
        )
    elif name == "q5_median_fare":
        out = _grouped(
            df,
            ["vendor_id", "payment_type"],
            {"fare_median": ("fare_amount", "median")},
        )
    elif name == "q6_hour_rollup":
        df = df.assign(pickup_hour_ts=df.pickup_ts.dt.floor("h"))
        out = _grouped(
            df,
            ["pickup_hour_ts"],
            {"trips": ("fare_amount", "size"), "fare_sum": ("fare_amount", "sum")},
        )
    elif name in ("q7_zone_join", "q8_topk_window"):
        z = dd.from_pandas(_zones(ds), npartitions=1)
        joined = df.merge(
            z,
            left_on="pu_location_id",
            right_on="location_id",
            how="inner",
            broadcast=True,
        )
        if name == "q7_zone_join":
            out = _grouped(
                joined,
                ["borough"],
                {
                    "trips": ("total_amount", "size"),
                    "total_sum": ("total_amount", "sum"),
                },
            )
        else:
            agg = _grouped(
                joined,
                ["borough", "pu_location_id"],
                {"trips": ("total_amount", "size")},
            )
            agg = agg.sort_values(
                ["borough", "trips", "pu_location_id"],
                ascending=[True, False, True],
            )
            agg["rk"] = agg.groupby("borough", dropna=False).cumcount() + 1
            out = agg[agg["rk"] <= 3][["borough", "pu_location_id", "trips", "rk"]]
    else:
        raise KeyError(name)
    return pa.Table.from_pandas(out, preserve_index=False)


def stop(session: DaskSession) -> None:
    session.close()
