"""PySpark adapter: the shared SQL from ``workloads.py`` on a local Spark session (``local[*]``).

Spark runs the SQL as written; the only difference is the distinct count, which uses Spark's
HyperLogLog++ ``approx_count_distinct(col, 0.01)``. Timing includes Spark's plan and codegen work,
so the figure is the engine's own cost for the query, not the JVM start-up (the harness times
``start`` separately).
"""

from __future__ import annotations

import pyarrow as pa

from dist_pipeline.engines.base import Dataset
from dist_pipeline.workloads import Query, render_sql

LABEL = "PySpark (Spark SQL, local[*])"
APPROX_DISTINCT = "approx_count_distinct(pu_location_id, 0.01)"


def start(ds: Dataset, opts: dict[str, object]):
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("dist-pipeline")
        .config("spark.driver.memory", str(opts.get("driver_memory", "6g")))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(opts.get("shuffle_partitions", 8)))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    if ds.fmt == "csv":
        trips = spark.read.option("header", "true").csv(ds.source, inferSchema=True)
    else:
        trips = spark.read.parquet(ds.location.as_posix())
    trips.createOrReplaceTempView("trips")
    zones = spark.read.option("header", "true").csv(ds.zones_csv.as_posix())
    zones.createOrReplaceTempView("zones_raw")
    spark.sql(
        "CREATE OR REPLACE TEMP VIEW zones AS SELECT CAST(LocationID AS INT) AS location_id, "
        "Borough AS borough FROM zones_raw"
    )
    return spark


def run(spark, query: Query, ds: Dataset, opts: dict[str, object]) -> pa.Table:
    sql = render_sql(query, trips="trips", zones="zones", distinct=APPROX_DISTINCT)
    return spark.sql(sql).toArrow()


def stop(spark) -> None:
    spark.catalog.clearCache()
    spark.stop()
