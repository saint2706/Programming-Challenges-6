# Distributed Data Processing Pipeline

**Category:** Data Analytics
**Difficulty:** Expert
**Status:** Implemented (Python)

Nine aggregation queries over NYC TLC yellow taxi trips (2019) run on Polars (streaming, in-memory and single-thread), DuckDB, Dask, Daft, PySpark (`local[*]`) and an own map-shuffle-reduce pipeline (`mrpipe`). Each engine-query runs in a fresh subprocess and records wall time and peak process-tree RSS. Every result is checked against a DuckDB exact oracle.

## Data

| Set    | Source                                                       | Rows       | Notes                                                                                      |
| ------ | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------------------------------------ |
| 1m     | March 2019 TLC yellow Parquet, 8 part files of about 1M rows | 7,866,571  | Normalised to 11 columns                                                                   |
| 12m    | All twelve 2019 months, 90 part files                        | 84,597,002 | Same normalisation                                                                         |
| CSV 1m | Our own CSV export of the 1m set (681 MB)                    | 7,866,571  | TLC does not serve CSV for these months (HTTP 403), so this is our export, not the TLC CSV |
| Zones  | `taxi_zone_lookup.csv`                                       | 265 zones  | Used by the join query                                                                     |

The brief said about 3M rows per month. The real March 2019 file has 7.9M rows, so the brief's figure was wrong; the sets above use the real counts.

Data lives under `data/` (gitignored, about 3.2 GB on disk: raw 1.2 GB, normalised 1.4 GB, CSV 650 MB, runs 6 MB). Only the 20,000-row test fixture under `tests/fixtures/` is committed.

## Method

- Queries q0 to q8 (`src/dist_pipeline/workloads.py`) cover a scan sum, a filter count, a skewed group, a high-cardinality origin-destination group, mean, sample variance and approximate distinct count (q4), a median (q5), an hourly rollup, a zone join and a top-3-per-borough window.
- q5 (median) is not mergeable from partial results, so mrpipe reports it as not supported by design.
- Each engine-query runs in its own `python -m dist_pipeline.worker` process. The parent samples the process tree's RSS every 50 ms and kills any leftover children. Timings are the median of 3 repeats after 1 warmup run. With 3 repeats the IQR is not available, so the results show min to max.
- Verification: each result is compared with the DuckDB exact oracle. Floats must agree to a relative or absolute 1e-9. The approximate distinct count must be within max(bound times exact, 5.0), where the bound is 0.05.
- The own pipeline (`mrpipe`) maps each part file to hash buckets of partial aggregates in Arrow IPC, shuffles them, and reduces each bucket. Distinct counts use HyperLogLog (p=12). Variance is merged with the pairwise update of Chan et al.
- Dask uses a LocalCluster of processes with 2 threads per worker. Its exact `nunique` runs as a separate groupby. Daft uses its DataFrame API. Spark runs Spark SQL in `local[*]` with its UI disabled.

## Machine and noise

Windows 11, Intel64 family 6 model 197, 16 physical and 16 logical cores, 31.4 GB RAM, Python 3.14.7. One laptop, one run of the matrix, run with nothing else heavy running. The Spark legs were run afterwards, separately, on the same quiet machine with the same 3 repeats and 1 warmup (`scripts/run_spark_leg.py`), once the Hadoop native files were installed; the other engines come from the first run. The min to max spread in `results/summary.md` is the noise estimate.

## Results (12m, median seconds, peak RSS in MB)

| Engine               | q0 scan sum  | q3 high-card group | q4 moments, distinct | q5 median            | q8 top-3 window |
| -------------------- | ------------ | ------------------ | -------------------- | -------------------- | --------------- |
| DuckDB               | 0.113 (257)  | 0.292 (391)        | 0.407 (607)          | 1.211 (1483)         | 0.158 (194)     |
| Polars streaming     | 0.078        | 0.222              | 0.348 (fails check)  | 1.754 (7598)         | 0.155           |
| Polars in-memory     | 0.092        | 1.262              | 3.114 (9362)         |                      |                 |
| Polars single thread | 0.711        |                    | 7.61                 |                      |                 |
| Daft                 | 0.230 (147)  | 0.524 (386)        | 2.617 (2746)         | 1.377 (2383)         | 0.802 (523)     |
| Dask                 | 0.596 (3590) | 4.005 (6267)       | 7.152 (6739)         | error (KilledWorker) | 2.999 (6149)    |
| mrpipe               | 0.199 (1948) | 0.971 (7547)       | 7.350 (13697)        | not supported        | 1.663 (6855)    |
| Spark                | 0.369 (2598) | 1.962 (2972)       | 3.653 (3542)         | 1.658 (2590)         | 1.198 (2254)    |

Polars streaming q4 fails the approximate-distinct check at 1m (CSV too) and at 12m: its distinct count is off by 14% at 1m and by 20% at 12m, beyond the bound. DuckDB, Daft and Spark pass (Spark's distinct count is off by 1.2% at 12m, inside the 5% bound). The recorded `Verified` column in `results/summary.md` is the source of truth.

Strong scaling of mrpipe on 12m, 1/2/4/8 workers: q3 3.04 / 1.51 / 1.04 / 0.90 s; q4 8.61 / 5.72 / 4.74 / 4.58 s. Gains flatten after 4 workers on this machine.

Weak scaling (2 parts per worker, 1/2/4 workers) is in `results/results.json` but was not verified against the oracle, so treat it as timing only.

## Where cluster engines lose and where they do not

On one laptop, the DuckDB and Polars streaming engines beat the distributed designs on almost every query. Dask and mrpipe lose most on q4, which needs a moment merge and a distinct merge over every partition, and Dask's exact `nunique` costs extra passes. Daft is competitive on q0 and q8. Spark verified on all 18 queries and never failed or ran out of memory, but it is slower than DuckDB on every 12m query (q0 0.369 s against 0.113 s, q4 3.653 s against 0.407 s, q8 1.198 s against 0.158 s); it is faster than Dask on q3, q4 and q8, faster than mrpipe on q4 and q8 (but slower on q0 and q3), and its q4 peak memory is about half of Dask's and a quarter of mrpipe's. Dask's 12m q5 median run failed with KilledWorker on 4 workers; the cause was not investigated.

## Where this is actually used

Map-then-merge aggregation, where each partition produces a partial result and a reduce step combines them, is the model described in the MapReduce paper (Dean and Ghemawat, OSDI 2004). Spark, Dask and Daft partition their work in the same spirit, across cores here and across machines in a cluster.

Approximate distinct counts are how these engines avoid shipping every distinct value between partitions, and each documents its own error budget:

- Spark's `approx_count_distinct` takes a maximum relative standard deviation, default 0.05.
- Trino's `approx_distinct` documents a default standard error of 2.3% and accepts a maximum standard error between 0.0040625 and 0.26.
- DuckDB's `approx_count_distinct` is documented as using HyperLogLog.
- Redis exposes HyperLogLog directly as `PFADD`, `PFCOUNT` and `PFMERGE`, at up to 12 KB per key and a standard error of 0.81%. Its documentation suggests counting unique visitors to a page, and `PFMERGE` combines sketches, which is the property that makes the partial-aggregate design work.

Polars documents `collect(engine="streaming")` for processing data in batches, so datasets larger than memory can be aggregated on one machine.

## Run

Run from this challenge folder, not the repo root. The core install runs the tests without the heavy engines.

```bash
uv run --locked pytest -q                       # tests (core only; engine tests skip when not installed)
uv sync --locked --extra engines                # Dask, pandas, PySpark, Daft
uv run dist-pipeline download                   # TLC Parquet into data/raw
uv run dist-pipeline prepare --scale 1m         # normalise into data/normalized
uv run dist-pipeline verify --scale 1m          # check engines against the DuckDB oracle
uv run dist-pipeline run --scale 1m             # timed matrix into results/results.json
uv run dist-pipeline report                     # results/summary.md and results/report.html
```

## Layout

```text
src/dist_pipeline/
  bench.py       fresh-subprocess harness, process-tree RSS sampling, scaling runs
  cli.py         typer commands: download, prepare, verify, run, report
  download.py    TLC Parquet download
  prepare.py     normalise to 11 columns, build the 1m and 12m sets and the CSV export
  workloads.py   queries q0 to q8, DuckDB SQL dialect
  verify.py      oracle comparison and tolerances
  mrpipe.py      own map-shuffle-reduce engine
  worker.py      one engine-query per subprocess
  engines/       adapters for Polars, DuckDB, Dask, Spark, Daft
  report.py      summary.md and report.html (escaped HTML, no external resources)
  stats.py       median, IQR (when n is at least 4), min, max
tests/           pytest; mini fixture of 20,000 real rows, plus real-data checks when data/ exists
scripts/make_fixtures.py   rebuilds the test fixture
results/         results.json, summary.md, report.html
```

## Files

- `results/results.json`: every run record, machine, matrix and scaling data
- `results/summary.md`: tables for the matrix, scaling and verification
- `results/report.html`: the same content with SVG charts, no external resources

## Spark

PySpark 4.2.0 bundles Hadoop 3.5.0 client jars. On Windows, reading Parquet also needs `winutils.exe` and `hadoop.dll` with `HADOOP_HOME` pointing at a folder that holds them in `bin\`. Without them the first read fails with `Py4JJavaError ... java.lang.UnsatisfiedLinkError: 'boolean org.apache.hadoop...NativeIO$Windows.access0...'` and the Spark legs are recorded as skipped. That is how the first matrix run recorded Spark.

The files used for the results above came from the community build repository `cdarlint/winutils` (not an Apache release), pinned to commit `7386986d5d8a079b5cd4464f4599766dd27e7d13`, folder `hadoop-3.3.6/bin`:

| File           | sha256                                                             |
| -------------- | ------------------------------------------------------------------ |
| `winutils.exe` | `496a591eb1e67df2a620f710d529ba6ddfe1c19149e6647cc4e320bb0efd8553` |
| `hadoop.dll`   | `d7ab36a68518748cef142be2da5069b4c763c2cd764c1d2e6ac48c7200405be3` |

They were put outside the repository and `HADOOP_HOME` was set only for the process that ran the Spark legs, not globally. The build is for Hadoop 3.3.6 while PySpark bundles 3.5.0 and no 3.5.0 build exists in that repository; local Parquet reads verified against the oracle on all 18 queries, but a different query shape could still hit an interface difference. The tests skip the Spark case unless `HADOOP_HOME` is set.

```bash
HADOOP_HOME=/path/to/hadoop uv run --locked python scripts/run_spark_leg.py
```

## Limits

- One laptop, not a cluster. Scaling results show what this machine's 16 cores do, not what a cluster would do.
- Spark was run with a Hadoop 3.3.6 native build against PySpark's Hadoop 3.5.0 jars (see the Spark section), and its timings come from a separate run on the same quiet machine rather than the first matrix. Ray is not included.
- Polars streaming fails the approximate-distinct check on q4 at 1m, 12m and CSV.
- Dask's 12m q5 median run failed (KilledWorker), so there is no Dask q5 timing at 12m.
- mrpipe does not run q5 (median), because it cannot be merged from partial results.
- Dask uses exact `nunique`, so its q4 timing reflects the extra pass.
- Weak scaling is timing only, not verified.
- Per-phase timings (map, reduce, shuffle) are excluded from the results. An accounting bug summed per-task map times across workers, and the fix is in `mrpipe.py` but the stored phase values in `results.json` were recorded before it.
- The CSV leg uses our own export of the 1m set, not the TLC CSV files.
- Three repeats are too few for an IQR, so the spread shown is min to max.

## Sources

- NYC TLC trip record data (yellow taxi 2019 Parquet): `https://d37ci6vzurychx.cloudfront.net` (base URL used by `src/dist_pipeline/download.py`)
- NYC TLC taxi zone lookup table, from the same TLC data page
- Dean, J. and Ghemawat, S., "MapReduce: Simplified Data Processing on Large Clusters", OSDI 2004
- Flajolet, P., Fusy, E., Gandouet, O., Meunier, F., "HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm", AOFA 2007
- Chan, T. F., Golub, G. H., LeVeque, R. J., "Updating formulae and a pairwise algorithm for computing sample variances", 1979
- Polars documentation (streaming engine): `https://docs.pola.rs/`
- DuckDB documentation (`approx_count_distinct`): `https://duckdb.org/docs/`
- Spark documentation (`approx_count_distinct`, default rsd 0.05): `https://spark.apache.org/docs/latest/`
- Dask documentation (DataFrame `nunique`): `https://docs.dask.org/`
- Daft documentation: `https://docs.daft.ai/`
