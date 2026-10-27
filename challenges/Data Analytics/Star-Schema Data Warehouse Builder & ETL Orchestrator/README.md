# Star-Schema Data Warehouse Builder & ETL Orchestrator

**Category:** Data Analytics
**Difficulty:** A

**Status:** Implemented (Python)

A star-schema warehouse built in DuckDB from the Online Retail II transaction feed, loaded by a small
task-DAG orchestrator with retries, idempotent re-runs, backfills, late-arrival restatement, crash
recovery and a cron scheduler. Every loaded result is checked against an independent polars
computation, and every failed data-quality gate stops the DAG.

## Data

- **Source:** UCI Machine Learning Repository, *Online Retail II* (a UK-based online gift retailer,
  invoices from 2009-12-01 to 2011-12-09). Licence **CC BY 4.0**; attribution is given here and in
  `sample_data/SOURCES.json`.
- **Fetch:** `scripts/fetch_data.py` downloads the archive, checks its SHA-256, and reads both sheets.
  The two sheets overlap, so 22,523 overlap rows are removed by invoice. The result is 1,044,848
  lines, written to `data/feed.parquet`. `data/` and `.cache/` are gitignored; the full feed is not
  committed.
- **Committed sample:** `sample_data/feed_sample.parquet`, 87,862 lines for 600 customers (seed
  20091201). Tests and the README commands run on it when the full feed is absent.
- **Simulated late arrivals:** the source has no arrival times, so the feed assigns each line an
  `arrival_date`. About 2% of invoices (a hash of the invoice number mod 100 < 2) arrive 30 to 119
  days after the invoice date, giving 19,710 late lines in the full feed.
- **Country to region:** `sample_data/ref_country_region.csv` is a hand-written lookup, copied from
  the Data Lineage Tracker challenge in this repository. It is not from an external source.

## Method

**Layers.** `raw.feed_lines` holds what the feed delivered. `staging.lines` holds cleaned lines with
an `exclusion` reason for rejected lines. `dw` is the star schema. `ops` holds run and task logs and
schedule state.

**Star schema (Kimball).**

- `dw.fact_sales` has one row per kept invoice line, with a surrogate key per line.
- `dw.fact_returns` holds the negative-quantity lines as a separate fact.
- `dw.dim_date` is shared by two date roles, `invoice_date_key` and `arrival_date_key`.
- `dw.dim_customer` and `dw.dim_product` are SCD type 2 with `valid_from`, `valid_to`, `is_current`
  and `version_no`. A customer's version tracks the country of the day's most frequent country.
  A product's version tracks its description and its list price in the month's price book.
- `dw.dim_country` and `dw.dim_line_flags` (a junk dimension for cancellation, guest and negative
  quantity flags) are small dimensions.
- Unknown customers (no customer id) use customer key 0.
- `dw.agg_country_month` is an aggregate rebuilt for every month the load touches.

**Orchestration.** A DAG of 13 tasks runs in deterministic topological order (Kahn's algorithm, first
declared first). A transient failure is retried with exponential backoff (`backoff * 2**(n-1)`). A
`PermanentError`, such as a failed data-quality gate, is not retried. A failed task skips only its
own downstream tasks, and the rest of the DAG still runs.

**Data-quality gates** run between stages: landed rows against the feed, staged rows against the
feed, unique line keys, the dimension and fact reconciliations, and the aggregate totals. A failed
gate fails the run.

**Data intervals.** A run loads the rows whose `arrival_date` falls in `[start, end)`. A calendar
month is the default. A cron fire at `F` covers the days from the previous fire up to `F`'s date.
Schedules that fire more than once a day are rejected.

**Idempotence and recovery.** Each task is one DuckDB transaction, so it commits all of its deletes
and inserts or none of them. Facts are deleted and re-inserted for the affected invoices. A run that
is killed leaves its `ops.run` row at `running`; the next run marks it `interrupted`. The uncommitted
transaction is discarded by DuckDB, so no duplicates remain. A content hash over the data tables
(excluding `ops`) shows that a re-run, and a monthly load compared with a full rebuild, produce
identical data.

**Late data.** A late line restates the older fact it belongs to. It also rebuilds the SCD2 history
of the customers and products it touches, and the aggregate for its own invoice month.

**Scheduler.** `warehouse tick` runs every due fire of a five-field cron expression that has not run
yet. The stored position advances only after a successful run, so a failed fire is retried by the
next tick. Call it from cron or Windows Task Scheduler.

**CLI.** The CLI uses Typer rather than argparse. The commands are typed functions whose options and
help come from their signatures, and `typer.testing.CliRunner` makes them testable in-process.

## Results

Measured by `scripts/benchmark.py` on the full feed (AMD64 Windows, Python 3.14.7, DuckDB 2.0.0,
polars 2.1.0). The full output is in `results/benchmark.md` and `results/benchmark.json`.

- **Load throughput:** the full window (1,044,848 lines) loads in 2.38 s, about 439,000 lines per second.
- **Incremental vs full rebuild:** 29 monthly runs take 27.6 s in total, with a median of 0.91 s per
  month. The last month on top of the loaded history takes 0.45 s. Monthly loads and a single full
  rebuild end with the same content hash. Rerunning a month without new data leaves the hash unchanged.
- **Star vs flat denormalised table** (median of 3 runs after a warm-up; the flat table holds the same
  kept lines with region, date and customer id copied in):

| Query                                     | Star ms | Flat ms | Faster |
| ----------------------------------------- | ------- | ------- | ------ |
| monthly sales by country                  | 15.2    | 8.8     | flat   |
| sales by stock code                       | 21.3    | 17.7    | flat   |
| distinct customers per region per quarter | 26.7    | 14.7    | flat   |
| total sales, no grouping                  | 1.1     | 1.5     | star   |
| one country's sale lines                  | 3.9     | 2.4     | flat   |
| returns by month                          | 1.8     | 2.7     | star   |

All six queries return the same answer on both layouts. The flat table wins on most grouped queries
here, because DuckDB joins small dimensions cheaply and the flat table avoids those joins. The star
wins on the single-column total and on the small returns fact. The flat table is also not
point-in-time: it cannot answer "which country was this customer in on that date" the way SCD2 can.

## Data findings

Counted from the full warehouse build:

- **Exclusions:** 11,931 of 1,044,848 lines are excluded. 5,993 have a stock code that is not a
  product code, and 5,938 fail the price rule. The remaining 1,032,917 lines are kept, and they
  reconcile exactly with 1,014,944 sale facts and 17,973 return facts.
- **Cancellations:** 19,165 lines are cancellations (invoices starting with C). Returns total
  -719,656.34 against sales of 19,699,768.84.
- **Missing customers:** 235,287 lines have no customer id. The 226,757 kept sale facts that belong to
  them use the unknown-customer key, so customer analysis covers a subset of revenue.
- **Customer churn in SCD2:** 5,875 customers produce 5,906 versions. Twelve customers have a second
  version because their most frequent country changed.
- **Price changes:** 4,891 stock codes produce 18,497 product versions, and 3,243 of them have more
  than one version. A product version changes when its description or its monthly list price changes.
- **Late data:** 19,710 lines arrived 30 or more days after their invoice date.

## Where this is actually used

Airflow is the closest production analogue of the orchestrator. Its documentation describes a DAG as
a graph of tasks with dependencies that also dictates how many times to retry them. Each run covers a
defined data interval, and the task context exposes its start (`data_interval_start`). The
orchestrator here follows the same ideas: a topological order, per-task retries, and runs keyed by an
interval.

dbt snapshots implement the same kind of slowly changing dimension as this warehouse's SCD2 tables.
The dbt documentation says snapshots "implement type-2 Slowly Changing Dimensions over mutable source
tables". Each row carries a `dbt_valid_from` timestamp and a `dbt_valid_to` timestamp, which is null
for the current version. This is the validity-window pattern used by `dw.dim_customer` and
`dw.dim_product`.

DuckDB is an in-process analytical SQL database with no server to run. It is the storage and query
engine here. That makes the whole warehouse a single file, and it makes the benchmark's comparison of
star and flat layouts a comparison inside one process.

## Run it

Run every command from this challenge directory.

```bash
uv sync --locked --group dev                 # install dependencies
uv run --locked pytest -q                    # tests (about 35 s)
uv run python scripts/fetch_data.py          # optional: download the full feed to data/
uv run warehouse run 2010-03                 # one calendar month (the sample feed if data/ is absent)
uv run warehouse run 2009-12-01/2010-03-01   # an explicit [start, end) interval
uv run warehouse backfill 2010-01 2011-01    # every month from the first up to, not including, the last
uv run warehouse tick "0 2 * * *"            # run the due fires of a daily schedule
uv run warehouse status                      # the latest runs
uv run warehouse report                      # writes out/report.html, a self-contained run report
uv run python scripts/benchmark.py           # regenerates results/benchmark.json and .md
```

The default warehouse is `out/warehouse.duckdb`. Pass `--db` to use another file, and `--feed` to use
another feed.

## Layout

```text
src/star_warehouse/
    paths.py       project root
    dag.py         task DAG: order, retries, skips, events
    cron.py        five-field cron parsing and fire sequences
    warehouse.py   DDL, transactions, content hash, interval parsing
    feed.py        the simulated feed: line keys and late arrivals
    dq.py          data-quality expectations
    tasks.py       the 13 SQL tasks and the DAG they form
    pipeline.py    runs, backfills, the scheduler tick
    report.py      self-contained HTML report with an inline SVG graph
    cli.py         Typer commands
scripts/
    fetch_data.py  download, verify and read the source workbook
    benchmark.py   measurements written to results/
sample_data/       committed sample feed, country lookup, sources
results/           benchmark.json and benchmark.md
tests/             star vs polars oracle, pipeline, DAG, cron, feed, report and CLI tests
```

## Files

- `README.md`: this file.
- `pyproject.toml`, `uv.lock`: the project and its locked dependencies.
- `pytest.ini`: test configuration.

## Sources

- UCI Machine Learning Repository, Online Retail II, CC BY 4.0.
- Apache Airflow documentation, DAGs: https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html
- dbt documentation, snapshots: https://docs.getdbt.com/docs/build/snapshots
- DuckDB, in-process analytical database (checked through Context7, library `/duckdb/duckdb-web`).
