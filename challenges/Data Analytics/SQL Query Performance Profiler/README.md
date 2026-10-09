# SQL Query Performance Profiler

**Category:** Data Analytics
**Difficulty:** I

**Status:** Implemented (Python)

Source modules live in `src/sql_profiler/`; the tests are in `tests/`.

Run a query on a real retail database, see its **actual** execution plan (measured row counts and
timings, not just the optimizer's guess), and get plain-English findings: what is slow, what the
optimizer got wrong, and what to try. Then test the fix: a rewrite is only compared if it returns
the same rows, and "faster" is only claimed when the runs don't overlap. It targets DuckDB 1.5 (the
profiler's JSON plan format), on the UCI *Online Retail II* data normalized into four tables:
`invoice_lines` (1,044,848 rows), `invoices` (53,628), `products` (4,950), `customers` (5,942).

```bash
uv run streamlit run src/sql_profiler/app.py            # profile, compare, experiments
uv run sqlprof explain "SELECT ..." --html report.html   # plan + findings, optional HTML report
uv run sqlprof compare "SELECT ..." "SELECT ..."         # same rows? faster? plan differences?
uv run sqlprof lab --markdown                            # the built-in rewrite experiments
uv run sqlprof index-lab                                 # before/after CREATE INDEX
uv run sqlprof schema
uv run pytest -q
```

The committed `sample_data/` (600 customers plus 300 guest invoices) makes everything work out of the
box. For the full tables: `uv run --group fetch python -m sql_profiler.fetch_data` (about 45 MB, written
to `data/`, which is not committed). Every number below is from the full tables.

## What it shows

`explain` prints and draws the plan the way plans are read, result at the top and tables at the
bottom: box colour is the operator's share of the work, line thickness is how many rows flow along it,
each box shows rows against the optimizer's estimate, and a box is outlined when a finding points at
it. The HTML report is one self-contained file (no scripts, light and dark themes) with the SQL, the
findings, the drawing and an operator table.

| Finding                                                                       | Fires when (all thresholds are in `rules.Thresholds`)                                                                  |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `estimate.off`                                                                | Actual rows are 10x (warn) or 100x (critical) off the estimate, at 1,000 rows or more.                                 |
| `filter.function_on_column`                                                   | A scan filter calls a function on the data (`strftime(col, ...)`, `substr(col, ...)`) and reads 10,000+ rows.          |
| `filter.late`                                                                 | A FILTER above its input drops 90%+ of 10,000+ rows: the predicate could not move down to the scan.                    |
| `scan.reads_much_keeps_little`                                                | A filtered scan reads 100,000+ rows and keeps 1% or fewer.                                                             |
| `join.no_equality_key`                                                        | A cross product, range or nested-loop join compares 1M+ row pairs (critical at 100M).                                  |
| `join.big_build_side`                                                         | A hash join builds its hash table on the larger input (2x, 100,000+ rows).                                             |
| `join.row_explosion`                                                          | A join outputs 10x more rows than its larger input.                                                                    |
| `sort.full`                                                                   | An ORDER BY sorts 100,000+ rows (no LIMIT turned it into a top-N).                                                     |
| `scan.repeated`                                                               | The same table is scanned twice or more.                                                                               |
| `aggregate.barely_reduces`, `subquery.decorrelated`, `time.dominant_operator` | Information: nearly every row is its own group; a correlated subquery became a join; one operator is 50%+ of the work. |

A rule stays quiet unless the measured data says it matters: a function call in a filter on a
5,000-row table is not a finding, and neither is a scan's estimate when the scan is filtered only by a
dynamic join filter, which no up-front estimate can know about.

## What the experiments say

`sqlprof lab` runs eight query/rewrite pairs on the full data: the rows must match (the one exception
is flagged), then each is timed 15 times after a warm-up and the plans are diffed. The headline is
that **DuckDB's optimizer already erases most of the textbook anti-patterns**, and the ones that
still matter are different from the folklore.

| Experiment                                     | A (ms) | B (ms) | A / B  | Verdict                      | What happened                                                                                                                                                                          |
| ---------------------------------------------- | ------ | ------ | ------ | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Self-join vs window function                   | 53.62  | 6.17   | 8.69x  | B faster                     | Counting each customer's earlier invoices with a join multiplies rows (flagged `join.row_explosion`, `scan.repeated`); a window function does one pass.                                |
| OR in a join condition vs UNION of equi-joins  | 24.73  | 3.49   | 7.08x  | B faster                     | `a.x = b.x OR a.y = b.y` cannot hash, so DuckDB falls back to a `BLOCKWISE_NL_JOIN` that compares every pair (`join.no_equality_key`).                                                 |
| Function on the key vs a range                 | 4.25   | 1.85   | 2.30x  | B faster                     | Same operators on the same table, but `substr(invoice, 1, 3) = '536'` reads all 1,044,848 rows and the range reads 624,824: min/max statistics skip row groups only for a bare column. |
| Sort everything vs `LIMIT 10`                  | 177.49 | 5.46   | 32.50x | B faster (not the same rows) | A full sort of a million rows (`sort.full`) vs a top-N heap. Not equivalent by design: the second returns ten rows.                                                                    |
| `IN (subquery)` vs `JOIN`                      | 5.31   | 5.24   | 1.01x  | no clear difference          | The plans differ in join type, the cost does not.                                                                                                                                      |
| Correlated subquery vs join to a pre-aggregate | 16.62  | 17.17  | 0.97x  | no clear difference          | DuckDB decorrelates the subquery into a delim join (reported as information); the explicit rewrite is no faster.                                                                       |
| Aggregate before the join vs after             | 15.63  | 15.01  | 1.04x  | no clear difference          | Shrinking 1M lines to 53k invoice totals first doesn't pay: the join already has a small build side.                                                                                   |
| `NOT IN` vs `NOT EXISTS`                       | 4.89   | 9.20   | 0.53x  | A faster                     | The folklore is reversed here: `NOT EXISTS` becomes a delim join and the optimizer's estimate for it is off by 100x or more (`estimate.off`, critical).                                |

**The equivalence check earned its keep.** Two of my own first drafts were wrong. A NULL-safe join
(`IS NOT DISTINCT FROM`) standing in for the correlated subquery returns different rows: guest
invoices have a NULL customer, the subquery gives them NULL, and the NULL-safe join gives them the
date of every other guest invoice (8,752 rows differ, and a test pins it). And a window frame over
`ROWS` in (date, invoice) order counted same-minute invoices as "earlier", where the join's strict
`<` does not (305 rows differ); the `RANGE ... INTERVAL 1 SECOND PRECEDING` frame in the experiment
matches exactly. Both would have shown a big speedup with no warning if only the timings were compared.

**Indexes help a little and sometimes hurt.** `sqlprof index-lab` times point lookups on
`invoice_lines` before and after `CREATE INDEX` (25 runs each):

| Lookup                  | Rows matched | Rows read before | Rows read after | Median before | Median after |
| ----------------------- | ------------ | ---------------- | --------------- | ------------- | ------------ |
| `invoice = '536365'`    | 7            | 514,698          | 7               | 1.93 ms       | 0.84 ms      |
| `stock_code = '21744'`  | 232          | 1,044,848        | 232             | 2.50 ms       | 1.62 ms      |
| `stock_code = '84828'`  | 362          | 1,044,848        | 362             | 2.21 ms       | 2.45 ms      |
| `stock_code = '85123A'` | 5,711        | 1,044,848        | 1,044,848       | 2.24 ms       | 2.20 ms      |

A 7-row lookup is 2.3x faster, though the table is stored in invoice order so min/max statistics had
already halved the scan. At 232 rows the gain is smaller. At 362 rows the index was used (rows read
fell from over a million to 362) and the lookup came out no faster, 2.21 to 2.45 ms: random access
cost what the scan saved. At about 5,700 rows the planner ignored the index and the plan did not change.
The scan is already fast enough that the index has little to save.

**The optimizer's range guess is a flat 20%.** `price > 50` on the 1,044,848-row table is estimated at
208,969 rows, exactly 20%, and 2,514 match (83x off). That is the usual source of `estimate.off`
findings, which appear in 6 of the 16 experiment plans, and the finding says so.

## Design notes

**Measured, not guessed.** The plan comes from `EXPLAIN (ANALYZE, FORMAT JSON)`, so every operator has
its real output rows, rows scanned, time, and the optimizer's estimate. Operator time is *CPU* time
summed over threads, so a parallel query's operators add up to more than wall clock; shares are
shares of that total, and the report shows wall clock and the parallel factor next to it. Profiling
has its own overhead: use it to find where, not to quote absolute speed.

**Two quirks of DuckDB's output that matter.** An `Estimated Cardinality` of `0` means "no estimate",
not "zero rows", so it is dropped. And `ORDER BY ... LIMIT n` plans a second scan to fetch the full rows
of the top n by row id; that scan carries an empty dynamic filter and returns a sliver, so it is
recognised as a fetch and not reported as a repeated scan. Both were false findings until the first
real plans showed them.

**Equivalence is exact.** Two queries match when `A EXCEPT ALL B` and `B EXCEPT ALL A` are both empty
(a multiset comparison, row order ignored, NULLs equal), after comparing row counts. Floating-point
columns compare exactly, so `ROUND` any computed sums in both queries.

**"Faster" needs every run to win.** With 7 to 15 runs, a ratio of medians reports noise as a speedup
whenever one run in a pair is slow. The verdict is "B faster" only when the slowest run of B beats the
fastest run of A, otherwise "no clear difference", and the app plots every run.

**Untrusted SQL is contained.** Only one `SELECT` is accepted (checked by DuckDB's own statement
parser, so `DROP`, `COPY`, `EXPLAIN` and multi-statement strings are refused, not string-matched), file access is
switched off after the tables load (`read_csv('C:/...')` and `COPY ... TO` fail), runaway queries
are interrupted after a timeout and the connection stays usable, and everything that comes from the
query (SQL text, table names, filter text) is HTML-escaped in the report. The app serialises
measurements behind a lock so two sessions can't distort each other's timings.

## Not done

- DuckDB only. Each engine's plan format is its own: the parser, the operator names and several
  rules are specific to DuckDB's profiler output, and other engines would need their own adapter.
- No plan-only mode (`EXPLAIN` without running): a query that takes minutes must be run to be profiled,
  and is stopped by the timeout.
- No cost model of its own and no index or rewrite *recommendations* beyond the text of each finding:
  it reports and checks, it doesn't generate rewrites.
- The experiments are on one dataset of about a million rows; they show what a rewrite does there,
  not a law. At 100x the data, differences the optimizer hides here (join order, build sides) may appear.

## Sources

- Chen, D. (2019). *Online Retail II*. UCI Machine Learning Repository,
  <https://archive.ics.uci.edu/dataset/502/online+retail+ii> (CC BY 4.0). Download SHA-256 in
  `sample_data/SOURCES.json`.
- DuckDB documentation: [profiling](https://duckdb.org/docs/stable/dev/profiling), `EXPLAIN ANALYZE`,
  and [indexes](https://duckdb.org/docs/stable/sql/indexes).
- Moerkotte, Neumann and Steidl (2009), *Preventing Bad Plans by Bounding the Impact of Cardinality
  Estimation Errors* (VLDB): the q-error, the factor by which an estimate is off, used by `estimate.off`.
- Leis et al. (2015), *How Good Are Query Optimizers, Really?* (VLDB): why wrong row estimates are the
  main cause of bad plans.
