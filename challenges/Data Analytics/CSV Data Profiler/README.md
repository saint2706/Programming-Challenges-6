# CSV Data Profiler

**Category:** Data Analytics
**Difficulty:** B

**Status:** Implemented (Python)

Point it at any CSV and it emits one self-contained HTML report: per-column
semantic type, null counts, distributions, and a correlation matrix across
numeric columns — no server, no external assets once the file is generated.

## What it does

- **Delimiter/encoding sniffing.** Tries `utf-8` then falls back to
  `latin-1` (which never fails to decode), and guesses the delimiter with
  `csv.Sniffer` over a sample of the file — comma, semicolon, tab, or pipe.
  `--sep` overrides the guess.
- **Semantic typing beyond the raw dtype.** Polars gives you a storage
  dtype; this adds one more layer — numeric/datetime pass straight
  through, a string column of `true`/`false`/`yes`/`no` becomes `boolean`,
  and everything else is `categorical` if fewer than half its values are
  unique, else `text` (free-form). That's the difference between a
  `department` column (a handful of repeated values) and a `notes` column
  (every value distinct).
- **Per-column stats + chart.** Numeric: min/max/mean/median/stddev, an
  IQR-rule outlier count (outside `Q1 - 1.5*IQR` .. `Q3 + 1.5*IQR`), and a
  histogram. Categorical: top-N value counts and a horizontal bar chart.
  Datetime: date range and a monthly row-count line chart. Boolean: a
  value-count bar chart.
- **Dataset-level checks.** Row/column counts, a duplicate-row count
  (`df.is_duplicated().sum()`), and a Pearson correlation heatmap across
  every numeric column.
- **One HTML file, no internet required to view it.** Plotly's JS bundle
  is embedded inline once; every chart below it reuses the same `Plotly`
  global instead of re-fetching a CDN script.

## Design notes

Uses **polars** (not pandas) and **Plotly** (not matplotlib) — interactive,
zoomable charts fit a profiling report better than static PNGs, and
polars' lazy-friendly API scales further past the toy examples here.
`numpy` is a transitive dependency of `DataFrame.corr()`.

Duplicate-row detection is a strict full-row comparison. Two transactions
with the same customer/amount/date but a different `order_id` are *not*
flagged — that's a legitimate repeat purchase, not a data-entry duplicate.
See `sample_data/messy_transactions.csv` for exactly this case.

## Run it

```bash
cd "challenges/Data Analytics/CSV Data Profiler"

uv run --with polars --with plotly --with numpy python profiler.py sample_data/clean_employees.csv -o clean_report.html
uv run --with polars --with plotly --with numpy python profiler.py sample_data/messy_transactions.csv -o messy_report.html

uv run --with pytest --with polars --with plotly --with numpy pytest -q   # 15 tests
```

Open the generated `.html` file in any browser — it's fully self-contained.

## Sample data

- `sample_data/clean_employees.csv` — a tidy dataset exercising every
  semantic type (numeric, categorical, datetime, boolean, free text).
- `sample_data/messy_transactions.csv` — missing values, extreme outliers,
  a missing date, and a genuine repeat purchase (not a duplicate row).

## Where this is actually used

Every "let's look at the data first" step of an analysis is this report by
another name — `pandas-profiling`/`ydata-profiling`, `Great Expectations`'
data docs, and Databricks' auto-generated column stats all do the same
three things: infer a type per column, summarize its distribution, and
flag what looks broken (nulls, outliers, duplicates) before anyone builds
a model or dashboard on top of it.
