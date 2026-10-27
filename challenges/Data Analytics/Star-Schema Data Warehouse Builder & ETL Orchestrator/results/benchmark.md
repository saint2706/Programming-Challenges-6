# Benchmark results

Feed: 1,044,848 lines, window 2009-12-01 to 2012-05-01. Python 3.14.7, DuckDB 2.0.0, polars 2.1.0, AMD64 Windows.

## Load throughput

| Measure                   | Value   |
| ------------------------- | ------- |
| Full-window load, seconds | 2.38    |
| Feed lines per second     | 438,956 |

## Incremental vs full rebuild

| Measure                                    | Value |
| ------------------------------------------ | ----- |
| Full rebuild (one run), seconds            | 2.38  |
| Monthly runs (29 months), total seconds    | 27.64 |
| Median monthly run, seconds                | 0.906 |
| Last month on top of history, seconds      | 0.451 |
| Last month rerun (no new data), seconds    | 0.43  |
| Monthly load hash equals full rebuild hash | True  |
| Rerun leaves the hash unchanged            | True  |

## Star schema vs flat denormalised table

The flat table has 1,032,917 kept lines with region, date and customer id copied in and no surrogate keys or SCD2 history. Median of 3 runs after one warm-up, in milliseconds.

| Query                                     | Star ms | Flat ms | Faster | Result rows | Same answer |
| ----------------------------------------- | ------- | ------- | ------ | ----------- | ----------- |
| monthly sales by country                  | 15.2    | 8.8     | flat   | 562         | True        |
| sales by stock code                       | 21.3    | 17.7    | flat   | 4877        | True        |
| distinct customers per region per quarter | 26.7    | 14.7    | flat   | 55          | True        |
| total sales, no grouping                  | 1.1     | 1.5     | star   | 1           | True        |
| one country's sale lines                  | 3.9     | 2.4     | flat   | 1           | True        |
| returns by month                          | 1.8     | 2.7     | star   | 25          | True        |
