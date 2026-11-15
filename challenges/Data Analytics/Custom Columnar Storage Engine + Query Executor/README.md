# Custom Columnar Storage Engine + Query Executor

**Category:** Data Analytics
**Difficulty:** X

**Status:** Implemented (Rust)

A mini Parquet plus DuckDB in about 2,500 lines of dependency-free Rust: its own columnar file
format with per-chunk encodings and zone maps, a SQL subset, a planner that pushes projections
and predicates into the scans and prunes row groups, and a vectorized executor with hash joins
and hash aggregation. On TPC-H SF1 (6M `lineitem` rows) every answer matches DuckDB.

## Run it

```bash
cargo test --release                       # 18 tests (unit + end-to-end against plain-Rust answers)
cargo run --release --bin colq -- import sales.csv sales.col [--delim '|'] [--row-group 65536]
cargo run --release --bin colq -- info sales.col
cargo run --release --bin colq -- query --table sales=sales.col --timing \
    "SELECT region, sum(qty * price) AS rev FROM sales WHERE day >= '2024-06-01' GROUP BY region ORDER BY rev DESC"
cargo run --release --bin colq -- query --table sales=sales.col "EXPLAIN SELECT ..."
uv run --no-project --with duckdb==1.5.3 python scripts/bench.py   # TPC-H SF1 vs DuckDB -> results/
```

Without `--table`, every `*.col` file in the current folder is a table named after the file.

## Design

**File format** (`src/format.rs`). `COL1`, then column chunks row group by row group, then a
footer, its length and `COL1` again: the same layout as Parquet (`PAR1` at both ends, metadata
after the data so the writer is one pass). The footer stores the schema and, per row group and
column, the chunk's byte range, encoding and min/max. A reader parses the footer once and then
seeks only to the chunks a query needs. Every read is bounds-checked, so a truncated or corrupt
file gives an error, never a panic or an out-of-range read. Import streams: one pass over the
CSV infers types, a second fills and writes one row group (64Ki rows by default) at a time.

**Encodings** (`src/encoding.rs`). The writer encodes each chunk every applicable way and keeps
the smallest, as Parquet writers do per chunk:

| Encoding | Types | Wins on                                                            |
| -------- | ----- | ------------------------------------------------------------------ |
| plain    | all   | high-cardinality floats and strings                                |
| dict     | all   | low-cardinality columns (flags, dates, prices, discounts)          |
| for      | int   | narrow ranges: min plus bit-packed offsets; constants cost 9 bytes |
| rle      | int   | long runs of large values                                          |
| delta    | int   | sorted keys: bit-packed differences                                |

**SQL** (`src/sql.rs`). `[EXPLAIN] SELECT ... FROM t [JOIN u ON a = b ...] [WHERE] [GROUP BY]
[ORDER BY ... ASC|DESC] [LIMIT]` with arithmetic, comparisons, `BETWEEN`, `AND/OR/NOT`, and
`COUNT(*) / COUNT / SUM / AVG / MIN / MAX`. Joined tables must have distinct column names (TPC-H
style `l_`, `o_`, `c_` prefixes), so `t.col` qualifiers are accepted and dropped.

**Planner** (`src/plan.rs`). Resolves each column to its table, then:

- projection pushdown: a scan reads only the columns the query touches;
- predicate pushdown: `WHERE` is split into conjuncts; those over one table run inside that
  table's scan, the rest run after the joins;
- zone-map pruning: a conjunct shaped `col op literal` skips any row group whose min/max cannot
  satisfy it;
- the FROM table streams; each joined table is scanned, filtered and built into a hash table once.

`EXPLAIN` prints the tree with the pushed filters and the surviving row groups:

```text
HashAggregate group=[cust_name] -> cust_name AS cust_name, SUM(qty) AS SUM(qty)
  Filter (qty > cust_id)
    HashJoin cust = cust_id
      Scan custs columns=[cust_id, cust_name] row_groups=4/4
      Scan sales columns=[id, qty, cust] filter=(id < 2000) row_groups=2/20
```

**Executor** (`src/exec.rs`). Operators work on a whole row group at a time. Expressions evaluate
column-at-a-time; a literal stays a scalar instead of being broadcast to a column, and two
literals are folded. Integer arithmetic is overflow-checked, division is floating point (as in
DuckDB). Aggregation is a hash table from group key to accumulator slot. Sort is a stable
multi-key sort; `LIMIT` without `ORDER BY` stops the scan early.

## Results: TPC-H SF1 against DuckDB

`results/bench.md`, DuckDB 1.5.3 with the tables loaded in memory, this engine reading its `.col`
files from disk, median of 3 wall-clock runs on an Intel Core Ultra 7 255H laptop:

| Query                             | colq    | DuckDB  | Answers match | Row groups read |
| --------------------------------- | ------- | ------- | ------------- | --------------- |
| Q1 pricing summary                | 3.61 s  | 0.022 s | yes           | 92 / 92         |
| Q6 forecast revenue               | 0.78 s  | 0.005 s | yes           | 92 / 92         |
| Q3 shipping priority (3-way join) | 1.34 s  | 0.036 s | yes           | 118 / 118       |
| Key range on `l_orderkey`         | 0.018 s | 0.001 s | yes           | 3 / 92          |

Answers agree to a relative 1e-9 (DuckDB sums exact DECIMALs, this engine sums f64).

**Zone maps work when the data is sorted on the filter column.** `lineitem` comes out of the
generator sorted by `l_orderkey`, so a 100,000-key range reads 3 of 92 row groups, 197k of 6M
rows, and finishes 43x faster than Q6. Q1 and Q6 filter on `l_shipdate`, which is spread across
every row group, so nothing is pruned: that is why warehouses cluster or sort tables on the
columns people filter by.

**Encodings beat Parquet's file size where structure is exploitable, and lose on raw floats.**

| Table    | CSV     | Parquet (DuckDB default, compressed) | .col    |
| -------- | ------- | ------------------------------------ | ------- |
| lineitem | 285 MB  | 63.1 MB                              | 76.8 MB |
| orders   | 40.6 MB | 15.3 MB                              | 7.3 MB  |
| customer | 2.4 MB  | 0.66 MB                              | 0.06 MB |

`c_custkey` is 1, 2, 3, ..., so delta encoding stores every difference in 0 bits and the whole
column costs a few bytes per row group. `l_extendedprice` has no structure the encodings can use
and is 48 MB of the 77 MB `lineitem` file, stored plain. Parquet wins there because it adds
general-purpose compression on top of its encodings; this format has none.

**DuckDB is 18 to 160 times faster.** The gap comes from things a mini engine skips:

- strings are decoded into owned `String`s, 6M allocations for `l_shipdate` alone, where DuckDB
  compares dictionary codes and inlines short strings;
- the hash aggregate builds a `Vec` key per row; DuckDB hashes whole vectors and uses a
  partitioned, cache-sized table;
- one thread; DuckDB runs morsel-driven parallelism across all cores;
- no compression of the encoded bytes, and no late materialisation.

## Limits

- No NULLs: every column is dense, so `AVG`, `MIN` and `MAX` of zero rows return NaN instead of
  NULL.
- Inner equi-joins only, one key each; no subqueries, `HAVING`, `LIKE`, `CASE` or window
  functions. Dates are ISO strings, which compare correctly as text.
- Dictionary-encoded chunks are fully decoded before filtering; executing directly on codes is the
  next step and the largest speed-up available.

## Where this is actually used

The layout here is Apache Parquet's in miniature. A Parquet file starts and ends with `PAR1`,
keeps column chunks grouped into row groups, and writes its metadata after the data so a writer
needs one pass; readers parse that footer first to find only the chunks they need. Parquet's
encodings include plain, dictionary, the RLE/bit-packing hybrid and delta binary packed, the same
families this engine chooses between per chunk.

Zone maps are standard in analytical databases. DuckDB's documentation says a min-max index
(zonemap) is created automatically for columns of all general-purpose data types. Snowflake
stores tables as columnar micro-partitions of 50 to 500 MB uncompressed and keeps per-column value
ranges for each one, so queries skip micro-partitions that cannot match a filter: the same
pruning as the key-range query above, and the reason clustering keys matter there.

Vectorized execution over column batches is the model DuckDB and most modern analytical engines
use. The gap in the benchmark shows what the production versions add on top: operating on
compressed data, parallel morsels, and careful memory layout.

## Files

- `src/format.rs`: file layout, streaming writer, footer, chunk reader
- `src/encoding.rs`: plain, dictionary, frame-of-reference, RLE, delta; bit packing
- `src/csv.rs`: RFC 4180 CSV, type inference, streaming import
- `src/sql.rs`: tokenizer and parser
- `src/plan.rs`: column resolution, pushdown, zone-map pruning, `EXPLAIN`
- `src/exec.rs`: expression evaluation, hash join, hash aggregate, sort, limit
- `src/main.rs`: the `colq` CLI
- `tests/engine.rs`: end-to-end tests against plain-Rust answers, encoding round trips, corrupt files
- `scripts/bench.py`, `results/`: TPC-H SF1 benchmark against DuckDB
