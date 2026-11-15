# TPC-H SF1: colq vs DuckDB (median of 3 runs, wall clock)

| Query                             | colq (s) | DuckDB (s) | Answers match | colq scan                                                        |
| --------------------------------- | -------- | ---------- | ------------- | ---------------------------------------------------------------- |
| Q1 pricing summary                | 3.609    | 0.022      | True          | 4 rows in 3.434s; scanned 6001215 rows, read 92/92 row groups    |
| Q6 forecast revenue               | 0.780    | 0.005      | True          | 1 rows in 0.734s; scanned 6001215 rows, read 92/92 row groups    |
| Q3 shipping priority (3-way join) | 1.343    | 0.036      | True          | 10 rows in 1.282s; scanned 7651215 rows, read 118/118 row groups |
| Key range (zone-map pruning)      | 0.018    | 0.001      | True          | 1 rows in 0.014s; scanned 196608 rows, read 3/92 row groups      |

## File sizes (bytes)

| Table    | CSV         | Parquet (DuckDB default) | .col       |
| -------- | ----------- | ------------------------ | ---------- |
| lineitem | 285,095,147 | 63,072,449               | 76,845,732 |
| orders   | 40,612,125  | 15,326,703               | 7,342,284  |
| customer | 2,438,528   | 658,899                  | 56,847     |

## lineitem column encodings (`colq info`)

```text
6001215 rows in 92 row groups
  l_orderkey           Int      3752783 bytes  delta
  l_quantity           Float      4538632 bytes  dict
  l_extendedprice      Float     48010180 bytes  plain
  l_discount           Float      3009624 bytes  dict
  l_tax                Float      3008152 bytes  dict
  l_returnflag         Str      1502604 bytes  dict
  l_linestatus         Str       751992 bytes  dict
  l_shipdate           Str     12245493 bytes  dict
```
