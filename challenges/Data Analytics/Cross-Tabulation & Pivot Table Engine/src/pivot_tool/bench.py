"""``pivot bench``: the engine against polars and DuckDB on the same questions.

The table is the Adult census repeated until it has the requested number of rows. The engine does
more than the others here (grand totals, a laid-out grid with headers), while polars and DuckDB are
timed on the plain group-by that produces the same numbers, so read the comparison as "what does a
pivot cost", not as a claim about group-by speed.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table as RichTable

from pivot_tool.engine import Table, read_bytes
from pivot_tool.presets import ADULT


def best(fn: Callable[[], Any], repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def synthetic_csv(rows: int) -> bytes:
    """The Adult census, repeated: the header once, then the body as many times as fits."""
    header, _, body = read_bytes(ADULT.path).partition(b"\n")
    lines = body.splitlines()
    full, rest = divmod(rows, len(lines))
    out = [header]
    for _ in range(full):
        out.extend(lines)
    out.extend(lines[:rest])
    return b"\n".join(out) + b"\n"


def run_benchmark(console: Console, rows: int, repeat: int) -> None:
    import duckdb
    import polars as pl

    data = synthetic_csv(rows)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "adult_big.csv"
        path.write_bytes(data)
        console.print(
            f"{rows:,} rows, {len(data) / 1e6:.0f} MB of CSV, best of {repeat}"
        )

        con = duckdb.connect()
        load = lambda: con.execute(
            "CREATE OR REPLACE TABLE t AS SELECT * FROM read_csv(?)", [str(path)]
        )
        engine_table: list[Table] = []

        def load_engine() -> None:
            engine_table[:] = [Table(data)]

        timings: list[tuple[str, float, float, float]] = [
            (
                "parse and type the CSV",
                best(load_engine, repeat),
                best(lambda: pl.read_csv(path), repeat),
                best(load, repeat),
            )
        ]
        frame = pl.read_csv(path)
        t = engine_table[0]

        def case(name: str, spec: dict, polars_fn: Callable[[], Any], sql: str) -> None:
            timings.append(
                (
                    name,
                    best(lambda: t.pivot(spec), repeat),
                    best(polars_fn, repeat),
                    best(lambda: con.execute(sql).fetchall(), repeat),
                )
            )

        case(
            "sum of hours by education x sex",
            {
                "rows": [{"field": "education"}],
                "cols": [{"field": "sex"}],
                "values": [{"field": "hours_per_week", "agg": "sum"}],
            },
            lambda: frame.group_by(["education", "sex"]).agg(
                pl.col("hours_per_week").sum()
            ),
            "SELECT education, sex, sum(hours_per_week) FROM t GROUP BY ALL",
        )
        case(
            "mean age by workclass x occupation x income",
            {
                "rows": [{"field": "workclass"}, {"field": "occupation"}],
                "cols": [{"field": "income"}],
                "values": [{"field": "age", "agg": "mean"}],
            },
            lambda: frame.group_by(["workclass", "occupation", "income"]).agg(
                pl.col("age").mean()
            ),
            "SELECT workclass, occupation, income, avg(age) FROM t GROUP BY ALL",
        )
        case(
            "median hours by occupation x sex",
            {
                "rows": [{"field": "occupation"}],
                "cols": [{"field": "sex"}],
                "values": [{"field": "hours_per_week", "agg": "median"}],
            },
            lambda: frame.group_by(["occupation", "sex"]).agg(
                pl.col("hours_per_week").median()
            ),
            "SELECT occupation, sex, median(hours_per_week) FROM t GROUP BY ALL",
        )

        table = RichTable(title="Seconds (lower is better)", header_style="bold")
        for c in ("task", "pivot engine", "polars", "DuckDB", "polars / engine"):
            table.add_column(c, justify="left" if c == "task" else "right")
        for name, e, p, d in timings:
            table.add_row(name, f"{e:.3f}", f"{p:.3f}", f"{d:.3f}", f"{p / e:.1f}x")
        console.print(table)
        t.close()
