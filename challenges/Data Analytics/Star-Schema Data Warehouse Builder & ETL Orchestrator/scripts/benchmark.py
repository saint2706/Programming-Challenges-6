"""Measure the warehouse and write results/benchmark.json and results/benchmark.md.

Three measurements, each on the feed that ``default_feed()`` picks (the full feed when fetched):

1. Load throughput: one run over the whole arrival window, feed lines per second.
2. Incremental vs full rebuild: monthly runs in order from an empty file, against one full-window run.
   Both must end with the same content hash. The last month is also timed as a single run on top
   of the loaded history, and a rerun of that month is timed to show the idempotent path.
3. Star vs a flat denormalised table built from the same kept lines. Six query shapes, each run
   three times after a warm-up, median reported. Shapes where the flat table wins are kept in the
   output on purpose.

Run from the challenge directory:  uv run python scripts/benchmark.py
The DuckDB files live under out/bench/ and are deleted at the end.
"""

from __future__ import annotations

import json
import platform
import shutil
import statistics
import time
from datetime import date, timedelta
from pathlib import Path

import duckdb
import polars as pl
from star_warehouse.paths import project_root
from star_warehouse.pipeline import default_feed, default_lookup, months, run_interval
from star_warehouse.warehouse import connect, content_hash

ROOT = project_root()
RESULTS = ROOT / "results"
WORK = ROOT / "out" / "bench"
REPEATS = 3

QUERIES = {
    "monthly sales by country": (
        """SELECT d.month_start, c.country, round(sum(f.sales_amount), 2)
           FROM dw.fact_sales f
           JOIN dw.dim_date d ON d.date_key = f.invoice_date_key
           JOIN dw.dim_country c ON c.country_key = f.country_key
           GROUP BY ALL""",
        """SELECT month_start, country, round(sum(revenue), 2)
           FROM flat WHERE kind = 'sale' GROUP BY ALL""",
    ),
    "sales by stock code": (
        """SELECT p.stock_code, round(sum(f.sales_amount), 2)
           FROM dw.fact_sales f JOIN dw.dim_product p ON p.product_key = f.product_key
           GROUP BY ALL""",
        """SELECT stock_code, round(sum(revenue), 2)
           FROM flat WHERE kind = 'sale' GROUP BY ALL""",
    ),
    "distinct customers per region per quarter": (
        """SELECT c.region, d.year, d.quarter, count(DISTINCT cu.customer_id)
           FROM dw.fact_sales f
           JOIN dw.dim_date d ON d.date_key = f.invoice_date_key
           JOIN dw.dim_country c ON c.country_key = f.country_key
           JOIN dw.dim_customer cu ON cu.customer_key = f.customer_key
           GROUP BY ALL""",
        """SELECT region, year(invoice_date), quarter(invoice_date), count(DISTINCT customer_id)
           FROM flat WHERE kind = 'sale' GROUP BY ALL""",
    ),
    "total sales, no grouping": (
        "SELECT round(sum(sales_amount), 2) FROM dw.fact_sales",
        "SELECT round(sum(revenue), 2) FROM flat WHERE kind = 'sale'",
    ),
    "one country's sale lines": (
        """SELECT count(*) FROM dw.fact_sales f
           JOIN dw.dim_country c ON c.country_key = f.country_key
           WHERE c.country = 'United Kingdom'""",
        "SELECT count(*) FROM flat WHERE kind = 'sale' AND country = 'United Kingdom'",
    ),
    "returns by month": (
        """SELECT d.month_start, count(*), round(sum(r.return_amount), 2)
           FROM dw.fact_returns r JOIN dw.dim_date d ON d.date_key = r.invoice_date_key
           GROUP BY ALL""",
        """SELECT month_start, count(*), round(sum(revenue), 2)
           FROM flat WHERE kind = 'return' GROUP BY ALL""",
    ),
}


def _timed(fn):
    start = time.perf_counter()
    value = fn()
    return time.perf_counter() - start, value


def _median_ms(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[float, list]:
    con.execute(sql).fetchall()  # warm-up
    samples = []
    rows: list = []
    for _ in range(REPEATS):
        elapsed, rows = _timed(lambda: con.execute(sql).fetchall())
        samples.append(elapsed * 1000)
    return statistics.median(samples), sorted(rows, key=repr)


def _window(feed: Path) -> tuple[date, date, int]:
    arrivals = pl.read_parquet(feed, columns=["arrival_date"])["arrival_date"]
    first = arrivals.min().replace(day=1)
    last = arrivals.max()
    end = (last.replace(day=28) + timedelta(days=4)).replace(day=1)
    return first, end, arrivals.len()


def load_and_incremental(
    feed: Path, lookup: Path, first: date, end: date, lines: int
) -> dict:
    full_db = WORK / "full.duckdb"
    seconds, outcome = _timed(
        lambda: run_interval(full_db, feed, lookup, first, end, backoff=0.0)
    )
    if outcome.status != "succeeded":
        raise SystemExit(f"full run failed: {outcome.failed_tasks}")
    con = connect(full_db)
    full_hash = content_hash(con)
    con.close()

    inc_db = WORK / "incremental.duckdb"
    per_month = []
    total = 0.0
    for start, stop in months(first, end):
        elapsed, outcome = _timed(
            lambda s=start, e=stop: run_interval(
                inc_db, feed, lookup, s, e, backoff=0.0
            )
        )
        if outcome.status != "succeeded":
            raise SystemExit(
                f"incremental run failed for {start}: {outcome.failed_tasks}"
            )
        total += elapsed
        per_month.append(elapsed)
    con = connect(inc_db)
    inc_hash = content_hash(con)
    con.close()

    last_start, last_end = months(first, end)[-1]
    rerun_elapsed, rerun = _timed(
        lambda: run_interval(inc_db, feed, lookup, last_start, last_end, backoff=0.0)
    )
    con = connect(inc_db)
    rerun_hash = content_hash(con)
    con.close()

    return {
        "full_db": full_db,
        "load": {
            "feed_lines": lines,
            "window": f"{first} to {end}",
            "full_load_seconds": round(seconds, 2),
            "lines_per_second": round(lines / seconds),
        },
        "incremental": {
            "months": len(per_month),
            "monthly_total_seconds": round(total, 2),
            "median_month_seconds": round(statistics.median(per_month), 3),
            "last_month_seconds": round(per_month[-1], 3),
            "last_month_rerun_seconds": round(rerun_elapsed, 3),
            "rerun_status": rerun.status,
            "same_hash_as_full_rebuild": inc_hash == full_hash,
            "rerun_leaves_hash_unchanged": rerun_hash == inc_hash,
            "full_rebuild_hash_prefix": full_hash[:16],
        },
    }


def star_vs_flat(full_db: Path) -> dict:
    con = duckdb.connect(str(full_db))
    con.execute(
        """CREATE OR REPLACE TABLE flat AS
           SELECT s.line_key, s.invoice, s.stock_code, s.customer_id, s.country,
                  s.invoice_date, date_trunc('month', s.invoice_date)::DATE AS month_start,
                  s.kind, s.revenue, s.quantity, c.region
           FROM staging.lines s
           JOIN dw.dim_country c ON c.country = s.country
           WHERE s.exclusion IS NULL"""
    )
    con.execute("CHECKPOINT")
    flat_rows = con.execute("SELECT count(*) FROM flat").fetchone()[0]
    rows = []
    for name, (star_sql, flat_sql) in QUERIES.items():
        star_ms, star_rows = _median_ms(con, star_sql)
        flat_ms, flat_rows_out = _median_ms(con, flat_sql)
        rows.append(
            {
                "query": name,
                "star_ms": round(star_ms, 1),
                "flat_ms": round(flat_ms, 1),
                "faster": "star" if star_ms < flat_ms else "flat",
                "result_rows": len(star_rows),
                "same_answer": star_rows == flat_rows_out,
            }
        )
    con.close()
    return {"flat_rows": flat_rows, "queries": rows}


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    cells = [headers] + rows
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]

    def line(row: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


def write_markdown(data: dict) -> str:
    load, inc, sf = data["load"], data["incremental"], data["star_vs_flat"]
    out = [
        "# Benchmark results",
        "",
        (
            f"Feed: {load['feed_lines']:,} lines, window {load['window']}. "
            f"Python {data['environment']['python']}, DuckDB {data['environment']['duckdb']}, "
            f"polars {data['environment']['polars']}, {data['environment']['machine']}."
        ),
        "",
        "## Load throughput",
        "",
        _md_table(
            ["Measure", "Value"],
            [
                ["Full-window load, seconds", f"{load['full_load_seconds']}"],
                ["Feed lines per second", f"{load['lines_per_second']:,}"],
            ],
        ),
        "",
        "## Incremental vs full rebuild",
        "",
        _md_table(
            ["Measure", "Value"],
            [
                ["Full rebuild (one run), seconds", f"{load['full_load_seconds']}"],
                [
                    f"Monthly runs ({inc['months']} months), total seconds",
                    f"{inc['monthly_total_seconds']}",
                ],
                ["Median monthly run, seconds", f"{inc['median_month_seconds']}"],
                [
                    "Last month on top of history, seconds",
                    f"{inc['last_month_seconds']}",
                ],
                [
                    "Last month rerun (no new data), seconds",
                    f"{inc['last_month_rerun_seconds']}",
                ],
                [
                    "Monthly load hash equals full rebuild hash",
                    str(inc["same_hash_as_full_rebuild"]),
                ],
                [
                    "Rerun leaves the hash unchanged",
                    str(inc["rerun_leaves_hash_unchanged"]),
                ],
            ],
        ),
        "",
        "## Star schema vs flat denormalised table",
        "",
        (
            f"The flat table has {sf['flat_rows']:,} kept lines with region, date and customer id "
            "copied in and no surrogate keys or SCD2 history. Median of "
            f"{REPEATS} runs after one warm-up, in milliseconds."
        ),
        "",
        _md_table(
            ["Query", "Star ms", "Flat ms", "Faster", "Result rows", "Same answer"],
            [
                [
                    q["query"],
                    f"{q['star_ms']}",
                    f"{q['flat_ms']}",
                    q["faster"],
                    f"{q['result_rows']}",
                    str(q["same_answer"]),
                ]
                for q in sf["queries"]
            ],
        ),
        "",
    ]
    return "\n".join(out)


def main() -> None:
    feed = default_feed()
    lookup = default_lookup()
    lines = pl.read_parquet(feed, columns=["line_key"]).height
    first, end, _ = _window(feed)
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    try:
        measured = load_and_incremental(feed, lookup, first, end, lines)
        full_db = measured.pop("full_db")
        star = star_vs_flat(full_db)
        data = {
            "environment": {
                "python": platform.python_version(),
                "duckdb": duckdb.__version__,
                "polars": pl.__version__,
                "machine": platform.machine() + " " + platform.system(),
                "feed": feed.name,
            },
            **measured,
            "star_vs_flat": star,
        }
    finally:
        shutil.rmtree(WORK, ignore_errors=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "benchmark.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (RESULTS / "benchmark.md").write_text(
        write_markdown(data) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {RESULTS / 'benchmark.json'} and {RESULTS / 'benchmark.md'}")


if __name__ == "__main__":
    main()
