"""TPC-H SF1 benchmark: colq against DuckDB on the same queries, answers cross-checked.

    uv run --with duckdb==1.5.3 python scripts/bench.py [--sf 1] [--runs 3]

Generates TPC-H with DuckDB's tpch extension into data/ (gitignored), imports the CSVs with
`colq import`, runs each query in both engines, checks the answers agree, and writes
results/bench.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXE = ROOT / "target" / "release" / ("colq.exe" if sys.platform == "win32" else "colq")

TABLES = {
    "lineitem": "l_orderkey, l_quantity, l_extendedprice, l_discount, l_tax, l_returnflag, "
    "l_linestatus, l_shipdate",
    "orders": "o_orderkey, o_custkey, o_orderdate, o_shippriority",
    "customer": "c_custkey, c_mktsegment",
}

QUERIES = {
    "Q1 pricing summary": """
        SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty,
               sum(l_extendedprice) AS sum_base,
               sum(l_extendedprice * (1 - l_discount)) AS sum_disc,
               sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
               avg(l_quantity) AS avg_qty, avg(l_discount) AS avg_disc, count(*) AS n
        FROM lineitem WHERE l_shipdate <= '1998-09-02'
        GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus""",
    "Q6 forecast revenue": """
        SELECT sum(l_extendedprice * l_discount) AS revenue FROM lineitem
        WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01'
          AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24""",
    "Q3 shipping priority (3-way join)": """
        SELECT l_orderkey, sum(l_extendedprice * (1 - l_discount)) AS revenue,
               o_orderdate, o_shippriority
        FROM lineitem JOIN orders ON l_orderkey = o_orderkey
             JOIN customer ON o_custkey = c_custkey
        WHERE c_mktsegment = 'BUILDING' AND o_orderdate < '1995-03-15'
          AND l_shipdate > '1995-03-15'
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate LIMIT 10""",
    "Key range (zone-map pruning)": """
        SELECT count(*) AS n, sum(l_quantity) AS q FROM lineitem
        WHERE l_orderkey >= 3000000 AND l_orderkey < 3100000""",
}


def generate(sf: float) -> None:
    DATA.mkdir(exist_ok=True)
    if all((DATA / f"{t}.csv").exists() for t in TABLES):
        return
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={sf})")
    for t, cols in TABLES.items():
        con.execute(
            f"COPY (SELECT {cols} FROM {t}) TO '{(DATA / t).as_posix()}.csv' (HEADER)"
        )
        con.execute(
            f"COPY (SELECT {cols} FROM {t}) TO '{(DATA / t).as_posix()}.parquet'"
        )


def colq(*args: str) -> str:
    out = subprocess.run([str(EXE), *args], capture_output=True, text=True, check=False)
    if out.returncode:
        raise SystemExit(f"colq {args[0]} failed: {out.stderr}")
    return out.stdout


def parse_tsv(text: str) -> list[tuple]:
    def cell(s: str) -> object:
        for f in (int, float):
            try:
                return f(s)
            except ValueError:
                pass
        return s

    return [tuple(cell(c) for c in line.split("\t")) for line in text.splitlines()[1:]]


def same(a: list[tuple], b: list[tuple]) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b, strict=True):
        for x, y in zip(ra, rb, strict=True):
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                if not math.isclose(float(x), float(y), rel_tol=1e-9, abs_tol=1e-9):
                    return False
            elif str(x) != str(y):
                return False
    return True


def timed(fn, runs: int) -> tuple[float, object]:
    times, result = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=1)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    generate(a.sf)
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "-q"], cwd=ROOT, check=True
    )

    sizes = {}
    for t in TABLES:
        col = DATA / f"{t}.col"
        if not col.exists():
            print(colq("import", str(DATA / f"{t}.csv"), str(col)).strip())
        sizes[t] = {
            ext: (DATA / f"{t}.{ext}").stat().st_size
            for ext in ("csv", "parquet", "col")
        }
    info = colq("info", str(DATA / "lineitem.col"))

    con = duckdb.connect()
    for t in TABLES:
        con.execute(
            f"CREATE TABLE {t} AS SELECT * FROM '{(DATA / t).as_posix()}.parquet'"
        )
    tables = [x for t in TABLES for x in ("--table", f"{t}={DATA / f'{t}.col'}")]

    rows = []
    for name, sql in QUERIES.items():
        sql = " ".join(sql.split())
        t_col, out = timed(lambda s=sql: colq("query", *tables, "--timing", s), a.runs)
        t_duck, want = timed(lambda s=sql: con.execute(s).fetchall(), a.runs)
        got = parse_tsv(out)
        ok = same(got, want)
        proc = subprocess.run(
            [str(EXE), "query", *tables, "--timing", sql],
            check=True,
            capture_output=True,
            text=True,
        )
        rows.append(
            {
                "query": name,
                "colq_s": round(t_col, 3),
                "duckdb_s": round(t_duck, 3),
                "match": ok,
                "rows": len(got),
                "scan": proc.stderr.strip(),
            }
        )
        print(rows[-1])
        if not ok:
            print("colq:", got[:3], "\nduckdb:", want[:3])

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    payload = {"sf": a.sf, "runs": a.runs, "sizes": sizes, "queries": rows}
    (out_dir / "bench.json").write_text(
        json.dumps(payload, indent=4) + "\n", encoding="utf-8", newline="\n"
    )

    def table(head: list[str], body: list[list[str]]) -> str:
        w = [max(len(r[i]) for r in [head, *body]) for i in range(len(head))]

        def fmt(r: list[str]) -> str:
            return "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |"

        return "\n".join([fmt(head), fmt(["-" * x for x in w]), *map(fmt, body)]) + "\n"

    md = f"# TPC-H SF{a.sf:g}: colq vs DuckDB (median of {a.runs} runs, wall clock)\n\n"
    md += table(
        ["Query", "colq (s)", "DuckDB (s)", "Answers match", "colq scan"],
        [
            [
                r["query"],
                f"{r['colq_s']:.3f}",
                f"{r['duckdb_s']:.3f}",
                str(r["match"]),
                r["scan"],
            ]
            for r in rows
        ],
    )
    md += "\n## File sizes (bytes)\n\n"
    md += table(
        ["Table", "CSV", "Parquet (DuckDB default)", ".col"],
        [
            [t, *(f"{s[e]:,}" for e in ("csv", "parquet", "col"))]
            for t, s in sizes.items()
        ],
    )
    md += "\n## lineitem column encodings (`colq info`)\n\n```text\n" + info + "```\n"
    (out_dir / "bench.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)


if __name__ == "__main__":
    main()
