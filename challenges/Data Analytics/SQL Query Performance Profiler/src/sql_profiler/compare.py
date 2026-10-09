"""Is the rewrite really faster, and is it really the same query?

A rewrite is only an improvement if it returns the same rows, so ``compare`` checks that first
(exactly, with ``EXCEPT ALL`` in both directions), then times both versions and reports whether the
difference is bigger than the run-to-run noise.

Timing is wall clock around fetching the full result, after one warm-up run. "Faster" is only claimed
when *every* run of one query beats *every* run of the other; anything less is "no clear difference",
because with a handful of runs a median ratio on its own happily reports noise as a speedup.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import duckdb

from sql_profiler.db import check_read_only
from sql_profiler.plan import Plan, profile, run_with_timeout


@dataclass(frozen=True)
class Timing:
    runs_s: tuple[float, ...]

    @property
    def median_s(self) -> float:
        return statistics.median(self.runs_s)

    @property
    def best_s(self) -> float:
        return min(self.runs_s)

    @property
    def worst_s(self) -> float:
        return max(self.runs_s)


@dataclass(frozen=True)
class ResultCheck:
    same: bool
    rows_a: int
    rows_b: int
    reason: str


@dataclass(frozen=True)
class Comparison:
    a: Timing
    b: Timing
    check: ResultCheck
    plan_a: Plan
    plan_b: Plan

    @property
    def speedup(self) -> float:
        """Median of A over median of B: above 1 means B is faster."""
        return self.a.median_s / self.b.median_s if self.b.median_s else float("inf")

    @property
    def verdict(self) -> str:
        """``b_faster``, ``a_faster`` or ``no_clear_difference`` (run ranges overlap)."""
        if self.b.worst_s < self.a.best_s:
            return "b_faster"
        if self.a.worst_s < self.b.best_s:
            return "a_faster"
        return "no_clear_difference"

    @property
    def same_plan(self) -> bool:
        return self.plan_a.signature() == self.plan_b.signature()

    @property
    def operators_only_in_a(self) -> dict[str, int]:
        return dict(self.plan_a.operator_counts() - self.plan_b.operator_counts())

    @property
    def operators_only_in_b(self) -> dict[str, int]:
        return dict(self.plan_b.operator_counts() - self.plan_a.operator_counts())


def _body(con: duckdb.DuckDBPyConnection, sql: str) -> str:
    return check_read_only(con, sql).strip().rstrip(";")


def time_query(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    repeats: int = 7,
    warmup: int = 1,
    timeout_s: float = 60.0,
) -> Timing:
    body = _body(con, sql)
    runs = []
    for i in range(warmup + repeats):
        start = time.perf_counter()
        run_with_timeout(con, timeout_s, lambda: con.execute(body).to_arrow_table())
        if i >= warmup:
            runs.append(time.perf_counter() - start)
    return Timing(tuple(runs))


def same_results(con: duckdb.DuckDBPyConnection, a: str, b: str) -> ResultCheck:
    """Whether both queries return the same multiset of rows (row order is ignored).

    Floating-point columns compare exactly, so ``ROUND`` computed sums in both queries.
    """
    a, b = _body(con, a), _body(con, b)
    rows_a = con.execute(f"SELECT count(*) FROM ({a})").fetchone()[0]
    rows_b = con.execute(f"SELECT count(*) FROM ({b})").fetchone()[0]
    try:
        only_a = con.execute(
            f"SELECT count(*) FROM (({a}) EXCEPT ALL ({b}))"
        ).fetchone()[0]
        only_b = con.execute(
            f"SELECT count(*) FROM (({b}) EXCEPT ALL ({a}))"
        ).fetchone()[0]
    except duckdb.Error as exc:
        return ResultCheck(False, rows_a, rows_b, f"result columns differ: {exc}")
    if only_a or only_b:
        return ResultCheck(
            False,
            rows_a,
            rows_b,
            f"{only_a:,} rows only in A, {only_b:,} rows only in B",
        )
    return ResultCheck(True, rows_a, rows_b, "identical rows")


def compare(
    con: duckdb.DuckDBPyConnection,
    a: str,
    b: str,
    repeats: int = 7,
    timeout_s: float = 60.0,
) -> Comparison:
    check = same_results(con, a, b)
    return Comparison(
        a=time_query(con, a, repeats, timeout_s=timeout_s),
        b=time_query(con, b, repeats, timeout_s=timeout_s),
        check=check,
        plan_a=profile(con, a, timeout_s),
        plan_b=profile(con, b, timeout_s),
    )


def plan_diff(plan_a: Plan, plan_b: Plan) -> list[str]:
    """Operator-count changes between two plans, e.g. ``ORDER_BY 1 -> 0``."""
    ca, cb = plan_a.operator_counts(), plan_b.operator_counts()
    return [
        f"{op} {ca[op]} -> {cb[op]}"
        for op in sorted(set(ca) | set(cb))
        if ca[op] != cb[op]
    ]
