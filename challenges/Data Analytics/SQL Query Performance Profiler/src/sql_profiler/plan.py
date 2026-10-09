"""A query plan as data: DuckDB's ``EXPLAIN (ANALYZE, FORMAT JSON)`` output parsed into a tree.

What each number means (DuckDB profiler):

* ``self_time_s``: time spent inside the operator itself, summed over every thread that ran it.
  It is CPU time, so the operators of a parallel query add up to more than the wall-clock
  ``latency_s``. Shares are shares of the operator total, not of wall time.
* ``rows``: rows the operator actually produced. ``est_rows``: what the optimizer expected before
  running; the gap between them is where bad plans come from.
* ``rows_scanned``: for a table scan, rows read from storage before filters were applied.
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

import duckdb

from sql_profiler.db import check_read_only

WRAPPERS = {"EXPLAIN_ANALYZE", "QUERY", ""}
# operators that report 0 rows because their output flows through a hash table or a CTE
NO_ESTIMATE_CHECK = {"CTE", "LEFT_DELIM_JOIN", "RIGHT_DELIM_JOIN", "DELIM_JOIN"}


@dataclass
class PlanNode:
    id: int
    name: str  # operator_type, e.g. HASH_JOIN, TABLE_SCAN
    label: str  # operator_name, e.g. SEQ_SCAN
    self_time_s: float
    rows: int
    est_rows: int | None
    rows_scanned: int
    result_bytes: int
    info: dict
    depth: int = 0
    children: list[PlanNode] = field(default_factory=list)
    parent: PlanNode | None = field(default=None, repr=False, compare=False)
    share: float = 0.0

    @property
    def table(self) -> str | None:
        t = self.info.get("Table")
        return t.rsplit(".", 1)[-1] if t else None

    @property
    def filters(self) -> str | None:
        return self.info.get("Filters")

    @property
    def conditions(self) -> str | None:
        return self.info.get("Conditions")

    @property
    def join_type(self) -> str | None:
        return self.info.get("Join Type")

    @property
    def q_error(self) -> float | None:
        """How many times off the estimate was (always >= 1; 1 means exact). ``None`` if no estimate."""
        if self.est_rows is None:
            return None
        est, act = (
            self.est_rows + 1,
            self.rows + 1,
        )  # +1: an estimate of 0 or an empty result is fine
        return max(est / act, act / est)

    @property
    def estimate_is_comparable(self) -> bool:
        """Whether ``est_rows`` and ``rows`` measure the same thing.

        Not for operators whose row count is not their output, nor for a scan filtered only by a
        dynamic filter (derived from the other side of a join while running), which no
        up-front estimate can know about.
        """
        if self.est_rows is None or self.name in NO_ESTIMATE_CHECK:
            return False
        return not (
            self.name == "TABLE_SCAN"
            and not self.filters
            and "Dynamic Filters" in self.info
        )

    def walk(self) -> Iterator[PlanNode]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class Plan:
    sql: str
    root: PlanNode
    latency_s: float
    cpu_s: float
    peak_memory_bytes: int
    raw: dict = field(repr=False)

    @property
    def nodes(self) -> list[PlanNode]:
        return list(self.root.walk())

    @property
    def rows_returned(self) -> int:
        return self.root.rows

    @property
    def operator_time_s(self) -> float:
        return sum(n.self_time_s for n in self.nodes)

    @property
    def rows_scanned(self) -> int:
        """Rows read from tables across every scan, before filters were applied."""
        return sum(n.rows_scanned for n in self.nodes if n.name == "TABLE_SCAN")

    @property
    def parallelism(self) -> float:
        """CPU seconds per wall-clock second: about 1 means the query ran on one core."""
        return self.cpu_s / self.latency_s if self.latency_s else 0.0

    def signature(self) -> tuple:
        """The plan's shape: operators and the tables they touch, no numbers. Equal means same plan."""

        def sig(n: PlanNode) -> tuple:
            return (n.name, n.table, n.join_type, tuple(sig(c) for c in n.children))

        return sig(self.root)

    def operator_counts(self) -> Counter[str]:
        return Counter(n.name for n in self.nodes)


def _int(value) -> int | None:
    if value is None:
        return None
    m = re.search(r"\d+", str(value).replace(",", ""))
    return int(m.group()) if m else None


def _build(
    raw: dict, counter: list[int], depth: int, parent: PlanNode | None
) -> PlanNode:
    info = raw.get("extra_info") or {}
    estimate = _int(info.get("Estimated Cardinality"))
    node = PlanNode(
        id=counter[0],
        name=raw.get("operator_type", ""),
        label=(raw.get("operator_name") or raw.get("operator_type", "")).strip(),
        self_time_s=float(raw.get("operator_timing", 0.0)),
        rows=int(raw.get("operator_cardinality", 0)),
        est_rows=estimate or None,  # DuckDB writes 0 where it made no estimate
        rows_scanned=int(raw.get("operator_rows_scanned", 0)),
        result_bytes=int(raw.get("result_set_size", 0)),
        info=info,
        depth=depth,
        parent=parent,
    )
    counter[0] += 1
    node.children = [
        _build(c, counter, depth + 1, node) for c in raw.get("children", [])
    ]
    return node


def parse_profile(raw: dict, sql: str = "") -> Plan:
    """Parse the JSON DuckDB writes for a profiled query (with or without the EXPLAIN wrapper)."""
    top = raw
    while (
        top.get("operator_type", "") in WRAPPERS and len(top.get("children", [])) == 1
    ):
        top = top["children"][0]
    if not top.get("operator_type"):
        raise ValueError("profile has no operators")
    root = _build(top, [0], 0, None)
    total = sum(n.self_time_s for n in root.walk()) or 1.0
    for n in root.walk():
        n.share = n.self_time_s / total
    return Plan(
        sql=sql,
        root=root,
        latency_s=float(raw.get("latency", 0.0)),
        cpu_s=float(raw.get("cpu_time", 0.0)),
        peak_memory_bytes=int(raw.get("system_peak_buffer_memory", 0)),
        raw=raw,
    )


def run_with_timeout(con: duckdb.DuckDBPyConnection, seconds: float, fn):
    """``fn()`` with the query interrupted after ``seconds``; a timeout raises ``TimeoutError``."""
    timer = threading.Timer(seconds, con.interrupt)
    timer.start()
    try:
        return fn()
    except duckdb.InterruptException as exc:
        raise TimeoutError(f"query stopped after {seconds:g}s") from exc
    finally:
        timer.cancel()


def profile(con: duckdb.DuckDBPyConnection, sql: str, timeout_s: float = 60.0) -> Plan:
    """Run ``sql`` once under the profiler and return its plan with real timings and row counts."""
    body = check_read_only(con, sql).strip().rstrip(";")
    rows = run_with_timeout(
        con,
        timeout_s,
        lambda: con.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {body}").fetchall(),
    )
    return parse_profile(json.loads(rows[0][1]), sql=body)
