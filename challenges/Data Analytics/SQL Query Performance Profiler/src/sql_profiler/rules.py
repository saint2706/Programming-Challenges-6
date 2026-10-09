"""Turn a profiled plan into plain-English findings.

Each rule looks at measured facts (rows in and out, estimate vs actual, time share), never at the SQL
text alone, and reports the numbers it used. A rule that cannot be shown to matter on the measured
data stays quiet: a function call in a filter on a 100-row table is not a finding.

Every threshold is in ``Thresholds`` so tests and callers can tighten or relax them.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from sql_profiler.plan import Plan, PlanNode

SEVERITIES = ("critical", "warn", "info")
PAIRWISE_JOINS = {
    "CROSS_PRODUCT",
    "BLOCKWISE_NL_JOIN",
    "NESTED_LOOP_JOIN",
    "PIECEWISE_MERGE_JOIN",
    "IE_JOIN",
}
JOINS = PAIRWISE_JOINS | {"HASH_JOIN", "ASOF_JOIN"}
GROUP_BYS = {"HASH_GROUP_BY", "PERFECT_HASH_GROUP_BY"}
SQL_WORDS = {
    "in",
    "and",
    "or",
    "not",
    "between",
    "is",
    "like",
    "ilike",
    "null",
    "true",
    "false",
}
FUNCTION_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


@dataclass(frozen=True)
class Thresholds:
    misestimate_warn: float = 10.0
    misestimate_critical: float = 100.0
    misestimate_min_rows: int = 1_000  # ignore estimates where both sides are tiny
    scan_min_rows: int = 10_000  # scans smaller than this are never worth a finding
    wide_scan_rows: int = 100_000
    low_selectivity: float = 0.01
    dominant_share: float = 0.5
    dominant_min_s: float = 0.005
    pairwise_warn: float = 1e6  # candidate pairs a nested-loop style join must compare
    pairwise_critical: float = 1e8
    build_ratio: float = 2.0
    build_min_rows: int = 100_000
    explosion_ratio: float = 10.0
    explosion_min_rows: int = 100_000
    sort_rows: int = 100_000
    group_ratio: float = 0.5
    group_min_rows: int = 100_000


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str  # critical | warn | info
    node_id: int | None
    title: str
    detail: str
    suggestion: str


def _n(x: float) -> str:
    return f"{x:,.0f}"


def _where(node: PlanNode) -> str:
    return f"{node.label} on {node.table}" if node.table else node.label


def misestimate(plan: Plan, t: Thresholds) -> Iterable[Finding]:
    for node in plan.nodes:
        q = node.q_error
        if q is None or not node.estimate_is_comparable or q < t.misestimate_warn:
            continue
        if max(node.est_rows or 0, node.rows) < t.misestimate_min_rows:
            continue
        direction = "over" if (node.est_rows or 0) > node.rows else "under"
        yield Finding(
            "estimate.off",
            "critical" if q >= t.misestimate_critical else "warn",
            node.id,
            f"{_where(node)}: row estimate {q:,.0f}x off",
            f"The optimizer expected {_n(node.est_rows or 0)} rows and got {_n(node.rows)} "
            f"({direction}estimate). Join order, join sides and memory are chosen from estimates, "
            "so everything above this node was planned on a wrong number.",
            "Check the filter: DuckDB assumes a fixed selectivity for range and expression "
            "predicates it has no statistics for. Equality on a column, a range on sorted data, or "
            "rewriting the predicate to a plain column comparison gives it something to estimate from.",
        )


def nonsargable_filter(plan: Plan, t: Thresholds) -> Iterable[Finding]:
    for node in plan.nodes:
        if (
            node.name != "TABLE_SCAN"
            or not node.filters
            or node.rows_scanned < t.scan_min_rows
        ):
            continue
        calls = [
            f for f in FUNCTION_CALL.findall(node.filters) if f.lower() not in SQL_WORDS
        ]
        if not calls:
            continue
        yield Finding(
            "filter.function_on_column",
            "warn" if node.rows_scanned >= t.wide_scan_rows else "info",
            node.id,
            f"{_where(node)}: filter calls {calls[0]}() on the data",
            f"Filter `{node.filters}` wraps a column in a function, so the min/max statistics "
            f"that let a scan skip whole row groups cannot be used: all {_n(node.rows_scanned)} rows "
            f"were read and the function ran on each, keeping {_n(node.rows)}.",
            "Compare the bare column with constants instead (a date range instead of "
            "`strftime(col, '%Y') = '2011'`, `col >= 'abc' AND col < 'abd'` instead of a prefix "
            "function) so the predicate can be skipped over, not just evaluated.",
        )


def filter_not_pushed(plan: Plan, t: Thresholds) -> Iterable[Finding]:
    for node in plan.nodes:
        if node.name != "FILTER" or not node.children:
            continue
        rows_in = node.children[0].rows
        if rows_in < t.scan_min_rows or node.rows > 0.1 * rows_in:
            continue
        yield Finding(
            "filter.late",
            "warn",
            node.id,
            f"FILTER drops {1 - node.rows / rows_in:.0%} of rows after they were produced",
            f"{_n(rows_in)} rows flow into the filter and {_n(node.rows)} come out. A predicate that "
            "stays above its input could not be moved into the scan or below the join.",
            "Put the condition on a bare column of one table so it can move down to the scan.",
        )


def low_selectivity_scan(plan: Plan, t: Thresholds) -> Iterable[Finding]:
    for node in plan.nodes:
        if (
            node.name != "TABLE_SCAN"
            or node.rows_scanned < t.wide_scan_rows
            or not node.filters
        ):
            continue
        if node.rows > t.low_selectivity * node.rows_scanned:
            continue
        yield Finding(
            "scan.reads_much_keeps_little",
            "info",
            node.id,
            f"{_where(node)}: read {_n(node.rows_scanned)} rows to keep {_n(node.rows)}",
            f"Only {node.rows / node.rows_scanned:.2%} of the rows read survive `{node.filters}`. "
            "The scan is a full pass because the filter column is not clustered, so row groups "
            "cannot be skipped.",
            "If this filter is hot, store the table sorted by that column (row groups then carry "
            "tight min/max ranges). An ART index only pays off for very selective point lookups.",
        )


def dominant_operator(plan: Plan, t: Thresholds) -> Iterable[Finding]:
    for node in plan.nodes:
        if node.share >= t.dominant_share and node.self_time_s >= t.dominant_min_s:
            yield Finding(
                "time.dominant_operator",
                "info",
                node.id,
                f"{_where(node)} is {node.share:.0%} of the work",
                f"{node.self_time_s * 1000:,.1f} ms of {plan.operator_time_s * 1000:,.1f} ms of operator "
                f"time (summed over threads) is spent here, on {_n(node.rows)} output rows.",
                "Look here first: shrink its input, or avoid the operator.",
            )


def pairwise_join(plan: Plan, t: Thresholds) -> Iterable[Finding]:
    for node in plan.nodes:
        if node.name not in PAIRWISE_JOINS or len(node.children) != 2:
            continue
        left, right = node.children[0].rows, node.children[1].rows
        pairs = left * right
        if pairs < t.pairwise_warn:
            continue
        kind = (
            "cartesian product"
            if node.name == "CROSS_PRODUCT"
            else "join with no equality key"
        )
        yield Finding(
            "join.no_equality_key",
            "critical" if pairs >= t.pairwise_critical else "warn",
            node.id,
            f"{node.label}: {kind} of {_n(left)} x {_n(right)} rows",
            f"With no `a = b` condition to hash on, up to {_n(pairs)} row pairs are candidates and "
            f"{_n(node.rows)} survive. Cost grows with the product of the inputs, not their sum.",
            "Add an equality condition (join on the key, then filter), or shrink one side first.",
        )


def join_build_side(plan: Plan, t: Thresholds) -> Iterable[Finding]:
    for node in plan.nodes:
        if node.name != "HASH_JOIN" or len(node.children) != 2:
            continue
        probe, build = node.children[0].rows, node.children[1].rows
        if build >= t.build_min_rows and build >= t.build_ratio * probe:
            yield Finding(
                "join.big_build_side",
                "warn",
                node.id,
                f"HASH_JOIN builds its hash table on the larger input ({_n(build)} vs {_n(probe)} rows)",
                "The right-hand input is loaded into memory and the left one streams past it. "
                "Building on the bigger side costs memory and time.",
                "Usually a sign the optimizer expected the sides the other way round (see the "
                "estimate findings). Filter the large side earlier, or check statistics.",
            )


def join_explosion(plan: Plan, t: Thresholds) -> Iterable[Finding]:
    for node in plan.nodes:
        if node.name not in JOINS or len(node.children) != 2:
            continue
        biggest = max(c.rows for c in node.children)
        if (
            node.rows >= t.explosion_min_rows
            and biggest
            and node.rows >= t.explosion_ratio * biggest
        ):
            yield Finding(
                "join.row_explosion",
                "warn",
                node.id,
                f"{node.label} multiplies rows {node.rows / biggest:.0f}x",
                f"The inputs have {_n(node.children[0].rows)} and {_n(node.children[1].rows)} rows and the "
                f"join produces {_n(node.rows)}. Neither side is unique on the join key.",
                "If you only need one row per key, aggregate or DISTINCT one side before the join.",
            )


def full_sort(plan: Plan, t: Thresholds) -> Iterable[Finding]:
    for node in plan.nodes:
        if node.name == "ORDER_BY" and node.rows >= t.sort_rows:
            yield Finding(
                "sort.full",
                "warn" if node.share >= 0.25 else "info",
                node.id,
                f"ORDER_BY fully sorts {_n(node.rows)} rows",
                f"{node.self_time_s * 1000:,.1f} ms ({node.share:.0%} of operator time). Sorting all "
                "rows costs O(n log n) and, unlike most operators, needs the whole result in memory.",
                "If only the first N rows are needed, add LIMIT N so it becomes TOP_N, which keeps "
                "a small heap instead of sorting everything. Drop the ORDER BY if nothing reads the order.",
            )


def repeated_scan(plan: Plan, t: Thresholds) -> Iterable[Finding]:
    scans = [
        n
        for n in plan.nodes
        if n.name == "TABLE_SCAN"
        and n.table
        and n.rows_scanned >= t.scan_min_rows
        # a scan with only a join/top-N derived filter that keeps a sliver is a row fetch (TOP_N's
        # late materialization looks like this), not a second pass over the data
        and (n.filters or n.rows >= 0.001 * n.rows_scanned)
    ]
    for table, count in Counter(n.table for n in scans).items():
        if count >= 2:
            first = next(n for n in scans if n.table == table)
            yield Finding(
                "scan.repeated",
                "info",
                first.id,
                f"{table} is scanned {count} times",
                f"Each of the {count} scans reads {_n(first.rows_scanned)} rows or more. A self-join, a "
                "UNION of two filtered copies or a repeated subquery all look like this.",
                "Compute it once (a CTE referenced twice, conditional aggregation `sum(x) FILTER (WHERE ...)`, "
                "or a window function) when the passes could share one scan.",
            )


def weak_aggregation(plan: Plan, t: Thresholds) -> Iterable[Finding]:
    for node in plan.nodes:
        if node.name not in GROUP_BYS or not node.children:
            continue
        rows_in = node.children[0].rows
        if rows_in >= t.group_min_rows and node.rows >= t.group_ratio * rows_in:
            yield Finding(
                "aggregate.barely_reduces",
                "info",
                node.id,
                f"{node.label} turns {_n(rows_in)} rows into {_n(node.rows)}",
                "Nearly every row is its own group, so the aggregation builds a hash table about as "
                "large as its input.",
                "Check that the grouping key is the one you mean, or pre-aggregate on a coarser key.",
            )


def decorrelated_subquery(plan: Plan, t: Thresholds) -> Iterable[Finding]:
    for node in plan.nodes:
        if node.name.endswith("DELIM_JOIN"):
            yield Finding(
                "subquery.decorrelated",
                "info",
                node.id,
                "A correlated subquery was turned into a join",
                "DuckDB removed the per-row subquery with a delim join, so it does not run once per "
                "outer row. The rewrite is automatic but adds a distinct-key pass.",
                "Writing the same logic as an explicit join to a pre-aggregated table is often "
                "clearer and gives the same plan.",
            )


RULES: tuple[Callable[[Plan, Thresholds], Iterable[Finding]], ...] = (
    misestimate,
    nonsargable_filter,
    filter_not_pushed,
    low_selectivity_scan,
    pairwise_join,
    join_build_side,
    join_explosion,
    full_sort,
    repeated_scan,
    weak_aggregation,
    decorrelated_subquery,
    dominant_operator,
)


def diagnose(plan: Plan, thresholds: Thresholds | None = None) -> list[Finding]:
    """All findings, most severe first, then in plan order."""
    t = thresholds or Thresholds()
    found = [f for rule in RULES for f in rule(plan, t)]
    return sorted(
        found,
        key=lambda f: (
            SEVERITIES.index(f.severity),
            f.node_id if f.node_id is not None else -1,
        ),
    )
