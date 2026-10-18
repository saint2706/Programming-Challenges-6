"""Static column-level lineage for one model, from its SQL text.

Two kinds of dependency are kept apart, as OpenLineage does:

* **direct**: the output value is computed from the input column (``SUM(price * quantity)`` depends
  directly on ``price`` and ``quantity``). sqlglot's ``lineage()`` finds these, through CTEs,
  subqueries, joins, unions, ``SELECT *`` and window functions.
* **indirect**: the input column decides *which rows* exist or how they are grouped (WHERE, JOIN
  ... ON, GROUP BY, HAVING, QUALIFY) without being part of any output expression. Changing it can
  change every output column, so it is recorded once per model, not once per output column.

sqlglot cannot see everything, so the analysis also reports what it knows it cannot trace
(``COLUMNS('regex')``, table functions that read files). Whether it missed anything else is what the
empirical check in ``verify`` is for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError
from sqlglot.lineage import Node, lineage
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, build_scope

DIALECT = "duckdb"
MAX_EXPR = 200

# From least to most transforming: the strongest hop on a path names the edge.
ROLE_RANK = {
    "copy": 0,
    "rename": 1,
    "cast": 2,
    "expression": 3,
    "conditional": 4,
    "subquery": 5,
    "window": 6,
    "aggregate": 7,
}
INDIRECT_REASONS = ("filter", "join", "group", "having", "qualify")


class AnalysisError(ValueError):
    """A model could not be analysed; the message names the model."""


@dataclass(frozen=True, slots=True)
class Dep:
    """One direct dependency of an output column on a column of an upstream table."""

    table: str
    column: str
    role: str
    steps: tuple[str, ...]


@dataclass(slots=True)
class ColumnLineage:
    name: str
    expr: str
    deps: list[Dep]
    opaque: bool = False  # the expression uses something sqlglot cannot trace

    expr_kind: str = (
        "expression"  # what an expression with no column inputs is, e.g. COUNT(*)
    )

    @property
    def role(self) -> str:
        if not self.deps:
            return self.expr_kind
        return max((d.role for d in self.deps), key=ROLE_RANK.__getitem__)


@dataclass(slots=True)
class ModelLineage:
    model: str
    columns: list[ColumnLineage]
    indirect: dict[tuple[str, str], tuple[str, ...]]
    tables: tuple[str, ...]
    warnings: list[str] = field(default_factory=list)


def _clean(sql: str) -> str:
    """Drop the quotes sqlglot adds around every identifier, and cap the length."""
    text = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"', r"\1", sql)
    return text if len(text) <= MAX_EXPR else text[: MAX_EXPR - 1] + "…"


def _unwrap(e: exp.Expr) -> exp.Expr:
    return e.this if isinstance(e, exp.Alias) else e


def _hop_role(e: exp.Expr) -> str:
    alias = e.alias_or_name if isinstance(e, exp.Alias) else None
    inner = _unwrap(e)
    if isinstance(inner, exp.Column):
        return "copy" if alias in (None, inner.name) else "rename"
    if inner.find(exp.Window):
        return "window"
    if inner.find(exp.AggFunc):
        return "aggregate"
    if inner.find(exp.Subquery):
        return "subquery"
    if inner.find(exp.Case, exp.If, exp.Coalesce):
        return "conditional"
    if isinstance(inner, exp.Cast) and isinstance(inner.this, exp.Column):
        return "cast"
    return "expression"


def _expr_sql(e: exp.Expr) -> str:
    return _clean(_unwrap(e).sql(DIALECT))


def _leaf_paths(root: Node):
    """Every path from an output column's node down to a leaf, without revisiting a node."""
    stack = [(root, (root,))]
    while stack:
        node, path = stack.pop()
        if not node.downstream:
            yield path
            continue
        for child in node.downstream:
            if child not in path:
                stack.append((child, (*path, child)))


def _direct_deps(root: Node) -> list[Dep]:
    best: dict[tuple[str, str], Dep] = {}
    for path in _leaf_paths(root):
        leaf = path[-1]
        if not isinstance(leaf.source, exp.Table):
            continue  # a constant, COUNT(*), or something sqlglot gave up on
        table = leaf.source.name
        alias = leaf.source.alias_or_name
        column = (
            leaf.name[len(alias) + 1 :]
            if leaf.name.startswith(alias + ".")
            else leaf.name
        )
        hops = [n.expression for n in path[:-1] if isinstance(n.expression, exp.Expr)]
        role = max(
            (_hop_role(h) for h in hops), key=ROLE_RANK.__getitem__, default="copy"
        )
        steps = tuple(_expr_sql(h) for h in hops)
        key = (table, column)
        if key not in best or ROLE_RANK[role] > ROLE_RANK[best[key].role]:
            best[key] = Dep(table, column, role, steps)
    return sorted(best.values(), key=lambda d: (d.table, d.column))


def _resolve(scope: Scope, col: exp.Column) -> set[tuple[str, str]]:
    """The physical (table, column) pairs behind ``col``, looking through CTEs and subqueries."""
    s: Scope | None = scope
    source = None
    while s is not None:
        source = s.sources.get(col.table)
        if source is not None:
            break
        s = s.parent
    if source is None:
        return set()
    if isinstance(source, exp.Table):
        return {(source.name, col.name)}
    out: set[tuple[str, str]] = set()
    branches = source.union_scopes if source.is_set_operation else [source]
    for branch in branches:
        for proj in branch.expression.selects:
            if proj.alias_or_name == col.name:
                for inner in proj.find_all(exp.Column):
                    out |= _resolve(branch, inner)
    return out


def _indirect_deps(qualified: exp.Expr) -> dict[tuple[str, str], tuple[str, ...]]:
    root = build_scope(qualified)
    found: dict[tuple[str, str], set[str]] = {}

    def collect(scope: Scope, node: exp.Expr | None, reason: str) -> None:
        if node is None:
            return
        for col in node.find_all(exp.Column):
            for key in _resolve(scope, col):
                found.setdefault(key, set()).add(reason)

    for scope in root.traverse() if root else []:
        select = scope.expression
        if not isinstance(select, exp.Select):
            continue
        collect(scope, select.args.get("where"), "filter")
        collect(scope, select.args.get("group"), "group")
        collect(scope, select.args.get("having"), "having")
        collect(scope, select.args.get("qualify"), "qualify")
        for join in select.args.get("joins") or []:
            collect(scope, join.args.get("on"), "join")
    return {
        k: tuple(r for r in INDIRECT_REASONS if r in v)
        for k, v in sorted(found.items())
    }


def _physical_tables(qualified: exp.Expr) -> tuple[str, ...]:
    root = build_scope(qualified)
    names: set[str] = set()
    for scope in root.traverse() if root else []:
        for source in scope.sources.values():
            if isinstance(source, exp.Table):
                names.add(source.name)
    return tuple(sorted(names))


def analyze_model(
    name: str, sql: str, schema: dict[str, dict[str, str]]
) -> ModelLineage:
    """Lineage of ``sql``'s output columns; ``schema`` maps every upstream table to its columns."""
    try:
        parsed = parse_one(sql, dialect=DIALECT)
    except SqlglotError as exc:
        raise AnalysisError(f"{name}: cannot parse the SQL: {exc}") from exc
    if not isinstance(parsed, exp.Query):
        raise AnalysisError(
            f"{name}: a model must be a single SELECT (got {type(parsed).__name__})"
        )

    warnings: list[str] = []
    if parsed.find(exp.Columns):
        warnings.append(
            "uses COLUMNS('regex'): the columns it expands to cannot be traced statically"
        )
    for table in parsed.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            warnings.append(
                f"reads {_clean(table.sql(DIALECT))}: a table function, so its columns have no lineage"
            )

    try:
        nodes = lineage(None, sql, schema=schema, dialect=DIALECT)
        qualified = qualify(parsed.copy(), schema=schema, dialect=DIALECT)
    except SqlglotError as exc:
        raise AnalysisError(f"{name}: {exc}") from exc

    columns = []
    for out, node in nodes.items():
        expr = node.expression
        opaque = isinstance(expr, exp.Expr) and bool(expr.find(exp.Columns))
        kind = "constant"
        if isinstance(expr, exp.Expr):
            inner = _unwrap(expr)
            kind = (
                "window"
                if inner.find(exp.Window)
                else "aggregate"
                if inner.find(exp.AggFunc)
                else "constant"
            )
        columns.append(
            ColumnLineage(
                out,
                _expr_sql(expr) if isinstance(expr, exp.Expr) else out,
                _direct_deps(node),
                opaque,
                kind,
            )
        )
    return ModelLineage(
        name, columns, _indirect_deps(qualified), _physical_tables(qualified), warnings
    )
