"""The column graph of a whole pipeline: build it in dependency order, then ask questions of it.

Nodes are ``(table, column)`` pairs for sources and models alike. Direct edges connect one column to
one column. Indirect edges connect a source column to a *model* (it shapes that model's rows), and
reach every column of that model when a question is asked with ``indirect=True``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import duckdb
from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import build_scope

from data_lineage.analyze import DIALECT, AnalysisError, ModelLineage, analyze_model
from data_lineage.project import Project, ProjectError

Node = tuple[str, str]


def label(node: Node) -> str:
    return f"{node[0]}.{node[1]}"


def parse_node(text: str) -> Node:
    table, sep, column = text.partition(".")
    if not sep or not table or not column:
        raise ValueError(f"expected table.column, got {text!r}")
    return table, column


@dataclass(slots=True)
class Graph:
    sources: dict[str, dict[str, str]]  # source -> {column: type}
    lineage: dict[str, ModelLineage]  # model -> lineage, in dependency order
    layer: dict[str, str]  # model -> folder name
    rank: dict[str, int]  # table -> 0 for sources, 1 + deepest parent for models
    direct_up: dict[Node, list[tuple[Node, str, tuple[str, ...]]]] = field(
        default_factory=dict
    )
    direct_down: dict[Node, list[Node]] = field(default_factory=dict)
    indirect_up: dict[str, dict[Node, tuple[str, ...]]] = field(default_factory=dict)
    indirect_down: dict[Node, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        down: dict[Node, list[Node]] = defaultdict(list)
        up: dict[Node, list[tuple[Node, str, tuple[str, ...]]]] = defaultdict(list)
        ind_down: dict[Node, set[str]] = defaultdict(set)
        for model, ml in self.lineage.items():
            for col in ml.columns:
                dst = (model, col.name)
                for dep in col.deps:
                    src = (dep.table, dep.column)
                    up[dst].append((src, dep.role, dep.steps))
                    down[src].append(dst)
            self.indirect_up[model] = dict(ml.indirect)
            for src in ml.indirect:
                ind_down[src].add(model)
        self.direct_up, self.direct_down, self.indirect_down = (
            dict(up),
            dict(down),
            dict(ind_down),
        )

    # -- structure -----------------------------------------------------------------------------
    @property
    def models(self) -> list[str]:
        return list(self.lineage)

    def columns_of(self, table: str) -> list[str]:
        if table in self.sources:
            return list(self.sources[table])
        return [c.name for c in self.lineage[table].columns]

    def has_node(self, node: Node) -> bool:
        return node[0] in self.rank and node[1] in self.columns_of(node[0])

    def all_nodes(self) -> list[Node]:
        return [(t, c) for t in self.rank for c in self.columns_of(t)]

    # -- questions -----------------------------------------------------------------------------
    def upstream(self, node: Node, indirect: bool = True) -> set[Node]:
        """Every column ``node`` is computed from, transitively."""
        seen: set[Node] = set()
        queue = deque([node])
        while queue:
            cur = queue.popleft()
            parents = [p for p, _, _ in self.direct_up.get(cur, ())]
            if indirect and cur[0] in self.indirect_up:
                parents += list(self.indirect_up[cur[0]])
            for p in parents:
                if p not in seen:
                    seen.add(p)
                    queue.append(p)
        return seen

    def downstream(self, node: Node, indirect: bool = True) -> set[Node]:
        """Every column that changes if ``node`` changes, transitively."""
        seen: set[Node] = set()
        queue = deque([node])
        while queue:
            cur = queue.popleft()
            children = list(self.direct_down.get(cur, ()))
            if indirect:
                for model in self.indirect_down.get(cur, ()):
                    children += [(model, c) for c in self.columns_of(model)]
            for c in children:
                if c not in seen:
                    seen.add(c)
                    queue.append(c)
        return seen

    def source_columns(self, node: Node, indirect: bool = True) -> set[Node]:
        return {n for n in self.upstream(node, indirect) if n[0] in self.sources}

    def to_json(self) -> dict:
        edges, indirect = [], []
        for model, ml in self.lineage.items():
            for col in ml.columns:
                for dep in col.deps:
                    edges.append(
                        {
                            "src": [dep.table, dep.column],
                            "dst": [model, col.name],
                            "role": dep.role,
                            "steps": list(dep.steps),
                        }
                    )
            for (table, column), reasons in ml.indirect.items():
                indirect.append(
                    {"src": [table, column], "model": model, "reasons": list(reasons)}
                )
        return {
            "sources": [
                {
                    "name": s,
                    "rank": 0,
                    "columns": [{"name": c, "type": t} for c, t in cols.items()],
                }
                for s, cols in self.sources.items()
            ],
            "models": [
                {
                    "name": m,
                    "layer": self.layer[m],
                    "rank": self.rank[m],
                    "warnings": ml.warnings,
                    "reads": list(ml.tables),
                    "columns": [
                        {
                            "name": c.name,
                            "expr": c.expr,
                            "role": c.role,
                            "opaque": c.opaque,
                        }
                        for c in ml.columns
                    ],
                }
                for m, ml in self.lineage.items()
            ],
            "edges": edges,
            "indirect": indirect,
        }


def source_schemas(project: Project) -> dict[str, dict[str, str]]:
    """Column names and DuckDB types of every source file (reads only the file headers)."""
    con = duckdb.connect(":memory:")
    out: dict[str, dict[str, str]] = {}
    try:
        for src in project.sources:
            try:
                rows = con.execute(
                    f"DESCRIBE SELECT * FROM {src.relation()}"
                ).fetchall()
            except duckdb.Error as exc:
                raise ProjectError(
                    f"source {src.name}: DuckDB cannot read {src.path}: {exc}"
                ) from exc
            out[src.name] = {r[0]: r[1] for r in rows}
    finally:
        con.close()
    return out


def referenced_tables(name: str, sql: str) -> set[str]:
    """Physical tables a model reads (CTE names excluded), without needing any schema."""
    try:
        root = build_scope(parse_one(sql, dialect=DIALECT))
    except SqlglotError as exc:
        raise AnalysisError(f"{name}: cannot parse the SQL: {exc}") from exc
    names: set[str] = set()
    for scope in root.traverse() if root else []:
        for source in scope.sources.values():
            if isinstance(source, exp.Table) and isinstance(
                source.this, exp.Identifier
            ):
                names.add(source.name)
    return names


def build_graph(
    project: Project, schemas: dict[str, dict[str, str]] | None = None
) -> Graph:
    sources = schemas if schemas is not None else source_schemas(project)
    deps = {m.name: referenced_tables(m.name, m.sql) for m in project.models}
    known = set(sources) | set(deps)
    for name, wanted in deps.items():
        missing = sorted(wanted - known)
        if missing:
            raise AnalysisError(
                f"{name} reads {', '.join(missing)}, which is neither a source nor a model"
            )
        if name in wanted:
            raise AnalysisError(f"{name} reads itself")

    order = _topological(deps)
    by_name = {m.name: m for m in project.models}
    schema: dict[str, dict[str, str]] = {s: dict(c) for s, c in sources.items()}
    lineage: dict[str, ModelLineage] = {}
    rank: dict[str, int] = dict.fromkeys(sources, 0)
    for name in order:
        ml = analyze_model(name, by_name[name].sql, schema)
        lineage[name] = ml
        schema[name] = {c.name: "unknown" for c in ml.columns}
        rank[name] = 1 + max((rank[t] for t in deps[name]), default=0)
    return Graph(sources, lineage, {m.name: m.layer for m in project.models}, rank)


def _topological(deps: dict[str, set[str]]) -> list[str]:
    models = set(deps)
    pending = {m: {d for d in ds if d in models} for m, ds in deps.items()}
    order: list[str] = []
    ready = sorted(m for m, ds in pending.items() if not ds)
    while ready:
        m = ready.pop(0)
        order.append(m)
        for other, ds in pending.items():
            if m in ds:
                ds.discard(m)
                if not ds and other not in order and other not in ready:
                    ready.append(other)
        ready.sort()
    if len(order) != len(models):
        cycle = sorted(models - set(order))
        raise AnalysisError(f"circular dependency between models: {', '.join(cycle)}")
    return order
