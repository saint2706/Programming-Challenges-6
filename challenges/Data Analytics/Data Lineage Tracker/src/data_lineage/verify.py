"""Check the static lineage against the data: change one source column, rerun, see what moved.

For every source column and every strategy in ``execute.STRATEGIES`` the whole pipeline is rebuilt
with that one column changed, and every model column is fingerprinted (a hash of its sorted values)
and compared with an unchanged run. The set of model columns that moved is the *empirical
downstream* of the source column. It is compared with the static downstream:

* **confirmed**: the lineage says it depends on the source column and changing it moved the column;
* **unexercised**: the lineage says it depends, but no change moved it. Either the dependency is
  real and the data never reaches it (a COALESCE fallback that is never needed, a filter that never
  rejects), or the static analysis over-reports. The check cannot tell which; it says so;
* **missed**: changing the source column moved a column the lineage does *not* link to it. This is
  the one that matters: a dependency the static analysis cannot see (``COLUMNS('regex')``, a UDF,
  a macro).

A column that differs between two unchanged runs is reported as noise and excluded, since a
difference there proves nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

from data_lineage import execute
from data_lineage.graph import Graph, Node, label
from data_lineage.project import Project


@dataclass(slots=True)
class Pair:
    source: Node
    target: Node
    kind: str  # "direct": a chain of direct edges reaches it; "indirect": only through row-shaping
    strategies: list[str] = field(default_factory=list)  # which changes moved it


@dataclass(slots=True)
class Verification:
    runs: int = 0
    strategies: tuple[str, ...] = execute.STRATEGIES
    noise: list[Node] = field(default_factory=list)
    keys: dict[str, str | None] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    schema_mismatches: list[str] = field(default_factory=list)
    confirmed: list[Pair] = field(default_factory=list)
    unexercised: list[Pair] = field(default_factory=list)
    missed: list[Pair] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missed and not self.schema_mismatches

    @property
    def considered(self) -> int:
        return len(self.confirmed) + len(self.unexercised)

    def to_json(self) -> dict:
        def pairs(items: list[Pair]) -> list[dict]:
            return [
                {
                    "source": list(p.source),
                    "target": list(p.target),
                    "kind": p.kind,
                    "strategies": p.strategies,
                }
                for p in items
            ]

        return {
            "runs": self.runs,
            "strategies": list(self.strategies),
            "noise": [list(n) for n in self.noise],
            "row_keys": self.keys,
            "failures": self.failures,
            "schema_mismatches": self.schema_mismatches,
            "confirmed": pairs(self.confirmed),
            "unexercised": pairs(self.unexercised),
            "missed": pairs(self.missed),
        }


Fingerprints = dict[str, execute.Prints]


def _run(
    project: Project,
    graph: Graph,
    perturb: tuple[str, str, str] | None,
    keys: dict[str, str | None] | None = None,
) -> tuple[Fingerprints, dict[str, dict[str, str]], dict[str, str | None]]:
    con = execute.connect()
    try:
        execute.load_sources(con, project, graph.sources, perturb)
        execute.run_models(con, project, graph.models)
        if keys is None:
            keys = {m: execute.unique_key(con, m) for m in graph.models}
        prints = {m: execute.fingerprints(con, m, keys[m]) for m in graph.models}
        actual = {m: execute.actual_columns(con, m) for m in graph.models}
    finally:
        con.close()
    return prints, actual, keys


def _moved(
    base: Fingerprints,
    run: Fingerprints,
    ignore: set[Node],
    plain_only: frozenset[str] = frozenset(),
) -> set[Node]:
    """Model columns whose values differ from the baseline.

    A column is compared row-aligned (by the table's key) only while the key itself is unchanged,
    and never for a model in ``plain_only``. Those are the models whose key is downstream of the
    changed source column: shuffling or shifting what a row is *called* re-pairs it with the other
    columns of the row, which is not a dependency of those columns on the source.
    """
    out: set[Node] = set()
    for model, b in base.items():
        r = run[model]
        key_same = (
            b.key is not None
            and model not in plain_only
            and r.plain.get(b.key) == b.plain[b.key]
        )
        for column, digest in b.plain.items():
            if (model, column) in ignore:
                continue
            changed = r.rows != b.rows or r.plain.get(column) != digest
            if not changed and key_same:
                changed = r.aligned.get(column) != b.aligned[column]
            if changed:
                out.add((model, column))
    return out


def verify(
    project: Project,
    graph: Graph,
    strategies: tuple[str, ...] = execute.STRATEGIES,
    only_sources: set[Node] | None = None,
    progress=None,
) -> Verification:
    result = Verification(strategies=strategies)
    base, actual, keys = _run(project, graph, None)
    again, _, _ = _run(project, graph, None, keys)
    result.runs = 2
    result.keys = keys

    # the static column list must equal what DuckDB actually produced
    for model in graph.models:
        static = [c.name for c in graph.lineage[model].columns]
        real = list(actual[model])
        if static != real:
            result.schema_mismatches.append(
                f"{model}: static analysis lists {static}, DuckDB produced {real}"
            )
    noise = _moved(base, again, set())
    result.noise = sorted(noise)

    sources = [
        (s, c)
        for s, cols in graph.sources.items()
        for c in cols
        if only_sources is None or (s, c) in only_sources
    ]
    for source in sources:
        static_all = graph.downstream(source, indirect=True) - noise
        static_direct = graph.downstream(source, indirect=False)
        plain_only = frozenset(
            m for m, k in keys.items() if k is not None and (m, k) in static_all
        )
        moved_by: dict[Node, list[str]] = {}
        for strategy in strategies:
            result.runs += 1
            if progress:
                progress(f"{label(source)} / {strategy}")
            try:
                changed, _, _ = _run(
                    project, graph, (source[0], source[1], strategy), keys
                )
            except duckdb.Error as exc:
                result.failures.append(
                    f"{label(source)} / {strategy}: {str(exc).splitlines()[0]}"
                )
                continue
            for node in _moved(base, changed, noise, plain_only):
                moved_by.setdefault(node, []).append(strategy)

        for target in sorted(static_all):
            kind = "direct" if target in static_direct else "indirect"
            pair = Pair(source, target, kind, moved_by.get(target, []))
            (result.confirmed if pair.strategies else result.unexercised).append(pair)
        for target in sorted(set(moved_by) - static_all):
            result.missed.append(Pair(source, target, "missed", moved_by[target]))
    return result
