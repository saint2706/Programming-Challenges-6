"""Command line: build the lineage graph and report, trace one column, check it against the data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from data_lineage import execute
from data_lineage.analyze import AnalysisError
from data_lineage.graph import Graph, Node, build_graph, label, parse_node
from data_lineage.paths import project_root
from data_lineage.project import Project, ProjectError, load_project
from data_lineage.report import build_html
from data_lineage.verify import Verification, verify

app = typer.Typer(
    help="Column-level lineage for a SQL pipeline, checked against the data.",
    no_args_is_help=True,
)
console = Console()

PipelineOpt = Annotated[
    Path,
    typer.Option(
        "--pipeline", "-p", help="Folder with sources.toml and the .sql models."
    ),
]
IndirectOpt = Annotated[
    bool,
    typer.Option(
        "--indirect", help="Also follow filter, join and group-by columns (rows only)."
    ),
]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() or path.exists() else project_root() / path


def _load(pipeline: Path) -> tuple[Project, Graph]:
    try:
        project = load_project(_resolve(pipeline))
        return project, build_graph(project)
    except (ProjectError, AnalysisError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc


def _node(graph: Graph, text: str) -> Node:
    try:
        node = parse_node(text)
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc
    if not graph.has_node(node):
        known = ", ".join(sorted(graph.rank))
        console.print(f"[red]error:[/red] no column {text}. Tables: {known}")
        raise typer.Exit(2)
    return node


def _write(out: Path, name: str, text: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _json(payload: dict) -> str:
    return json.dumps(payload, indent=4, ensure_ascii=False) + "\n"


def _summary(v: Verification) -> None:
    table = Table(title=f"Static lineage vs the data ({v.runs} pipeline runs)")
    table.add_column("verdict")
    table.add_column("pairs", justify="right")
    table.add_column("meaning")
    table.add_row(
        "confirmed",
        str(len(v.confirmed)),
        "lineage says it depends, and changing the source moved it",
    )
    table.add_row(
        "unexercised",
        str(len(v.unexercised)),
        "lineage says it depends, no change moved it",
    )
    table.add_row(
        "missed",
        str(len(v.missed)),
        "changing the source moved a column the lineage does not link to it",
    )
    console.print(table)
    if v.noise:
        names = ", ".join(label(n) for n in v.noise)
        console.print(
            f"[yellow]noise:[/yellow] {names} differ between two unchanged runs, excluded"
        )
    for failure in v.failures:
        console.print(f"[yellow]run failed:[/yellow] {failure}")
    for mismatch in v.schema_mismatches:
        console.print(f"[red]schema mismatch:[/red] {mismatch}")
    for p in v.unexercised:
        console.print(
            f"[yellow]?[/yellow] {label(p.source)} -> {label(p.target)} ({p.kind}): never moved"
        )
    for p in v.missed:
        how = ", ".join(p.strategies)
        console.print(
            f"[red]![/red] {label(p.source)} -> {label(p.target)}: moved by {how}, not in the lineage"
        )


@app.command()
def build(
    pipeline: PipelineOpt = Path("pipeline"),
    out: Annotated[
        Path, typer.Option(help="Folder for lineage.json and lineage.html.")
    ] = Path("out"),
    check: Annotated[
        bool,
        typer.Option(
            "--verify/--no-verify", help="Also run the empirical check and embed it."
        ),
    ] = False,
    title: Annotated[str, typer.Option(help="Heading of the report.")] = "Data lineage",
) -> None:
    """Analyse every model, write lineage.json and the interactive lineage.html."""
    project, graph = _load(pipeline)
    verification = verify(project, graph) if check else None
    out = _resolve(out)
    _write(out, "lineage.json", _json(graph.to_json()))
    html = _write(out, "lineage.html", build_html(graph, verification, title))
    n_edges = sum(len(c.deps) for ml in graph.lineage.values() for c in ml.columns)
    console.print(
        f"{len(graph.sources)} sources, {len(graph.models)} models, "
        f"{len(graph.all_nodes())} columns, {n_edges} column edges -> {html}"
    )
    for model, ml in graph.lineage.items():
        for w in ml.warnings:
            console.print(f"[yellow]warning:[/yellow] {model}: {w}")
    if verification:
        _write(out, "verification.json", _json(verification.to_json()))
        _summary(verification)
        if not verification.ok:
            raise typer.Exit(1)


@app.command()
def models(pipeline: PipelineOpt = Path("pipeline")) -> None:
    """List the sources and models in dependency order."""
    _, graph = _load(pipeline)
    table = Table()
    for c in ("table", "layer", "columns", "reads", "warnings"):
        table.add_column(c)
    for s, cols in graph.sources.items():
        table.add_row(s, "source", str(len(cols)), "", "")
    for m, ml in graph.lineage.items():
        table.add_row(
            m,
            graph.layer[m],
            str(len(ml.columns)),
            ", ".join(ml.tables),
            str(len(ml.warnings)) if ml.warnings else "",
        )
    console.print(table)


@app.command()
def trace(
    column: Annotated[str, typer.Argument(help="model.column to trace upstream.")],
    pipeline: PipelineOpt = Path("pipeline"),
    indirect: IndirectOpt = False,
) -> None:
    """Where a column's value comes from, hop by hop, down to the source columns."""
    _, graph = _load(pipeline)
    root = _node(graph, column)
    tree = Tree(f"[bold]{label(root)}[/bold]")

    def grow(branch: Tree, node: Node, seen: frozenset[Node]) -> None:
        for src, role, steps in graph.direct_up.get(node, ()):
            detail = f": {steps[0]}" if role != "copy" and steps else ""
            child = branch.add(f"{label(src)} [dim]{role}{detail}[/dim]")
            if src not in seen:
                grow(child, src, seen | {src})

    grow(tree, root, frozenset({root}))
    console.print(tree)
    sources = sorted(graph.source_columns(root, indirect=False))
    console.print(
        f"values computed from: {', '.join(label(n) for n in sources) or 'no source column'}"
    )
    if indirect:
        rows_only = sorted(graph.source_columns(root, indirect=True) - set(sources))
        console.print(
            f"rows shaped by: {', '.join(label(n) for n in rows_only) or 'none'}"
        )


@app.command()
def impact(
    column: Annotated[str, typer.Argument(help="table.column to change.")],
    pipeline: PipelineOpt = Path("pipeline"),
    indirect: IndirectOpt = False,
) -> None:
    """Every column that changes if this one changes."""
    _, graph = _load(pipeline)
    root = _node(graph, column)
    down = sorted(graph.downstream(root, indirect=indirect))
    by_model: dict[str, list[str]] = {}
    for table, col in down:
        by_model.setdefault(table, []).append(col)
    console.print(
        f"[bold]{label(root)}[/bold] reaches {len(down)} columns in {len(by_model)} models"
    )
    for model in sorted(by_model, key=lambda m: graph.rank[m]):
        console.print(f"  {model}: {', '.join(by_model[model])}")


@app.command("verify")
def verify_cmd(
    pipeline: PipelineOpt = Path("pipeline"),
    out: Annotated[Path, typer.Option(help="Folder for verification.json.")] = Path(
        "out"
    ),
    source: Annotated[
        list[str] | None,
        typer.Option(
            "--source", help="Only this source column (table.column); repeatable."
        ),
    ] = None,
    strategy: Annotated[
        list[str] | None,
        typer.Option(
            "--strategy", help=f"One of {', '.join(execute.STRATEGIES)}; repeatable."
        ),
    ] = None,
) -> None:
    """Change each source column, rerun the pipeline, and compare what moved with the lineage."""
    project, graph = _load(pipeline)
    chosen = tuple(strategy) if strategy else execute.STRATEGIES
    bad = [s for s in chosen if s not in execute.STRATEGIES]
    if bad:
        console.print(f"[red]error:[/red] unknown strategy {', '.join(bad)}")
        raise typer.Exit(2)
    only = {_node(graph, s) for s in source} if source else None
    result = verify(project, graph, chosen, only)
    _write(_resolve(out), "verification.json", _json(result.to_json()))
    _summary(result)
    if not result.ok:
        raise typer.Exit(1)
