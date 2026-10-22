"""Command line: write a report for a bundled table or any file, list findings, show the checks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from report_generator import fetch as fetch_mod
from report_generator.config import Config
from report_generator.data import DATASETS, data_dir, load
from report_generator.engine import Report, build_report
from report_generator.paths import project_root
from report_generator.render import to_html, to_json, to_markdown

app = typer.Typer(
    help="Turn a table into a short, verified, plain-English report of its key findings.",
    no_args_is_help=True,
)
console = Console()

Source = Annotated[
    str,
    typer.Argument(help="bike, retail, ames, or the path of a CSV/TSV/Parquet file."),
]
TimeCol = Annotated[
    str | None, typer.Option("--time", help="The date column (found automatically).")
]
Ignore = Annotated[
    list[str] | None,
    typer.Option("--ignore", help="Leave this column out; repeatable."),
]
SumCols = Annotated[
    list[str] | None,
    typer.Option("--sum", help="Treat as additive (sum when coarsening)."),
]
MeanCols = Annotated[
    list[str] | None,
    typer.Option("--mean", help="Treat as a level (average when coarsening)."),
]
RowLabel = Annotated[
    str | None, typer.Option("--rows-label", help="What one row is, e.g. 'sales'.")
]
Title = Annotated[str | None, typer.Option("--title", help="Report title.")]
MaxFindings = Annotated[
    int, typer.Option("--max-findings", "-n", help="Headline findings to keep.")
]
NoVerify = Annotated[
    bool, typer.Option("--no-verify", help="Skip the DuckDB re-derivation.")
]
OutDir = Annotated[
    Path,
    typer.Option(
        "--out", "-o", help="Folder for report.md, report.html and report.json."
    ),
]


def _build(
    source: str,
    *,
    time: str | None,
    ignore: list[str] | None,
    sum_cols: list[str] | None,
    mean_cols: list[str] | None,
    rows_label: str | None,
    title: str | None,
    max_findings: int,
    verify: bool,
) -> Report:
    try:
        spec, frame = load(source)
        return build_report(
            frame,
            title=title or (spec.title if spec else Path(source).stem),
            config=replace(Config(), max_findings=max_findings),
            time=time or (spec.time if spec else None),
            ignore=tuple(ignore or ()) + (spec.ignore if spec else ()),
            sum_columns=tuple(sum_cols or ()) + (spec.sum_columns if spec else ()),
            mean_columns=tuple(mean_cols or ()) + (spec.mean_columns if spec else ()),
            row_label=rows_label or (spec.row_label if spec else "rows"),
            verify=verify,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc


@app.command()
def datasets() -> None:
    """List the bundled tables."""
    table = Table(title="Bundled tables")
    for col in ("name", "rows are", "title", "note"):
        table.add_column(col)
    for d in DATASETS.values():
        missing = "" if (data_dir() / d.file).exists() else " [missing]"
        table.add_row(d.name, d.row_label, d.title + missing, d.note)
    console.print(table)


@app.command()
def fetch() -> None:
    """Rebuild bike.parquet and ames.parquet from their public sources (needs network)."""
    root = project_root()
    rows = fetch_mod.fetch_all(root / "sample_data", root.parents[1])
    fetch_mod.write_sources(root / "sample_data", rows)
    for name, n in rows.items():
        console.print(f"wrote {name} ({n:,} rows)")


@app.command()
def generate(
    source: Source,
    time: TimeCol = None,
    title: Title = None,
    rows_label: RowLabel = None,
    ignore: Ignore = None,
    sum_cols: Annotated[
        list[str] | None, typer.Option("--sum", help="Additive column; repeatable.")
    ] = None,
    mean_cols: Annotated[
        list[str] | None, typer.Option("--mean", help="Level column; repeatable.")
    ] = None,
    max_findings: MaxFindings = 12,
    no_verify: NoVerify = False,
    out: OutDir = Path("out"),
) -> None:
    """Write report.md, report.html and report.json for a table."""
    report = _build(
        source, time=time, ignore=ignore, sum_cols=sum_cols, mean_cols=mean_cols, rows_label=rows_label,
        title=title, max_findings=max_findings, verify=not no_verify,
    )  # fmt: skip
    out = out if out.is_absolute() else project_root() / out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(to_markdown(report), encoding="utf-8", newline="\n")
    (out / "report.html").write_text(to_html(report), encoding="utf-8", newline="\n")
    (out / "report.json").write_text(to_json(report), encoding="utf-8", newline="\n")
    console.print(
        f"[bold]{report.title}[/bold]: {len(report.findings)} findings and {len(report.notes)} data notes"
    )
    for f in report.findings:
        console.print(f"  - {f.text}")
    if report.dropped:
        console.print(
            f"[yellow]{len(report.dropped)} statements failed a check and were left out[/yellow]"
        )
    console.print(f"wrote {out / 'report.html'} (plus .md and .json)")


@app.command()
def findings(
    source: Source,
    time: TimeCol = None,
    rows_label: RowLabel = None,
    ignore: Ignore = None,
    all_candidates: Annotated[
        bool, typer.Option("--all", help="Include candidates that were not selected.")
    ] = False,
) -> None:
    """List findings with their scores, as selected for the report or (--all) every candidate."""
    report = _build(
        source, time=time, ignore=ignore, sum_cols=None, mean_cols=None, rows_label=rows_label,
        title=None, max_findings=12, verify=False,
    )  # fmt: skip
    chosen = {f.id for f in report.findings}
    table = Table(
        title=f"{report.title}: {report.candidates} candidates, {len(chosen)} selected"
    )
    for col in ("score", "id", "section", "picked", "sentence"):
        table.add_column(col, overflow="fold")
    for f in report.candidate_findings if all_candidates else report.findings:
        table.add_row(
            f"{f.score:.2f}", f.id, f.section, "yes" if f.id in chosen else "", f.text
        )
    console.print(table)


@app.command()
def verify(
    source: Source,
    time: TimeCol = None,
    rows_label: RowLabel = None,
    ignore: Ignore = None,
) -> None:
    """Show each statement next to the numbers recomputed from the raw table."""
    report = _build(
        source, time=time, ignore=ignore, sum_cols=None, mean_cols=None, rows_label=rows_label,
        title=None, max_findings=12, verify=True,
    )  # fmt: skip
    table = Table(title=f"{report.title}: re-derived with DuckDB and scipy")
    for col in ("statement", "check", "report", "recomputed", ""):
        table.add_column(col, overflow="fold")
    by_id = {f.id: f for f in [report.overview, *report.findings, *report.notes]}
    for fid, checks in report.checks.items():
        for i, c in enumerate(checks):
            table.add_row(
                by_id[fid].id if i == 0 else "", c.name, f"{c.claimed:,.6g}", f"{c.recomputed:,.6g}",
                "[green]ok[/green]" if c.ok else "[red]DIFFERS[/red]",
            )  # fmt: skip
    console.print(table)
    for d in report.dropped:
        console.print(f"[yellow]left out:[/yellow] {d['id']} ({d['reason']})")
