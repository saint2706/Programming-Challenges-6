"""Command line: profile a query, compare a rewrite, or run the built-in experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from sql_profiler import compare as comparison
from sql_profiler import db, lab, render, rules
from sql_profiler.plan import profile

app = typer.Typer(
    help="Profile and explain DuckDB query plans on a real retail database.",
    no_args_is_help=True,
)
console = Console()

SqlArg = Annotated[str, typer.Argument(help="SQL text, or the path of a .sql file.")]
SEVERITY_STYLE = {"critical": "bold red", "warn": "yellow", "info": "cyan"}


def _sql(arg: str) -> str:
    """The text of ``arg`` if it names an existing file, otherwise ``arg`` itself."""
    try:
        path = Path(arg)
        if len(arg) < 260 and path.is_file():
            return path.read_text(encoding="utf-8")
    except OSError:
        pass
    return arg


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error:[/red] {message}")
    return typer.Exit(2)


@app.command()
def explain(
    sql: SqlArg,
    html: Annotated[
        Path | None, typer.Option(help="Also write a self-contained HTML report here.")
    ] = None,
    timeout: Annotated[
        float, typer.Option(help="Stop the query after this many seconds.")
    ] = 60.0,
) -> None:
    """Run a query under the profiler and explain where the time and the surprises are."""
    con = db.connect()
    try:
        plan = profile(con, _sql(sql), timeout)
    except (ValueError, TimeoutError) as exc:
        raise _fail(str(exc)) from exc
    findings = rules.diagnose(plan)
    console.print(
        f"[bold]{plan.latency_s * 1000:,.1f} ms[/bold] wall, {plan.cpu_s * 1000:,.1f} ms CPU "
        f"({plan.parallelism:.1f}x parallel), {plan.rows_returned:,} rows returned"
    )
    console.print(render.text_tree(plan), markup=False, highlight=False)
    if findings:
        console.print()
    for f in findings:
        style = SEVERITY_STYLE[f.severity]
        console.print(
            f"[{style}]{f.severity.upper()}[/{style}] {f.title}", highlight=False
        )
        console.print(
            f"    {f.detail}\n    [dim]Try:[/dim] {f.suggestion}", highlight=False
        )
    if not findings:
        console.print("\nNo findings: nothing in this plan crosses a threshold.")
    if html:
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text(render.report_html(plan, findings), encoding="utf-8")
        console.print(f"\nreport written to {html}")


@app.command()
def compare(
    a: SqlArg,
    b: SqlArg,
    repeats: Annotated[
        int, typer.Option(min=3, help="Timed runs per query (after one warm-up).")
    ] = 9,
) -> None:
    """Check that B returns A's rows, then time both and say whether the difference is real."""
    con = db.connect()
    try:
        result = comparison.compare(con, _sql(a), _sql(b), repeats=repeats)
    except (ValueError, TimeoutError) as exc:
        raise _fail(str(exc)) from exc
    console.print(
        f"rows: {result.check.reason} (A {result.check.rows_a:,}, B {result.check.rows_b:,})"
    )
    console.print(
        f"A median {result.a.median_s * 1000:,.2f} ms  (best {result.a.best_s * 1000:,.2f}, worst {result.a.worst_s * 1000:,.2f})"
    )
    console.print(
        f"B median {result.b.median_s * 1000:,.2f} ms  (best {result.b.best_s * 1000:,.2f}, worst {result.b.worst_s * 1000:,.2f})"
    )
    verdict = {
        "b_faster": f"B is faster: every run of B beat every run of A ({result.speedup:.2f}x by median)",
        "a_faster": f"A is faster: every run of A beat every run of B ({1 / result.speedup:.2f}x by median)",
        "no_clear_difference": "no clear difference: the runs overlap",
    }[result.verdict]
    console.print(verdict)
    console.print(
        "plan: "
        + (
            "the same operators on the same tables"
            if result.same_plan
            else "; ".join(comparison.plan_diff(result.plan_a, result.plan_b))
        )
    )
    console.print(
        f"rows read from tables: A {result.plan_a.rows_scanned:,}, B {result.plan_b.rows_scanned:,}"
    )


@app.command("lab")
def run_lab(
    repeats: Annotated[int, typer.Option(min=3)] = 9,
    markdown: Annotated[
        bool, typer.Option(help="Print a Markdown table instead of the terminal one.")
    ] = False,
) -> None:
    """Run every built-in rewrite experiment (see lab.py) and print the verdicts."""
    results = lab.run_lab(db.connect(), repeats)
    rows = []
    for case, c in results:
        rows.append(
            (
                case.id,
                "yes"
                if c.check.same
                else ("by design" if not case.equivalent else "NO"),
                f"{c.a.median_s * 1000:,.2f}",
                f"{c.b.median_s * 1000:,.2f}",
                f"{c.speedup:.2f}x",
                {
                    "b_faster": "B faster",
                    "a_faster": "A faster",
                    "no_clear_difference": "no clear difference",
                }[c.verdict],
                "same" if c.same_plan else "differs",
            )
        )
    head = ("case", "same rows", "A ms", "B ms", "A/B", "verdict", "plan")
    if markdown:
        console.print("| " + " | ".join(head) + " |", markup=False)
        console.print("| " + " | ".join("---" for _ in head) + " |", markup=False)
        for r in rows:
            console.print("| " + " | ".join(r) + " |", markup=False)
        return
    table = Table(title="rewrite experiments")
    for h in head:
        table.add_column(h)
    for r in rows:
        table.add_row(*r)
    console.print(table)


@app.command("index-lab")
def run_index_lab(repeats: Annotated[int, typer.Option(min=3)] = 15) -> None:
    """Time point lookups before and after CREATE INDEX."""
    table = Table(title="CREATE INDEX on invoice_lines")
    for h in (
        "lookup",
        "rows",
        "scan before",
        "scan after",
        "ms before",
        "ms after",
        "rows read before",
        "rows read after",
    ):
        table.add_column(h)
    for r in lab.index_lab(repeats=repeats):
        table.add_row(
            r.label,
            f"{r.rows:,}",
            "Seq",
            r.scan_type_with.split()[0],
            f"{r.without.median_s * 1000:.2f}",
            f"{r.with_index.median_s * 1000:.2f}",
            f"{r.scanned_without:,}",
            f"{r.scanned_with:,}",
        )
    console.print(table)


@app.command()
def schema() -> None:
    """List the tables, their columns and row counts."""
    con = db.connect()
    for (name,) in con.execute("SHOW TABLES").fetchall():
        rows = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        cols = ", ".join(
            f"{c} {t}" for c, t, *_ in con.execute(f"DESCRIBE {name}").fetchall()
        )
        console.print(f"[bold]{name}[/bold] ({rows:,} rows): {cols}", highlight=False)


if __name__ == "__main__":
    app()
