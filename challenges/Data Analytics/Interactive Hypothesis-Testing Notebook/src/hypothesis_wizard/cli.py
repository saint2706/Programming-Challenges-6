"""Command line: ``hypo datasets`` lists the bundled tables; ``hypo ask`` runs the wizard once.

Output is plain text (no colour or table layout), so it reads the same in a pipe or a log.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer

from hypothesis_wizard import data, methods, wizard

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Pick, run and read a hypothesis test from a table and a research question.",
)


def _frame(source: str) -> pl.DataFrame:
    path = Path(source)
    if path.suffix.lower() == ".csv" and path.exists():
        return data.read_table(path)
    return data.load(source)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def _result_line(result: methods.TestResult) -> str:
    text = f"{result.test}: {result.statistic_name} = {_fmt(result.statistic)}"
    if result.df is not None:
        text += f", df = {_fmt(result.df)}"
    text += f", p = {_fmt(result.p_value)}"
    if result.effect is not None:
        text += f"; {result.effect_name} = {_fmt(result.effect)}"
    if result.ci is not None:
        text += f"; {result.ci_name} [{_fmt(result.ci[0])}, {_fmt(result.ci[1])}]"
    return text


def _print_report(report: wizard.Report, session: wizard.Session) -> None:
    plan = report.plan
    typer.echo(f"Question: {report.kind}  groups: {', '.join(report.names)}")
    typer.echo("Why this test:")
    for step in plan.steps:
        typer.echo(f"  - {step.branch}: {step.reason}")
    for warning in plan.warnings:
        typer.echo(f"  ! {warning}")
    typer.echo(f"Primary: {_result_line(report.primary)}")
    if plan.alternatives:
        typer.echo(
            f"Alternatives (run with the session counter): {', '.join(plan.alternatives)}"
        )
    if report.posthoc:
        typer.echo("Post-hoc comparisons:")
        for row in report.posthoc:
            p_value = row.get("p", row.get("p_raw"))
            typer.echo(f"  {row['a']} vs {row['b']}: adjusted p = {_fmt(p_value)}")
    typer.echo(
        f"Significance threshold for the primary test: {_fmt(report.threshold)} "
        f"(alpha / {len(session.tried)} tests tried)"
    )
    if report.mde is not None:
        typer.echo(
            f"Minimum detectable difference at 80% power: {_fmt(report.mde, 3)} SD"
        )
    typer.echo(f"Verdict: {report.verdict}")


@app.command()
def datasets() -> None:
    """List the bundled sample tables and their licences."""
    for key, ds in data.DATASETS.items():
        typer.echo(f"{key}: {ds.title} [{ds.license}]")


@app.command()
def ask(
    source: str = typer.Argument(
        ..., help="Dataset key (see `datasets`) or a CSV path."
    ),
    outcome: str = typer.Option(..., "--outcome", "-o", help="Outcome column."),
    group: str | None = typer.Option(
        None, "--group", "-g", help="Group or second column."
    ),
    pair: str | None = typer.Option(None, "--pair", help="Subject ID column."),
    paired: bool = typer.Option(
        False, "--paired", help="Analyse as paired measurements."
    ),
    stratum: str | None = typer.Option(
        None, "--stratum", help="Stratum column (2x2 tables)."
    ),
    ordinal: list[str] = typer.Option(  # noqa: B008
        [], "--ordinal", help="Treat a column as ordinal."
    ),
    kind: str | None = typer.Option(
        None,
        "--kind",
        help="compare, association or correlation (inferred if omitted).",
    ),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level."),
    alternative: list[str] = typer.Option(  # noqa: B008
        [], "--alternative", help="Also run this listed alternative (repeatable)."
    ),
) -> None:
    """Run the wizard on one table and print the plan, results and verdict."""
    df = _frame(source)
    spec = wizard.Spec(
        outcome=outcome,
        group=group,
        pair=pair,
        paired=paired,
        stratum=stratum,
        kind=kind,
        ordinal=tuple(ordinal),
        alpha=alpha,
    )
    session = wizard.Session(alpha=alpha)
    try:
        report = wizard.analyse(df, spec, session)
        _print_report(report, session)
        for test in alternative:
            result = wizard.run_alternative(report, test, session, spec)
            typer.echo(
                f"Alternative {_result_line(result)}; threshold now "
                f"{_fmt(session.threshold)} ({len(session.tried)} tests tried)"
            )
    except (ValueError, KeyError, pl.exceptions.PolarsError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()
