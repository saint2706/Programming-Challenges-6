"""Command line: one interval, a backfill, one scheduler tick, the run log and the HTML report.

Typer was chosen over argparse because the commands are typed functions with options and help text
generated from their signatures, and ``typer.testing.CliRunner`` makes them testable in-process.
Every command is a thin wrapper: the work lives in ``pipeline`` and ``report``.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import typer

from star_warehouse import report
from star_warehouse.paths import project_root
from star_warehouse.pipeline import (
    backfill as run_backfill,
)
from star_warehouse.pipeline import (
    default_feed,
    default_lookup,
    run_interval,
)
from star_warehouse.pipeline import (
    tick as run_tick,
)
from star_warehouse.tasks import Batch, build_tasks
from star_warehouse.warehouse import connect, parse_interval

app = typer.Typer(
    help="Star-schema warehouse builder and DAG orchestrator (Online Retail II).",
    no_args_is_help=True,
)

DbOption = Annotated[
    Path | None,
    typer.Option(help="DuckDB file. Default: out/warehouse.duckdb in the project."),
]
FeedOption = Annotated[
    Path | None,
    typer.Option(
        help="Feed parquet. Default: data/feed.parquet if fetched, else the sample."
    ),
]


def _db(db: Path | None) -> Path:
    return db or project_root() / "out" / "warehouse.duckdb"


def _feed(feed: Path | None) -> Path:
    return feed or default_feed()


def _check_feed(feed: Path) -> None:
    if not feed.exists():
        raise typer.BadParameter(f"feed not found: {feed} (run scripts/fetch_data.py)")


@app.command()
def run(
    interval: Annotated[
        str,
        typer.Argument(
            help="YYYY-MM (calendar month) or YYYY-MM-DD/YYYY-MM-DD ([start, end))."
        ),
    ],
    db: DbOption = None,
    feed: FeedOption = None,
) -> None:
    """Load the feed rows that arrived in INTERVAL and bring the star schema up to date."""
    try:
        start, end = parse_interval(interval)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    feed_path = _feed(feed)
    _check_feed(feed_path)
    outcome = run_interval(_db(db), feed_path, default_lookup(), start, end)
    typer.echo(f"run {outcome.run_id[:8]} {start} to {end}: {outcome.status}")
    for name in outcome.failed_tasks:
        typer.echo(
            f"  failed: {name}: {outcome.results[name].attempts[-1].error}", err=True
        )
    if outcome.status != "succeeded":
        raise typer.Exit(1)


@app.command()
def backfill(
    first: Annotated[str, typer.Argument(help="First month, YYYY-MM.")],
    last_exclusive: Annotated[
        str, typer.Argument(help="Month after the last one, YYYY-MM.")
    ],
    db: DbOption = None,
    feed: FeedOption = None,
) -> None:
    """Run every calendar month from FIRST up to LAST_EXCLUSIVE, oldest first, stopping at a failure."""
    try:
        first_day, _ = parse_interval(first)
        last_day, _ = parse_interval(last_exclusive)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    feed_path = _feed(feed)
    _check_feed(feed_path)
    outcomes = run_backfill(_db(db), feed_path, default_lookup(), first_day, last_day)
    for outcome in outcomes:
        typer.echo(f"{outcome.start} to {outcome.end}: {outcome.status}")
    if not outcomes or outcomes[-1].status != "succeeded":
        raise typer.Exit(1)


@app.command()
def tick(
    expr: Annotated[
        str, typer.Argument(help="Five-field cron expression, at most one fire a day.")
    ],
    name: Annotated[
        str, typer.Option(help="Schedule name in ops.schedule_state.")
    ] = "scheduled",
    now: Annotated[
        str | None,
        typer.Option(
            help="Pretend the current time is this ISO timestamp (for repeatable runs)."
        ),
    ] = None,
    db: DbOption = None,
    feed: FeedOption = None,
) -> None:
    """Run every due fire of EXPR that has not run yet. Call it from cron or Task Scheduler."""
    try:
        moment = datetime.fromisoformat(now) if now else datetime.now()
    except ValueError as exc:
        raise typer.BadParameter(f"--now: {exc}") from exc
    feed_path = _feed(feed)
    _check_feed(feed_path)
    try:
        outcomes = run_tick(
            _db(db), feed_path, default_lookup(), expr, moment, name=name
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not outcomes:
        typer.echo("nothing due")
    for outcome in outcomes:
        typer.echo(f"{outcome.start} to {outcome.end}: {outcome.status}")
    if any(o.status != "succeeded" for o in outcomes):
        raise typer.Exit(1)


@app.command()
def status(
    limit: Annotated[int, typer.Option(min=1, help="How many runs to show.")] = 10,
    db: DbOption = None,
) -> None:
    """Show the most recent runs from the run log."""
    path = _db(db)
    if not path.exists():
        raise typer.BadParameter(f"no warehouse yet at {path}")
    con = connect(path)
    try:
        rows = con.execute(
            "SELECT run_id, interval_start, interval_end, trigger, status FROM ops.run "
            "ORDER BY started_at DESC LIMIT ?",
            [limit],
        ).fetchall()
    finally:
        con.close()
    if not rows:
        typer.echo("no runs yet")
    for run_id, start, end, trigger, state in rows:
        typer.echo(f"{run_id[:8]}  {start} to {end}  {trigger:<16} {state}")


@app.command("report")
def report_cmd(
    out: Annotated[Path | None, typer.Option(help="HTML file to write.")] = None,
    db: DbOption = None,
    feed: FeedOption = None,
) -> None:
    """Write the self-contained HTML run and lineage report."""
    path = _db(db)
    if not path.exists():
        raise typer.BadParameter(f"no warehouse yet at {path}")
    con = connect(path)
    try:
        # The graph is fixed; the batch bounds only matter when a task runs, and none do here.
        batch = Batch(
            con=con,
            feed=_feed(feed),
            lookup=default_lookup(),
            start=date(2009, 1, 1),
            end=date(2009, 1, 2),
        )
        target = out or path.parent / "report.html"
        written = report.write(
            con, build_tasks(batch), target, "Star-schema warehouse run report"
        )
    finally:
        con.close()
    typer.echo(f"wrote {written}")


if __name__ == "__main__":
    app()
