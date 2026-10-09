"""``cronwatch``: a crontab editor with run history, missed-run detection and failure alerts."""

import os
import socket
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from cronwatch import display
from cronwatch.alerts import Alert, Alerter, build_channels
from cronwatch.config import Config, ConfigError, load_config
from cronwatch.crontab import (
    Backend,
    CrontabError,
    FileCrontab,
    Session,
    SystemCrontab,
    default_wrapper,
)
from cronwatch.monitor import check_missed
from cronwatch.paths import project_root
from cronwatch.runner import run_wrapped
from cronwatch.store import Store
from cronwatch.timeline import build_timeline, render_timeline
from cronwatch.timeutil import schedule_zone

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=False,
    help=__doc__,
)

CrontabOption = Annotated[
    Path | None,
    typer.Option(
        "--crontab",
        envvar="CRONWATCH_CRONTAB",
        help="Edit this file instead of the user's crontab.",
    ),
]
UserOption = Annotated[
    str | None,
    typer.Option(
        "--user",
        "-u",
        help="Whose crontab to use (needs privileges), via `crontab -u`.",
    ),
]


def _fail(message: str, code: int = 2) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


def home() -> Path:
    path = project_root()
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_settings() -> Config:
    try:
        return load_config(home() / "config.toml")
    except ConfigError as exc:
        raise _fail(str(exc)) from exc


def make_backend(crontab: Path | None, user: str | None) -> Backend:
    return FileCrontab(crontab) if crontab else SystemCrontab(user)


def open_session(crontab: Path | None, user: str | None) -> Session:
    try:
        return Session(make_backend(crontab, user), home() / "backups")
    except CrontabError as exc:
        raise _fail(str(exc)) from exc


def make_alerter(config: Config, store: Store | None) -> Alerter:
    return Alerter(build_channels(config, os.environ), store)


@app.callback()
def main(
    ctx: typer.Context, crontab: CrontabOption = None, user: UserOption = None
) -> None:
    """With no command, opens the interactive TUI."""
    if ctx.invoked_subcommand is None:
        launch_tui(crontab, user)


def launch_tui(crontab: Path | None, user: str | None) -> None:
    from cronwatch.app import CronWatchApp

    config = load_settings()
    store = Store(home() / "cronwatch.db")
    try:
        tz = schedule_zone(config.timezone)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    CronWatchApp(
        open_session(crontab, user),
        store,
        config,
        tz,
        make_alerter(config, store),
        default_wrapper(),
    ).run()


@app.command("tui")
def tui(crontab: CrontabOption = None, user: UserOption = None) -> None:
    """Open the interactive crontab editor and run-history browser."""
    launch_tui(crontab, user)


@app.command(context_settings={"allow_interspersed_args": False})
def run(
    command: Annotated[
        list[str], typer.Argument(help="The command to run, after `--`.")
    ],
    job: Annotated[
        str, typer.Option("--job", "-j", help="Job id used in history and alerts.")
    ],
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            "-t",
            min=0.1,
            help="Kill the job after this many seconds (exit 124).",
        ),
    ] = None,
    ok_codes: Annotated[
        str,
        typer.Option(
            "--ok-codes", help="Comma-separated exit codes that count as success."
        ),
    ] = "0",
) -> None:
    """Run COMMAND like cron would and record the result: `cronwatch run --job backup -- ./backup.sh`.

    Output and exit status pass straight through. Failures of cronwatch itself (database, alert
    channels) are warnings and never change the job's exit status.
    """
    try:
        codes = frozenset(int(part) for part in ok_codes.split(",") if part.strip())
    except ValueError as exc:
        raise _fail("--ok-codes must be comma-separated integers") from exc
    # Everything about cronwatch's own state is optional here: whatever breaks, the job still runs.
    config, store = Config(), None
    try:
        base = home()
    except OSError as exc:
        print(
            f"cronwatch: warning: history and alerts are unavailable: {exc}",
            file=sys.stderr,
        )
        base = None
    if base is not None:
        try:
            config = load_config(base / "config.toml")
        except ConfigError as exc:
            print(f"cronwatch: warning: ignoring config: {exc}", file=sys.stderr)
        try:
            store = Store(base / "cronwatch.db")
        except Exception as exc:  # noqa: BLE001
            print(f"cronwatch: warning: history is unavailable: {exc}", file=sys.stderr)
    code = run_wrapped(
        command,
        job,
        store,
        config,
        make_alerter(config, store),
        timeout=timeout,
        ok_codes=codes or frozenset({0}),
    )
    raise typer.Exit(code)


def _table(*columns: str) -> Table:
    table = Table(show_edge=False, header_style="bold", pad_edge=False)
    for column in columns:
        table.add_column(column, overflow="fold")
    return table


@app.command("list")
def list_jobs(crontab: CrontabOption = None, user: UserOption = None) -> None:
    """Show every job with its schedule in plain English, next run and last result."""
    config = load_settings()
    session = open_session(crontab, user)
    store = Store(home() / "cronwatch.db")
    tz, now, last = schedule_zone(config.timezone), datetime.now(UTC), store.last_runs()
    table = _table("", "Job", "Schedule", "Meaning", "Next run", "Last run", "Command")
    for job in session.crontab.jobs():
        text, style = (
            display.last_run_text(last.get(job.job_id or ""), now, tz)
            if job.monitored
            else ("not monitored", "dim")
        )
        meaning = job.error or (job.expr.describe() if job.expr else "")
        table.add_row(
            "●" if job.enabled else "○",
            job.label,
            job.schedule,
            meaning,
            display.next_run_text(job, now, tz),
            f"[{style}]{text}[/]" if style else text,
            job.command,
        )
    Console().print(table if session.crontab.jobs() else "No jobs in this crontab.")


@app.command()
def history(
    job: Annotated[
        str | None, typer.Option("--job", "-j", help="Only this job id.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1)] = 20,
    show: Annotated[
        int | None,
        typer.Option("--run", help="Print the captured output of this run id."),
    ] = None,
) -> None:
    """Recent runs, newest first; `--run ID` prints one run's captured output."""
    store = Store(home() / "cronwatch.db")
    config = load_settings()
    tz = schedule_zone(config.timezone)
    if show is not None:
        found = store.get_run(show)
        if found is None:
            raise _fail(f"no run with id {show}", 1)
        typer.echo(
            f"# {found.job_id} on {found.host}, {display.local(found.started_at, tz)}, status {found.status}, exit {found.exit_code}"
        )
        typer.echo(found.output_tail, nl=not found.output_tail.endswith("\n"))
        return
    table = _table("Id", "Job", "Started", "Status", "Exit", "Took", "Host")
    for r in store.runs(job, limit):
        glyph, style = display.STATUS_GLYPHS.get(r.status, ("?", ""))
        table.add_row(
            str(r.id),
            r.job_id,
            display.local(r.started_at, tz),
            f"[{style}]{glyph} {r.status}[/]",
            "—" if r.exit_code is None else str(r.exit_code),
            display.duration(r.duration_s),
            r.host,
        )
    Console().print(table)


@app.command()
def timeline(
    crontab: CrontabOption = None,
    user: UserOption = None,
    past: Annotated[float, typer.Option(help="Hours of history to show.", min=0.5)] = 6,
    future: Annotated[
        float, typer.Option(help="Hours of schedule to show.", min=0.5)
    ] = 18,
    width: Annotated[
        int, typer.Option(help="Columns for the time axis.", min=24, max=400)
    ] = 72,
) -> None:
    """Draw what ran and what is about to, one row per job."""
    config = load_settings()
    session = open_session(crontab, user)
    store = Store(home() / "cronwatch.db")
    now = datetime.now(UTC)
    jobs = session.crontab.jobs()
    runs = {
        j.job_id: store.runs(j.job_id, 500, now - timedelta(hours=past))
        for j in jobs
        if j.job_id
    }
    since = {
        j.job_id: s.monitor_cursor
        for j in jobs
        if j.job_id and (s := store.get_state(j.job_id)).monitor_cursor
    }
    result = build_timeline(
        jobs,
        runs,
        now,
        schedule_zone(config.timezone),
        width,
        timedelta(hours=past),
        timedelta(hours=future),
        config.grace,
        monitored_since=since,
    )
    Console().print(render_timeline(result), soft_wrap=True)


@app.command()
def check(
    crontab: CrontabOption = None,
    user: UserOption = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report missed runs without alerting.")
    ] = False,
) -> None:
    """Find monitored jobs that should have started but left no run, and alert. Run it from cron every few minutes."""
    config = load_settings()
    session = open_session(crontab, user)
    store = Store(home() / "cronwatch.db")
    missed = check_missed(
        session.crontab.jobs(),
        store,
        make_alerter(config, store),
        datetime.now(UTC),
        config.grace,
        schedule_zone(config.timezone),
        notify=not dry_run,
    )
    for item in missed:
        typer.echo(
            f"{item.job_id}: {len(item.instants)} missed run(s), latest scheduled for {item.instants[-1]:%Y-%m-%d %H:%M} UTC"
        )
    if not missed:
        typer.echo("No missed runs.")
    raise typer.Exit(1 if missed else 0)


@app.command("test-alert")
def test_alert() -> None:
    """Send a test alert through every configured channel and say which worked."""
    config = load_settings()
    store = Store(home() / "cronwatch.db")
    alerter = make_alerter(config, store)
    if not alerter.channels:
        raise _fail(
            f"no alert channels are configured; add [alerts.webhook], [alerts.email] or [alerts.desktop] to {home() / 'config.toml'}",
            1,
        )
    results = alerter.dispatch(
        Alert(
            "test",
            "cronwatch-test",
            socket.gethostname(),
            datetime.now(UTC),
            "echo test",
            status="failed",
            exit_code=1,
        )
    )
    for result in results:
        typer.echo(
            f"{result.channel}: "
            + ("sent" if result.ok else f"FAILED - {result.detail}")
        )
    raise typer.Exit(0 if all(r.ok for r in results) else 1)


@app.command("install-monitor")
def install_monitor(
    crontab: CrontabOption = None,
    user: UserOption = None,
    every: Annotated[
        int, typer.Option(min=1, max=59, help="Minutes between checks.")
    ] = 5,
) -> None:
    """Add a crontab entry that runs `cronwatch check`, plus CRONWATCH_HOME so cron finds the same history."""
    session = open_session(crontab, user)
    if any(
        "cronwatch" in j.command and " check" in j.command
        for j in session.crontab.jobs()
    ):
        typer.echo("A `cronwatch check` entry is already installed.")
        return
    wrapper = default_wrapper()

    def add(c):
        c.set_env("CRONWATCH_HOME", str(home()))
        c.add_job(
            f"*/{every} * * * *", f"{wrapper} check", "cronwatch missed-run monitor"
        )

    try:
        session.apply(add)
    except CrontabError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"Installed: every {every} minutes, cronwatch check")
