"""CLI for the multi-timezone meeting scheduler.

Run: uv run --with typer --with rich --with tzdata python cli.py find ...
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from scheduler import (
    find_overlaps,
    format_window,
    parse_participant_spec,
    search_timezones,
)

app = typer.Typer(
    add_completion=False,
    help="Find meeting times that overlap working hours across timezones.",
)
console = Console()


@app.command()
def find(
    date_str: Annotated[
        str, typer.Option("--date", help="Reference calendar date, YYYY-MM-DD")
    ],
    ref_tz: Annotated[
        str, typer.Option("--ref-tz", help="Timezone the --date is interpreted in")
    ],
    participant: Annotated[
        list[str],
        typer.Option(
            "--participant",
            "-p",
            help='Repeatable: "Name|Area/City|HH:MM-HH:MM[|days]"',
        ),
    ],
    display_tz: Annotated[
        str | None,
        typer.Option(help="Timezone to display results in (default: --ref-tz)"),
    ] = None,
    min_participants: Annotated[
        int | None,
        typer.Option(
            "--min-participants",
            help="Minimum participants required (default: everyone)",
        ),
    ] = None,
) -> None:
    """Find overlapping working-hour windows for a set of participants."""
    try:
        ref_date = date.fromisoformat(date_str)
    except ValueError as exc:
        console.print(f"[red]Invalid --date {date_str!r}: {exc}[/red]")
        raise typer.Exit(1) from exc

    try:
        participants = [parse_participant_spec(spec) for spec in participant]
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    display = display_tz or ref_tz

    try:
        windows = find_overlaps(participants, ref_date, ref_tz, min_participants)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not windows:
        console.print(
            "[yellow]No overlapping window found for the given constraints.[/yellow]"
        )
        raise typer.Exit(0)

    table = Table(title=f"Overlaps on {ref_date} ({ref_tz}), shown in {display}")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Duration")
    table.add_column("Available")
    for w in windows:
        start_str, end_str = format_window(w, display)
        table.add_row(
            start_str,
            end_str,
            f"{w.duration_minutes():.0f} min",
            ", ".join(w.available),
        )
    console.print(table)


@app.command()
def zones(
    query: Annotated[
        str, typer.Argument(help="Substring to search IANA timezone names for")
    ],
) -> None:
    """Search available IANA timezone names (helper for building --participant specs)."""
    matches = search_timezones(query, limit=30)
    if not matches:
        console.print(f"[yellow]No timezones matching {query!r}.[/yellow]")
        raise typer.Exit(0)
    for m in matches:
        console.print(m)


if __name__ == "__main__":
    app()
