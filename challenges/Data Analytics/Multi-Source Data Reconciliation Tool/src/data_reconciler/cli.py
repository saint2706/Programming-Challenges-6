"""Command line: run a reconciliation job, or look up one key."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from data_reconciler import compare as cmp
from data_reconciler.config import ConfigError, load
from data_reconciler.paths import project_root
from data_reconciler.reconcile import VERDICTS, lookup, reconcile
from data_reconciler.report import write_outputs

app = typer.Typer(
    help="Reconcile two datasets on a key and categorize every difference.",
    no_args_is_help=True,
)
console = Console()
ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", help="The reconciliation job (TOML).")
]


def _job(config: Path):
    path = (
        config if config.is_absolute() or config.exists() else project_root() / config
    )
    try:
        return reconcile(load(path))
    except (ConfigError, FileNotFoundError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc


@app.command()
def run(
    config: ConfigOpt = Path("airports.toml"),
    out: Annotated[
        Path, typer.Option(help="Folder for the CSV, JSON and HTML outputs.")
    ] = Path("out"),
    no_files: Annotated[
        bool, typer.Option("--no-files", help="Print the summary only.")
    ] = False,
) -> None:
    """Match, compare, print the summary and write pairs, differences, unmatched rows and an HTML report."""
    rec = _job(config)
    s, cfg = rec.summary, rec.config
    console.print(
        f"[bold]{cfg.left.name}[/bold] {s['left']['rows']:,} rows vs [bold]{cfg.right.name}[/bold] {s['right']['rows']:,} rows"
    )
    rows = Table(title="where every row went")
    rows.add_column("status")
    rows.add_column(cfg.left.name, justify="right")
    rows.add_column(cfg.right.name, justify="right")
    for status in ("matched", "only_left", "only_right", "duplicate_key", "unkeyed"):
        rows.add_row(
            status,
            f"{s['left']['status'].get(status, 0):,}",
            f"{s['right']['status'].get(status, 0):,}",
        )
    console.print(rows)
    console.print(
        "matched by key: "
        + ", ".join(f"{n:,} by {k}" for k, n in s["matched_by_key"].items())
    )
    console.print(
        "pairs: "
        + ", ".join(f"{s['verdicts'][v]:,} {v.replace('_', ' ')}" for v in VERDICTS)
    )
    fields = Table(title="fields of matched pairs that are the same entity")
    shown = [
        c
        for c in cmp.CATEGORIES
        if any(s["fields_same_entity"][f][c] for f in s["fields_same_entity"])
    ]
    fields.add_column("field")
    for c in shown:
        fields.add_column(c, justify="right")
    for name, counts in s["fields_same_entity"].items():
        fields.add_row(name, *(f"{counts[c]:,}" for c in shown))
    console.print(fields)
    if not no_files:
        paths = write_outputs(rec, out)
        console.print(f"wrote {', '.join(p.name for p in paths.values())} to {out}")


@app.command()
def show(
    key: Annotated[str, typer.Argument(help="A key value, e.g. an IATA code.")],
    config: ConfigOpt = Path("airports.toml"),
) -> None:
    """Everything known about one key: how its pair compares, or why it has none."""
    rec = _job(config)
    found = lookup(rec, key)
    pair = found["pair"]
    if pair:
        table = Table(
            title=f"{pair['key']}: matched by key #{pair['stage'] + 1}, verdict {pair['verdict']}"
        )
        for c in (
            "field",
            rec.config.left.name,
            rec.config.right.name,
            "category",
            "metric",
            "note",
        ):
            table.add_column(c)
        for spec in rec.config.fields:
            d = pair[f"{spec.name}__metric"]
            table.add_row(
                spec.name,
                str(pair[f"{spec.name}__left"] or ""),
                str(pair[f"{spec.name}__right"] or ""),
                pair[f"{spec.name}__category"],
                "" if d is None else f"{d:,.3f}",
                str(pair[f"{spec.name}__note"] or ""),
            )
        console.print(table)
        return
    if not found["rows"]:
        console.print(f"no row in either source carries {key!r}")
        raise typer.Exit(1)
    for row in found["rows"]:
        values = {k: v for k, v in row["values"].items() if v is not None}
        console.print(
            f"[bold]{row['side']}[/bold] ({row['status']}): {values}", highlight=False
        )


if __name__ == "__main__":
    app()
