"""CLI for the offline dictionary/thesaurus.

Run: uv run --with wn --with rapidfuzz --with typer --with rich python cli.py define run

First-run setup (one-time, needs internet, ~13MB):
    uv run --with wn python -c "import wn; wn.download('oewn:2021')"
"""

from __future__ import annotations

from typing import Annotated

import typer
from lexicon import (
    LexiconNotInstalledError,
    antonyms,
    define,
    exists,
    pos_name,
    search,
    suggest,
    synonyms,
)
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    add_completion=False,
    help="Offline dictionary and thesaurus, backed by the Open English WordNet.",
)
console = Console()
err_console = Console(stderr=True)


def _print_suggestions(word: str) -> None:
    hits = suggest(word)
    if hits:
        err_console.print(f"[yellow]{word!r} not found. Did you mean:[/yellow]")
        for hit in hits:
            err_console.print(f"  [cyan]{hit}[/cyan]")
    else:
        err_console.print(f"[yellow]{word!r} not found, and no close matches.[/yellow]")


def _fail_lexicon_missing(exc: LexiconNotInstalledError) -> typer.Exit:
    err_console.print(f"[red]{exc}[/red]")
    return typer.Exit(2)


@app.command("define")
def define_cmd(
    word: Annotated[str, typer.Argument(help="Word to define", metavar="WORD")],
) -> None:
    """Show every sense (definition + examples), grouped by part of speech."""
    try:
        if not exists(word):
            _print_suggestions(word)
            raise typer.Exit(1)
        senses = define(word)
    except LexiconNotInstalledError as exc:
        raise _fail_lexicon_missing(exc) from exc

    by_pos: dict[str, list] = {}
    for sense in senses:
        by_pos.setdefault(sense.pos, []).append(sense)

    console.print(f"[bold]{word}[/bold]")
    for pos, pos_senses in by_pos.items():
        console.print(f"\n[bold cyan]{pos_name(pos)}[/bold cyan]")
        for i, sense in enumerate(pos_senses, start=1):
            console.print(f"  {i}. {sense.definition}")
            for example in sense.examples:
                console.print(f'     [dim]"{example}"[/dim]')


@app.command("synonyms")
def synonyms_cmd(
    word: Annotated[
        str, typer.Argument(help="Word to find synonyms for", metavar="WORD")
    ],
) -> None:
    """List synonyms pooled across every sense of WORD."""
    try:
        if not exists(word):
            _print_suggestions(word)
            raise typer.Exit(1)
        syns = synonyms(word)
    except LexiconNotInstalledError as exc:
        raise _fail_lexicon_missing(exc) from exc

    if not syns:
        console.print(f"[yellow]No synonyms recorded for {word!r}.[/yellow]")
        raise typer.Exit(0)
    console.print(Panel(", ".join(syns), title=f"Synonyms of {word}"))


@app.command("antonyms")
def antonyms_cmd(
    word: Annotated[
        str, typer.Argument(help="Word to find antonyms for", metavar="WORD")
    ],
) -> None:
    """List antonyms pooled across every sense of WORD."""
    try:
        if not exists(word):
            _print_suggestions(word)
            raise typer.Exit(1)
        ants = antonyms(word)
    except LexiconNotInstalledError as exc:
        raise _fail_lexicon_missing(exc) from exc

    if not ants:
        console.print(f"[yellow]No antonyms recorded for {word!r}.[/yellow]")
        raise typer.Exit(0)
    console.print(Panel(", ".join(ants), title=f"Antonyms of {word}"))


@app.command("search")
def search_cmd(
    query: Annotated[
        str, typer.Argument(help="Substring or approximate spelling to search for")
    ],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max results")] = 20,
) -> None:
    """Fuzzy/substring search across the whole vocabulary."""
    try:
        hits = search(query, limit=limit)
    except LexiconNotInstalledError as exc:
        raise _fail_lexicon_missing(exc) from exc

    if not hits:
        console.print(f"[yellow]No matches for {query!r}.[/yellow]")
        raise typer.Exit(0)
    for hit in hits:
        console.print(hit)


if __name__ == "__main__":
    app()
