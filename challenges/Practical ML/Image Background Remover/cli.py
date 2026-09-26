"""Batch background-removal CLI, backed by rembg's pretrained matting models.

Run: uv run --with "rembg[cpu]" --with typer python cli.py PHOTOS/ out/
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from remove_bg import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    process_batch,
    resolve_input_files,
)
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

app = typer.Typer(
    add_completion=False,
    help="Remove backgrounds from a batch of images using a pretrained rembg model.",
)
console = Console()
err_console = Console(stderr=True)


@app.command()
def remove(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            help="An image file, or a directory of images to process.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory to write background-removed PNGs to."),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help=f"rembg model to use. One of: {', '.join(AVAILABLE_MODELS)}.",
        ),
    ] = DEFAULT_MODEL,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive/--no-recursive",
            help="When INPUT_PATH is a directory, also scan its subdirectories.",
        ),
    ] = False,
) -> None:
    """Remove the background from every supported image under INPUT_PATH,
    writing one transparent PNG per input into OUTPUT_DIR.
    """
    if model not in AVAILABLE_MODELS:
        err_console.print(
            f"[red]Unknown model {model!r}. Available models: "
            f"{', '.join(AVAILABLE_MODELS)}[/red]"
        )
        raise typer.Exit(2)

    files = resolve_input_files(input_path, recursive)
    if not files:
        err_console.print(
            f"[yellow]No supported images found under {input_path}.[/yellow]"
        )
        raise typer.Exit(1)

    console.print(
        f"Found [bold]{len(files)}[/bold] image(s). Loading model [cyan]{model}[/cyan] "
        "(first use downloads and caches it, may take a moment)..."
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Removing backgrounds", total=len(files))

        def on_result(result) -> None:
            progress.advance(task)
            if result.status == "skipped":
                err_console.print(
                    f"[yellow]skip[/yellow] {result.path}: {result.reason}"
                )

        batch = process_batch(
            input_path,
            output_dir,
            model_name=model,
            recursive=recursive,
            progress_callback=on_result,
        )

    table = Table(title="Batch summary")
    table.add_column("Outcome")
    table.add_column("Count", justify="right")
    table.add_row("Processed", str(len(batch.processed)))
    table.add_row("Skipped", str(len(batch.skipped)))
    console.print(table)

    if batch.skipped:
        reasons = Counter(r.reason.split(":", 1)[0] for r in batch.skipped if r.reason)
        for reason, count in reasons.most_common():
            console.print(f"  [yellow]{reason}[/yellow]: {count}")

    if batch.processed:
        console.print(f"Output written to [bold]{output_dir}[/bold]")

    if not batch.processed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
