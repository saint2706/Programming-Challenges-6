"""Command line: download, prepare, verify, run, report.

``uv run dist-pipeline --help`` lists the commands. Every command runs from the challenge folder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from dist_pipeline.paths import results_dir

app = typer.Typer(
    help="Distributed data processing pipeline benchmark.", no_args_is_help=True
)
console = Console()

VALID_SCALES = ("1m", "12m")


def _check_scale(value: str) -> str:
    # The scale becomes part of file names and a DuckDB path literal, so accept only known keys.
    if value not in VALID_SCALES:
        raise typer.BadParameter(f"must be one of {', '.join(VALID_SCALES)}")
    return value


ScaleOpt = Annotated[
    str,
    typer.Option(
        help="Dataset scale: 1m (March 2019) or 12m (all 2019).", callback=_check_scale
    ),
]


@app.command()
def download() -> None:
    """Fetch the TLC yellow taxi 2019 Parquet files and the zone lookup into data/raw/."""
    from dist_pipeline.download import download_all

    manifest = download_all()
    console.print(
        f"downloaded {len(manifest)} files; manifest in data/raw/manifest.json"
    )


@app.command()
def prepare(
    scale: Annotated[
        str, typer.Option(help="1m, 12m or csv-1m (the CSV leg of 1m).")
    ] = "1m",
) -> None:
    """Normalise the raw files into part files (1M rows each) or the CSV export."""
    from dist_pipeline.prepare import build_csv, build_scale

    if scale == "csv-1m":
        info = build_csv("1m")
    else:
        info = build_scale(scale)
    console.print_json(json.dumps(info, default=str))


@app.command()
def verify(
    scale: ScaleOpt = "1m",
    engines: Annotated[
        str, typer.Option(help="Comma-separated engine keys; default all installed.")
    ] = "",
    queries: Annotated[
        str, typer.Option(help="Comma-separated query names; default all.")
    ] = "",
) -> None:
    """Run the engines once against the exact DuckDB oracle and print each verdict."""
    from dist_pipeline.bench import (
        ALL_ENGINES,
        ALL_QUERIES,
        default_workdir,
        run_matrix,
    )

    eng = tuple(e for e in engines.split(",") if e) or ALL_ENGINES
    qs = tuple(q for q in queries.split(",") if q) or ALL_QUERIES
    rows = run_matrix(
        (scale,), eng, qs, repeats=1, warmup=0, workdir=default_workdir(), csv_scales=()
    )
    failed = 0
    for r in rows:
        if r["verified"] is False:
            failed += 1
        console.print(
            f"{r['engine']:<13} {r['query']:<20} {r['status']:<14} "
            f"verified={r['verified']}  {r['verify_detail'] or r['error'] or ''}"
        )
    if failed:
        raise typer.Exit(code=1)


@app.command()
def run(
    scale: ScaleOpt = "12m",
    repeats: Annotated[int, typer.Option(min=1)] = 3,
    warmup: Annotated[int, typer.Option(min=0)] = 1,
    scaling: Annotated[
        bool, typer.Option(help="Also run mrpipe strong and weak scaling.")
    ] = True,
    csv: Annotated[bool, typer.Option(help="Also run the CSV leg on 1m.")] = True,
) -> None:
    """Run the full matrix and the scaling runs, and write results/results.json."""
    from dist_pipeline.bench import (
        ALL_ENGINES,
        ALL_QUERIES,
        default_workdir,
        machine,
        run_matrix,
        run_scaling,
    )

    workdir = default_workdir()
    scales = ("1m", scale) if scale != "1m" else ("1m",)
    rows = run_matrix(
        scales,
        ALL_ENGINES,
        ALL_QUERIES,
        repeats=repeats,
        warmup=warmup,
        workdir=workdir,
        csv_scales=("1m",) if csv else (),
        progress=lambda r: console.print(
            f"{r['scale']:<4} {r['fmt']:<7} {r['engine']:<13} {r['query']:<20} {r['status']}"
        ),
    )
    payload: dict[str, object] = {"machine": machine(), "matrix": rows, "scaling": None}
    if scaling:
        payload["scaling"] = run_scaling(
            scale,
            ("q3_high_card_od", "q4_moments_distinct"),
            strong_workers=(1, 2, 4, 8),
            weak_workers=(1, 2, 4),
            parts_per_worker=2,
            repeats=repeats,
            warmup=warmup,
            workdir=workdir,
        )
    results_dir().mkdir(parents=True, exist_ok=True)
    out = results_dir() / "results.json"
    out.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    console.print(f"wrote {out}")


@app.command()
def report(
    results: Annotated[Path | None, typer.Option(help="results.json to read.")] = None,
) -> None:
    """Write summary.md and report.html beside results.json (offline, XSS-escaped)."""
    from dist_pipeline.report import write_report

    source = results or results_dir() / "results.json"
    md, page = write_report(source, source.parent)
    console.print(f"wrote {md}")
    console.print(f"wrote {page}")


if __name__ == "__main__":
    app()
