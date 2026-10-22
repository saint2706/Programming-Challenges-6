"""Command line: fetch the Criteo data, compare models on it or on your own paths, score them on simulated truth."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from attribution import criteo, report
from attribution.compare import LABELS, MODELS, Comparison, Settings, compare
from attribution.evaluate import ScenarioResult, evaluate
from attribution.journeys import AttributionError, Journeys, from_paths
from attribution.paths import project_root
from attribution.simulate import SCENARIOS, simulate

app = typer.Typer(
    help="Compare first-touch, last-touch, linear, time-decay, position, Markov, Shapley and logistic attribution.",
    no_args_is_help=True,
)
console = Console()

DAY = 86400.0
Source = Annotated[
    str,
    typer.Argument(
        help="criteo, sim:<scenario> (independent, retargeting, awareness), or a CSV of paths."
    ),
]
Draws = Annotated[
    int,
    typer.Option("--draws", help="Bootstrap resamples for the intervals (0 = none)."),
]
HalfLife = Annotated[
    float | None,
    typer.Option(
        "--half-life",
        help="Time-decay half-life in days (steps for a paths CSV); default 7.",
    ),
]
OutDir = Annotated[
    Path, typer.Option("--out", "-o", help="Folder for the report files.")
]
ModelList = Annotated[
    list[str] | None,
    typer.Option("--model", "-m", help="Only these models; repeatable."),
]


def _fail(exc: Exception) -> typer.Exit:
    console.print(f"[red]error:[/red] {exc}")
    return typer.Exit(2)


def _models(chosen: list[str] | None) -> tuple[str, ...]:
    if not chosen:
        return MODELS
    bad = [m for m in chosen if m not in MODELS]
    if bad:
        raise AttributionError(
            f"unknown model {bad[0]!r}; choose from {', '.join(MODELS)}"
        )
    return tuple(chosen)


def read_paths_csv(path: Path) -> Journeys:
    """A CSV with ``path`` (channels joined by '>'), ``conversions`` and optionally ``nulls``."""
    frame = pl.read_csv(path)
    for col in ("path", "conversions"):
        if col not in frame.columns:
            raise AttributionError(
                f"{path.name} needs a {col!r} column; it has {', '.join(frame.columns)}"
            )
    paths = [
        [c.strip() for c in p.split(">") if c.strip()] for p in frame["path"].to_list()
    ]
    nulls = frame["nulls"].fill_null(0).to_list() if "nulls" in frame.columns else None
    return from_paths(paths, frame["conversions"].fill_null(0).to_list(), nulls)


def _journeys(
    source: str, channel: str, top: int, sample: float | None
) -> tuple[Journeys, str, float]:
    """The journeys, a title, and how many time units make one day (1 where time is only an order)."""
    if source == "criteo":
        return (
            criteo.load_journeys(channel, top, sample_users=sample),
            "Criteo display advertising, 30 days",
            DAY,
        )
    if source.startswith("sim:"):
        name = source[4:]
        if name not in SCENARIOS:
            raise AttributionError(
                f"unknown scenario {name!r}; choose from {', '.join(SCENARIOS)}"
            )
        return (
            simulate(name, 100_000, seed=0).journeys,
            f"Simulated customers: {name}",
            1.0,
        )
    path = Path(source)
    if not path.exists():
        raise AttributionError(
            f"{source!r} is neither 'criteo', 'sim:<scenario>' nor a file"
        )
    return read_paths_csv(path), path.stem, 1.0


@app.command()
def fetch(
    force: Annotated[bool, typer.Option("--force", help="Download again.")] = False,
) -> None:
    """Download the Criteo dataset (623 MB, CC BY-NC-SA 4.0) and slim it to Parquet."""
    if force or not criteo.raw_path().exists():
        console.print("downloading 623 MB from Hugging Face ...")
        criteo.download()
    out = criteo.to_parquet()
    console.print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


@app.command()
def models() -> None:
    """List the models."""
    table = Table()
    table.add_column("name")
    table.add_column("label")
    for m in MODELS:
        table.add_row(m, LABELS[m])
    console.print(table)


def _print_comparison(c: Comparison) -> None:
    j = c.journeys
    table = Table(
        title=f"Share of {j.total_value:,.0f} conversions by channel and model"
    )
    table.add_column("channel")
    for m in c.models:
        table.add_column(LABELS[m], justify="right")
    for ci in np.argsort(-c.share.mean(axis=0)):
        table.add_row(
            report.short_name(j.channels[ci]),
            *[f"{100 * c.share[mi, ci]:.1f}%" for mi in range(len(c.models))],
        )
    console.print(table)
    tv = c.distance()
    far = np.unravel_index(np.argmax(tv), tv.shape)
    console.print(
        f"most different pair: {LABELS[c.models[far[0]]]} and {LABELS[c.models[far[1]]]} "
        f"({100 * tv[far]:.0f}% of conversions would move channel)"
    )


@app.command("compare")
def compare_cmd(
    source: Source,
    channel: Annotated[
        str,
        typer.Option("--channel", help="Criteo context column used as the channel."),
    ] = "cat9",
    top: Annotated[
        int,
        typer.Option(
            "--top", help="Criteo: channels kept, the rest become 'other' (max 12)."
        ),
    ] = 11,
    draws: Draws = 0,
    half_life: HalfLife = None,
    sample: Annotated[
        float | None,
        typer.Option("--sample", help="Criteo: fraction of users to keep."),
    ] = None,
    model: ModelList = None,
    out: OutDir = Path("out"),
) -> None:
    """Run the models on the same journeys and write a report."""
    try:
        chosen = _models(model)
        j, title, unit = _journeys(source, channel, top, sample)
        days = half_life if half_life is not None else 7.0
        c = compare(j, Settings(half_life=days * unit), chosen, draws=draws)
    except AttributionError as exc:
        raise _fail(exc) from exc
    _print_comparison(c)
    out = out if out.is_absolute() else project_root() / out
    out.mkdir(parents=True, exist_ok=True)
    notes = [
        f"Time decay halves credit every {days:g} {'steps' if j.unit == 'steps' else 'days'}."
    ]
    for mi, m in enumerate(c.models):
        zeros = int((c.share[mi] <= 0).sum())
        if zeros >= max(2, j.k // 3):
            notes.append(
                f"{LABELS[m]} gives no credit to {zeros} of {j.k} channels: it fits conversion to which channels "
                "a journey touched, and for those channels the touch goes with *less* conversion. Treat its "
                "shares as a sign the data cannot separate the channels, not as a verdict."
            )
    (out / "report.html").write_text(
        report.build_html(title, c, [], chosen, notes), encoding="utf-8", newline="\n"
    )
    (out / "results.json").write_text(
        report.to_json(c, [], chosen), encoding="utf-8", newline="\n"
    )
    console.print(f"wrote {out / 'report.html'} and results.json")


@app.command("simulate")
def simulate_cmd(
    scenario: Annotated[
        list[str] | None,
        typer.Option("--scenario", "-s", help="Scenario; repeatable (all by default)."),
    ] = None,
    customers: Annotated[
        int, typer.Option("--customers", help="Simulated customers per run.")
    ] = 100_000,
    seeds: Annotated[
        int, typer.Option("--seeds", help="Independent runs per scenario.")
    ] = 5,
    model: ModelList = None,
    out: OutDir = Path("out"),
) -> None:
    """Score every model against the true incremental effect in simulated worlds."""
    try:
        chosen = _models(model)
        names = scenario or list(SCENARIOS)
        bad = [n for n in names if n not in SCENARIOS]
        if bad:
            raise AttributionError(
                f"unknown scenario {bad[0]!r}; choose from {', '.join(SCENARIOS)}"
            )
        results: list[ScenarioResult] = [
            evaluate(n, customers, seeds, chosen) for n in names
        ]
    except AttributionError as exc:
        raise _fail(exc) from exc
    table = Table(
        title=f"Share of conversions put on the wrong channel (mean of {seeds} runs, {customers:,} customers)"
    )
    table.add_column("scenario")
    for m in chosen:
        table.add_column(LABELS[m], justify="right")
    table.add_column("best")
    for r in results:
        best = int(np.argmin(r.error))
        table.add_row(
            r.name, *[f"{100 * v:.1f}%" for v in r.error], LABELS[chosen[best]]
        )
    console.print(table)
    out = out if out.is_absolute() else project_root() / out
    out.mkdir(parents=True, exist_ok=True)
    page = report.build_html(
        "Attribution models against a known truth", None, results, chosen
    )
    (out / "simulation.html").write_text(page, encoding="utf-8", newline="\n")
    (out / "simulation.json").write_text(
        report.to_json(None, results, chosen), encoding="utf-8", newline="\n"
    )
    console.print(f"wrote {out / 'simulation.html'} and simulation.json")
