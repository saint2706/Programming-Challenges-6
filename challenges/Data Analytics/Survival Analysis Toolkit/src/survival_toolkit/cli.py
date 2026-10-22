"""Command line: curves, tests, Cox models and the HTML report for a bundled dataset or any file."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from survival_toolkit import fetch as fetch_mod
from survival_toolkit.analysis import (
    Analysis,
    analyze,
    resolve,
    to_dict,
)
from survival_toolkit.data import DATASETS, Spec, data_dir
from survival_toolkit.paths import project_root
from survival_toolkit.report import build_html, fmt_p, fmt_time
from survival_toolkit.types import SurvivalError

app = typer.Typer(
    help="Survival analysis: Kaplan-Meier curves, log-rank tests and Cox regression.",
    no_args_is_help=True,
)
console = Console()

Dataset = Annotated[
    str,
    typer.Argument(help="telco, lung, gbsg, or the path of a CSV/TSV/Parquet file."),
]
Time = Annotated[
    str | None, typer.Option("--time", help="Duration column (for a file).")
]
Event = Annotated[
    str | None, typer.Option("--event", help="Event column (for a file).")
]
EventValue = Annotated[
    str | None,
    typer.Option("--event-value", help="Value of --event that means an event."),
]
Entry = Annotated[
    str | None,
    typer.Option(
        "--entry", help="Column with the time a subject came under observation."
    ),
]
Unit = Annotated[
    str, typer.Option("--unit", help="Time unit shown on axes (for a file).")
]
By = Annotated[
    str | None, typer.Option("--by", "-b", help="Split the curves by this column.")
]
Bins = Annotated[
    int | None,
    typer.Option(
        "--bins", help="Cut a numeric --by column into this many quantile bins."
    ),
]
Ci = Annotated[str, typer.Option("--ci", help="log-log, log, plain or arcsin.")]
Horizons = Annotated[
    list[float] | None,
    typer.Option("--horizon", "-H", help="Report S(t) at this time; repeatable."),
]
Tau = Annotated[
    float | None, typer.Option("--tau", help="Horizon for restricted mean survival.")
]
Covariates = Annotated[
    list[str] | None,
    typer.Option("--covariate", "-c", help="Cox covariate; repeatable."),
]
Reference = Annotated[
    list[str] | None,
    typer.Option("--reference", help="Reference level as column=level; repeatable."),
]
Strata = Annotated[
    str | None, typer.Option("--strata", help="Stratify the Cox model by this column.")
]
Ties = Annotated[str, typer.Option("--ties", help="efron or breslow.")]
Penalizer = Annotated[
    float,
    typer.Option("--penalizer", help="Ridge penalty on standardised coefficients."),
]
Transform = Annotated[
    str,
    typer.Option("--transform", help="PH-test time scale: km, rank, identity, log."),
]
OutDir = Annotated[
    Path, typer.Option("--out", "-o", help="Folder for the report files.")
]


def _default_by(dataset: str, by: str | None) -> str | None:
    """The bundled datasets know which column to split by; a file must be told."""
    return by or (DATASETS[dataset].group if dataset in DATASETS else None)


def _fail(exc: Exception) -> typer.Exit:
    console.print(f"[red]error:[/red] {exc}")
    return typer.Exit(2)


def _reference(items: list[str] | None, spec: Spec | None) -> dict[str, str]:
    ref = dict(spec.reference) if spec else {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SurvivalError(f"--reference wants column=level, got {item!r}")
        ref[key] = value
    return ref


def _run(
    dataset: str,
    *,
    time: str | None,
    event: str | None,
    event_value: str | None,
    entry: str | None,
    unit: str,
    by: str | None,
    bins: int | None,
    ci: str,
    horizon: list[float] | None,
    tau: float | None,
    covariate: list[str] | None = None,
    reference: list[str] | None = None,
    strata: str | None = None,
    ties: str = "efron",
    penalizer: float = 0.0,
    transform: str = "km",
    cox_default: bool = False,
) -> Analysis:
    try:
        spec, prepared = resolve(
            dataset,
            time=time,
            event=event,
            entry=entry,
            event_value=event_value,
            unit=unit,
        )
        covs = covariate or (list(spec.covariates) if spec and cox_default else None)
        if cox_default and not covs:
            raise SurvivalError(
                "pass at least one --covariate (a file has no default set)"
            )
        result = analyze(
            prepared,
            title=spec.title if spec else Path(dataset).name,
            by=by,
            bins=bins,
            horizons=horizon
            or (list(spec.horizons) if spec and spec.horizons else None),
            tau=tau if tau is not None else (spec.tau if spec else None),
            ci=ci,
            covariates=covs,
            reference=_reference(reference, spec),
            strata=strata,
            ties=ties,
            penalizer=penalizer,
            transform=transform,
        )
        if spec and spec.caveat:
            result.notes.append(spec.caveat)
        return result
    except SurvivalError as exc:
        raise _fail(exc) from exc


@app.command()
def datasets() -> None:
    """List the bundled datasets."""
    table = Table(title="Bundled datasets")
    for col in ("name", "time", "event", "unit", "title"):
        table.add_column(col)
    for s in DATASETS.values():
        present = (data_dir() / s.file).exists()
        table.add_row(
            s.name, s.time, s.event, s.unit, s.title + ("" if present else " [missing]")
        )
    console.print(table)


@app.command()
def fetch() -> None:
    """Download the bundled datasets into sample_data/ (needs network)."""
    rows = fetch_mod.fetch_all(project_root() / "sample_data")
    for name, n in rows.items():
        console.print(f"wrote {name} ({n:,} rows)")


def _print_groups(a: Analysis) -> None:
    table = Table(title=f"{a.title}: {a.n:,} subjects, {a.events:,} events ({a.unit})")
    table.add_column("group")
    for col in ("n", "events", "median (95% CI)"):
        table.add_column(col, justify="right")
    for h in a.horizons:
        table.add_column(f"S({h:g})", justify="right")
    if a.tau is not None:
        table.add_column(f"RMST {a.tau:g}", justify="right")
    for g in a.groups:
        med = fmt_time(g.median)
        if math.isfinite(g.median):
            med += f" ({fmt_time(g.median_ci[0])} to {fmt_time(g.median_ci[1])})"
        cells = [g.label, f"{g.n:,}", f"{g.events:,}", med]
        cells += ["–" if r["s"] is None else f"{r['s']:.3f}" for r in g.retention]
        if a.tau is not None:
            cells.append("–" if g.rmst is None else f"{g.rmst[0]:.1f}")
        table.add_row(*cells)
    console.print(table)
    for k, v in a.dropped.items():
        console.print(f"[dim]dropped {v:,} rows: {k}[/dim]")


@app.command()
def km(
    dataset: Dataset,
    by: By = None,
    bins: Bins = None,
    ci: Ci = "log-log",
    horizon: Horizons = None,
    tau: Tau = None,
    entry: Entry = None,
    time: Time = None,
    event: Event = None,
    event_value: EventValue = None,
    unit: Unit = "time units",
) -> None:
    """Kaplan-Meier curves: median, retention at horizons and restricted mean survival."""
    a = _run(
        dataset, time=time, event=event, event_value=event_value, entry=entry, unit=unit,
        by=_default_by(dataset, by), bins=bins, ci=ci, horizon=horizon, tau=tau,
    )  # fmt: skip
    _print_groups(a)


@app.command()
def compare(
    dataset: Dataset,
    by: By = None,
    bins: Bins = None,
    horizon: Horizons = None,
    tau: Tau = None,
    entry: Entry = None,
    time: Time = None,
    event: Event = None,
    event_value: EventValue = None,
    unit: Unit = "time units",
) -> None:
    """Test whether the groups' curves differ: five weighted log-rank tests, pairs, RMST."""
    a = _run(
        dataset, time=time, event=event, event_value=event_value, entry=entry, unit=unit,
        by=_default_by(dataset, by), bins=bins, ci="log-log", horizon=horizon, tau=tau,
    )  # fmt: skip
    if not a.tests:
        raise _fail(
            SurvivalError("need --by with at least two groups that have events")
        )
    _print_groups(a)
    table = Table(title="Weighted log-rank tests")
    table.add_column("weight")
    for col in ("chi2", "df", "p"):
        table.add_column(col, justify="right")
    for k, v in a.tests.items():
        table.add_row(k, f"{v.statistic:.2f}", str(v.df), fmt_p(v.p_value))
    console.print(table)
    if a.pairwise:
        pt = Table(title="Pairs (log-rank, Holm)")
        pt.add_column("pair")
        for col in ("chi2", "p", "adjusted p"):
            pt.add_column(col, justify="right")
        for r in a.pairwise:
            pt.add_row(
                f"{r['a']} vs {r['b']}",
                f"{r['statistic']:.2f}",
                fmt_p(r["p"]),
                fmt_p(r["p_adjusted"]),
            )
        console.print(pt)
    for r in a.rmst_pairs:
        console.print(
            f"RMST to {a.tau:g}: {r['a']} - {r['b']} = {r['difference']:.2f} {a.unit} "
            f"({r['lower']:.2f} to {r['upper']:.2f}), p {fmt_p(r['p'])}"
        )


def _print_cox(label: str, fit) -> None:
    m = fit.model
    lr_stat, df, p = m.likelihood_ratio()
    console.print(
        f"[bold]{label}[/bold]  n={m.n:,} events={m.n_events:,} ties={m.ties} "
        f"LR chi2={lr_stat:.1f} (df {df}, p {fmt_p(p)}) concordance={m.concordance()['c']:.3f} "
        f"converged={m.converged}"
    )
    table = Table()
    table.add_column("covariate")
    for col in ("HR", "95% CI", "p", "PH chi2", "PH p"):
        table.add_column(col, justify="right")
    ph = {t["name"]: t for t in fit.ph["terms"]}
    for r in m.summary():
        t = ph[r["name"]]
        flag = "[red]" if t["p"] < 0.05 else ""
        table.add_row(
            r["name"], f"{r['hr']:.3f}", f"{r['hr_lower']:.3f}-{r['hr_upper']:.3f}", fmt_p(r["p"]),
            f"{flag}{t['chi2']:.1f}", f"{flag}{fmt_p(t['p'])}",
        )  # fmt: skip
    g = fit.ph["global"]
    table.add_row("GLOBAL", "", "", "", f"{g['chi2']:.1f}", fmt_p(g["p"]))
    console.print(table)
    if fit.dropped:
        console.print(
            f"[dim]{fit.dropped:,} rows left out for a missing covariate[/dim]"
        )


@app.command()
def cox(
    dataset: Dataset,
    covariate: Covariates = None,
    reference: Reference = None,
    strata: Strata = None,
    ties: Ties = "efron",
    penalizer: Penalizer = 0.0,
    transform: Transform = "km",
    entry: Entry = None,
    time: Time = None,
    event: Event = None,
    event_value: EventValue = None,
    unit: Unit = "time units",
) -> None:
    """Cox proportional hazards regression with its proportional-hazards test."""
    a = _run(
        dataset, time=time, event=event, event_value=event_value, entry=entry, unit=unit, by=None,
        bins=None, ci="log-log", horizon=None, tau=None, covariate=covariate, reference=reference,
        strata=strata, ties=ties, penalizer=penalizer, transform=transform, cox_default=True,
    )  # fmt: skip
    _print_cox("Cox", a.cox)
    if a.cox_stratified:
        _print_cox(f"Cox stratified by {strata}", a.cox_stratified)


@app.command()
def report(
    dataset: Dataset,
    by: By = None,
    bins: Bins = None,
    ci: Ci = "log-log",
    horizon: Horizons = None,
    tau: Tau = None,
    covariate: Covariates = None,
    reference: Reference = None,
    strata: Strata = None,
    ties: Ties = "efron",
    penalizer: Penalizer = 0.0,
    transform: Transform = "km",
    entry: Entry = None,
    time: Time = None,
    event: Event = None,
    event_value: EventValue = None,
    unit: Unit = "time units",
    out: OutDir = Path("out"),
) -> None:
    """Write survival_report.html (one self-contained file) and analysis.json."""
    spec = DATASETS.get(dataset)
    a = _run(
        dataset, time=time, event=event, event_value=event_value, entry=entry, unit=unit,
        by=_default_by(dataset, by), bins=bins, ci=ci, horizon=horizon, tau=tau,
        covariate=covariate or (list(spec.covariates) if spec else None), reference=reference,
        strata=strata or (spec.strata if spec else None), ties=ties, penalizer=penalizer,
        transform=transform,
    )  # fmt: skip
    out = out if out.is_absolute() else project_root() / out
    out.mkdir(parents=True, exist_ok=True)
    page = out / "survival_report.html"
    page.write_text(build_html(a), encoding="utf-8", newline="\n")
    (out / "analysis.json").write_text(
        json.dumps(to_dict(a), indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    console.print(
        f"wrote {page} ({page.stat().st_size / 1e6:.1f} MB) and analysis.json"
    )
