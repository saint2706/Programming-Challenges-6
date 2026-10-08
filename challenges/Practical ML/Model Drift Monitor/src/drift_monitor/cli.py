"""Typer CLI: fetch the data, train and benchmark the monitor, scan the live stream."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from drift_monitor import data, pipeline

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

RESULTS = pipeline.RESULTS_DIR / "report.json"
HEADLINE_DETECTORS = (
    "score_ks",
    "score_psi",
    "features_any",
    "page_hinkley_score",
    "adwin_score",
    "adwin_error",
)


def get_context():
    """Indirection so tests can swap in a small trained pipeline."""
    return pipeline.load_context()


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=2)


@app.command()
def fetch() -> None:
    """Download the electricity dataset from OpenML (no token needed)."""
    path = data.fetch()
    typer.echo(f"data in {path}")


@app.command()
def train(
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    n_seeds: Annotated[
        int, typer.Option(min=1, help="Benchmark streams per scenario.")
    ] = 20,
) -> None:
    """Fit the model, calibrate thresholds, monitor the live split, benchmark; writes results/report.json."""
    report = pipeline.run_all(
        data.DATA_DIR, pipeline.RESULTS_DIR, seed=seed, n_seeds=n_seeds
    )
    typer.echo(format_report(report))


def _fmt(v, spec=".3f") -> str:
    return "-" if v is None else format(v, spec)


def format_report(report: dict) -> str:
    m, nat = report["model"], report["natural"]
    lines = [
        (
            f"windows of {report['window']} rows, labels delayed {report['label_delay']} windows; "
            f"{report['split']['live']} live rows in {nat['n_windows']} windows"
        ),
        (
            f"accuracy: out-of-fold train {m['accuracy_train_oof']:.3f}, "
            f"reference {m['accuracy_reference']:.3f}, live {m['accuracy_live']:.3f}"
        ),
        "",
        "natural run: windows in which each detector alarmed",
    ]
    nv = report.get("null_validation")
    if nv:
        lines.insert(
            2,
            "cells above threshold on unseen reference windows (target 1%): "
            f"iid null {nv['iid']['cell_alert_rate']:.1%}, block null {nv['blocks']['cell_alert_rate']:.1%}",
        )
    for det in HEADLINE_DETECTORS:
        wins = nat["alarm_windows"][det]
        lines.append(
            f"  {det:<20}{len(wins):>3} windows, first at {nat['first_alarm_window'][det]}"
        )
    lines += [
        "",
        "injected drift (detection rate vs chance, mean delay in windows, false alarms per window)",
    ]
    summary = report["benchmark"]["summary"]
    for det in ("score_ks", "features_any", "adwin_error"):
        lines.append(f"  {det}")
        for r in summary:
            if r["detector"] == det and r["scenario"] != "none":
                lines.append(
                    f"    {r['scenario']:<32}{r['magnitude']:<6}{r['detection_rate']:.2f} vs {_fmt(r['chance_rate'], '.2f')}"
                    f"  delay {_fmt(r['mean_delay_windows'], '.1f')}"
                )
    return "\n".join(lines)


@app.command()
def report() -> None:
    """Print the last benchmark from results/report.json."""
    if not RESULTS.is_file():
        raise _fail("no results/report.json yet; run `train` first")
    typer.echo(format_report(json.loads(RESULTS.read_text(encoding="utf-8"))))


@app.command()
def monitor(
    from_window: Annotated[int, typer.Option(min=0, help="First window to show.")] = 0,
    to_window: Annotated[
        int | None, typer.Option(help="Last window to show (inclusive).")
    ] = None,
) -> None:
    """Scan the live stream window by window and list the alerts."""
    art, live = get_context()
    res = pipeline.run_monitor(art, live)
    last = res.table.height - 1
    to = last if to_window is None else to_window
    if from_window > to or to > last:
        raise _fail(
            f"window range {from_window}..{to} is invalid; the live stream has windows 0..{last}"
        )
    typer.echo(
        f"{'window':>6}  {'alerts':>6}  {'score KS':>8}  {'accuracy*':>9}   alerting signals"
    )
    for row in res.table.filter(
        (res.table["window"] >= from_window) & (res.table["window"] <= to)
    ).iter_rows(named=True):
        wins = res.alerts.filter(res.alerts["window"] == row["window"])
        signals = ", ".join(sorted(set(wins["signal"])))
        typer.echo(
            f"{row['window']:>6}  {wins.height:>6}  {row['score_ks']:>8.3f}  {_fmt(row['accuracy']):>9}   {signals}"
        )
    typer.echo(
        f"* accuracy of window t-{art['delay']} (labels arrive {art['delay']} windows late)"
    )
    if res.dropped_rows:
        typer.echo(
            f"{res.dropped_rows} trailing rows did not fill a window and were not scored"
        )


if __name__ == "__main__":
    app()
