"""Command line: score customers to a CSV, or validate the segments against the next period."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from rfm_explorer import data, scoring
from rfm_explorer import validate as validation
from rfm_explorer.pipeline import segment_customers
from rfm_explorer.rfm import Monetary, last_day
from rfm_explorer.scoring import Method

app = typer.Typer(
    help="RFM customer segmentation on the UCI Online Retail II transactions.",
    no_args_is_help=True,
)
console = Console()

DataOpt = Annotated[
    Path | None,
    typer.Option(
        "--data", help="Parquet file; default is the full dataset, else the sample."
    ),
]
LookbackOpt = Annotated[
    int,
    typer.Option(help="Days of history before the snapshot; 0 = all history.", min=0),
]
MonetaryOpt = Annotated[
    Monetary, typer.Option(help="net (purchases minus refunds) or gross.")
]


def _lines(path: Path | None) -> tuple[pl.DataFrame, pl.DataFrame, Path]:
    path = path or data.find_data_file()
    lines, ledger = data.clean(data.load_raw(path))
    return lines, ledger, path


def _cell(v) -> str:
    if isinstance(v, float):
        return f"{v:,.3f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def _show(frame: pl.DataFrame, title: str) -> None:
    table = Table(title=title)
    for name in frame.columns:
        table.add_column(name)
    for row in frame.iter_rows():
        table.add_row(*(_cell(v) for v in row))
    console.print(table)


@app.command()
def score(
    method: Annotated[Method, typer.Option(help="Scoring method.")] = "quintile",
    snapshot: Annotated[
        str | None,
        typer.Option(help="YYYY-MM-DD; default the day after the last transaction."),
    ] = None,
    lookback_days: LookbackOpt = 365,
    monetary: MonetaryOpt = "net",
    k: Annotated[int, typer.Option(help="Clusters for k-means.", min=2)] = 5,
    out: Annotated[Path, typer.Option(help="CSV of every scored customer.")] = Path(
        "out/rfm_customers.csv"
    ),
    data_file: DataOpt = None,
) -> None:
    """Score customers and print the segment table; write the customer list to ``--out``."""
    lines, _, path = _lines(data_file)
    day = (
        date.fromisoformat(snapshot)
        if snapshot
        else last_day(lines) + timedelta(days=1)
    )
    scored = segment_customers(lines, day, lookback_days or None, monetary, method, k)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.write_csv(out)
    console.print(
        f"{path.name}: {scored.height:,} customers scored as of {day}, written to {out}"
    )
    _show(
        scoring.summarize(scored).select(
            "segment",
            "customers",
            "customer_share",
            "money_share",
            "mean_recency",
            "mean_frequency",
            "mean_monetary",
        ),
        f"{method} segments",
    )


@app.command()
def validate(
    horizon_days: Annotated[
        int, typer.Option(help="Days of future behaviour to compare.", min=1)
    ] = 90,
    lookback_days: LookbackOpt = 365,
    monetary: MonetaryOpt = "net",
    k: Annotated[int, typer.Option(help="Clusters for k-means.", min=2)] = 5,
    resamples: Annotated[int, typer.Option(help="Bootstrap resamples.", min=50)] = 500,
    data_file: DataOpt = None,
) -> None:
    """Score as of ``horizon_days`` before the end of the data, then compare with what happened."""
    lines, _, path = _lines(data_file)
    day = last_day(lines) + timedelta(days=1) - timedelta(days=horizon_days)
    result = validation.validate(
        lines,
        day,
        horizon_days,
        lookback_days or None,
        monetary,
        k,
        resamples=resamples,
    )
    console.print(
        f"{path.name}: scored as of {day}; {result.customers:,} customers, "
        f"{result.repeat_rate:.1%} bought in the next {horizon_days} days"
    )
    _show(
        result.segments["quintile"].drop("future_spend", "priority"),
        "quintile segments, next-period behaviour",
    )
    _show(result.orderings, "share of future spend from the top 20% (random = 20%)")


@app.command()
def ledger(data_file: DataOpt = None) -> None:
    """Show which rows are dropped before scoring, and why."""
    _, led, path = _lines(data_file)
    _show(led, f"{path.name}: rows removed before scoring")


if __name__ == "__main__":
    app()
