"""``stockout``: fetch the data, run the experiment, score one store-product, print a worklist."""

from __future__ import annotations

import json
from datetime import date
from typing import Annotated

import polars as pl
import typer

from stockout_risk import data, models, pipeline, predict
from stockout_risk.paths import data_dir, results_dir

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

DateOpt = Annotated[
    str | None,
    typer.Option(
        "--date",
        help="Day the features are as of (YYYY-MM-DD); default: the last day whose next day is in the data.",
    ),
]


def _day(text: str | None, feat) -> date:
    if text is None:
        return pipeline.last_day(feat)
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise typer.BadParameter(f"{text!r} is not a YYYY-MM-DD date") from None


def _model():
    try:
        return models.load_model(data_dir() / "model")
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None


@app.command("data")
def data_cmd(
    force: Annotated[bool, typer.Option(help="Download again.")] = False,
) -> None:
    """Download FreshRetailNet-50K into data/ (sha256-checked) and print what is in it."""
    data.download(force=force)
    for key, value in data.summary(data.load_raw()).items():
        typer.echo(f"{key}: {value}")


@app.command()
def evaluate(
    ablation: Annotated[
        bool, typer.Option(help="Refit without each feature group.")
    ] = True,
    other_labels: Annotated[
        bool, typer.Option(help="Also fit the 1h and 8h labels.")
    ] = True,
    refresh: Annotated[
        bool, typer.Option(help="Rebuild the cached feature table.")
    ] = False,
) -> None:
    """Train on the early days, tune on the middle, report on the last week; write results/."""
    out = pipeline.run(
        pipeline.load_features(refresh=refresh),
        ablate=ablation,
        other_labels=other_labels,
        log=lambda m: typer.echo(m, err=True),
    )
    pipeline.save(out)
    r = out["result"]
    typer.echo(
        f"test PR-AUC {r['lightgbm']['pr_auc']:.3f} (base rate {r['base_rate']['test']:.3f}), "
        f"ROC-AUC {r['lightgbm']['roc_auc']:.3f}; wrote {results_dir()}"
    )


@app.command()
def score(
    store_id: Annotated[int, typer.Option(help="Store id.")],
    product_id: Annotated[int, typer.Option(help="Product id.")],
    date_: DateOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Risk of a stockout tomorrow for one store-product, with the reasons."""
    booster, meta = _model()
    feat = pipeline.load_features()
    try:
        out = predict.score_one(
            booster, meta, feat, store_id, product_id, _day(date_, feat)
        )
    except LookupError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    if as_json:
        typer.echo(json.dumps(out, indent=2))
        return
    typer.echo(
        f"store {store_id}, product {product_id}: {out['risk']:.1%} chance of {out['label_hours']}+ "
        f"out-of-stock hours on {out['predicts_day']}"
    )
    if out["actual_next_oos_hours"] is not None:
        typer.echo(
            f"  what happened: {out['actual_next_oos_hours']} out-of-stock hours"
        )
    if out["warning"]:
        typer.echo(f"  warning: {out['warning']}", err=True)
    for r in out["reasons"]:
        value = "n/a" if r["value"] is None else f"{r['value']:.3g}"
        typer.echo(f"  {r['log_odds']:+.2f}  {r['meaning']} = {value}")


@app.command()
def worklist(
    date_: DateOpt = None,
    top: Annotated[int, typer.Option(min=1, max=5000, help="How many rows.")] = 20,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """The store-products most likely to stock out tomorrow."""
    booster, meta = _model()
    feat = pipeline.load_features()
    day = _day(date_, feat)
    try:
        rows = predict.worklist(booster, meta, feat, day, top)
    except LookupError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    if not predict.tomorrow_known(feat.filter(pl.col("dt") == day)):
        typer.echo(f"warning: {predict.MISSING_TOMORROW}", err=True)
    if as_json:
        typer.echo(json.dumps(rows.to_dicts(), indent=2))
        return
    typer.echo("store  product  risk   oos_today  14d_mean_oos  product_share_heavy")
    for r in rows.iter_rows(named=True):
        typer.echo(
            f"{r['store_id']:>5}  {r['product_id']:>7}  {r['risk']:.1%}  {r['oos_t']:>9.0f}  "
            f"{r['oos_mean14']:>12.1f}  {r['product_heavy_rate']:>19.0%}"
        )
