"""``fraud``: fetch the data, run the experiment, score one transaction, flag a file of them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from fraud_flagging import data, models, pipeline, policy
from fraud_flagging.paths import data_dir, results_dir

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

SUFFIXES = {".csv", ".parquet"}
MAX_INPUT_BYTES = 500 * 1024 * 1024
PolicyOpt = Annotated[str, typer.Option(help=f"One of {', '.join(policy.POLICIES)}.")]
Precision = Annotated[float, typer.Option(help="Precision target (policy precision).")]
Recall = Annotated[float, typer.Option(help="Recall target (policy recall).")]
ReviewCost = Annotated[float, typer.Option(help="Cost of one review (policy cost).")]
PerHour = Annotated[int, typer.Option(help="Reviews per hour (policy capacity).")]


def _model():
    try:
        return models.load_model(data_dir() / "model")
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None


def _decide(frame, meta, name, precision, recall, review_cost, per_hour):
    try:
        return policy.decide(
            frame,
            meta,
            name,
            precision=precision,
            recall=recall,
            review_cost=review_cost,
            per_hour=per_hour,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def _read(path: Path) -> pl.DataFrame:
    """A transaction table from a ``.csv`` or ``.parquet`` file, after cheap sanity checks."""
    if path.suffix.lower() not in SUFFIXES:
        raise typer.BadParameter(
            f"{path.name}: only {', '.join(sorted(SUFFIXES))} files are read"
        )
    if not path.is_file():
        raise typer.BadParameter(f"{path} is not a file")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise typer.BadParameter(
            f"{path.name} is larger than {MAX_INPUT_BYTES // 2**20} MB"
        )
    try:
        return (
            pl.read_parquet(path)
            if path.suffix.lower() == ".parquet"
            else pl.read_csv(path)
        )
    except Exception as exc:  # noqa: BLE001  (polars raises many types for a malformed file)
        raise typer.BadParameter(f"could not read {path.name}: {exc}") from None


@app.command("data")
def data_cmd(
    force: Annotated[bool, typer.Option(help="Download again.")] = False,
) -> None:
    """Download the data into data/ (md5 and sha256 checked) and print what is in it."""
    data.download(force=force)
    for key, value in data.summary(data.load_raw()).items():
        typer.echo(f"{key}: {value}")


@app.command()
def evaluate(
    ablation: Annotated[
        bool, typer.Option(help="Refit without each feature group.")
    ] = True,
    rolling: Annotated[
        bool, typer.Option(help="Score four 6-hour blocks, training on the past.")
    ] = True,
) -> None:
    """Train on the first day, tune on the next 12 hours, report on the last 12; write results/."""
    out = pipeline.run(
        pipeline.load_frame(),
        ablate=ablation,
        rolling=rolling,
        log=lambda m: typer.echo(m, err=True),
    )
    pipeline.save(out)
    r = out["result"]
    main = r["candidates"][f"lightgbm_{r['main_strategy']}"]["test"]
    typer.echo(
        f"main model LightGBM ({r['main_strategy']}): test PR-AUC {main['pr_auc']:.3f} "
        f"(base rate {main['base_rate']:.4f}), ROC-AUC {main['roc_auc']:.3f}; wrote {results_dir()}"
    )


@app.command()
def score(
    row: Annotated[
        int, typer.Option(help="Row of the published file (the tx_id of the data).")
    ],
    precision: Precision = 0.9,
    recall: Recall = 0.8,
    review_cost: ReviewCost = 25.0,
    per_hour: PerHour = 3,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Score one transaction of the data: risk, the decision under each policy, and why."""
    booster, meta = _model()
    frame = data.load().filter(pl.col("tx_id") == row)
    if frame.height == 0:
        typer.echo(
            f"no transaction with row {row} (it may be an exact duplicate that was dropped)",
            err=True,
        )
        raise typer.Exit(1)
    scored = policy.score_frame(booster, meta, frame)
    decisions = {
        name: bool(
            _decide(scored, meta, name, precision, recall, review_cost, per_hour)[0]
        )
        for name in ("precision", "recall", "cost")
    }
    contrib = models.contributions(booster, scored)[0]
    top = sorted(zip(meta["features"], contrib, strict=True), key=lambda t: -abs(t[1]))[
        :5
    ]
    out = {
        "row": row,
        "time_hours": float(scored["Time"][0] / 3600),
        "amount": float(scored["Amount"][0]),
        "risk": float(scored["p"][0]),
        "was_fraud": bool(scored["y"][0]),
        "flagged_by": decisions,
        "reasons": [{"feature": f, "log_odds": float(c)} for f, c in top],
    }
    if as_json:
        typer.echo(json.dumps(out, indent=2))
        return
    typer.echo(
        f"row {row}: amount {out['amount']:.2f} at hour {out['time_hours']:.1f}, "
        f"{out['risk']:.2%} chance of fraud (it {'was' if out['was_fraud'] else 'was not'} fraud)"
    )
    for name, flagged in decisions.items():
        typer.echo(f"  {name:<10} {'FLAG' if flagged else 'pass'}")
    typer.echo("  largest contributions (log-odds; V1..V28 are anonymised components):")
    for f, c in top:
        typer.echo(f"    {c:+.2f}  {f}")


@app.command()
def flag(
    path: Annotated[
        Path, typer.Argument(help="A .csv or .parquet with Time, V1..V28 and Amount.")
    ],
    policy_: Annotated[
        str, typer.Option("--policy", help=f"One of {', '.join(policy.POLICIES)}.")
    ] = "cost",
    precision: Precision = 0.9,
    recall: Recall = 0.8,
    review_cost: ReviewCost = 25.0,
    per_hour: PerHour = 3,
    out: Annotated[
        Path | None,
        typer.Option(help="Write the scored table here (.csv or .parquet)."),
    ] = None,
    only_flagged: Annotated[
        bool, typer.Option(help="Write only the flagged rows.")
    ] = False,
    force: Annotated[bool, typer.Option(help="Overwrite --out if it exists.")] = False,
) -> None:
    """Score a table of transactions and flag them under a policy."""
    booster, meta = _model()
    if out is not None:
        if out.suffix.lower() not in SUFFIXES:
            raise typer.BadParameter(f"--out must end in {', '.join(sorted(SUFFIXES))}")
        if out.exists() and not force:
            raise typer.BadParameter(f"{out} exists; use --force to overwrite it")
        if not out.parent.is_dir():
            raise typer.BadParameter(f"{out.parent} is not a folder")
    frame = _read(path)
    try:
        scored = policy.score_frame(booster, meta, frame)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    flags = _decide(scored, meta, policy_, precision, recall, review_cost, per_hour)
    scored = scored.with_columns(pl.Series("flag", flags)).drop(
        "hour", "p_raw", strict=False
    )
    typer.echo(
        f"{scored.height:,} transactions, {int(flags.sum()):,} flagged under policy {policy_}; "
        f"flagged amount {scored.filter(pl.col('flag'))['Amount'].sum():,.2f}"
    )
    if out is not None:
        table = scored.filter(pl.col("flag")) if only_flagged else scored
        table = table.drop(
            [
                c
                for c in table.columns
                if c
                in ("log_amount", "amount_frac", "amount_zero", "tod_sin", "tod_cos")
            ]
        )
        table.write_parquet(
            out
        ) if out.suffix.lower() == ".parquet" else table.write_csv(out)
        typer.echo(f"wrote {table.height:,} rows to {out}")
