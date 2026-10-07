"""Typer CLI: fetch the Criteo data, check randomization, benchmark the learners, score and target."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import data
import learners
import metrics
import numpy as np
import pipeline
import polars as pl
import policy
import propensity
import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
SCORERS = [*learners.LEARNERS]


def get_artifacts(with_models: bool = False):
    """Indirection so tests can swap in a small benchmark."""
    return pipeline.load_artifacts(pipeline.RESULTS_DIR, with_models=with_models)


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=2)


def _artifacts(with_models: bool = False):
    try:
        return get_artifacts(with_models)
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc


@app.command()
def fetch(
    rows: Annotated[
        int, typer.Option(min=1000, help="Rows to sample.")
    ] = data.SAMPLE_ROWS,
    seed: Annotated[int, typer.Option(help="Sampling seed.")] = 0,
) -> None:
    """Download Criteo Uplift v2.1 (~300 MB, once) and cache a random sample as parquet."""
    typer.echo(f"data in {data.fetch(n=rows, seed=seed)}")


@app.command()
def check() -> None:
    """Randomization check: standardized mean difference per feature between the arms."""
    try:
        df = data.load()
        bal = data.assert_randomized(df)
    except (FileNotFoundError, data.RandomizationError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"{'feature':<8}{'smd':>8}")
    for row in bal.to_dicts():
        typer.echo(f"{row['feature']:<8}{row['smd']:>+8.3f}")
    typer.echo(f"ok: no gross imbalance over {df.height:,} rows")
    train, val, _ = data.split(df)
    X, t, _ = data.xy(train)
    Xv, tv, _ = data.xy(val)
    diag = propensity.diagnose(propensity.PropensityModel().fit(X, t), Xv, tv)
    shares = [b["treated_share"] for b in diag["bins"]]
    typer.echo(
        f"treatment predictable from the features: held-out AUC {diag['auc']:.3f} "
        f"[{diag['auc_lo']:.3f}, {diag['auc_hi']:.3f}] (0.5 = pure randomization); "
        f"treated share by predicted-propensity quintile {min(shares):.3f} to {max(shares):.3f}"
    )
    if diag["auc_lo"] > 0.5:
        typer.echo(
            "note: balanced feature by feature but not jointly, so the benchmark weights "
            "customers by inverse-propensity and keeps the unadjusted numbers as a sensitivity"
        )


def _ci(c, scale=1.0, spec=".4f") -> str:
    if c is None:
        return "-"
    est, lo, hi = (c[k] * scale for k in ("est", "lo", "hi"))
    return f"{est:{spec}} [{lo:{spec}}, {hi:{spec}}]"


def format_report(report: dict) -> str:
    lines = []
    prop = report.get("propensity")
    if prop:
        lines.append(
            f"treatment predictable from the features: held-out AUC {prop['auc']:.3f} "
            f"[{prop['auc_lo']:.3f}, {prop['auc_hi']:.3f}] (0.5 = pure randomization); "
            "every number below is inverse-propensity weighted"
        )
    for outcome, summ in report["outcomes"].items():
        ate = _ci(summ["ate"], 1000, ".2f")
        raw = _ci(summ["ate_unadjusted"], 1000, ".2f")
        header = (
            f"== {outcome}  (test n={summ['n_test']:,}, ATE {ate} per 1000; "
            f"unadjusted {raw}) =="
        )
        lines.append("\n" + header)
        lines.append(
            f"{'learner':<10}{'Qini':>26}{'AUUC':>26}"
            f"{'incremental@20 /1000':>30}{'Qini - T':>26}"
        )
        for name, e in summ["learners"].items():
            lines.append(
                f"{name:<10}{_ci(e['qini']):>26}{_ci(e['auuc']):>26}"
                f"{_ci(e['incremental@20'], 1000, '.2f'):>30}"
                f"{_ci(e['qini_vs_T']):>26}"
            )
    return "\n".join(lines)


@app.command()
def benchmark(
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    n_boot: Annotated[int, typer.Option(min=10, help="Bootstrap resamples.")] = 200,
    n_seeds: Annotated[
        int, typer.Option(min=1, help="Seeds for the semi-synthetic benchmark.")
    ] = 5,
    stage: Annotated[
        list[str] | None,
        typer.Option(help="Stage(s) to run: visit, conversion, synthetic."),
    ] = None,
    fresh: Annotated[bool, typer.Option(help="Recompute finished stages.")] = False,
) -> None:
    """Tune on validation, score the test split once, benchmark; writes results/report.json."""
    stages = tuple(stage) if stage else pipeline.STAGES
    unknown = set(stages) - set(pipeline.STAGES)
    if unknown:
        raise _fail(
            f"unknown stage(s) {sorted(unknown)}; choose from {pipeline.STAGES}"
        )
    try:
        report = pipeline.run_all(
            data.DATA_DIR,
            pipeline.RESULTS_DIR,
            seed=seed,
            n_boot=n_boot,
            n_seeds=n_seeds,
            stages=stages,
            fresh=fresh,
        )
    except (FileNotFoundError, data.RandomizationError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(format_report(report))


@app.command()
def score(
    input: Annotated[Path, typer.Argument(help="Parquet with the 12 feature columns.")],
    output: Annotated[Path, typer.Option(help="Where to write the scored parquet.")],
    outcome: Annotated[str, typer.Option(help="visit or conversion.")] = "visit",
    learner: Annotated[str, typer.Option(help="Which learner to score with.")] = "T",
) -> None:
    """Add `uplift` (predicted incremental outcome) and `uplift_rank` (1 = most persuadable)."""
    art = _artifacts(with_models=True)
    if outcome not in art.models or learner not in art.models[outcome]:
        raise _fail(f"no fitted learner {learner!r} for outcome {outcome!r}")
    if not input.exists():
        raise _fail(f"{input} does not exist")
    df = pl.read_parquet(input)
    missing = [f for f in data.FEATURES if f not in df.columns]
    if missing:
        raise _fail(f"input is missing feature columns: {missing}")
    X = df.select(data.FEATURES).to_numpy().astype(np.float32)
    uplift = art.models[outcome][learner].predict(X)
    ranks = np.empty(len(uplift), dtype=np.int64)
    ranks[np.argsort(-uplift, kind="stable")] = np.arange(1, len(uplift) + 1)
    df.with_columns(
        uplift=pl.Series(uplift), uplift_rank=pl.Series(ranks)
    ).write_parquet(output)
    typer.echo(
        f"scored {df.height:,} rows with learner {learner} ({outcome}) -> {output}"
    )


@app.command()
def target(
    budget: Annotated[
        float, typer.Option(help="Share of customers to contact, in (0, 1].")
    ],
    outcome: Annotated[str, typer.Option(help="visit or conversion.")] = "visit",
    learner: Annotated[str, typer.Option(help="Which learner's ranking to use.")] = "T",
    value: Annotated[
        float | None, typer.Option(help="Value of one incremental outcome.")
    ] = None,
    cost: Annotated[float, typer.Option(min=0.0, help="Cost of one contact.")] = 0.0,
    n_boot: Annotated[
        int, typer.Option(min=10, help="Bootstrap resamples for the interval.")
    ] = 200,
) -> None:
    """Expected incremental outcomes from contacting the top `budget` share, with a CI."""
    if not 0.0 < budget <= 1.0:
        raise _fail(f"budget must be in (0, 1], got {budget}")
    art = _artifacts()
    if outcome not in art.scores:
        raise _fail(f"no results for outcome {outcome!r}; have {sorted(art.scores)}")
    ranked = pipeline.ranked_scores(art, outcome)
    if learner not in ranked or learner == "random":
        raise _fail(f"unknown learner {learner!r}; choose from {SCORERS}")
    r = ranked[learner]
    boot = metrics.bootstrap(
        {learner: r},
        n_boot=n_boot,
        stat=lambda rr, w=None: {"inc": policy.incremental_at(rr, budget, w)},
    )
    inc = metrics.interval(boot, learner, "inc")
    ate = metrics.ate(r)
    n = len(r.t)
    typer.echo(
        f"{learner}-learner on {outcome}: contact the top {budget:.0%} of {n:,} customers "
        "(inverse-propensity weighted)"
    )
    typer.echo(f"  incremental {outcome}s per 1,000 customers: {_ci(inc, 1000, '.2f')}")
    typer.echo(
        f"  random targeting of the same share:        {budget * ate * 1000:.2f}"
    )
    typer.echo(f"  treat everyone:                            {ate * 1000:.2f}")
    if value is not None:
        be = policy.break_even_cost(r, budget, value)
        frac, prof = policy.best_fraction(r, value, cost, n)
        typer.echo(
            f"  break-even cost per contact at this share: {be:.4f} (you pay {cost:.4f})"
        )
        typer.echo(
            f"  best share at cost {cost:.4f}: {frac:.0%} (profit {prof:,.1f} on {n:,} customers)"
        )


if __name__ == "__main__":
    app()
