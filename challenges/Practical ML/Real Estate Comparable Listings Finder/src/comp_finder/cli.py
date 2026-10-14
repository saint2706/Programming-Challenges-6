"""Typer CLI: run the experiment, or find comps for a sale or for a custom property.

comps evaluate | find | data
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from comp_finder import data, explain, pipeline, report
from comp_finder.engine import METHODS, Settings
from comp_finder.paths import project_root

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

MethodOption = Annotated[str, typer.Option(help=f"Similarity: {', '.join(METHODS)}.")]


@app.callback()
def _main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(
                errors="replace"
            )  # a redirected Windows console is cp1252
        except (AttributeError, ValueError):
            pass


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(2)


def _out_dir(value: str) -> Path:
    return Path(value) if Path(value).is_absolute() else project_root() / value


@app.command()
def evaluate(
    out: Annotated[str, typer.Option(help="Results folder.")] = "results",
    reps: Annotated[int, typer.Option(min=100, help="Bootstrap resamples.")] = 1000,
) -> None:
    """Tune on the validation window, score the test window, write results/report.md and charts."""
    result = pipeline.run(reps=reps, log=typer.echo)
    path = report.write_report(result, _out_dir(out))
    for m, v in result.metrics["main"].items():
        lo, hi = v["ci"]["mdape"]
        typer.echo(
            f"{report.NAMES[m]:42} MdAPE {v['mdape']:.2%} [{lo:.2%}, {hi:.2%}]  "
            f"within 10% {v['within10']:.1%}  within 20% {v['within20']:.1%}"
        )
    typer.echo(f"report: {path}")


@app.command("data")
def data_summary() -> None:
    """Fetch (first use) and summarise the sales table."""
    sales = data.load()
    n_fit, n_val, n_test = data.window_counts(sales, data.Windows())
    typer.echo(
        f"{sales.height:,} sales, {sales['sold'].min()} to {sales['sold'].max()}, "
        f"{sales['zipcode'].n_unique()} zip codes, {int((sales['n_prior'] > 0).sum())} resales"
    )
    typer.echo(f"fit {n_fit:,} / validation {n_val:,} / test {n_test:,}")


def _money(v: float) -> str:
    return f"${v:,.0f}"


@app.command()
def find(
    sale_id: Annotated[
        int | None, typer.Option(help="Property id of a sale in the data.")
    ] = None,
    property_json: Annotated[
        str | None,
        typer.Option(
            "--property", help="Custom property as JSON (see README for fields)."
        ),
    ] = None,
    on: Annotated[
        str | None,
        typer.Option("--date", help="Custom property's as-of date (YYYY-MM-DD)."),
    ] = None,
    method: MethodOption = "leaf",
    k: Annotated[int, typer.Option(min=1, max=15)] = 5,
    bracket: Annotated[
        bool, typer.Option(help="Make sure comps bracket the subject.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Machine-readable output.")
    ] = False,
) -> None:
    """Comps for a sale from the data (``--sale-id``) or for an unsold property (``--property``)."""
    if method not in METHODS:
        raise _fail(f"method must be one of {sorted(METHODS)}")
    if (sale_id is None) == (property_json is None):
        raise _fail("give exactly one of --sale-id or --property")
    engine, _ = pipeline.build_engine()
    settings = Settings(
        **{
            **vars(pipeline.saved_settings()),
            "k": k,
            "selection": "bracket" if bracket else "top",
        }
    )
    actual = None
    note = ""
    if sale_id is not None:
        hits = np.flatnonzero(engine.ids == sale_id)
        if len(hits) == 0:
            raise _fail(f"no sale with id {sale_id}")
        q = int(hits[-1])
        actual = float(engine.price[q])
        if q < engine.n_fit:
            note = (
                "note: this sale is inside the fit window, so the models saw its price."
            )
    else:
        try:
            record = json.loads(property_json)
            if not isinstance(record, dict):
                raise TypeError("--property must be a JSON object")
            last = engine.sales["sold"].max()
            sold = date.fromisoformat(on) if on else last + timedelta(days=1)
            if sold < last:
                raise ValueError(f"--date must be on or after the last sale ({last})")
            engine = engine.extended(data.subject_frame(record, sold, engine.sales))
        except (ValueError, TypeError) as exc:  # JSONDecodeError is a ValueError
            raise _fail(str(exc)) from exc
        q = engine.sales.height - 1
    rows = np.array([q])
    kmax = k * 3 if bracket else k
    comps = engine.rank(method, rows, kmax, settings.pool)
    est = engine.estimate(rows, comps, settings)
    details = explain.explain_comps(engine, method, q, est.comps[0], settings)
    avm = float(engine.avm(rows, settings.extrapolate)[0])
    summary = {
        "method": method,
        "estimate": float(est.price[0]),
        "gbm_avm": avm,
        "actual": actual,
        "bracketed": bool(est.bracketed[0]),
        "comps": details,
    }
    if as_json:
        typer.echo(json.dumps(summary, indent=2))
        return
    sold_on = engine.sales["sold"][q]
    typer.echo(f"{METHODS[method]}, k={k}, as of {sold_on}")
    if note:
        typer.echo(note)
    typer.echo(
        f"comps-based estimate {_money(summary['estimate'])}   GBM AVM {_money(avm)}"
    )
    if actual is not None:
        typer.echo(
            f"actual sale price    {_money(actual)}  (comps error {summary['estimate'] / actual - 1:+.1%})"
        )
    typer.echo(f"bracketed: {'yes' if summary['bracketed'] else 'no'}")
    for i, c in enumerate(details, 1):
        typer.echo(
            f"\n#{i} sold {c['sold']} for {_money(c['price'])}  "
            f"{c['km']:.2f} km, {c['days_before']} days earlier"
        )
        typer.echo(
            f"   market {c['time_pct']:+.1f}%, attributes {c['net_attribute_pct']:+.1f}% "
            f"(gross {c['gross_attribute_pct']:.1f}%) -> adjusted {_money(c['adjusted_price'])}"
        )
        for r in c["reasons"]:
            typer.echo(f"   + {r['label']}: {r['text']}")


if __name__ == "__main__":  # pragma: no cover
    app()
