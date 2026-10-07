"""run_all: randomization check, split, benchmark each stage (cached), write artifacts.

Stages (``visit``, ``conversion``, ``synthetic``) are cached as JSON in ``results/stages/`` so an
interrupted run resumes and a finished stage is not recomputed (``fresh=True`` forces it).
Generated scores and fitted models are git-ignored; ``results/report.json`` is tracked.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import data
import evaluate
import joblib
import metrics
import numpy as np
import polars as pl

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
STAGES = ("visit", "conversion", "synthetic")


def clean(obj):
    """Make a nested result strict-JSON: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _stage(
    name,
    results_dir,
    train,
    val,
    test,
    *,
    n_boot,
    seed,
    grid,
    n_seeds,
    synth_n,
    synth_params,
):
    if name == "synthetic":
        X, t, _ = data.xy(train.head(synth_n))
        return evaluate.evaluate_synth(
            X, t, n_seeds=n_seeds, params=synth_params, seed=seed
        )
    summary, scores, models, (t_te, y_te) = evaluate.evaluate_real(
        train, val, test, name, grid=grid, n_boot=n_boot, seed=seed
    )
    pl.DataFrame({"t": t_te, "y": y_te, **scores}).write_parquet(
        results_dir / f"scores_{name}.parquet"
    )
    joblib.dump(models, results_dir / f"models_{name}.joblib")
    return summary


def run_all(
    data_dir,
    results_dir,
    seed=0,
    n_boot=200,
    n_seeds=5,
    df=None,
    grid=evaluate.GRID,
    synth_n=200_000,
    synth_params=evaluate.SYNTH_PARAMS,
    stages=STAGES,
    fresh=False,
) -> dict:
    results_dir = Path(results_dir)
    df = data.load(Path(data_dir)) if df is None else df
    balance = data.assert_randomized(df)  # aborts before anything is written
    train, val, test = data.split(df, seed)
    (results_dir / "stages").mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "seed": seed,
            "n_boot": n_boot,
            "n_seeds": n_seeds,
            "grid": grid,
            "synth_n": synth_n,
            "propensity": data.propensity(train),
        },
        "randomization": balance.to_dicts(),
        "rows": {"train": train.height, "val": val.height, "test": test.height},
        "outcomes": {},
    }
    for name in stages:
        path = results_dir / "stages" / f"{name}.json"
        if path.exists() and not fresh:
            payload = json.loads(path.read_text())
        else:
            payload = clean(
                _stage(
                    name,
                    results_dir,
                    train,
                    val,
                    test,
                    n_boot=n_boot,
                    seed=seed,
                    grid=grid,
                    n_seeds=n_seeds,
                    synth_n=synth_n,
                    synth_params=synth_params,
                )
            )
            path.write_text(json.dumps(payload, indent=2))
        if name == "synthetic":
            report["synthetic"] = payload
        else:
            report["outcomes"][name] = payload
    report = clean(report)
    (results_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


@dataclass
class Artifacts:
    report: dict
    scores: dict[str, pl.DataFrame] = field(default_factory=dict)
    models: dict[str, dict] = field(default_factory=dict)


def load_artifacts(results_dir=RESULTS_DIR, with_models: bool = False) -> Artifacts:
    results_dir = Path(results_dir)
    report_path = results_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"{report_path} not found; run `python cli.py benchmark` first"
        )
    art = Artifacts(report=json.loads(report_path.read_text()))
    for outcome in art.report["outcomes"]:
        scores = results_dir / f"scores_{outcome}.parquet"
        if scores.exists():
            art.scores[outcome] = pl.read_parquet(scores)
        models = results_dir / f"models_{outcome}.joblib"
        if with_models and models.exists():
            # joblib unpickles: safe only because these files are written by our own
            # `benchmark` into the gitignored results/ dir. Never point this at files from elsewhere.
            art.models[outcome] = joblib.load(models)
    return art


def ranked_scores(art: Artifacts, outcome: str) -> dict[str, metrics.Ranked]:
    frame = art.scores[outcome]
    t, y = frame["t"].to_numpy(), frame["y"].to_numpy()
    return {
        c: metrics.rank(frame[c].to_numpy(), t, y)
        for c in frame.columns
        if c not in ("t", "y")
    }
