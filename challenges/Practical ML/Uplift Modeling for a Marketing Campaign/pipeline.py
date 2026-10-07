"""run_all: randomization check, split, propensity model, benchmark each stage (cached), artifacts.

Stages (``propensity``, ``visit``, ``conversion``, ``synthetic``) are cached as JSON in
``results/stages/``. A stage file stores the *key* it was computed under (seed, bootstrap size,
grid, ... and a fingerprint of the data) and is reused only if the key matches, so changing any
of them recomputes it, while a stage that was not asked for this time but is still valid stays in
``report.json``. ``fresh=True`` recomputes the stages that were asked for.

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
import propensity

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
REAL_STAGES = ("visit", "conversion")
STAGES = ("propensity", *REAL_STAGES, "synthetic")
STAGE_VERSION = 2  # bump when a stage's payload format or meaning changes


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


def fingerprint(df: pl.DataFrame) -> str:
    """Row count plus an order-independent hash of the rows: a different sample changes it."""
    return f"{df.height}:{int(np.bitwise_xor.reduce(df.hash_rows(seed=0).to_numpy()))}"


def _key(name, fp, *, seed, n_boot, n_seeds, grid, synth_n, synth_params) -> dict:
    """Everything a stage's result depends on, and nothing else."""
    key = {"version": STAGE_VERSION, "data": fp, "seed": seed}
    if name == "synthetic":
        key |= {"n_seeds": n_seeds, "synth_n": synth_n, "synth_params": synth_params}
    elif name in REAL_STAGES:
        key |= {"n_boot": n_boot, "grid": grid}
    return clean(key)


def _read_stage(path: Path, key: dict):
    """The cached payload if the file exists and was computed under ``key``, else ``None``."""
    if not path.exists():
        return None
    stored = json.loads(path.read_text())
    return stored["payload"] if stored.get("key") == key else None


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
    fp = fingerprint(df)
    wanted = set(stages)
    if wanted & set(REAL_STAGES):
        wanted.add("propensity")  # the real stages need e(x)
    report = {
        "config": {
            "seed": seed,
            "n_boot": n_boot,
            "n_seeds": n_seeds,
            "grid": grid,
            "synth_n": synth_n,
            "treated_share": data.propensity(train),
        },
        "randomization": balance.to_dicts(),
        "rows": {"train": train.height, "val": val.height, "test": test.height},
        "outcomes": {},
    }
    model_path = results_dir / "propensity.joblib"
    model = None

    def propensity_model():
        nonlocal model
        if model is None:
            # joblib unpickles: safe only because this file is written by our own run into the
            # gitignored results/ dir. Never point this at files from elsewhere.
            model = joblib.load(model_path)
        return model

    for name in STAGES:
        path = results_dir / "stages" / f"{name}.json"
        key = _key(
            name,
            fp,
            seed=seed,
            n_boot=n_boot,
            n_seeds=n_seeds,
            grid=grid,
            synth_n=synth_n,
            synth_params=synth_params,
        )
        payload = None if fresh and name in wanted else _read_stage(path, key)
        if payload is None:
            if name not in wanted:
                continue  # not asked for, and stale or missing: leave it out of the report
            if name == "propensity":
                X_tr, t_tr, _ = data.xy(train)
                model = propensity.PropensityModel(seed=seed).fit(X_tr, t_tr)
                joblib.dump(model, model_path)
                Xt, tt, _ = data.xy(test)
                payload = propensity.diagnose(model, Xt, tt)
            elif name == "synthetic":
                X, t, _ = data.xy(train.head(synth_n))
                payload = evaluate.evaluate_synth(
                    X, t, n_seeds=n_seeds, params=synth_params, seed=seed
                )
            else:
                summary, scores, models, (t_te, y_te, w_te) = evaluate.evaluate_real(
                    train,
                    val,
                    test,
                    name,
                    propensity_model=propensity_model(),
                    grid=grid,
                    n_boot=n_boot,
                    seed=seed,
                )
                pl.DataFrame({"t": t_te, "y": y_te, "w": w_te, **scores}).write_parquet(
                    results_dir / f"scores_{name}.parquet"
                )
                joblib.dump(models, results_dir / f"models_{name}.joblib")
                payload = summary
            payload = clean(payload)
            path.write_text(json.dumps({"key": key, "payload": payload}, indent=2))
        if name == "propensity":
            report["propensity"] = payload
        elif name == "synthetic":
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
    """One ranking per scorer on the test split, with the stored IPW arm weights applied."""
    frame = art.scores[outcome]
    t, y = frame["t"].to_numpy(), frame["y"].to_numpy()
    w = frame["w"].to_numpy() if "w" in frame.columns else None
    return {
        c: metrics.rank(frame[c].to_numpy(), t, y, w)
        for c in frame.columns
        if c not in ("t", "y", "w")
    }
