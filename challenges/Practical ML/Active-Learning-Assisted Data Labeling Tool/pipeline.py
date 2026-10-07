"""run_all: load the problem, run every (strategy, seed) job once (cached on disk), summarize.

Each job is one JSON file in ``results/jobs/`` named after, and keyed by, everything its curve
depends on: the data fingerprint, the head's ``C``, budget, init size, cluster count, and the job's
strategy / seed / batch size / eval split. A file is reused only if its stored key matches, so an
interrupted run resumes, a changed config or dataset never reuses stale curves, and a changed seed
count computes only the new seeds. ``report.json`` is rebuilt from the cache on every run, so it
always contains every stage whose jobs are all present and lists the rest under ``"missing"``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import data
import embed
import evaluate
import loop
import model
import numpy as np
import stats
import strategies
from joblib import Parallel, delayed

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
STAGES = ("main", "batch", "stop")
TUNE_SEED_BASE = 1000
STAGE_VERSION = 1  # bump when a job's payload format or meaning changes


@dataclass(frozen=True)
class Config:
    seeds: int = 10
    tune_seeds: int = 3
    budget: int = 1500
    batch: int = 50
    init: int = 77
    batch_sizes: tuple[int, ...] = (10, 50, 200)
    sens_seeds: int = 5
    sens_strategies: tuple[str, ...] = ("random", "margin", "badge", "cluster-margin")
    n_clusters: int = 150
    n_boot: int = 2000
    tol: float = 0.01

    def __post_init__(self):
        if "random" not in self.sens_strategies:
            raise ValueError(
                "sens_strategies must include 'random', the paired baseline"
            )
        for name in self.sens_strategies:
            if name not in strategies.NAMES:
                raise ValueError(
                    f"unknown strategy {name!r}; choose from {strategies.NAMES}"
                )


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
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    return obj


def fingerprint(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def job_key(fp: str, C: float, cfg: Config, job: dict) -> str:
    payload = {
        "version": STAGE_VERSION,
        "data": fp,
        "C": C,
        "budget": cfg.budget,
        "init": cfg.init,
        "n_clusters": cfg.n_clusters,
        **job,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _job(strategy, seed, b, split):
    return {"strategy": strategy, "seed": seed, "b": b, "eval": split}


def _ident(job):
    return (job["strategy"], job["seed"], job["b"], job["eval"])


def stage_jobs(stage: str, cfg: Config) -> list[dict]:
    if stage == "main":
        return [
            _job(n, s, cfg.batch, "test")
            for n in strategies.NAMES
            for s in range(cfg.seeds)
        ]
    if stage == "batch":
        seeds = range(min(cfg.sens_seeds, cfg.seeds))
        return [
            _job(n, s, b, "test")
            for b in cfg.batch_sizes
            for n in cfg.sens_strategies
            for s in seeds
        ]
    if stage == "stop":
        return [
            _job("margin", TUNE_SEED_BASE + i, cfg.batch, "val")
            for i in range(cfg.tune_seeds)
        ]
    raise ValueError(f"unknown stage {stage!r}; choose from {STAGES}")


def _path(jobs_dir: Path, job: dict, key: str) -> Path:
    s, seed, b, split = _ident(job)
    return jobs_dir / f"{s}-s{seed}-b{b}-{split}-{key[:12]}.json"


def _read(path: Path, key: str):
    if not path.exists():
        return None
    stored = json.loads(path.read_text(encoding="utf-8"))
    return stored["curve"] if stored.get("key") == key else None


def _work(problem, job, budget, init, clusters):
    res = evaluate.run_job(
        problem, job["strategy"], job["seed"], budget, job["b"], init, clusters
    )
    return job, res["curve"]


def _compute(problem, todo, cfg, jobs_dir, fp, n_jobs):
    """Run ``todo`` (jobs), writing each finished curve to disk as it completes."""
    if not todo:
        return
    by_split = {
        "test": problem,
        "val": replace(problem, X_eval=problem.X_val, y_eval=problem.y_val),
    }
    clusters = None
    if any(j["strategy"] == "cluster-margin" for j in todo):
        clusters = strategies.cluster_pool(
            problem.X, cfg.n_clusters
        )  # once, shared by all jobs
    args = [
        (
            by_split[j["eval"]],
            j,
            cfg.budget,
            cfg.init,
            clusters if j["strategy"] == "cluster-margin" else None,
        )
        for j in todo
    ]
    if n_jobs == 1:
        results = (_work(*a) for a in args)
    else:
        results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_work)(*a) for a in args
        )
    for job, curve in results:
        key = job_key(fp, problem.C, cfg, job)
        _path(jobs_dir, job, key).write_text(
            json.dumps({"key": key, "curve": clean(curve)}), encoding="utf-8"
        )


def _collect(problem, jobs, cfg, jobs_dir, fp):
    found, missing = {}, []
    for j in jobs:
        key = job_key(fp, problem.C, cfg, j)
        curve = _read(_path(jobs_dir, j, key), key)
        if curve is None:
            missing.append(j)
        else:
            found[_ident(j)] = curve
    return found, missing


def load_problem(
    data_dir=data.DATA_DIR,
    *,
    get_choice=embed.default_choice,
    seed=0,
    val_size=data.VAL_SIZE,
):
    """Banking77 -> ``loop.Problem``: split, embed (cached, one verified backend), pick ``C`` once."""
    raw = data.load_raw(data_dir)
    splits = data.make_splits(raw, seed=seed, val_size=val_size)
    choice = functools.cache(get_choice)
    cache_dir = Path(data_dir) / "embeddings"
    arrays, infos = {}, {}
    for name, texts in (
        ("pool", splits.pool_texts),
        ("val", splits.val_texts),
        ("test", splits.test_texts),
    ):
        arrays[name], infos[name] = embed.embed_cached(texts, cache_dir, choice)
    K = len(splits.classes)
    C = model.select_C(arrays["pool"], splits.pool_y, arrays["val"], splits.val_y, K)
    problem = loop.Problem(
        X=arrays["pool"].astype(np.float64),
        y=splits.pool_y,
        X_eval=arrays["test"].astype(np.float64),
        y_eval=splits.test_y,
        n_classes=K,
        C=C,
        X_val=arrays["val"].astype(np.float64),
        y_val=splits.val_y,
    )
    info = {
        "backend": infos["pool"]["backend"],
        "tried": infos["pool"]["tried"],
        "model": infos["pool"]["model"],
        "duplicates_collapsed": splits.n_duplicates,
        "test_overlap": splits.test_overlap,
        "val": len(splits.val_texts),
    }
    return problem, splits, info


def _curves(found, jobs, strategy_names):
    out: dict = {n: {} for n in strategy_names}
    for j in jobs:
        out[j["strategy"]][j["seed"]] = found[_ident(j)]
    return out


def run_all(
    problem,
    results_dir,
    cfg: Config | None = None,
    *,
    stages=STAGES,
    fresh=False,
    n_jobs=1,
    meta=None,
) -> dict:
    cfg = cfg or Config()
    results_dir = Path(results_dir)
    jobs_dir = results_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(
        problem.X,
        problem.y,
        problem.X_eval,
        problem.y_eval,
        problem.X_val,
        problem.y_val,
    )
    computed: set = set()
    available: dict = {}
    missing_stages = []
    for stage in STAGES:
        jobs = stage_jobs(stage, cfg)
        found, missing = _collect(problem, jobs, cfg, jobs_dir, fp)
        if stage in stages:
            todo = jobs if fresh else missing
            todo = [j for j in todo if _ident(j) not in computed]
            if todo:
                _compute(problem, todo, cfg, jobs_dir, fp, n_jobs)
                computed |= {_ident(j) for j in todo}
                found, missing = _collect(problem, jobs, cfg, jobs_dir, fp)
        available[stage] = (jobs, found)
        if missing:
            missing_stages.append(stage)

    ceiling_acc = evaluate.ceiling(problem)
    report = {
        "version": STAGE_VERSION,
        "config": clean(asdict(cfg)),
        "data": {
            **(meta or {}),
            "fingerprint": fp,
            "pool": len(problem.X),
            "val": len(problem.X_val),
            "eval": len(problem.X_eval),
            "classes": problem.n_classes,
            "C": problem.C,
        },
        "ceiling": {"accuracy": ceiling_acc},
        "missing": missing_stages,
    }
    rule = None
    if "stop" not in missing_stages:
        jobs, found = available["stop"]
        curves = [found[_ident(j)] for j in jobs]
        rule = stats.pick_stop_rule(curves, tol=cfg.tol)
        report["stop"] = {"rule": rule, "curve": evaluate.mean_curve(curves)}
    if "main" not in missing_stages:
        jobs, found = available["main"]
        curves = _curves(found, jobs, strategies.NAMES)
        report["main"] = {
            "summary": evaluate.summarize(
                curves, ceiling_acc, rule=rule, n_boot=cfg.n_boot
            ),
            "curves": {
                n: evaluate.mean_curve(list(by.values())) for n, by in curves.items()
            },
        }
    if "batch" not in missing_stages:
        jobs, found = available["batch"]
        report["batch"] = {}
        for b in cfg.batch_sizes:
            sub = [j for j in jobs if j["b"] == b]
            curves = _curves(found, sub, cfg.sens_strategies)
            report["batch"][str(b)] = {
                "summary": evaluate.summarize(curves, ceiling_acc, n_boot=cfg.n_boot),
                "curves": {
                    n: evaluate.mean_curve(list(by.values()))
                    for n, by in curves.items()
                },
            }
    report = clean(report)
    (results_dir / "report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    return report
