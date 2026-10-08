"""The benchmark: tune on validation, score the test split once, plus the known-effect check.

Real data has no ground truth for an individual's effect, so the learners are judged by
uplift curves on the test split (with a paired bootstrap) and by decile calibration; the
semi-synthetic benchmark (``synth``) is where "did it recover tau" is answerable.

Treatment on Criteo is *nearly* but not exactly randomized (see ``propensity.py``), so every
real-data number is inverse-propensity weighted with an e(x) fit on the train split. The plain
(unweighted) difference-in-means numbers are kept as an ``unadjusted`` sensitivity next to them.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from uplift import data, learners, metrics, propensity, synth

GRID = [
    {"num_leaves": nl, "min_child_samples": mcs}
    for nl in (15, 63)
    for mcs in (200, 2000)
]
FRACS = (0.1, 0.2, 0.3, 0.5)
SYNTH_PARAMS = {"n_estimators": 150, "min_child_samples": 100, "num_leaves": 15}


def tune(cls, train, val, grid, e, seed: int, val_weights=None) -> dict:
    """The grid entry with the best Qini coefficient on the validation split.

    ``e`` is a constant or a propensity model; ``val_weights`` are the validation IPW weights."""
    X, t, y = train
    Xv, tv, yv = val
    best, best_score = grid[0], -np.inf
    for params in grid:
        score = metrics.qini_coefficient(
            metrics.rank(
                cls(e, params, seed).fit(X, t, y).predict(Xv), tv, yv, val_weights
            )
        )
        if score > best_score:
            best, best_score = params, score
    return best


UNADJUSTED_STATS = ("qini", "auuc", "incremental@20")


def evaluate_real(
    train,
    val,
    test,
    outcome: str,
    *,
    propensity_model,
    grid=GRID,
    n_boot: int = 200,
    seed: int = 0,
):
    """Fit every learner on ``train`` (params tuned on ``val``), score ``test`` once.

    ``propensity_model`` is e(x) fit on ``train``. Returns ``(summary, scores, models,
    (t, y, w))`` where ``w`` are the test IPW weights."""
    tr, va, te = (data.xy(df, outcome) for df in (train, val, test))
    t_te, y_te = te[1], te[2]
    w_va = propensity.ipw_weights(va[1], propensity_model.predict(va[0]))
    w_te = propensity.ipw_weights(t_te, propensity_model.predict(te[0]))
    models, params, scores = {}, {}, {}
    for name, cls in {**learners.LEARNERS, **learners.BASELINES}.items():
        params[name] = (
            {}
            if name == "random"
            else tune(cls, tr, va, grid, propensity_model, seed, w_va)
        )
        models[name] = cls(propensity_model, params[name], seed).fit(*tr)
        scores[name] = models[name].predict(te[0])
    ranked = {n: metrics.rank(s, t_te, y_te, w_te) for n, s in scores.items()}
    plain = {n: metrics.rank(s, t_te, y_te) for n, s in scores.items()}
    boot = metrics.bootstrap(ranked, n_boot=n_boot, seed=seed)
    boot_plain = metrics.bootstrap(plain, n_boot=n_boot, seed=seed)
    out = {}
    for name in ranked:
        entry = {
            stat: metrics.interval(boot, name, stat)
            for stat in boot.point[name]
            if stat != "ate"
        }
        entry["qini_vs_T"] = (
            None if name == "T" else metrics.diff_interval(boot, name, "T", "qini")
        )
        entry["unadjusted"] = {
            stat: metrics.interval(boot_plain, name, stat) for stat in UNADJUSTED_STATS
        }
        entry["params"] = params[name]
        entry["calibration"] = metrics.decile_calibration(
            scores[name], t_te, y_te, arm_weights=w_te
        )
        out[name] = entry
    summary = {
        "n_test": len(t_te),
        "ate": metrics.interval(boot, "T", "ate"),
        "ate_unadjusted": metrics.interval(boot_plain, "T", "ate"),
        "learners": out,
    }
    return summary, scores, models, (t_te, y_te, w_te)


def mean_ci(values) -> dict | None:
    """Mean and a t-based 95% CI over seeds (NaNs ignored); ``None`` if nothing is finite."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    mean = float(v.mean())
    if v.size == 1:
        return {"mean": mean, "lo": mean, "hi": mean}
    half = float(stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return {"mean": mean, "lo": mean - half, "hi": mean + half}


def evaluate_synth(
    X,
    t,
    *,
    scenarios=synth.SCENARIOS,
    n_seeds: int = 5,
    params=SYNTH_PARAMS,
    seed: int = 0,
    base_rate: float = 0.05,
    strength: float = 1.0,
) -> dict:
    """Spearman / RMSE against the true tau and Qini against the oracle ordering, over seeds.

    ``base_rate`` defaults to roughly Criteo's visit rate (4.7%)."""
    e, half = float(np.mean(t)), len(X) // 2
    tr, te = slice(0, half), slice(half, None)
    out = {}
    for scenario in scenarios:
        runs = {n: {"spearman": [], "rmse": [], "qini": []} for n in learners.LEARNERS}
        oracle = []
        for s in range(n_seeds):
            y, tau = synth.simulate(
                X,
                t,
                seed=seed * 1000 + s,
                scenario=scenario,
                base_rate=base_rate,
                strength=strength,
            )
            oracle.append(metrics.qini_coefficient(metrics.rank(tau[te], t[te], y[te])))
            for name, cls in learners.LEARNERS.items():
                pred = cls(e, params, seed + s).fit(X[tr], t[tr], y[tr]).predict(X[te])
                runs[name]["rmse"].append(
                    float(np.sqrt(np.mean((pred - tau[te]) ** 2)))
                )
                runs[name]["spearman"].append(
                    float(stats.spearmanr(pred, tau[te]).statistic)
                    if scenario == "heterogeneous"
                    else float("nan")
                )
                runs[name]["qini"].append(
                    metrics.qini_coefficient(metrics.rank(pred, t[te], y[te]))
                )
        out[scenario] = {
            **{n: {k: mean_ci(v) for k, v in r.items()} for n, r in runs.items()},
            "oracle": {"qini": mean_ci(oracle)},
        }
    return out
