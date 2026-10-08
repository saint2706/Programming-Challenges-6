"""Benchmark pieces: one job (strategy x seed), the ceiling, and the cross-seed summary."""

from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from active_labeling import loop, stats, strategies
from active_labeling.model import Head

TARGETS = (0.90, 0.95)


def run_job(problem, name, seed, budget, b, init, clusters=None) -> dict:
    with threadpool_limits(
        1
    ):  # one BLAS thread per job: the jobs are what we parallelize
        strategy = strategies.make(name, clusters=clusters)
        curve = loop.run_loop(problem, strategy, seed, budget=budget, b=b, init=init)
    return {"strategy": name, "seed": seed, "b": b, "curve": curve}


def ceiling(problem) -> float:
    """Accuracy of the head trained on every pool label: what labeling everything would buy."""
    head = Head(problem.n_classes, problem.C).fit(problem.X, problem.y)
    return float((head.predict(problem.X_eval) == problem.y_eval).mean())


def _nanmean(values):
    v = stats.as_float(values)
    return float(np.nanmean(v)) if np.isfinite(v).any() else None


def mean_curve(curves) -> dict:
    acc = np.array([stats.as_float(c["acc"]) for c in curves])
    mean = acc.mean(axis=0)
    se = (
        acc.std(axis=0, ddof=1) / np.sqrt(len(acc))
        if len(acc) > 1
        else np.zeros_like(mean)
    )
    return {
        "n": list(curves[0]["n"]),
        "acc": mean.tolist(),
        "lo": (mean - 1.96 * se).tolist(),
        "hi": (mean + 1.96 * se).tolist(),
        "coverage": np.mean([c["coverage"] for c in curves], axis=0).tolist(),
        "skew": np.mean([c["skew"] for c in curves], axis=0).tolist(),
    }


def summarize(
    curves, ceiling_acc, *, rule=None, targets=TARGETS, n_boot=2000, baseline="random"
) -> dict:
    """``curves = {strategy: {seed: curve}}``. Every comparison is paired by seed with ``baseline``."""
    seeds = sorted(curves[baseline])
    base = [curves[baseline][s] for s in seeds]
    base_alc = np.array([stats.alc(c["n"], c["acc"]) for c in base])

    def hits(cs, t):
        return np.array(
            [stats.labels_to_target(c["n"], c["acc"], t * ceiling_acc) for c in cs]
        )

    base_hit = {t: hits(base, t) for t in targets}
    out = {}
    for name, by_seed in curves.items():
        cs = [by_seed[s] for s in seeds]
        alcs = np.array([stats.alc(c["n"], c["acc"]) for c in cs])
        entry = {
            "seeds": len(seeds),
            "acc_final": stats.mean_ci(
                [stats.as_float(c["acc"])[-1] for c in cs], n_boot
            ),
            "f1_final": stats.mean_ci(
                [stats.as_float(c["f1"])[-1] for c in cs], n_boot
            ),
            "alc": stats.mean_ci(alcs, n_boot),
            "alc_vs_random": None
            if name == baseline
            else stats.mean_ci(alcs - base_alc, n_boot),
            "coverage_final": float(np.mean([c["coverage"][-1] for c in cs])),
            "skew_final": float(np.mean([c["skew"][-1] for c in cs])),
            "select_seconds": _nanmean([_nanmean(c["sel_time"]) for c in cs]),
            "sel_err": _nanmean([_nanmean(c["sel_err"]) for c in cs]),
            "pool_err": _nanmean([_nanmean(c["pool_err"]) for c in cs]),
            "targets": {},
        }
        for t in targets:
            h = hits(cs, t)
            reached = ~np.isnan(h)
            both = reached & ~np.isnan(base_hit[t])
            entry["targets"][f"{t:.2f}"] = {
                "reached": int(reached.sum()),
                "labels": float(h[reached].mean()) if reached.any() else None,
                "saving_vs_random": None
                if name == baseline
                else stats.mean_ci((base_hit[t] - h)[both], n_boot),
            }
        if rule is not None:
            outs = [stats.stop_outcome(c, rule["threshold"], rule["k"]) for c in cs]
            entry["stop"] = {
                "fired": int(sum(o["fired"] for o in outs)),
                "labels": float(np.mean([o["n"] for o in outs])),
                "acc_gap": stats.mean_ci([o["gap"] for o in outs], n_boot),
            }
        out[name] = entry
    return out
