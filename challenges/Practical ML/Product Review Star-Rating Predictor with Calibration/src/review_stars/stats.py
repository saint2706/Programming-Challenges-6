"""Bootstrap uncertainty: whole-product (cluster) resampling, paired by construction.

Reviews of one product are correlated (same item, often the same defects and the same buyers), so
resampling reviews as if independent makes every interval too narrow. Each replicate here draws
clusters (``parent_asin``) with replacement and takes all of a drawn cluster's rows.

Replicate ``b`` of a given ``(groups, seed)`` is always the same draw, whichever model is being
scored, so two models evaluated with the same seed are paired and a difference of their replicate
metrics is a proper paired bootstrap of the difference.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from joblib import Parallel, delayed

from review_stars import metrics


def _cluster_codes(groups):
    groups = np.asarray(groups)
    if len(groups) == 0:
        raise ValueError("cannot bootstrap an empty split")
    _, inverse = np.unique(groups, return_inverse=True)
    return inverse.astype(np.int64), int(inverse.max()) + 1


def _resample(inverse: np.ndarray, n_clusters: int, seed: int, b: int) -> np.ndarray:
    rng = np.random.default_rng((seed, b))
    times = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters))
    return np.repeat(np.arange(len(inverse)), times[inverse])


def boot_indices(groups, n_boot: int, seed: int) -> Iterator[np.ndarray]:
    """Row indices of each cluster-bootstrap replicate."""
    inverse, n_clusters = _cluster_codes(groups)
    for b in range(n_boot):
        yield _resample(inverse, n_clusters, seed, b)


def _run_chunk(preds, y, inverse, n_clusters, seed, bs, bins, curve_bins):
    out = {
        name: {k: np.full(len(bs), np.nan) for k in metrics.BUNDLE_KEYS}
        for name in preds
    }
    if curve_bins:
        for name in preds:
            out[name]["reliability_n"] = np.zeros((len(bs), curve_bins))
            out[name]["reliability_k"] = np.zeros((len(bs), curve_bins))
    for j, b in enumerate(bs):
        idx = _resample(inverse, n_clusters, seed, b)
        for name, P in preds.items():
            for k, v in metrics.bundle(P[idx], y[idx], bins).items():
                out[name][k][j] = v
            if curve_bins:
                conf, correct = metrics.top_label.raw(P[idx], y[idx])
                n, k_ok = metrics.fixed_bin_counts(conf, correct, curve_bins)
                out[name]["reliability_n"][j], out[name]["reliability_k"][j] = n, k_ok
    return out


def boot_metrics(
    preds,
    y,
    groups,
    n_boot: int,
    seed: int,
    bins: int = 15,
    n_jobs: int = 1,
    curve_bins: int | None = None,
):
    """``{model: {metric: replicate values [n_boot]}}`` for every model in ``preds`` (name -> P).

    With ``curve_bins`` each model also gets ``reliability_n`` and ``reliability_k``: per replicate,
    the reviews and the correct reviews in each of ``curve_bins`` equal-width confidence bins
    (arrays ``[n_boot, curve_bins]``), from which reliability bands are read."""
    y = np.asarray(y)
    inverse, n_clusters = _cluster_codes(groups)
    parts = np.array_split(np.arange(n_boot), max(1, min(n_jobs, n_boot)))
    if len(parts) == 1:
        results = [
            _run_chunk(preds, y, inverse, n_clusters, seed, parts[0], bins, curve_bins)
        ]
    else:
        results = Parallel(n_jobs=len(parts))(
            delayed(_run_chunk)(
                preds, y, inverse, n_clusters, seed, bs, bins, curve_bins
            )
            for bs in parts
        )
    return {
        name: {
            k: np.concatenate([r[name][k] for r in results]) for k in results[0][name]
        }
        for name in preds
    }


def ci(reps, point: float | None = None, level: float = 0.95) -> dict:
    """Percentile interval of replicate values (NaN replicates are skipped and counted)."""
    reps = np.asarray(reps, dtype=np.float64)
    ok = reps[~np.isnan(reps)]
    n_nan = int(len(reps) - len(ok))
    if len(ok) == 0:
        nan = float("nan")
        return {
            "est": nan if point is None else float(point),
            "lo": nan,
            "hi": nan,
            "n": len(reps),
            "n_nan": n_nan,
        }
    lo, hi = np.percentile(ok, [50 * (1 - level), 50 * (1 + level)])
    est = float(ok.mean()) if point is None else float(point)
    return {
        "est": est,
        "lo": float(lo),
        "hi": float(hi),
        "n": len(reps),
        "n_nan": n_nan,
    }
