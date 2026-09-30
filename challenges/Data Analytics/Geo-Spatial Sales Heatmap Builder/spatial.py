"""Spatial autocorrelation: contiguity weights from polygons, global and local Moran's I.

"Is high-sales territory clustered, or scattered at random?" is a question about spatial
autocorrelation.  Global Moran's I answers it for the whole map; local Moran's I (LISA,
Anselin 1995) says *where* the clusters are.

Everything is row-standardised binary contiguity: each region's neighbours share its
weight equally, so ``lag_i`` is the mean of its neighbours' deviations from the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

Ring = list[list[float]]


def _rings(geometry: dict):
    if geometry["type"] == "Polygon":
        yield from geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        for poly in geometry["coordinates"]:
            yield from poly
    else:
        raise ValueError(f"unsupported geometry type {geometry['type']!r}")


def contiguity(features: list[dict], kind: str = "queen") -> list[set[int]]:
    """Neighbour sets by shared boundary, computed by hashing vertices (queen) or edges (rook).

    Queen: two regions are neighbours if they share at least one vertex.  Rook: they share
    a boundary segment (two consecutive vertices).  This relies on adjacent polygons carrying
    *identical* coordinates along their shared border, which holds for the topologically
    consistent Census cartographic boundary files (and is verified by the tests).
    """
    if kind not in ("queen", "rook"):
        raise ValueError("kind must be 'queen' or 'rook'")
    owners: dict[tuple, list[int]] = {}
    for idx, feat in enumerate(features):
        seen: set[tuple] = set()
        for ring in _rings(feat["geometry"]):
            pts = [tuple(p) for p in ring]
            if kind == "queen":
                seen.update(pts)
            else:
                for a, b in pairwise(pts):
                    if a != b:
                        seen.add((a, b) if a <= b else (b, a))
        for key in seen:
            owners.setdefault(key, []).append(idx)
    neighbors: list[set[int]] = [set() for _ in features]
    for members in owners.values():
        if len(members) > 1:
            for i in members:
                neighbors[i].update(m for m in members if m != i)
    return neighbors


@dataclass
class Weights:
    """Row-standardised weights over the regions that have data, as padded index arrays."""

    index: np.ndarray  # positions (into the caller's arrays) of regions kept
    nb: (
        np.ndarray
    )  # (n, max_degree) neighbour positions *within* `index`, padded with 0
    mask: np.ndarray  # (n, max_degree) True where `nb` holds a real neighbour
    degree: np.ndarray  # (n,) number of neighbours
    islands: int  # regions with data but no neighbour that also has data (dropped)


def build_weights(neighbors: list[set[int]], valid: np.ndarray) -> Weights:
    """Restrict a neighbour structure to regions with data and drop any left with no neighbour."""
    valid = np.asarray(valid, dtype=bool)
    keep = valid.copy()
    for i, nbrs in enumerate(neighbors):
        if valid[i] and not any(valid[j] for j in nbrs):
            keep[i] = False
    index = np.flatnonzero(keep)
    pos = {int(g): p for p, g in enumerate(index)}
    lists = [[pos[j] for j in sorted(neighbors[int(g)]) if j in pos] for g in index]
    degree = np.array([len(item) for item in lists], dtype=int)
    width = max(int(degree.max()), 1) if len(index) else 1
    nb = np.zeros((len(index), width), dtype=int)
    mask = np.zeros((len(index), width), dtype=bool)
    for r, item in enumerate(lists):
        nb[r, : len(item)] = item
        mask[r, : len(item)] = True
    return Weights(index, nb, mask, degree, islands=int((valid & ~keep).sum()))


def _lag(z: np.ndarray, w: Weights) -> np.ndarray:
    return (z[w.nb] * w.mask).sum(axis=1) / np.maximum(w.degree, 1)


@dataclass
class GlobalMoran:
    i: float
    expected: float
    z_score: float  # against the permutation distribution
    p_value: float  # two-sided pseudo p-value
    n: int
    islands: int
    permutations: int


def morans_i(values, w: Weights, permutations: int = 999, seed: int = 0) -> GlobalMoran:
    """Global Moran's I with a random-permutation reference distribution.

    With row-standardised weights, ``I = sum_i z_i * lag_i / sum_i z_i^2`` where ``z`` is the
    deviation from the mean.  ``I`` near +1 means like values sit next to like values, near
    ``E[I] = -1/(n-1)`` means no spatial pattern, near -1 a checkerboard.
    """
    x = np.asarray(values, dtype=float)[w.index]
    n = len(x)
    if n < 3:
        raise ValueError("need at least 3 connected regions with data")
    z = x - x.mean()
    denom = float((z * z).sum())
    if denom == 0.0:
        raise ValueError("all values identical: Moran's I is undefined")
    observed = float((z * _lag(z, w)).sum() / denom)
    rng = np.random.default_rng(seed)
    sims = np.empty(permutations)
    for b in range(permutations):
        zp = z[rng.permutation(n)]
        sims[b] = float((zp * _lag(zp, w)).sum() / denom)
    expected = -1.0 / (n - 1)
    spread = float(sims.std(ddof=1)) if permutations > 1 else float("nan")
    z_score = (observed - float(sims.mean())) / spread if spread else float("nan")
    extreme = int((np.abs(sims - expected) >= abs(observed - expected)).sum())
    p_value = (extreme + 1) / (permutations + 1)
    return GlobalMoran(observed, expected, z_score, p_value, n, w.islands, permutations)


CLUSTER_LABELS = {
    0: "Not significant",
    1: "High-High (hot spot)",
    2: "Low-Low (cold spot)",
    3: "High-Low outlier",
    4: "Low-High outlier",
}


@dataclass
class LocalMoran:
    local_i: np.ndarray  # per kept region
    lag: np.ndarray
    p_value: np.ndarray  # conditional-permutation pseudo p-values
    quadrant: np.ndarray  # 1 HH, 2 LH, 3 LL, 4 HL   (Anselin's quadrant numbering)
    cluster: np.ndarray  # CLUSTER_LABELS codes after multiple-testing control
    alpha: float
    n_significant: int  # after the multiple-testing rule in force
    n_uncorrected: int  # regions with pseudo p <= alpha before any correction
    permutations: int


def benjamini_hochberg(p: np.ndarray, alpha: float) -> np.ndarray:
    """Boolean mask of hypotheses rejected by the Benjamini-Hochberg step-up procedure."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    passed = p[order] <= alpha * (np.arange(1, m + 1) / m)
    out = np.zeros(m, dtype=bool)
    if passed.any():
        cutoff = int(np.flatnonzero(passed).max())
        out[order[: cutoff + 1]] = True
    return out


def _sample_others(
    rng: np.random.Generator, n: int, degree: np.ndarray, width: int
) -> np.ndarray:
    """For each row i, ``degree[i]`` distinct indices drawn uniformly from ``range(n)`` minus ``i``."""
    rows = np.arange(n)[:, None]
    col = np.arange(width)[None, :]
    live = col < degree[:, None]
    draw = rng.integers(0, n - 1, size=(n, width))
    draw += draw >= rows  # skip i itself
    while True:
        sortable = np.where(
            live, draw, -1 - col
        )  # unique sentinels so padding never collides
        srt = np.sort(sortable, axis=1)
        dup_rows = (srt[:, 1:] == srt[:, :-1]).any(axis=1)
        if not dup_rows.any():
            return draw
        k = int(dup_rows.sum())
        redraw = rng.integers(0, n - 1, size=(k, width))
        redraw += redraw >= np.flatnonzero(dup_rows)[:, None]
        draw[dup_rows] = redraw


def local_moran(
    values,
    w: Weights,
    permutations: int = 999,
    alpha: float = 0.05,
    seed: int = 0,
    fdr: bool = True,
) -> LocalMoran:
    """Local Moran's I (LISA) with conditional permutation and Benjamini-Hochberg FDR control.

    ``I_i = z_i * lag_i / m2`` with ``m2 = mean(z^2)``.  Significance holds ``z_i`` fixed and
    reshuffles *other* regions' values among ``i``'s neighbour slots (conditional permutation).
    Testing thousands of regions at ``alpha = .05`` would flag ~5% by chance alone, so by
    default the flags are FDR-controlled; pass ``fdr=False`` for the raw per-region test.
    """
    x = np.asarray(values, dtype=float)[w.index]
    n = len(x)
    if n < 3:
        raise ValueError("need at least 3 connected regions with data")
    z = x - x.mean()
    m2 = float((z * z).mean())
    if m2 == 0.0:
        raise ValueError("all values identical: Moran's I is undefined")
    lag = _lag(z, w)
    local_i = z * lag / m2

    rng = np.random.default_rng(seed)
    width = w.nb.shape[1]
    degree = np.maximum(w.degree, 1)
    ge = np.zeros(n, dtype=int)
    for _ in range(permutations):
        picks = _sample_others(rng, n, w.degree, width)
        sim_lag = (z[picks] * w.mask).sum(axis=1) / degree
        sim_i = z * sim_lag / m2
        # one-sided in the direction of the observed statistic, as in esda
        ge += np.where(local_i >= 0, sim_i >= local_i, sim_i <= local_i)
    p = (ge + 1) / (permutations + 1)

    quadrant = np.where(z >= 0, np.where(lag >= 0, 1, 4), np.where(lag >= 0, 2, 3))
    significant = benjamini_hochberg(p, alpha) if fdr else p <= alpha
    # cluster code: 1 HH, 2 LL, 3 HL, 4 LH, 0 not significant
    code_of_quadrant = {1: 1, 3: 2, 4: 3, 2: 4}
    cluster = np.where(significant, np.vectorize(code_of_quadrant.get)(quadrant), 0)
    return LocalMoran(
        local_i,
        lag,
        p,
        quadrant,
        cluster,
        alpha,
        int(significant.sum()),
        int((p <= alpha).sum()),
        permutations,
    )
