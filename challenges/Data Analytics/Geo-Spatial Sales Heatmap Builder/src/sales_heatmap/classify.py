"""Choropleth classification schemes: quantile, equal interval, standard deviation, Fisher-Jenks.

Every scheme returns *edges*: ``[min, upper_1, ..., upper_k-1, max]``.  A value ``v``
falls in class ``i`` when ``edges[i] < v <= edges[i + 1]`` (the first class also
includes ``min``).  That right-closed convention matches ``jenkspy`` and
``mapclassify``, which is what the tests compare against.

The number of classes actually produced can be *smaller* than requested: heavy ties
make several quantile edges coincide, and a standard-deviation scheme drops edges
that fall outside the data range.  Callers should use ``len(edges) - 1``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

SCHEMES = ("quantile", "equal_interval", "jenks", "std_dev")
SCHEME_LABELS = {
    "quantile": "Quantile",
    "equal_interval": "Equal interval",
    "jenks": "Natural breaks (Jenks)",
    "std_dev": "Standard deviation",
}

# Fisher-Jenks is O(k * n^2) in the number of *distinct* values; above this we run it on
# an evenly spaced sample of the sorted distinct values (same idea as mapclassify's
# FisherJenksSampled) rather than let a million-row input stall the report.
JENKS_MAX_DISTINCT = 4000


def _clean(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("cannot classify: no finite values")
    return arr


def _dedupe_edges(edges: np.ndarray) -> np.ndarray:
    """Drop repeated edges (zero-width classes) while keeping min and max."""
    out = [edges[0]]
    for e in edges[1:]:
        if e > out[-1]:
            out.append(e)
    if len(out) == 1:  # every value identical: one degenerate class
        out.append(out[0])
    return np.asarray(out, dtype=float)


def quantile_edges(values, k: int) -> np.ndarray:
    arr = _clean(values)
    edges = np.quantile(arr, np.linspace(0.0, 1.0, k + 1))
    return _dedupe_edges(edges)


def equal_interval_edges(values, k: int) -> np.ndarray:
    arr = _clean(values)
    lo, hi = float(arr.min()), float(arr.max())
    if lo == hi:
        return np.array([lo, hi])
    return np.linspace(lo, hi, k + 1)


def std_dev_edges(values, k: int) -> np.ndarray:
    """Classes one standard deviation wide, centred on the mean (odd ``k`` puts the mean mid-class).

    Uses the population standard deviation (ddof=0), matching ``mapclassify.StdMean``.
    Edges outside the data range are dropped, so a right-skewed variable yields fewer
    than ``k`` classes: that is the scheme telling you the distribution isn't symmetric.
    """
    arr = _clean(values)
    lo, hi = float(arr.min()), float(arr.max())
    sd = float(arr.std())
    if sd == 0.0:
        return np.array([lo, hi])
    mean = float(arr.mean())
    interior = np.array([mean + sd * (i - k / 2.0) for i in range(1, k)])
    interior = interior[(interior > lo) & (interior < hi)]
    return np.concatenate(([lo], interior, [hi]))


def jenks_edges(values, k: int) -> np.ndarray:
    """Exact Fisher-Jenks natural breaks: the partition into ``k`` classes minimising within-class variance.

    Dynamic programming over the sorted *distinct* values, each weighted by its
    multiplicity, so ties are never split and the optimum equals the one over the full
    (duplicated) data.  ``cost(i, j)`` is the weighted sum of squared deviations of
    distinct values ``i..j`` from their weighted mean, from prefix sums.
    """
    arr = _clean(values)
    uniq, counts = np.unique(arr, return_counts=True)
    m = len(uniq)
    if m <= k:
        return (
            np.concatenate(([uniq[0]], uniq)) if m > 1 else np.array([uniq[0], uniq[0]])
        )
    if m > JENKS_MAX_DISTINCT:
        pick = np.unique(np.linspace(0, m - 1, JENKS_MAX_DISTINCT).astype(int))
        uniq, counts = uniq[pick], np.ones(len(pick))
        m = len(uniq)

    w = counts.astype(float)
    # Shift by the mean before squaring: prefix-sum variance is numerically fragile for
    # large offsets (retail sales in the billions vs. differences of a few thousand).
    x = uniq - np.average(uniq, weights=w)
    cw = np.concatenate(([0.0], np.cumsum(w)))
    cx = np.concatenate(([0.0], np.cumsum(w * x)))
    cxx = np.concatenate(([0.0], np.cumsum(w * x * x)))

    def cost_from(i: np.ndarray, j: int) -> np.ndarray:
        """SSD of distinct values i..j inclusive, for an array of start indices i."""
        sw = cw[j + 1] - cw[i]
        sx = cx[j + 1] - cx[i]
        sxx = cxx[j + 1] - cxx[i]
        return np.maximum(sxx - sx * sx / sw, 0.0)

    inf = np.inf
    best = np.full((k, m), inf)
    back = np.zeros((k, m), dtype=int)
    for j in range(m):
        best[0, j] = cost_from(np.array([0]), j)[0]
    for c in range(1, k):
        for j in range(c, m):
            starts = np.arange(c, j + 1)  # class c covers distinct values start..j
            total = best[c - 1, starts - 1] + cost_from(starts, j)
            pick = int(np.argmin(total))
            best[c, j] = total[pick]
            back[c, j] = starts[pick]

    uppers = []
    j = m - 1
    for c in range(k - 1, 0, -1):
        start = back[c, j]
        uppers.append(uniq[start - 1])  # last value of the previous class
        j = start - 1
    uppers.reverse()
    return np.concatenate(([uniq[0]], uppers, [uniq[-1]]))


_SCHEME_FUNCS: dict[str, Callable[[object, int], np.ndarray]] = {
    "quantile": quantile_edges,
    "equal_interval": equal_interval_edges,
    "jenks": jenks_edges,
    "std_dev": std_dev_edges,
}


def compute_edges(values, scheme: str, k: int) -> np.ndarray:
    if scheme not in _SCHEME_FUNCS:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")
    if k < 2:
        raise ValueError("need at least 2 classes")
    return _SCHEME_FUNCS[scheme](values, k)


def assign(values, edges) -> np.ndarray:
    """Class index per value (right-closed, first class includes the minimum).  NaN -> -1."""
    arr = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    n_classes = max(len(edges) - 1, 1)
    out = np.searchsorted(edges[1:-1], arr, side="left")
    out = np.clip(out, 0, n_classes - 1)
    return np.where(np.isfinite(arr), out, -1)


def gvf(values, classes) -> float:
    """Goodness of Variance Fit = 1 - SDCM / SDAM (Jenks): 1.0 means classes explain all the variance."""
    arr = np.asarray(values, dtype=float)
    cls = np.asarray(classes)
    mask = np.isfinite(arr) & (cls >= 0)
    arr, cls = arr[mask], cls[mask]
    sdam = float(((arr - arr.mean()) ** 2).sum())
    if sdam == 0.0:
        return 1.0
    sdcm = 0.0
    for c in np.unique(cls):
        grp = arr[cls == c]
        sdcm += float(((grp - grp.mean()) ** 2).sum())
    return 1.0 - sdcm / sdam


def class_counts(classes, n_classes: int) -> list[int]:
    cls = np.asarray(classes)
    return [int((cls == i).sum()) for i in range(n_classes)]
