"""Window drift statistics against a training baseline.

Numeric columns (the score and numeric features) are binned once on the
**training** data into baseline-quantile bins with open-ended outer bins, so live
values beyond the training range land in the outer bins instead of vanishing.
PSI and Jensen-Shannon compare bin proportions; KS and Wasserstein compare the
raw values against a stored reference sample. Categorical columns use their
training categories plus an "other" bin for anything unseen. Every statistic is
larger when the window drifts further, finite for empty bins and constant
columns, and 0 when the window matches the baseline exactly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps
from scipy.spatial import distance

N_BINS = 10
EPSILON = 1e-4  # smoothing floor for an empty bin


def _smooth(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPSILON, None)
    return p / p.sum()


def psi(base: np.ndarray, live: np.ndarray) -> float:
    """Population stability index between two bin-proportion vectors."""
    p, q = _smooth(base), _smooth(live)
    return float(np.sum((q - p) * np.log(q / p)))


def js(base: np.ndarray, live: np.ndarray) -> float:
    """Jensen-Shannon *distance* (base 2, so in [0, 1]) between proportion vectors."""
    return float(distance.jensenshannon(_smooth(base), _smooth(live), base=2))


def ks(reference: np.ndarray, live: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    return float(sps.ks_2samp(reference, live).statistic)


def wasserstein(reference: np.ndarray, live: np.ndarray) -> float:
    """First Wasserstein distance over the reference standard deviation.

    Dividing by the baseline std makes columns comparable (the same convention as
    Evidently); a constant baseline column is measured in absolute units.
    """
    scale = float(np.std(reference))
    return float(
        sps.wasserstein_distance(reference, live) / (scale if scale > 0 else 1.0)
    )


def class_rate_shift(base_rate: float, live_binary: np.ndarray) -> float:
    """Absolute difference between the live and baseline positive rate."""
    live_binary = np.asarray(live_binary)
    return 0.0 if live_binary.size == 0 else float(abs(live_binary.mean() - base_rate))


@dataclass
class _Numeric:
    reference: np.ndarray
    edges: np.ndarray  # interior bin edges; outer bins are open-ended
    proportions: np.ndarray

    def bin_counts(self, values: np.ndarray) -> np.ndarray:
        idx = np.digitize(values, self.edges)
        return np.bincount(idx, minlength=len(self.edges) + 1)


@dataclass
class _Categorical:
    categories: np.ndarray  # sorted training categories; last bin is "other"
    proportions: np.ndarray

    def bin_counts(self, values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.categories, values)
        known = (idx < len(self.categories)) & (
            self.categories[np.minimum(idx, len(self.categories) - 1)] == values
        )
        idx = np.where(known, idx, len(self.categories))
        return np.bincount(idx, minlength=len(self.categories) + 1)


class Baseline:
    """Per-column reference built from the training split only."""

    def __init__(self, columns: dict[str, _Numeric | _Categorical]):
        self.columns = columns

    @classmethod
    def fit(
        cls,
        columns: Mapping[str, np.ndarray],
        categorical: Iterable[str] = (),
        n_bins: int = N_BINS,
    ) -> Baseline:
        categorical = set(categorical)
        out: dict[str, _Numeric | _Categorical] = {}
        for name, values in columns.items():
            values = np.asarray(values)
            if name in categorical:
                cats = np.unique(values)
                counts = np.array(
                    [(values == c).sum() for c in cats] + [0], dtype=float
                )
                out[name] = _Categorical(cats, counts / counts.sum())
            else:
                x = values.astype(float)
                edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
                col = _Numeric(x, edges, np.zeros(len(edges) + 1))
                col.proportions = col.bin_counts(x) / len(x)
                out[name] = col
        return cls(out)

    def window_stats(self, column: str, live: np.ndarray) -> dict[str, float]:
        """Drift statistics of one window of ``column`` against the baseline."""
        if column not in self.columns:
            raise KeyError(f"unknown column {column!r}; known: {sorted(self.columns)}")
        col = self.columns[column]
        live = np.asarray(live)
        counts = col.bin_counts(
            live if isinstance(col, _Categorical) else live.astype(float)
        )
        live_p = counts / max(counts.sum(), 1)
        if isinstance(col, _Categorical):
            expected = _smooth(col.proportions) * counts.sum()
            return {
                "psi": psi(col.proportions, live_p),
                "js": js(col.proportions, live_p),
                "chi2": float(sps.chisquare(counts, expected).statistic),
            }
        x = live.astype(float)
        return {
            "psi": psi(col.proportions, live_p),
            "ks": ks(col.reference, x),
            "js": js(col.proportions, live_p),
            "wasserstein": wasserstein(col.reference, x),
        }
