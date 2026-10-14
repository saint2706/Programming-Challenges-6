"""Fit the models on the fit window, then find, adjust and combine comps for any sale.

The rule that makes this honest: a subject's comps are drawn only from sales **strictly before**
the subject's sale date. Because the table is date-sorted, that is a prefix of the table whose
length is ``before[i]``; no model is consulted about later rows. Everything fitted (scaler, linear
hedonic and its market index, GBM, similarity weights) sees the first ``n_fit`` rows only, so
the evaluation windows stay out of every weight. The comps pool itself is allowed to grow past
``n_fit`` (``pool="prior"``): an appraiser uses last week's sales too. ``pool="fit"`` freezes it
to the fit window for comparison.

Appraisal-style estimate for a subject ``s`` and comp ``c``::

    adjusted_c = price_c * exp(index[month_s] - index[month_c])      # market time
                         * exp(sum_f beta_f * (z_s,f - z_c,f))       # per-attribute adjustments

and the estimate is the median (or geometric mean) of the adjusted comps.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from comp_finder import features
from comp_finder.hedonic import LinearHedonic
from comp_finder.similarity import LeafProximity, WeightedEuclid, top_k
from comp_finder.trees import HedonicGBM

METHODS = {
    "knn": "Scaled kNN",
    "ridge": "Ridge-weighted distance",
    "hedonic": "Hedonic-weighted distance (GBM SHAP)",
    "leaf": "GBM leaf proximity",
}
ADJUST_MODES = ("none", "time", "full", "gbm")
COMBINE = ("median", "mean")
POOLS = ("prior", "fit")
SELECTIONS = ("top", "bracket")


@dataclass(frozen=True)
class Settings:
    k: int = 5
    extrapolate: str = "trend"
    combine: str = "median"
    adjust: str = "full"
    pool: str = "prior"
    selection: str = "top"

    def __post_init__(self):
        if self.k < 1:
            raise ValueError("k must be at least 1")
        for value, allowed in (
            (self.combine, COMBINE),
            (self.adjust, ADJUST_MODES),
            (self.pool, POOLS),
            (self.selection, SELECTIONS),
        ):
            if value not in allowed:
                raise ValueError(f"{value!r} not in {allowed}")


@dataclass
class Estimate:
    rows: np.ndarray  # subject rows
    comps: np.ndarray  # (m, k) comp rows, -1 where fewer than k earlier sales exist
    adjusted: np.ndarray  # (m, k) adjusted comp prices, NaN where invalid
    price: np.ndarray  # (m,) estimate, NaN when no comp is available
    time_log: np.ndarray  # (m, k)
    attr_log: np.ndarray  # (m, k, n_attributes) per-attribute log adjustments
    gross: (
        np.ndarray
    )  # (m, k) sum of |attribute adjustments| (log points, about percent/100)
    net: (
        np.ndarray
    )  # (m, k) sum of attribute adjustments; > 0 means the comp is inferior

    @property
    def bracketed(self) -> np.ndarray:
        """True when the comps include a better and a worse property than the subject."""
        valid = self.comps >= 0
        return ((self.net > 0) & valid).any(axis=1) & ((self.net < 0) & valid).any(
            axis=1
        )


class Engine:
    """Fitted models plus every sale's representation; ``fitted`` reuses models for extra rows."""

    def __init__(self, sales: pl.DataFrame, n_fit: int, fitted: tuple | None = None):
        if not 50 <= n_fit <= sales.height:
            raise ValueError("n_fit must leave at least 50 fit rows")
        sold = sales["sold"].to_numpy()
        if not (sold[:-1] <= sold[1:]).all():
            raise ValueError("sales must be sorted by sale date")
        self.sales, self.n_fit = sales, n_fit
        self.sold_days = sold.astype("datetime64[D]").astype(np.int64)
        self.before = np.searchsorted(self.sold_days, self.sold_days, side="left")
        self.ids = sales["id"].to_numpy()
        self.zip = sales["zipcode"].to_numpy()
        self.month = sales["month"].to_numpy()
        self.price = sales["price"].to_numpy()
        self.log_price = np.log(self.price)

        raw = features.build(sales)
        self.xy = raw[:, :2].copy()  # unclipped km east/north, for real distances
        fit = slice(0, n_fit)
        if fitted is None:
            scaler = features.Scaler.fit(raw[fit])
            z = scaler.transform(raw)
            hedonic = LinearHedonic.fit(
                z[fit], self.zip[fit], self.month[fit], self.log_price[fit]
            )
            gbm = HedonicGBM.fit(
                z[fit], self.log_price[fit] - hedonic.index(self.month[fit])
            )
        else:
            scaler, hedonic, gbm = fitted
        self.scaler, self.hedonic, self.gbm = scaler, hedonic, gbm
        self.raw = scaler.clip(raw)
        self.z = scaler.transform(raw)
        self.gbm_pred = gbm.predict(self.z)  # time-adjusted log price for every row
        self.metrics = {
            "knn": WeightedEuclid(self.z),
            "ridge": WeightedEuclid(self.z, hedonic.ridge_weights()),
            "hedonic": WeightedEuclid(self.z, gbm.shap_weight),
            "leaf": LeafProximity(gbm, self.z),
        }

    def extended(self, extra: pl.DataFrame) -> Engine:
        """The same fitted models over the table plus later ``extra`` rows (a custom subject)."""
        both = pl.concat([self.sales, extra.select(self.sales.columns)])
        return Engine(both, self.n_fit, (self.scaler, self.hedonic, self.gbm))

    # -- finding comps ---------------------------------------------------------------------

    def pool_limit(self, rows: np.ndarray, pool: str = "prior") -> np.ndarray:
        """How many leading table rows each subject may draw comps from."""
        lim = self.before[rows]
        return np.minimum(lim, self.n_fit) if pool == "fit" else lim

    def rank(
        self,
        method: str,
        rows: np.ndarray,
        kmax: int = 12,
        pool: str = "prior",
        include_own: bool = False,
        chunk: int = 256,
    ) -> np.ndarray:
        """The ``kmax`` most similar earlier sales per subject, best first; -1 pads short pools."""
        metric = self.metrics[method]
        rows = np.asarray(rows)
        out = np.full((len(rows), kmax), -1, dtype=np.int64)
        for a in range(0, len(rows), chunk):
            r = rows[a : a + chunk]
            lim = self.pool_limit(r, pool)
            n_pool = int(lim.max())
            if n_pool == 0:
                continue
            cand = np.arange(n_pool)
            s = metric.sim(r, cand).astype(np.float64)
            s[cand[None, :] >= lim[:, None]] = -np.inf
            if not include_own:
                s[self.ids[r][:, None] == self.ids[None, :n_pool]] = -np.inf
            idx, sc = top_k(s, kmax)
            idx[~np.isfinite(sc)] = -1
            out[a : a + chunk, : idx.shape[1]] = idx
        return out

    # -- adjusting and combining -----------------------------------------------------------

    def estimate(self, rows: np.ndarray, comps: np.ndarray, s: Settings) -> Estimate:
        """Adjust the first ``s.k`` comps (or a bracket-aware choice of ``s.k``) and combine them."""
        rows = np.asarray(rows)
        if s.selection == "bracket":
            comps = self._bracket(rows, comps, s)
        comps = comps[:, : s.k]
        valid = comps >= 0
        safe = np.where(valid, comps, 0)
        index = self.hedonic.index(self.month, s.extrapolate)
        time_log = index[rows][:, None] - index[safe]
        attr_log = self.hedonic.adjustments(self.z[rows][:, None, :], self.z[safe])
        gross, net = np.abs(attr_log).sum(axis=2), attr_log.sum(axis=2)
        if s.adjust == "none":
            log_adj = np.zeros_like(time_log)
        elif s.adjust == "time":
            log_adj = time_log
        elif s.adjust == "full":
            log_adj = time_log + net
        else:  # "gbm": time-adjust, then let the GBM price every difference, location included
            log_adj = time_log + (self.gbm_pred[rows][:, None] - self.gbm_pred[safe])
        adjusted = np.where(valid, self.price[safe] * np.exp(log_adj), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", RuntimeWarning
            )  # all-NaN rows are reported as NaN
            if s.combine == "median":
                price = np.nanmedian(adjusted, axis=1)
            else:
                price = np.exp(np.nanmean(np.log(adjusted), axis=1))
        return Estimate(rows, comps, adjusted, price, time_log, attr_log, gross, net)

    def _bracket(self, rows: np.ndarray, cands: np.ndarray, s: Settings) -> np.ndarray:
        """Top ``k`` by similarity, swapping so a better and a worse property are both present."""
        k = s.k
        if k < 2:
            return cands[:, :k].copy()
        valid = cands >= 0
        safe = np.where(valid, cands, 0)
        net = self.hedonic.adjustments(self.z[rows][:, None, :], self.z[safe]).sum(
            axis=2
        )
        out = cands[:, :k].copy()
        for i in range(len(rows)):
            sign = np.sign(net[i])
            sign[~valid[i]] = 0
            have = set(sign[:k][sign[:k] != 0])
            for need in (1.0, -1.0):
                if need in have or len(have) == 0:
                    continue
                spare = np.flatnonzero(sign[k:] == need)
                if len(spare) == 0:
                    continue
                other = np.flatnonzero(sign[:k] == -need)
                if len(other) == 0:
                    continue
                # swap in the best-ranked spare for the worst-ranked comp of the over-represented side
                out[i, other[-1]] = cands[i, k + spare[0]]
                sign[other[-1]] = need
                have = set(sign[:k][sign[:k] != 0])
        return out

    def avm(self, rows: np.ndarray, extrapolate: str = "trend") -> np.ndarray:
        """The pure model: GBM on time-adjusted log price, rolled to the sale month."""
        rows = np.asarray(rows)
        return np.exp(
            self.gbm_pred[rows] + self.hedonic.index(self.month[rows], extrapolate)
        )

    def predict(
        self, method: str, rows: np.ndarray, s: Settings, include_own: bool = False
    ) -> Estimate:
        """Comps plus estimate for ``rows`` under ``s``."""
        kmax = s.k * 3 if s.selection == "bracket" else s.k
        comps = self.rank(method, rows, kmax, s.pool, include_own)
        return self.estimate(rows, comps, s)


def with_settings(s: Settings, **kw) -> Settings:
    return replace(s, **kw)
