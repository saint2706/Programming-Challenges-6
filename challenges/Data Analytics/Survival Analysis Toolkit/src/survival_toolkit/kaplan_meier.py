"""Kaplan-Meier survival, Nelson-Aalen hazard, quantiles and restricted mean survival time.

Everything is computed from the counts at each distinct event time: ``n`` at risk (``entry < t <=
time``, so late entry is handled), ``d`` events. Greenwood's formula gives the variance; four
confidence-interval transforms are offered. Defaults follow R's ``survival`` (log-log intervals for
a plot, the midpoint rule for a median that sits exactly on 0.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from survival_toolkit.types import Survival, SurvivalError, n_at_risk

CIMethod = Literal["log-log", "log", "plain", "arcsin"]
CI_METHODS: tuple[str, ...] = ("log-log", "log", "plain", "arcsin")


@dataclass(frozen=True)
class KaplanMeier:
    """A fitted Kaplan-Meier curve; arrays are indexed by distinct event time."""

    data: Survival
    time: np.ndarray
    n_risk: np.ndarray
    n_event: np.ndarray
    n_censor: np.ndarray  # censored in [t_i, t_{i+1}), the next event time or the end
    survival: np.ndarray
    greenwood: np.ndarray  # cumulative sum of d / (n (n - d)); inf once S reaches 0
    lower: np.ndarray
    upper: np.ndarray
    ci_method: str
    alpha: float

    @property
    def se(self) -> np.ndarray:
        """Greenwood standard error of S(t); NaN where S(t) is 0."""
        with np.errstate(invalid="ignore"):
            return np.where(
                self.survival > 0, self.survival * np.sqrt(self.greenwood), np.nan
            )

    @property
    def cumulative_hazard(self) -> np.ndarray:
        """Nelson-Aalen estimate of H(t): the running sum of d / n."""
        return np.cumsum(self.n_event / self.n_risk)

    @property
    def cumulative_hazard_se(self) -> np.ndarray:
        return np.sqrt(np.cumsum(self.n_event / self.n_risk**2))

    @property
    def fleming_harrington(self) -> np.ndarray:
        """Survival from the Nelson-Aalen hazard, exp(-H(t)); never below KM, close to it."""
        return np.exp(-self.cumulative_hazard)

    def _index(self, t: float | np.ndarray) -> np.ndarray:
        return np.searchsorted(self.time, np.asarray(t, dtype=float), side="right") - 1

    def at(self, t: float | np.ndarray) -> np.ndarray:
        """S(t); 1 before the first event, the last value after the last."""
        idx = self._index(t)
        return np.where(idx < 0, 1.0, self.survival[np.maximum(idx, 0)])

    def ci_at(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = self._index(t)
        keep = np.maximum(idx, 0)
        return (
            np.where(idx < 0, 1.0, self.lower[keep]),
            np.where(idx < 0, 1.0, self.upper[keep]),
        )

    def quantile(self, p: float) -> float:
        """Smallest t with S(t) <= 1 - p (so ``quantile(0.5)`` is the median); inf if never.

        When S equals the target exactly over an interval, the interval's midpoint is returned,
        as R does.
        """
        return float(_first_at_or_below(self.time, self.survival, 1.0 - p))

    def median(self) -> float:
        return self.quantile(0.5)

    def median_ci(self) -> tuple[float, float]:
        """Brookmeyer-Crowley interval: where the confidence band crosses 0.5.

        The lower limit is when the lower band first reaches 0.5 (the median could be that early),
        the upper limit is when the upper band does; inf when it never gets there.
        """
        lo_band = np.where(self.survival == 0, 0.0, self.lower)
        hi_band = np.where(self.survival == 0, 0.0, self.upper)
        lo = _first_at_or_below(self.time, lo_band, 0.5, midpoint=False)
        hi = _first_at_or_below(self.time, hi_band, 0.5, midpoint=False)
        return float(lo), float(hi)

    def restricted_mean(self, tau: float) -> tuple[float, float]:
        """Restricted mean survival time up to ``tau`` and its Greenwood-based standard error.

        The area under S(t) on [0, tau]: the average event-free time within the horizon. It needs
        no extrapolation, unlike the mean, which is undefined when the curve never reaches 0.
        """
        horizon = float(self.data.time.max())
        if not 0 < tau <= horizon:
            raise SurvivalError(
                f"tau must be in (0, {horizon:g}] (the longest follow-up)"
            )
        in_range = self.time < tau
        knots = np.concatenate([[0.0], self.time[in_range], [tau]])
        levels = np.concatenate([[1.0], self.survival[in_range]])
        widths = np.diff(knots)
        area = float(np.sum(levels * widths))
        # A_i = area under S from t_i to tau, for each event time before tau
        tail = np.cumsum((levels * widths)[::-1])[
            ::-1
        ]  # tail[j] = area from knots[j] to tau
        a = tail[1:] if in_range.any() else np.empty(0)
        d = self.n_event[in_range]
        n = self.n_risk[in_range]
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(n > d, d / (n * (n - d)), 0.0)
        return area, float(np.sqrt(np.sum(a**2 * terms)))

    def table(self, at: np.ndarray) -> np.ndarray:
        """Number at risk at each of the times in ``at`` (for a risk table under a plot)."""
        return n_at_risk(self.data, np.asarray(at, dtype=float), closed_entry=True)


def _first_at_or_below(
    time: np.ndarray, curve: np.ndarray, level: float, *, midpoint: bool = True
) -> float:
    """First time the curve is <= level; the midpoint rule applies when it sits on level."""
    hit = np.flatnonzero(curve <= level + 1e-12)
    if hit.size == 0:
        return np.inf
    i = int(hit[0])
    if midpoint and abs(curve[i] - level) <= 1e-12 and i + 1 < time.size:
        return float((time[i] + time[i + 1]) / 2)
    return float(time[i])


def fit(
    data: Survival, *, ci: CIMethod = "log-log", alpha: float = 0.05
) -> KaplanMeier:
    """Fit the Kaplan-Meier estimator.

    Raises ``SurvivalError`` for an unknown ``ci`` or when there are no events (a curve of 1).
    """
    if ci not in CI_METHODS:
        raise SurvivalError(f"ci must be one of {', '.join(CI_METHODS)}")
    if not 0 < alpha < 1:
        raise SurvivalError("alpha must be between 0 and 1")
    if data.n_events == 0:
        raise SurvivalError("no events were observed, so there is no curve to estimate")

    event_times, deaths = np.unique(data.time[data.event], return_counts=True)
    n = n_at_risk(data, event_times)
    survival = np.cumprod(1.0 - deaths / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        greenwood = np.cumsum(np.where(n > deaths, deaths / (n * (n - deaths)), np.inf))
    censor_times = data.time[~data.event]
    bins = np.searchsorted(event_times, censor_times, side="right") - 1
    censored = np.bincount(bins[bins >= 0], minlength=event_times.size)
    lower, upper = _band(survival, greenwood, ci, stats.norm.ppf(1 - alpha / 2))
    return KaplanMeier(
        data=data,
        time=event_times,
        n_risk=n,
        n_event=deaths,
        n_censor=censored,
        survival=survival,
        greenwood=greenwood,
        lower=lower,
        upper=upper,
        ci_method=ci,
        alpha=alpha,
    )


def _band(
    s: np.ndarray, greenwood: np.ndarray, method: str, z: float
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(s.shape, np.nan)
    upper = np.full(s.shape, np.nan)
    ok = (s > 0) & np.isfinite(greenwood)
    flat = ok & (
        s >= 1.0
    )  # before any drop the variance is 0: the interval is the point
    lower[flat] = upper[flat] = 1.0
    ok &= ~flat
    g = greenwood[ok]
    sv = s[ok]
    if method == "plain":
        half = z * sv * np.sqrt(g)
        lo, hi = sv - half, sv + half
    elif method == "log":
        half = z * np.sqrt(g)
        lo, hi = sv * np.exp(-half), sv * np.exp(half)
    elif method == "log-log":
        spread = z * np.sqrt(g) / np.abs(np.log(sv))
        lo, hi = sv ** np.exp(spread), sv ** np.exp(-spread)
    else:  # arcsin square root
        theta = np.arcsin(np.sqrt(sv))
        half = z * np.sqrt(g) * np.sqrt(sv) / (2 * np.sqrt(1 - sv))
        lo = np.sin(np.maximum(0.0, theta - half)) ** 2
        hi = np.sin(np.minimum(np.pi / 2, theta + half)) ** 2
    lower[ok] = np.clip(lo, 0.0, 1.0)
    upper[ok] = np.clip(hi, 0.0, 1.0)
    return lower, upper
