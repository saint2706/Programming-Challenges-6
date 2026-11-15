"""Phase I (estimate the in-control process) and Phase II (watch the stream point by point).

Real metric streams are autocorrelated and often seasonal, which breaks the iid assumption every
control chart is designed under. `Prewhitener` turns the raw value into a one-step-ahead
residual (optional seasonal difference, then an AR(1) fit) so the chart watches surprises,
not the daily cycle.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from spc_stream.charts import CUSUM, EWMA, Point, Shewhart

D2 = 1.128  # E[|X1 - X2|] / sigma for normal data; the moving-range sigma estimator divides by it


@dataclass(frozen=True, slots=True)
class Baseline:
    mean: float
    sigma: float


def estimate_baseline(values: list[float]) -> Baseline:
    """Mean and moving-range sigma (MR-bar / d2), the standard individuals-chart estimate.

    The moving range ignores slow level changes inside the baseline, so it is less inflated by
    drift than the plain standard deviation.
    """
    if len(values) < 2:
        raise ValueError("Phase I needs at least two points")
    mean = sum(values) / len(values)
    mr = sum(abs(b - a) for a, b in itertools.pairwise(values)) / (len(values) - 1)
    sigma = mr / D2
    if sigma <= 0 or not math.isfinite(sigma):
        raise ValueError("Phase I data has no variation; cannot set control limits")
    return Baseline(mean, sigma)


class Prewhitener:
    """x_t -> residual of (seasonal difference, then AR(1)). season=0 and ar1=False is identity."""

    def __init__(self, season: int = 0, ar1: bool = False) -> None:
        self.season, self.ar1 = season, ar1
        self.mu = 0.0
        self.phi = 0.0
        self._raw: deque[float] = deque(maxlen=max(season, 1))
        self._prev: float | None = None

    def _diff(self, x: float) -> float | None:
        if not self.season:
            return x
        out = x - self._raw[0] if len(self._raw) == self.season else None
        self._raw.append(x)
        return out

    def fit(self, values: list[float]) -> list[float]:
        """Fit on Phase I data and return the Phase I residuals (warm-up points are dropped)."""
        d = [v for v in map(self._diff, values) if v is not None]
        if self.ar1 and len(d) > 2:
            self.mu = sum(d) / len(d)
            c = [v - self.mu for v in d]
            denom = sum(v * v for v in c)
            self.phi = (
                sum(a * b for a, b in itertools.pairwise(c)) / denom if denom else 0.0
            )
        out = [r for r in map(self._ar, d) if r is not None]
        return out

    def _ar(self, d: float) -> float | None:
        if not self.ar1:
            return d
        prev, self._prev = self._prev, d
        if prev is None:
            return None
        return (d - self.mu) - self.phi * (prev - self.mu)

    def __call__(self, x: float) -> float | None:
        d = self._diff(x)
        return None if d is None else self._ar(d)


@dataclass(frozen=True, slots=True)
class Event:
    i: int  # index in the stream
    x: float  # raw value
    point: Point | None  # None while the chart is warming up or inside Phase I
    alarm: str | None  # alarm after cooldown suppression


class Monitor:
    """Phase I on the first `baseline_n` values, then one chart update per value.

    After an alarm the chart restarts and further alarms are held back for `cooldown` points,
    so one incident pages once instead of every few minutes.
    """

    def __init__(
        self,
        chart: Shewhart | EWMA | CUSUM,
        baseline_n: int,
        season: int = 0,
        ar1: bool = False,
        cooldown: int = 0,
    ) -> None:
        if baseline_n < season + 3:
            raise ValueError(
                "baseline must be longer than the season plus three points"
            )
        self.chart, self.baseline_n, self.cooldown = chart, baseline_n, cooldown
        self.pre = Prewhitener(season, ar1)
        self.base: Baseline | None = None
        self._phase1: list[float] = []
        self._i = 0
        self._quiet_until = -1

    def update(self, x: float) -> Event:
        i = self._i
        self._i += 1
        if self.base is None:
            self._phase1.append(x)
            if len(self._phase1) == self.baseline_n:
                self.base = estimate_baseline(self.pre.fit(self._phase1))
                self._phase1 = []
            return Event(i, x, None, None)
        r = self.pre(x)
        if r is None:
            return Event(i, x, None, None)
        point = self.chart.update((r - self.base.mean) / self.base.sigma)
        alarm = point.alarm if point.alarm and i > self._quiet_until else None
        if point.alarm:
            self._quiet_until = i + self.cooldown
        return Event(i, x, point, alarm)

    def run(self, values: Iterable[float]) -> Iterator[Event]:
        for x in values:
            yield self.update(x)
