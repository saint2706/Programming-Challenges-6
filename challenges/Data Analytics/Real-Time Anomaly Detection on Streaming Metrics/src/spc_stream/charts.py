"""Online control charts. Each one sees one standardized value z = (x - mu) / sigma at a time.

All three share one interface: `update(z) -> Point` and `reset()`. Working in sigma units means
the caller owns the Phase I estimate of mu and sigma (see `monitor.py`) and every chart plots on
the same axis.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    stat: float  # the plotted statistic, in sigma units
    lcl: float
    ucl: float
    alarm: str | None  # rule that fired, or None


class Shewhart:
    """Individuals chart with Western Electric run rules.

    Rules: 1 = one point beyond 3 sigma, 2 = two of three beyond 2 sigma on one side,
    3 = four of five beyond 1 sigma on one side, 4 = eight in a row on one side.
    """

    name = "shewhart"

    def __init__(self, rules: tuple[int, ...] = (1, 2, 3, 4)) -> None:
        if not set(rules) <= {1, 2, 3, 4}:
            raise ValueError(f"unknown Western Electric rule in {rules}")
        self.rules = rules
        self.reset()

    def reset(self) -> None:
        self._hist: deque[float] = deque(maxlen=8)

    def _rule(self) -> str | None:
        h = list(self._hist)
        z = h[-1]
        if 1 in self.rules and abs(z) > 3:
            return "WE1: beyond 3 sigma"
        for side in (1, -1):
            if (
                2 in self.rules
                and sum(side * v > 2 for v in h[-3:]) >= 2
                and side * z > 2
            ):
                return "WE2: 2 of 3 beyond 2 sigma"
            if (
                3 in self.rules
                and sum(side * v > 1 for v in h[-5:]) >= 4
                and side * z > 1
            ):
                return "WE3: 4 of 5 beyond 1 sigma"
            if 4 in self.rules and len(h) == 8 and all(side * v > 0 for v in h):
                return "WE4: 8 in a row on one side"
        return None

    def update(self, z: float) -> Point:
        self._hist.append(z)
        alarm = self._rule()
        if alarm:
            self.reset()  # a run rule must not fire again on the same points
        return Point(z, -3.0, 3.0, alarm)


class EWMA:
    """EWMA chart. Defaults (lambda 0.1, L 2.814) give an in-control ARL of about 500.

    `exact_limits` uses the time-varying variance lam/(2-lam) * (1 - (1-lam)^(2i)), which
    tightens the limits right after a start or reset; False uses the asymptotic limits that the
    published ARL tables assume.
    """

    name = "ewma"

    def __init__(
        self, lam: float = 0.1, L: float = 2.814, exact_limits: bool = True
    ) -> None:
        if not 0 < lam <= 1:
            raise ValueError("lambda must be in (0, 1]")
        self.lam, self.L, self.exact = lam, L, exact_limits
        self.reset()

    def reset(self) -> None:
        self._z = 0.0
        self._i = 0

    def update(self, z: float) -> Point:
        self._i += 1
        self._z = self.lam * z + (1 - self.lam) * self._z
        var = self.lam / (2 - self.lam)
        if self.exact:
            var *= 1 - (1 - self.lam) ** (2 * self._i)
        limit = self.L * math.sqrt(var)
        stat = self._z
        alarm = f"EWMA beyond {self.L} sigma limit" if abs(stat) > limit else None
        if alarm:
            self.reset()
        return Point(stat, -limit, limit, alarm)


class CUSUM:
    """Two-sided tabular CUSUM. Defaults (k 0.5, h 5) give an in-control ARL of about 465.

    The plotted statistic is the upper sum when it leads, else minus the lower sum.
    """

    name = "cusum"

    def __init__(self, k: float = 0.5, h: float = 5.0) -> None:
        self.k, self.h = k, h
        self.reset()

    def reset(self) -> None:
        self._hi = self._lo = 0.0

    def update(self, z: float) -> Point:
        self._hi = max(0.0, self._hi + z - self.k)
        self._lo = max(0.0, self._lo - z - self.k)
        stat = self._hi if self._hi >= self._lo else -self._lo
        alarm = None
        if self._hi > self.h:
            alarm = "CUSUM upward shift"
        elif self._lo > self.h:
            alarm = "CUSUM downward shift"
        if alarm:
            self.reset()
        return Point(stat, -self.h, self.h, alarm)


CHARTS = {"shewhart": Shewhart, "ewma": EWMA, "cusum": CUSUM}


def make_chart(name: str) -> Shewhart | EWMA | CUSUM:
    try:
        return CHARTS[name]()
    except KeyError:
        raise ValueError(
            f"unknown chart {name!r}; choose from {sorted(CHARTS)}"
        ) from None
