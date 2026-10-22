"""The one input every estimator takes: durations, event flags and optional late entry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SurvivalError(ValueError):
    """The data cannot be analysed as given (bad times, empty groups, a singular model)."""


@dataclass(frozen=True)
class Survival:
    """Right-censored times, with optional left truncation (delayed entry).

    ``time`` is when the subject left observation (event or censoring), ``event`` is True for an
    observed event, and ``entry`` is when the subject came under observation (default 0). A
    subject is at risk at ``t`` when ``entry < t <= time``, so it never counts before entry and
    counts at its own exit time.
    """

    time: np.ndarray
    event: np.ndarray
    entry: np.ndarray | None = None

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        event = np.asarray(self.event).astype(bool)
        if time.ndim != 1 or time.shape != event.shape:
            raise SurvivalError("time and event must be 1-D arrays of the same length")
        if time.size == 0:
            raise SurvivalError("no observations")
        if not np.all(np.isfinite(time)):
            raise SurvivalError(
                "time has missing or infinite values; drop or impute them first"
            )
        entry = None
        if self.entry is not None:
            entry = np.asarray(self.entry, dtype=float)
            if entry.shape != time.shape or not np.all(np.isfinite(entry)):
                raise SurvivalError("entry must be finite and match time in length")
            if np.any(entry < 0):
                raise SurvivalError("entry times must be >= 0")
            if np.any(entry >= time):
                raise SurvivalError(
                    "every subject needs entry < time (zero or negative exposure was found)"
                )
        elif np.any(time <= 0):
            raise SurvivalError(
                "times must be > 0 (a subject observed for no time carries no information; drop it)"
            )
        if np.any(time < 0):
            raise SurvivalError("times must be >= 0")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "entry", entry)

    def __len__(self) -> int:
        return int(self.time.size)

    @property
    def n_events(self) -> int:
        return int(self.event.sum())

    def subset(self, mask: np.ndarray) -> Survival:
        mask = np.asarray(mask, dtype=bool)
        return Survival(
            self.time[mask],
            self.event[mask],
            None if self.entry is None else self.entry[mask],
        )


def n_at_risk(
    data: Survival, at: np.ndarray, *, closed_entry: bool = False
) -> np.ndarray:
    """Subjects with ``entry < t <= time`` for each ``t`` in ``at``.

    ``closed_entry`` counts a subject who enters exactly at ``t`` too (``entry <= t``): that is
    the convention for a risk table printed under a plot, where "at risk at 0" is everyone.
    """
    at = np.asarray(at, dtype=float)
    left = np.searchsorted(np.sort(data.time), at, side="left")  # left before t
    entered = (
        np.full(at.shape, len(data))
        if data.entry is None
        else np.searchsorted(
            np.sort(data.entry), at, side="right" if closed_entry else "left"
        )
    )
    if data.entry is None and not closed_entry:
        entered = np.where(at > 0, len(data), 0)
    return (entered - left).astype(int)
