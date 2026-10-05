"""Sequential change detectors (ADWIN, Page-Hinkley) with calibrated parameters.

The detectors from ``river`` watch a stream one value at a time. Their raw
parameters are not comparable across streams, so the stream is standardized with
the reference mean and std and the parameters are chosen on resampled reference
data: the most sensitive setting whose false alarms stay within ``alpha`` per
window of 336 rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from river import drift

WINDOW = 336
ALPHA = 0.01
PH_THRESHOLDS = (5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 120.0, 200.0, 400.0)
ADWIN_DELTAS = (0.01, 0.002, 0.0005, 0.0001, 1e-5, 1e-6, 1e-8)
PH_DELTA = 0.005  # in standard deviations of the reference stream


@dataclass(frozen=True)
class SeqParams:
    loc: float
    scale: float
    ph_delta: float
    ph_threshold: float
    adwin_delta: float


class SeqDetector:
    """A river detector fed standardized values, with a uniform ``update``/``reset``."""

    def __init__(self, name: str, make: Callable[[], object], loc: float, scale: float):
        self.name, self._make, self.loc, self.scale = name, make, loc, scale
        self._det = make()

    def update(self, x: float) -> bool:
        self._det.update((float(x) - self.loc) / self.scale)
        return bool(self._det.drift_detected)

    def reset(self) -> None:
        self._det = self._make()


def page_hinkley(delta: float, threshold: float) -> Callable[[], object]:
    return lambda: drift.PageHinkley(delta=delta, threshold=threshold)


def adwin(delta: float) -> Callable[[], object]:
    return lambda: drift.ADWIN(delta=delta)


def make_detectors(p: SeqParams) -> dict[str, SeqDetector]:
    return {
        "page_hinkley": SeqDetector(
            "page_hinkley", page_hinkley(p.ph_delta, p.ph_threshold), p.loc, p.scale
        ),
        "adwin": SeqDetector("adwin", adwin(p.adwin_delta), p.loc, p.scale),
    }


def run(detector: SeqDetector, stream: Sequence[float]) -> list[int]:
    """Indices (in order) at which the detector raised an alarm."""
    return [i for i, x in enumerate(stream) if detector.update(x)]


def _alarms(
    make: Callable[[], object], loc: float, scale: float, streams: list[np.ndarray]
) -> int:
    det = SeqDetector("tmp", make, loc, scale)
    n = 0
    for s in streams:
        det.reset()
        n += len(run(det, s))
    return n


def calibrate_sequential(
    reference: np.ndarray,
    window: int = WINDOW,
    alpha: float = ALPHA,
    reps: int = 10,
    seed: int = 0,
) -> SeqParams:
    """Most sensitive parameters with <= ``alpha`` false alarms per ``window`` rows."""
    reference = np.asarray(reference, dtype=float)
    loc = float(reference.mean())
    scale = float(reference.std()) or 1.0
    rng = np.random.default_rng(seed)
    streams = [rng.choice(reference, size=len(reference)) for _ in range(reps)]
    allowed = alpha * len(reference) * reps / window

    # PH: smaller threshold = more sensitive; ADWIN: larger delta = more sensitive.
    ph_threshold = next(
        (
            t
            for t in PH_THRESHOLDS
            if _alarms(page_hinkley(PH_DELTA, t), loc, scale, streams) <= allowed
        ),
        PH_THRESHOLDS[-1],
    )
    adwin_delta = next(
        (d for d in ADWIN_DELTAS if _alarms(adwin(d), loc, scale, streams) <= allowed),
        ADWIN_DELTAS[-1],
    )
    return SeqParams(loc, scale, PH_DELTA, ph_threshold, adwin_delta)
