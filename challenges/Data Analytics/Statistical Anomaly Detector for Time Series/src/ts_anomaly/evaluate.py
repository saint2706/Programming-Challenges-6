"""Scoring anomaly flags against ground truth.

Three lenses, because they answer different questions and disagree:

* point-wise precision / recall / F1: every sample is a prediction. Harsh on
  anything longer than one point (a level shift is 50 anomalous samples; a
  detector that flags only its first 3 gets 6% recall).
* event-wise precision / recall / F1: runs of consecutive flags are one event,
  runs of truth are one event, and an overlap (within `tolerance` points)
  counts. "Did it notice the thing, and how many false alarms did it raise?"
* NAB-style window score: the Numenta Anomaly Benchmark rewards *early*
  detection inside a labelled window and penalises false positives by how
  far they fall from the last window. See `nab_score` for exactly how this
  reimplementation differs from the official scorer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# NAB "standard" application profile: TP weight, FP weight, FN weight.
NAB_STANDARD = (1.0, 0.11, 1.0)


@dataclass
class Counts:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def point_counts(flags: np.ndarray, truth: np.ndarray) -> Counts:
    flags = np.asarray(flags, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if flags.shape != truth.shape:
        raise ValueError("flags and truth must have the same length")
    return Counts(
        int((flags & truth).sum()),
        int((flags & ~truth).sum()),
        int((~flags & truth).sum()),
    )


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True as inclusive (start, end) index pairs."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    padded = np.concatenate(([False], m, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def event_counts(flags: np.ndarray, truth: np.ndarray, tolerance: int = 0) -> Counts:
    """Event-level counts. tp = truth events hit by a flag; fp = flag events that
    touch no truth event; fn = truth events never hit. Overlap is tested after
    widening each *truth* event by `tolerance` points on both sides (so a
    detector that fires a few samples late/early still gets credit)."""
    flags = np.asarray(flags, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if flags.shape != truth.shape:
        raise ValueError("flags and truth must have the same length")
    n = len(truth)
    f_events = runs(flags)
    t_events = runs(truth)
    hit_t = 0
    for s, e in t_events:
        lo, hi = max(0, s - tolerance), min(n - 1, e + tolerance)
        if flags[lo : hi + 1].any():
            hit_t += 1
    false_f = 0
    for s, e in f_events:
        touches = any(
            s <= te + tolerance and e >= ts - tolerance for ts, te in t_events
        )
        if not touches:
            false_f += 1
    return Counts(hit_t, false_f, len(t_events) - hit_t)


# ------------------------------------------------------------------ NAB-style


def windows_to_index(
    timestamps: np.ndarray, windows: list[tuple[str, str]]
) -> list[tuple[int, int]]:
    """Convert (start, end) timestamp strings to inclusive row-index windows.

    Windows containing no rows are dropped.
    """
    ts = np.asarray(timestamps).astype("datetime64[us]")
    out = []
    for a, b in windows:
        lo = int(np.searchsorted(ts, np.datetime64(a), side="left"))
        hi = int(np.searchsorted(ts, np.datetime64(b), side="right")) - 1
        if hi >= lo:
            out.append((lo, hi))
    return out


def windows_to_truth(n: int, index_windows: list[tuple[int, int]]) -> np.ndarray:
    truth = np.zeros(n, dtype=bool)
    for lo, hi in index_windows:
        truth[lo : hi + 1] = True
    return truth


def _sigmoid(y: float) -> float:
    """NAB's scaled sigmoid: +0.987 at y=-1, 0 at y=0, -> -1 as y grows."""
    return 2.0 / (1.0 + np.exp(5.0 * y)) - 1.0


@dataclass
class NabResult:
    raw: float
    normalized: float  # 100 = perfect, 0 = the "detect nothing" baseline, <0 worse; nan if no windows
    windows_hit: int
    windows_total: int
    false_positive_events: int


def nab_score(
    flags: np.ndarray,
    index_windows: list[tuple[int, int]],
    profile: tuple[float, float, float] = NAB_STANDARD,
) -> NabResult:
    """A reimplementation of NAB's window scoring (Lavin & Ahmad 2015).

    * A run of consecutive flags is collapsed to one *detection* at its first
      sample, so a detector isn't fined once per sample of the same false alarm.
      (The official scorer sees a threshold-free anomaly-probability stream and
      applies its own threshold; this tool's detectors emit binary flags.)
    * In each window only the *first* detection scores, as
      `A_tp * sigmoid(y)` with y = -(distance to window end) / window width,
      so an early hit is worth up to ~0.99 and one at the last sample ~0.
    * A detection outside every window is a false positive, scored
      `A_fp * sigmoid(y)` (negative) with y = distance past the previous window's
      end / that window's width: just after a window it is cheap, far away it
      costs the full `A_fp`. With no earlier window it costs `A_fp` outright.
    * An undetected window costs `A_fn`.
    * normalized = 100 * (raw - null) / (perfect - null), where `null` is
      what a detector that never fires scores (-A_fn per window).
    """
    a_tp, a_fp, a_fn = profile
    flags = np.asarray(flags, dtype=bool)
    detections = [s for s, _ in runs(flags)]
    windows = sorted(index_windows)
    raw = 0.0
    hit = [False] * len(windows)
    fp_events = 0
    for d in detections:
        inside = next((i for i, (lo, hi) in enumerate(windows) if lo <= d <= hi), None)
        if inside is not None:
            if not hit[inside]:
                lo, hi = windows[inside]
                width = hi - lo + 1
                y = -(hi - d) / width
                raw += a_tp * _sigmoid(y)
                hit[inside] = True
            continue
        fp_events += 1
        prev = [(lo, hi) for lo, hi in windows if hi < d]
        if not prev:
            raw -= a_fp
        else:
            lo, hi = prev[-1]
            width = hi - lo + 1
            raw += a_fp * _sigmoid((d - hi) / width)
    raw -= a_fn * (len(windows) - sum(hit))

    perfect = a_tp * len(windows)
    null = -a_fn * len(windows)
    normalized = (
        100.0 * (raw - null) / (perfect - null) if perfect != null else float("nan")
    )
    return NabResult(raw, normalized, sum(hit), len(windows), fp_events)
