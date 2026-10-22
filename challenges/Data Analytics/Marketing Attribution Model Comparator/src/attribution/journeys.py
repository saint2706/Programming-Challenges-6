"""Customer journeys: the one structure every attribution model reads.

A journey is the ordered list of marketing touches a customer received, ending in a conversion or
not. Converted journeys are stored in full (compressed-sparse-row arrays: all touches of all
journeys back to back, plus the offset where each journey starts). Journeys that did not convert
are far more numerous, and the models only need counts from them, so they are kept as aggregates:
how many started and ended at each channel, how often one channel followed another, and how many
journeys touched exactly each set of channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np


class AttributionError(ValueError):
    """The journeys cannot be analysed as given."""


@dataclass
class Journeys:
    channels: list[str]
    offsets: np.ndarray  # (P + 1,) start of each converted journey in the touch arrays
    channel: (
        np.ndarray
    )  # (T,) channel index of each touch, in time order within a journey
    time: (
        np.ndarray
    )  # (T,) when the touch happened (seconds, days or steps: see ``unit``)
    clicked: np.ndarray  # (T,) the touch was a click rather than a view
    cost: np.ndarray  # (T,) what the touch cost
    conv_time: np.ndarray  # (P,) when the conversion happened
    value: np.ndarray  # (P,) what the conversion is worth (1 for a plain count)
    unit: str = "seconds"
    # journeys that never converted, as aggregates
    null_count: float = 0.0
    null_start: np.ndarray = field(
        default=None
    )  # (K,) journeys starting at each channel
    null_end: np.ndarray = field(default=None)  # (K,) journeys ending at each channel
    null_trans: np.ndarray = field(
        default=None
    )  # (K, K) channel i directly followed by j
    null_sets: dict[int, float] = field(
        default_factory=dict
    )  # channel-set bitmask -> journeys
    null_cost: np.ndarray = field(default=None)  # (K,) cost of touches in null journeys
    null_touches: np.ndarray = field(default=None)  # (K,) touches in null journeys

    def __post_init__(self) -> None:
        k = len(self.channels)
        for name in ("null_start", "null_end", "null_cost", "null_touches"):
            if getattr(self, name) is None:
                setattr(self, name, np.zeros(k))
        if self.null_trans is None:
            self.null_trans = np.zeros((k, k))
        if (
            self.offsets.size < 1
            or self.offsets[0] != 0
            or self.offsets[-1] != self.channel.size
        ):
            raise AttributionError(
                "offsets must start at 0 and end at the number of touches"
            )
        if np.any(np.diff(self.offsets) < 1):
            raise AttributionError("every converted journey needs at least one touch")
        if self.channel.size and (self.channel.min() < 0 or self.channel.max() >= k):
            raise AttributionError("a touch refers to a channel that does not exist")

    # ---- shape --------------------------------------------------------------------------------
    @property
    def k(self) -> int:
        return len(self.channels)

    @property
    def n_converted(self) -> int:
        return self.offsets.size - 1

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def path_of_touch(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_converted), self.lengths)

    @property
    def total_value(self) -> float:
        return float(self.value.sum())

    @property
    def conversion_rate(self) -> float:
        return self.n_converted / (self.n_converted + self.null_count)

    def touch_spend(self) -> np.ndarray:
        """Cost per channel over every touch, converted journeys and not."""
        return (
            np.bincount(self.channel, weights=self.cost, minlength=self.k)
            + self.null_cost
        )

    def touch_counts(self) -> np.ndarray:
        return np.bincount(self.channel, minlength=self.k) + self.null_touches

    def validate_names(self) -> None:
        if len(set(self.channels)) != self.k:
            raise AttributionError("channel names must be unique")


def from_paths(
    paths: list[list[str]],
    conversions: list[float],
    nulls: list[float] | None = None,
    *,
    values: list[float] | None = None,
) -> Journeys:
    """Journeys from the usual aggregated format: a path, how many times it converted, how often not.

    ``paths[i]`` is an ordered list of channel names. A path that converted ``conversions[i]``
    times (and did not ``nulls[i]`` times) contributes that many journeys. Times are the touch's
    position, in steps, so time-decay and position models are defined too; every touch is a view
    and costs nothing.
    """
    nulls = nulls or [0.0] * len(paths)
    if not len(paths) == len(conversions) == len(nulls):
        raise AttributionError("paths, conversions and nulls need one entry per path")
    names = sorted({c for p in paths for c in p})
    index = {c: i for i, c in enumerate(names)}
    k = len(names)
    ch, off, tm, ctime, val = [], [0], [], [], []
    null_start = np.zeros(k)
    null_end = np.zeros(k)
    null_trans = np.zeros((k, k))
    null_sets: dict[int, float] = {}
    null_touches = np.zeros(k)
    null_count = 0.0
    for i, (p, c, n) in enumerate(zip(paths, conversions, nulls, strict=True)):
        if not p:
            raise AttributionError(f"path {i} is empty")
        ids = [index[x] for x in p]
        reps = round(c)
        if reps != c or c < 0:
            raise AttributionError("conversions must be whole numbers >= 0")
        for _ in range(reps):
            ch.extend(ids)
            tm.extend(range(len(ids)))
            ctime.append(float(len(ids)))
            off.append(len(ch))
            val.append(1.0 if values is None else float(values[i]))
        if n > 0:
            mask = 0
            for a in ids:
                mask |= 1 << a
            null_sets[mask] = null_sets.get(mask, 0.0) + n
            null_start[ids[0]] += n
            null_end[ids[-1]] += n
            for a, b in pairwise(ids):
                null_trans[a, b] += n
            for a in ids:
                null_touches[a] += n
            null_count += n
    return Journeys(
        names, np.array(off, dtype=np.int64), np.array(ch, dtype=np.int32), np.array(tm, dtype=float),
        np.zeros(len(ch), dtype=bool), np.zeros(len(ch)), np.array(ctime), np.array(val), "steps",
        null_count, null_start, null_end, null_trans, null_sets, np.zeros(k), null_touches,
    )  # fmt: skip


def resample_weights(j: Journeys, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap weights for the converted journeys: how many times each is drawn (multinomial)."""
    return rng.multinomial(
        j.n_converted, np.full(j.n_converted, 1.0 / j.n_converted)
    ).astype(float)


def mask_of_paths(j: Journeys) -> np.ndarray:
    """The bitmask of distinct channels each converted journey touched."""
    bits = np.left_shift(np.int64(1), j.channel.astype(np.int64))
    return np.bitwise_or.reduceat(bits, j.offsets[:-1])
