"""Building ``Journeys`` from a flat table of touches (one row per ad exposure or visit)."""

from __future__ import annotations

import numpy as np

from attribution.journeys import AttributionError, Journeys


def from_touches(
    journey: np.ndarray,
    time: np.ndarray,
    channel: np.ndarray,
    converted: np.ndarray,
    channels: list[str],
    *,
    clicked: np.ndarray | None = None,
    cost: np.ndarray | None = None,
    conv_time: np.ndarray | None = None,
    value: np.ndarray | None = None,
    unit: str = "seconds",
) -> Journeys:
    """Group touches into journeys, ordered by time.

    ``journey`` labels the touches of one customer path, ``converted`` says (per touch, constant
    within a journey) whether that path ended in a conversion, ``conv_time`` and ``value`` say
    when and how much. ``channel`` indexes ``channels``.
    """
    t = time.size
    if t == 0:
        raise AttributionError("no touches")
    clicked = np.zeros(t, bool) if clicked is None else clicked.astype(bool)
    cost = np.zeros(t) if cost is None else cost.astype(float)
    value = np.ones(t) if value is None else value.astype(float)
    conv_time = time.astype(float) if conv_time is None else conv_time.astype(float)
    order = np.lexsort((time, journey))
    journey, time, channel = (
        journey[order],
        time[order],
        channel[order].astype(np.int32),
    )
    converted, clicked, cost = (
        converted[order].astype(bool),
        clicked[order],
        cost[order],
    )
    conv_time, value = conv_time[order], value[order]
    k = len(channels)
    first = np.concatenate([[0], np.flatnonzero(np.diff(journey)) + 1])
    last = np.concatenate([first[1:], [t]]) - 1
    lengths = last - first + 1
    path_conv = converted[first]

    # converted journeys, in full
    keep = np.repeat(path_conv, lengths)
    c_lengths = lengths[path_conv]
    offsets = np.concatenate([[0], np.cumsum(c_lengths)]).astype(np.int64)

    # journeys that never converted, as aggregates
    null = ~path_conv
    ns, ne = first[null], last[null]
    null_start = np.bincount(channel[ns], minlength=k).astype(float)
    null_end = np.bincount(channel[ne], minlength=k).astype(float)
    inside = np.flatnonzero((np.diff(journey) == 0) & np.repeat(null, lengths)[:-1])
    null_trans = np.zeros((k, k))
    np.add.at(null_trans, (channel[inside], channel[inside + 1]), 1.0)
    bits = np.left_shift(np.int64(1), channel.astype(np.int64))
    masks = np.bitwise_or.reduceat(bits, first)[null]
    uniq, counts = np.unique(masks, return_counts=True)
    null_sets = {int(m): float(c) for m, c in zip(uniq, counts, strict=True)}
    nt = np.repeat(null, lengths)
    return Journeys(
        channels=channels,
        offsets=offsets,
        channel=channel[keep],
        time=time[keep].astype(float),
        clicked=clicked[keep],
        cost=cost[keep],
        conv_time=conv_time[first][path_conv],
        value=value[first][path_conv],
        unit=unit,
        null_count=float(null.sum()),
        null_start=null_start,
        null_end=null_end,
        null_trans=null_trans,
        null_sets=null_sets,
        null_cost=np.bincount(channel[nt], weights=cost[nt], minlength=k),
        null_touches=np.bincount(channel[nt], minlength=k).astype(float),
    )
