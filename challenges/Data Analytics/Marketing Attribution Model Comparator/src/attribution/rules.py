"""Rule-based attribution: every model is a rule for dividing one conversion among its touches.

Each rule gives every touch a fraction of its journey's conversion, and the fractions in a journey
sum to one. Credit for a channel is then the sum of its touches' fractions times the journey's
weight (a bootstrap count, or 1) and value. Because credit is conserved the shares of every model
add up to the same total, which is what makes the models comparable.

==============  ================================================================================
first_touch     all credit to the first touch
last_touch      all credit to the last touch (the common default in ad platforms)
linear          an equal share to every touch
time_decay      credit halves for every ``half_life`` between the touch and the conversion
position_based  U-shaped: 40% first, 40% last, the other 20% spread over the middle
last_click      all credit to the last *click*; a journey with no click falls back to its last touch
==============  ================================================================================
"""

from __future__ import annotations

import numpy as np

from attribution.journeys import AttributionError, Journeys

RULES = (
    "first_touch",
    "last_touch",
    "linear",
    "time_decay",
    "position_based",
    "last_click",
)


def touch_fractions(
    j: Journeys, rule: str, *, half_life: float | None = None
) -> np.ndarray:
    """Fraction of its journey's conversion each touch receives under ``rule``."""
    starts, ends = j.offsets[:-1], j.offsets[1:] - 1
    n = j.lengths
    out = np.zeros(j.channel.size)
    if rule == "first_touch":
        out[starts] = 1.0
    elif rule == "last_touch":
        out[ends] = 1.0
    elif rule == "linear":
        out = np.repeat(1.0 / n, n)
    elif rule == "time_decay":
        if half_life is None or half_life <= 0:
            raise AttributionError("time_decay needs a positive half_life")
        age = np.maximum(np.repeat(j.conv_time, n) - j.time, 0.0)
        raw = 0.5 ** (age / half_life)
        out = raw / np.repeat(np.add.reduceat(raw, starts), n)
    elif rule == "position_based":
        pos = np.arange(j.channel.size) - np.repeat(starts, n)
        length = np.repeat(n, n)
        first, last = pos == 0, pos == length - 1
        out = np.where(
            length == 1, 1.0,
            np.where(length == 2, 0.5,
                     np.where(first | last, 0.4, 0.2 / np.maximum(length - 2, 1))),
        )  # fmt: skip
    elif rule == "last_click":
        idx = np.arange(j.channel.size)
        last_clicked = np.maximum.reduceat(np.where(j.clicked, idx, -1), starts)
        target = np.where(last_clicked >= 0, last_clicked, ends)
        out[target] = 1.0
    else:
        raise AttributionError(f"unknown rule {rule!r}; choose from {', '.join(RULES)}")
    return out


def credit(
    j: Journeys,
    rule: str,
    weights: np.ndarray | None = None,
    *,
    half_life: float | None = None,
) -> np.ndarray:
    """Conversions (or conversion value) credited to each channel by a rule."""
    w = np.ones(j.n_converted) if weights is None else weights
    per_touch = touch_fractions(j, rule, half_life=half_life) * np.repeat(
        w * j.value, j.lengths
    )
    return np.bincount(j.channel, weights=per_touch, minlength=j.k)
