"""Small journeys with answers worked out by hand, and a slow reference for every rule."""

from __future__ import annotations

import numpy as np
from attribution.journeys import Journeys, from_paths


def toy() -> Journeys:
    """A>B converts twice, A alone converts once and is lost three times, B alone is lost twice."""
    return from_paths([["A", "B"], ["A"], ["B"]], [2, 1, 0], [0, 3, 2])


def random_journeys(seed: int, n: int = 300, k: int = 5) -> Journeys:
    rng = np.random.default_rng(seed)
    names = [f"c{i}" for i in range(k)]
    paths, conv, null = [], [], []
    for _ in range(n):
        length = int(rng.integers(1, 6))
        paths.append([names[i] for i in rng.integers(0, k, length)])
        conv.append(int(rng.integers(0, 3)))
        null.append(int(rng.integers(0, 6)))
    return from_paths(paths, conv, null)


def reference_credit(j: Journeys, rule: str, half_life: float = 2.0) -> np.ndarray:
    """The rules written as plain per-journey loops (the slow, obvious version)."""
    out = np.zeros(j.k)
    for p in range(j.n_converted):
        a, b = j.offsets[p], j.offsets[p + 1]
        ch, t, clicked = j.channel[a:b], j.time[a:b], j.clicked[a:b]
        n = b - a
        w = np.zeros(n)
        if rule == "first_touch":
            w[0] = 1
        elif rule == "last_touch":
            w[-1] = 1
        elif rule == "linear":
            w[:] = 1 / n
        elif rule == "time_decay":
            w = np.array([0.5 ** (max(j.conv_time[p] - ti, 0) / half_life) for ti in t])
            w /= w.sum()
        elif rule == "position_based":
            if n == 1:
                w[0] = 1
            elif n == 2:
                w[:] = 0.5
            else:
                w[:] = 0.2 / (n - 2)
                w[0] = w[-1] = 0.4
        elif rule == "last_click":
            hits = np.flatnonzero(clicked)
            w[hits[-1] if hits.size else n - 1] = 1
        for c, wi in zip(ch, w, strict=True):
            out[c] += wi * j.value[p]
    return out
