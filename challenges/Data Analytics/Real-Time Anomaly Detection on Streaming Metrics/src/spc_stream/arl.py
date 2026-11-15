"""Average run length by simulation, checked against the published values.

ARL0 is the mean number of in-control points before a false alarm; ARL1 at shift d is the mean
delay to detect a step of d sigma that is present from the first point (zero-state ARL).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from spc_stream.charts import CUSUM, EWMA, Shewhart

SHIFTS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)

# Montgomery, Introduction to Statistical Quality Control (Table 9.2 for CUSUM k=0.5 h=5,
# Table 9.11 for EWMA lambda=0.1 L=2.814); Champ & Woodall (1987) for Shewhart + WE rules 1-4.
PUBLISHED = {
    "cusum k=0.5 h=5": {
        0.0: 465.0,
        0.5: 38.0,
        1.0: 10.4,
        1.5: 5.75,
        2.0: 4.01,
        3.0: 2.57,
    },
    "ewma lambda=0.1 L=2.814": {
        0.0: 500.0,
        0.5: 31.3,
        1.0: 10.3,
        1.5: 6.09,
        2.0: 4.36,
        3.0: 2.9,
    },
    "shewhart WE 1-4": {0.0: 91.75},
}

CONFIGS: dict[str, Callable[[], Shewhart | EWMA | CUSUM]] = {
    "shewhart 3 sigma": lambda: Shewhart(rules=(1,)),
    "shewhart WE 1-4": Shewhart,
    "ewma lambda=0.1 L=2.814": lambda: EWMA(exact_limits=False),
    "cusum k=0.5 h=5": CUSUM,
}


def shewhart_exact(shift: float) -> float:
    """Closed form for the plain 3-sigma chart: 1 / P(|z + shift| > 3)."""
    phi = lambda v: 0.5 * (1 + math.erf(v / math.sqrt(2)))
    return 1 / (phi(-3 - shift) + 1 - phi(3 - shift))


def run_length(
    chart: Shewhart | EWMA | CUSUM, rng: np.random.Generator, shift: float
) -> int:
    n = 0
    while True:
        for z in rng.standard_normal(1024) + shift:
            n += 1
            if chart.update(float(z)).alarm:
                return n


def simulate(
    reps: int = 2000, seed: int = 0
) -> dict[str, dict[float, tuple[float, float]]]:
    """{config: {shift: (mean run length, standard error)}}."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict[float, tuple[float, float]]] = {}
    for name, make in CONFIGS.items():
        out[name] = {}
        for d in SHIFTS:
            runs = np.array([run_length(make(), rng, d) for _ in range(reps)])
            out[name][d] = (
                float(runs.mean()),
                float(runs.std(ddof=1) / math.sqrt(reps)),
            )
    return out
