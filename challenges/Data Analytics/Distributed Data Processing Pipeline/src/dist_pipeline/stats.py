"""Summary statistics for repeated timings: median and inter-quartile range."""

from __future__ import annotations

import statistics


def summarise(xs: list[float]) -> dict[str, float | int | None]:
    """Median, IQR (Q3 - Q1, inclusive method), min and max of ``xs``.

    An IQR is only meaningful from about four samples; with fewer it is reported as ``None``
    rather than as a misleading number.
    """
    if not xs:
        return {"n": 0, "median": None, "iqr": None, "min": None, "max": None}
    ordered = sorted(xs)
    iqr = None
    if len(ordered) >= 4:
        q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
        iqr = q3 - q1
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "iqr": iqr,
        "min": ordered[0],
        "max": ordered[-1],
    }
