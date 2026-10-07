"""From a ranking to a targeting decision.

``u(x)`` is the uplift curve: incremental outcomes per customer of the population if the top
``x`` share is contacted, estimated on the randomized test split by comparing the arms. The
value of an incremental outcome and the cost of one contact are *parameters*: the data has
neither, so nothing is invented.
"""

from __future__ import annotations

import metrics
import numpy as np

GRID = np.linspace(0.0, 1.0, 101)


def incremental_at(r: metrics.Ranked, frac: float, w=None) -> float:
    x, _, u = metrics.curve(r, w)
    return float(np.interp(frac, x, u))


def budget_table(r: metrics.Ranked, fracs, n_customers: int, w=None) -> list[dict]:
    rows = []
    for f in fracs:
        inc = n_customers * incremental_at(r, f, w)
        contacts = round(f * n_customers)
        rows.append(
            {
                "share": float(f),
                "contacts": contacts,
                "incremental": inc,
                "per_contact": inc / contacts if contacts else 0.0,
            }
        )
    return rows


def profit(r, frac, value, cost, n_customers, w=None) -> float:
    return float(n_customers * (value * incremental_at(r, frac, w) - cost * frac))


def best_fraction(
    r, value, cost, n_customers, grid=None, w=None
) -> tuple[float, float]:
    """The contact share with the highest profit (0 and 0.0 if every share loses money)."""
    grid = GRID if grid is None else np.asarray(grid)
    x, _, u = metrics.curve(r, w)
    p = n_customers * (value * np.interp(grid, x, u) - cost * grid)
    i = int(np.argmax(p))
    return (float(grid[i]), float(p[i])) if p[i] > 0 else (0.0, 0.0)


def break_even_cost(r, frac, value, w=None) -> float:
    """The highest cost per contact at which contacting the top ``frac`` still pays."""
    return float(value * incremental_at(r, frac, w) / frac)
