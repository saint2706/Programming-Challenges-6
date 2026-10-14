"""Why is this sale a comp for that subject, and what was done to its price?

Every number here is recomputed from the sales table and the fitted models, never stored: the
similarity is split over the features (``similarity.contributions``, which adds up to the whole),
each feature is compared with what a random earlier sale would contribute, and the reasons are the
features where this comp is most alike *beyond* that baseline. The sentence for each reason is built
from raw table values, so it can be checked by hand (the tests do).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from comp_finder import features
from comp_finder.engine import Engine, Settings

LOCATION_KEY = "location"
GROUPS: dict[str, list[str]] = {
    LOCATION_KEY: features.LOCATION,
    **{c: [c] for c in features.ATTRIBUTES},
}
LABELS = {LOCATION_KEY: "Location", **features.LABELS}


def _same(subject, comp, noun: str, fmt: str = "{:.0f}") -> str:
    return f"{noun} {fmt.format(comp)} (subject {fmt.format(subject)})"


def phrase(engine: Engine, group: str, q: int, c: int) -> str:
    """One sentence comparing comp ``c`` with subject ``q`` on a feature group, from raw values."""
    s, p = engine.sales.row(q, named=True), engine.sales.row(c, named=True)
    if group == LOCATION_KEY:
        km = float(np.hypot(*(engine.xy[q] - engine.xy[c])))
        zip_note = f", both in {s['zipcode']}" if s["zipcode"] == p["zipcode"] else ""
        return f"{km:.1f} km away{zip_note}"
    if group in ("log_living", "log_lot", "log_living15", "log_lot15"):
        col = {
            "log_living": "sqft_living",
            "log_lot": "sqft_lot",
            "log_living15": "sqft_living15",
            "log_lot15": "sqft_lot15",
        }[group]
        noun = {
            "log_living": "sq ft living area",
            "log_lot": "sq ft lot",
            "log_living15": "sq ft neighbour living area",
            "log_lot15": "sq ft neighbour lot",
        }[group]
        pct = (p[col] / s[col] - 1) * 100
        return f"{p[col]:,} {noun} (subject {s[col]:,}, {pct:+.0f}%)"
    if group == "eff_year":

        def year(r):
            return max(r["yr_built"], r["yr_renovated"])

        return f"built/renovated {year(p)} (subject {year(s)})"
    if group == "has_basement":
        a, b = s["sqft_basement"] > 0, p["sqft_basement"] > 0
        return (
            "both have a basement"
            if a and b
            else "neither has a basement"
            if not (a or b)
            else "basement differs"
        )
    if group == "waterfront":
        a, b = s["waterfront"], p["waterfront"]
        return (
            "both waterfront"
            if a and b
            else "neither is waterfront"
            if not (a or b)
            else "waterfront differs"
        )
    noun = {
        "bedrooms": "bedrooms",
        "bathrooms": "bathrooms",
        "floors": "floors",
        "grade": "build grade",
        "condition": "condition",
        "view": "view",
    }[group]
    fmt = "{:g}" if group in ("bathrooms", "floors") else "{:.0f}"
    return _same(s[group], p[group], noun, fmt)


def explain_comps(
    engine: Engine,
    method: str,
    q: int,
    comps: np.ndarray,
    settings: Settings,
    top: int = 3,
) -> list[dict]:
    """A dict per comp: similarity split by feature, the best reasons, and the price grid."""
    metric = engine.metrics[method]
    comps = np.asarray([c for c in comps if c >= 0], dtype=np.int64)
    if len(comps) == 0:
        return []
    pool = np.arange(int(engine.before[q]))
    typical = metric.typical(pool, q)
    est = engine.estimate(
        np.array([q]), comps[None, :], replace(settings, k=len(comps), selection="top")
    )
    names = features.COLUMNS
    out = []
    for j, c in enumerate(comps):
        contrib = metric.contributions(q, int(c))
        better = (
            (typical - contrib) if not metric.higher_is_closer else (contrib - typical)
        )
        total = float(contrib.sum())
        scored = {
            g: float(sum(better[names.index(f)] for f in fs))
            for g, fs in GROUPS.items()
        }
        reasons = [
            {
                "group": g,
                "label": LABELS[g],
                "text": phrase(engine, g, q, int(c)),
                "beyond_typical": v,
            }
            for g, v in sorted(scored.items(), key=lambda kv: -kv[1])[:top]
            if v > 0
        ]
        grid = [
            {
                "feature": f,
                "label": features.LABELS[f],
                "log_adjustment": float(est.attr_log[0, j, i]),
                "pct": float(np.expm1(est.attr_log[0, j, i]) * 100),
            }
            for i, f in enumerate(features.ATTRIBUTES)
        ]
        out.append(
            {
                "row": int(c),
                "sale_id": int(engine.ids[c]),
                "sold": str(engine.sales["sold"][int(c)]),
                "price": float(engine.price[c]),
                "similarity": total,
                "contributions": dict(zip(names, map(float, contrib), strict=True)),
                "reasons": reasons,
                "km": float(np.hypot(*(engine.xy[q] - engine.xy[c]))),
                "days_before": int(engine.sold_days[q] - engine.sold_days[c]),
                "same_zip": bool(engine.zip[q] == engine.zip[c]),
                "time_pct": float(np.expm1(est.time_log[0, j]) * 100),
                "attribute_adjustments": grid,
                "net_attribute_pct": float(np.expm1(est.net[0, j]) * 100),
                "gross_attribute_pct": float(est.gross[0, j] * 100),
                "adjusted_price": float(est.adjusted[0, j]),
            }
        )
    return out
