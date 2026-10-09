"""Do the segments mean anything? Score customers at a past date, then look at what they did next.

Segments are only useful if customers in a "Champions" box really behave differently afterwards
from those in a "Lost" box. This scores everyone using only data before ``snapshot`` and compares
what happened in the next ``horizon_days``: did they buy again, and how much did they spend. It is
a check on the segmentation, not a forecast: nothing is fitted to the future.

The comparison across ways of ordering customers uses *top-share*: the fraction of all future spend
earned by the top 20% of customers (random ordering gives 20%). Tied priorities are split by
expected value, so a coarse score is not rewarded or punished for where its ties happen to fall.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from rfm_explorer.rfm import Monetary, as_of, build_rfm
from rfm_explorer.scoring import METHODS, score_customers

REFERENCE = "quintile: R+F+M"  # the ordering every other one is compared against


def future_outcomes(
    lines: pl.DataFrame, customer_ids: pl.Series, snapshot: date, horizon_days: int
) -> pl.DataFrame:
    """``customer_id, bought, future_spend`` for activity in ``[snapshot, snapshot + horizon)``.

    Spend counts purchases only; a customer with no purchase in the window has spend 0.
    """
    start = as_of(snapshot)
    end = start + timedelta(days=horizon_days)
    future = lines.filter(
        (pl.col("invoice_date") >= start)
        & (pl.col("invoice_date") < end)
        & ~pl.col("is_return")
    )
    spent = future.group_by("customer_id").agg(future_spend=pl.col("revenue").sum())
    return (
        pl.DataFrame({"customer_id": customer_ids})
        .join(spent, on="customer_id", how="left")
        .with_columns(future_spend=pl.col("future_spend").fill_null(0.0))
        .with_columns(bought=pl.col("future_spend") > 0)
    )


def top_share(priority, outcome, fraction: float = 0.2) -> float:
    """Share of total ``outcome`` earned by the top ``fraction`` of customers by ``priority``.

    A priority group that straddles the cut contributes in proportion to how much of it fits.
    """
    p = np.asarray(priority, dtype=float)
    y = np.asarray(outcome, dtype=float)
    total = y.sum()
    if p.size == 0 or total <= 0:
        return float("nan")
    values, inverse = np.unique(p, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=y)
    need = fraction * p.size
    taken = captured = 0.0
    for g in range(len(values) - 1, -1, -1):  # best priority first
        if taken >= need:
            break
        use = min(counts[g], need - taken)
        captured += sums[g] * use / counts[g]
        taken += use
    return float(captured / total)


def _bootstrap(
    orderings: dict[str, np.ndarray],
    spend: np.ndarray,
    fraction: float,
    resamples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Top-share of every ordering on the same customer resamples (so differences are paired)."""
    rng = np.random.default_rng(seed)
    n = len(spend)
    draws = {name: np.empty(resamples) for name in orderings}
    for b in range(resamples):
        idx = rng.integers(0, n, n)
        for name, priority in orderings.items():
            draws[name][b] = top_share(priority[idx], spend[idx], fraction)
    return draws


def segment_table(joined: pl.DataFrame) -> pl.DataFrame:
    """Per segment: size, repeat-purchase rate, mean future spend, share of all future spend."""
    total = joined["future_spend"].sum()
    return (
        joined.group_by("segment")
        .agg(
            customers=pl.len(),
            repeat_rate=pl.col("bought").mean(),
            mean_future_spend=pl.col("future_spend").mean(),
            future_spend=pl.col("future_spend").sum(),
            priority=pl.col("priority").mean(),
        )
        .with_columns(
            future_spend_share=pl.col("future_spend") / total
            if total
            else pl.lit(float("nan"))
        )
        .sort("priority", descending=True)
    )


@dataclass(frozen=True)
class Validation:
    snapshot: date
    horizon_days: int
    customers: int
    repeat_rate: float
    segments: dict[str, pl.DataFrame]
    orderings: pl.DataFrame


def validate(
    lines: pl.DataFrame,
    snapshot: date,
    horizon_days: int = 90,
    lookback_days: int | None = 365,
    monetary: Monetary = "net",
    k: int = 5,
    fraction: float = 0.2,
    resamples: int = 300,
    seed: int = 0,
) -> Validation:
    rfm = build_rfm(lines, snapshot, lookback_days, monetary)
    if rfm.is_empty():
        raise ValueError(f"no customers bought in the window before {snapshot}")
    outcome = future_outcomes(lines, rfm["customer_id"], snapshot, horizon_days)
    spend = outcome["future_spend"].to_numpy()

    segments: dict[str, pl.DataFrame] = {}
    orderings: dict[str, np.ndarray] = {}
    scored_by_method = {m: score_customers(rfm, m, k=k, seed=seed) for m in METHODS}
    for method, scored in scored_by_method.items():
        segments[method] = segment_table(scored.join(outcome, on="customer_id"))
        orderings[
            f"{method}: R+F+M" if method != "kmeans" else "kmeans: cluster rank"
        ] = scored["priority"].to_numpy()
    q = scored_by_method["quintile"]
    orderings["recency only"] = q["r"].to_numpy().astype(float)
    orderings["frequency only"] = q["f"].to_numpy().astype(float)
    orderings["monetary only"] = q["m"].to_numpy().astype(float)
    orderings["frequency + monetary (no recency)"] = (
        (q["f"] + q["m"]).to_numpy().astype(float)
    )

    draws = _bootstrap(orderings, spend, fraction, resamples, seed)
    reference = draws[REFERENCE]
    rows = []
    for name, priority in orderings.items():
        rho = (
            spearmanr(priority, spend).statistic
            if np.ptp(priority) > 0
            else float("nan")
        )
        lo, hi = np.nanpercentile(draws[name], [2.5, 97.5])
        d_lo, d_hi = np.nanpercentile(draws[name] - reference, [2.5, 97.5])
        rows.append(
            {
                "ordering": name,
                "spearman": float(rho),
                "top_share": top_share(priority, spend, fraction),
                "top_share_lo": float(lo),
                "top_share_hi": float(hi),
                "vs_rfm_lo": float(d_lo),
                "vs_rfm_hi": float(d_hi),
            }
        )
    return Validation(
        snapshot=snapshot,
        horizon_days=horizon_days,
        customers=rfm.height,
        repeat_rate=float(outcome["bought"].mean()),
        segments=segments,
        orderings=pl.DataFrame(rows).sort("top_share", descending=True),
    )
