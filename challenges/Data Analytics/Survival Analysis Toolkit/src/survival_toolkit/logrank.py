"""Comparing survival curves: the weighted log-rank family, trend test, pairwise tests, RMST.

At each distinct event time a 2 x K table says how many of the ``d`` events fell in each group,
given its share of the ``n`` at risk. The weighted sum of observed minus expected events is the test
statistic; the weight picks which part of the follow-up is emphasised:

==============  =====================================  ==========================
name            weight w(t)                            most sensitive to
==============  =====================================  ==========================
logrank         1                                      a constant hazard ratio
wilcoxon        n(t)  (Gehan-Breslow)                  early differences
tarone-ware     sqrt(n(t))                             in between
peto            pooled survival S~(t)                  early differences, robust to censoring
fh:rho,gamma    S(t-)^rho (1 - S(t-))^gamma            anything: (0,1) late, (1,0) early
==============  =====================================  ==========================

Late entry (left truncation) works because the risk sets come from ``entry < t <= time``. A
``strata`` array stratifies: the observed-minus-expected vectors and variances add across strata,
so groups are compared within a stratum and never across.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats

from survival_toolkit.kaplan_meier import fit as fit_km
from survival_toolkit.types import Survival, SurvivalError, n_at_risk

WEIGHTS = ("logrank", "wilcoxon", "tarone-ware", "peto", "fh:rho,gamma")


@dataclass(frozen=True)
class LogRankResult:
    groups: list[str]
    n: np.ndarray
    observed: np.ndarray
    expected: np.ndarray
    statistic: float
    df: int
    p_value: float
    weight: str
    z: (
        float | None
    )  # signed z for two groups: positive when groups[0] has excess events

    def summary_rows(self) -> list[dict]:
        return [
            {
                "group": g,
                "n": int(n),
                "observed": int(o),
                "expected": float(e),
                "o_over_e": float(o / e) if e > 0 else float("nan"),
            }
            for g, n, o, e in zip(
                self.groups, self.n, self.observed, self.expected, strict=True
            )
        ]


def parse_weight(spec: str) -> tuple[str, float, float]:
    """``logrank``, ``wilcoxon``, ``tarone-ware``, ``peto`` or ``fh:rho,gamma`` (e.g. ``fh:0,1``)."""
    name = spec.strip().lower()
    if name in {"logrank", "wilcoxon", "tarone-ware", "peto"}:
        return name, 0.0, 0.0
    if name.startswith("fh:"):
        try:
            rho, gamma = (float(x) for x in name[3:].split(","))
        except ValueError as exc:
            raise SurvivalError(
                "fh weights look like fh:rho,gamma, e.g. fh:0,1"
            ) from exc
        if rho < 0 or gamma < 0:
            raise SurvivalError("rho and gamma must be >= 0")
        return "fh", rho, gamma
    raise SurvivalError(f"unknown weight {spec!r}; use {', '.join(WEIGHTS)}")


def _codes(labels: np.ndarray) -> tuple[np.ndarray, list]:
    uniques, codes = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return codes, [str(u) for u in uniques]


def _weights(
    kind: str, rho: float, gamma: float, n: np.ndarray, d: np.ndarray
) -> np.ndarray:
    if kind == "logrank":
        return np.ones_like(n, dtype=float)
    if kind == "wilcoxon":
        return n.astype(float)
    if kind == "tarone-ware":
        return np.sqrt(n)
    if kind == "peto":
        return np.cumprod(1.0 - d / (n + 1.0))
    s = np.cumprod(1.0 - d / n)
    s_before = np.concatenate([[1.0], s[:-1]])
    return s_before**rho * (1.0 - s_before) ** gamma


def logrank(
    data: Survival,
    groups: np.ndarray,
    *,
    weight: str = "logrank",
    strata: np.ndarray | None = None,
    trend: bool = False,
    scores: np.ndarray | None = None,
) -> LogRankResult:
    """K-sample weighted log-rank test, optionally stratified, or a test for trend.

    ``trend=True`` tests whether the hazard moves monotonically with ``scores`` (default 1..K in
    sorted label order) with 1 degree of freedom, which beats the omnibus test for ordered groups.
    """
    kind, rho, gamma = parse_weight(weight)
    g_codes, labels = _codes(groups)
    if g_codes.shape != data.time.shape:
        raise SurvivalError("groups must have one label per subject")
    k = len(labels)
    if k < 2:
        raise SurvivalError("need at least two groups to compare")
    s_codes = np.zeros(len(data), dtype=int) if strata is None else _codes(strata)[0]

    u = np.zeros(k)
    v = np.zeros((k, k))
    observed = np.zeros(k)
    expected = np.zeros(k)
    for stratum in np.unique(s_codes):
        in_stratum = s_codes == stratum
        sub = data.subset(in_stratum)
        sub_g = g_codes[in_stratum]
        event_times = np.unique(sub.time[sub.event])
        if event_times.size == 0:
            continue
        n_k = np.zeros((k, event_times.size))
        d_k = np.zeros((k, event_times.size))
        for j in range(k):
            members = sub.subset(sub_g == j) if np.any(sub_g == j) else None
            if members is None:
                continue
            n_k[j] = n_at_risk(members, event_times)
            hit = members.time[members.event]
            if hit.size:
                d_k[j] = np.bincount(
                    np.searchsorted(event_times, hit), minlength=event_times.size
                )
        n = n_k.sum(axis=0)
        d = d_k.sum(axis=0)
        w = _weights(kind, rho, gamma, n, d)
        share = n_k / n
        u += (w * (d_k - d * share)).sum(axis=1)
        observed += d_k.sum(axis=1)
        expected += (d * share).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            hyper = np.where(n > 1, d * (n - d) / (n - 1.0), 0.0)
        weighted = w**2 * hyper
        v += np.diag((share * weighted).sum(axis=1)) - (share * weighted) @ share.T

    sizes = np.bincount(g_codes, minlength=k)
    if np.any(sizes == 0):
        raise SurvivalError("a group has no subjects")
    z: float | None = None
    if trend:
        a = (
            np.arange(1, k + 1, dtype=float)
            if scores is None
            else np.asarray(scores, float)
        )
        if a.shape != (k,):
            raise SurvivalError(
                f"scores needs {k} values, one per group in sorted label order"
            )
        var = float(a @ v @ a)
        if var <= 0:
            raise SurvivalError("the test has no variance (no events?)")
        z = float(a @ u / np.sqrt(var))
        statistic, df = z * z, 1
    else:
        u_r, v_r = u[:-1], v[:-1, :-1]
        try:
            statistic = float(u_r @ np.linalg.solve(v_r, u_r))
        except np.linalg.LinAlgError as exc:
            raise SurvivalError(
                "the variance matrix is singular: a group is never at risk when events happen"
            ) from exc
        df = k - 1
        if k == 2:
            z = float(u[0] / np.sqrt(v[0, 0]))
    return LogRankResult(
        groups=labels,
        n=sizes,
        observed=observed,
        expected=expected,
        statistic=statistic,
        df=df,
        p_value=float(stats.chi2.sf(statistic, df)),
        weight=weight,
        z=z,
    )


def adjust_p(p: np.ndarray, method: str = "holm") -> np.ndarray:
    """Holm (family-wise error) or Benjamini-Hochberg (false discovery rate) adjusted p-values."""
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p)
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    elif method == "bh":
        adj = np.minimum.accumulate((m / np.arange(1, m + 1) * p[order])[::-1])[::-1]
    elif method == "none":
        return p.copy()
    else:
        raise SurvivalError("adjust must be holm, bh or none")
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def pairwise(
    data: Survival,
    groups: np.ndarray,
    *,
    weight: str = "logrank",
    strata: np.ndarray | None = None,
    adjust: str = "holm",
) -> list[dict]:
    """Every pair of groups tested on its own, with multiplicity-adjusted p-values."""
    g = np.asarray(groups).astype(str)
    rows = []
    for a, b in combinations(sorted(set(g)), 2):
        mask = (g == a) | (g == b)
        res = logrank(
            data.subset(mask),
            g[mask],
            weight=weight,
            strata=None if strata is None else np.asarray(strata)[mask],
        )
        rows.append({"a": a, "b": b, "statistic": res.statistic, "p": res.p_value})
    if rows:
        adj = adjust_p(np.array([r["p"] for r in rows]), adjust)
        for r, q in zip(rows, adj, strict=True):
            r["p_adjusted"] = float(q)
    return rows


def rmst_difference(
    a: Survival, b: Survival, tau: float, *, alpha: float = 0.05
) -> dict:
    """Restricted mean survival time up to ``tau`` in two groups, and their difference a - b."""
    mean_a, se_a = fit_km(a).restricted_mean(tau)
    mean_b, se_b = fit_km(b).restricted_mean(tau)
    diff = mean_a - mean_b
    se = float(np.hypot(se_a, se_b))
    z = diff / se if se > 0 else float("nan")
    half = stats.norm.ppf(1 - alpha / 2) * se
    return {
        "rmst_a": mean_a,
        "rmst_b": mean_b,
        "difference": diff,
        "se": se,
        "lower": diff - half,
        "upper": diff + half,
        "p": float(2 * stats.norm.sf(abs(z))) if se > 0 else float("nan"),
    }
