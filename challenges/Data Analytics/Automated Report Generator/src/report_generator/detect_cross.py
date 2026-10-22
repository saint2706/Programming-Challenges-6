"""Findings that need no clock: concentration, group differences, relationships, shape, data quality."""

from __future__ import annotations

import math
import re
from itertools import combinations

import numpy as np
import polars as pl
from scipy import stats

from report_generator.config import Config
from report_generator.findings import Finding, count, label, num, pct, pvalue, ratio
from report_generator.profile import Column, Profile

CALENDAR_NAME = re.compile(r"season|month|quarter", re.IGNORECASE)
MAX_LEVELS = 30
NEGATIVE_OK = re.compile(
    r"qty|quantity|count|cnt|revenue|amount|units|rentals|orders", re.IGNORECASE
)


def sat(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def readable(name: str) -> str:
    """'Sale_Type' -> 'sale type': a column name as a noun in a sentence."""
    return re.sub(r"[_\s]+", " ", name).strip().lower()


def pluralize(noun: str) -> str:
    if noun.endswith("y") and not noun.endswith(("ay", "ey", "oy", "uy")):
        return noun[:-1] + "ies"
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def entity_noun(name: str) -> str:
    """'customer_id' -> 'customers', 'stock_code' -> 'stock codes'."""
    noun = re.sub(r"\s*\bid$", "", readable(name)) or readable(name)
    return pluralize(noun)


def overview_finding(profile: Profile, grain: str | None) -> Finding:
    frame = profile.frame
    facts = {
        "rows": count(frame.height),
        "row_label": label(profile.row_label),
        "cols": count(frame.width),
    }
    template = "overview"
    if profile.time:
        d = frame[profile.time].cast(pl.Date).drop_nulls()
        facts.update(
            {
                "start": label(str(d.min())),
                "end": label(str(d.max())),
                "time_col": label(profile.time),
            }
        )
        template = "overview.time"
    return Finding(
        id="overview", type="overview", section="Overview", columns=(), template=template, facts=facts, score=1.0,
        stats={"grain": grain, "event_table": profile.event_table},
    )  # fmt: skip


def _group_totals(frame: pl.DataFrame, by: str, measure: Column | None) -> pl.DataFrame:
    expr = (
        pl.len().alias("total")
        if measure is None
        else pl.col(measure.name).sum().alias("total")
    )
    return frame.group_by(by).agg(expr).drop_nulls(by).sort("total", descending=True)


def concentration_findings(profile: Profile, cfg: Config) -> list[Finding]:
    frame = profile.frame
    additive: list[Column | None] = [None] if profile.event_table else []
    additive += [c for c in profile.measures if c.agg == "sum"]
    out: list[Finding] = []
    for col in [*profile.dimensions, *profile.entities]:
        if col.n_unique < 5 or col.calendar or CALENDAR_NAME.search(col.name):
            continue
        entity = col.role == "entity"
        best: Finding | None = None
        for m in additive:
            tot = _group_totals(frame, col.name, m)
            vals = np.clip(tot["total"].to_numpy().astype(float), 0, None)
            total = vals.sum()
            n = vals.size
            if total <= 0 or n < 5:
                continue
            what = (
                f"total {m.name}" if m is not None else f"number of {profile.row_label}"
            )
            names = tot[col.name].cast(pl.String).to_list()
            shares = vals / total
            if entity:
                top_n = max(1, math.ceil(0.2 * n))
                top = float(shares[:top_n].sum())
                if n < 50 or top < cfg.pareto_share:
                    continue
                f = Finding(
                    id=f"concentration:{col.name}:{m.name if m else 'rows'}", type="concentration", section="Composition",
                    columns=(col.name, m.name if m else "rows"), template="concentration.entity",
                    facts={"what": label(what), "entity": label(entity_noun(col.name)), "top_n": count(top_n), "n": count(n), "share": pct(top)},
                    score=0.9 * sat((top - 0.2) / 0.8),
                    stats={"top_share": top, "n_groups": n, "top_group": names[0], "top_group_share": float(shares[0])},
                    chart={"kind": "lorenz", "title": f"Cumulative share of {what} by {readable(col.name)}",
                           "x": [float(i / n) for i in range(0, n + 1, max(1, n // 100))],
                           "y": [float(shares[:i].sum()) for i in range(0, n + 1, max(1, n // 100))]},
                    detail="groups ranked by total; share = cumulative total of the top 20% of groups",
                )  # fmt: skip
            else:
                top1 = float(shares[0])
                k = min(3, n - 1)
                topk = float(shares[:k].sum())
                if top1 < cfg.top_share or top1 < 2 / n:
                    continue
                f = Finding(
                    id=f"concentration:{col.name}:{m.name if m else 'rows'}", type="concentration", section="Composition",
                    columns=(col.name, m.name if m else "rows"), template="concentration.dim",
                    facts={"what": label(what), "dim": label(readable(col.name)), "dims": label(pluralize(readable(col.name))), "top": label(names[0]), "share": pct(top1),
                           "k": count(k), "share_k": pct(topk), "n": count(n)},
                    score=0.9 * sat((top1 - 1 / n) / (1 - 1 / n)),
                    stats={"top_share": top1, "top_k_share": topk, "n_groups": n},
                    chart={"kind": "bars", "title": f"Share of {what} by {readable(col.name)}",
                           "labels": names[:10], "values": [float(x) for x in shares[:10]], "percent": True},
                    detail="shares of the sum of positive group totals",
                )  # fmt: skip
            if best is None or f.score > best.score:
                best = f
        if best is not None:
            out.append(best)
    return out


def _levels(
    frame: pl.DataFrame, dim: str, measure: str, min_rows: int
) -> dict[str, np.ndarray]:
    sub = frame.select(
        pl.col(dim).cast(pl.String).alias("g"),
        pl.col(measure).cast(pl.Float64).alias("v"),
    )
    sub = sub.drop_nulls().filter(pl.col("v").is_finite())
    out = {}
    for key, part in sub.group_by("g"):
        if part.height >= min_rows:
            out[str(key[0])] = part["v"].to_numpy()
    return out


def group_findings(
    profile: Profile, cfg: Config, seasonal_columns: set[str]
) -> list[Finding]:
    frame = profile.frame
    cand = []
    for dim in profile.dimensions:
        calendar = dim.calendar in ("month", "quarter") or CALENDAR_NAME.search(
            dim.name
        )
        for m in profile.measures:
            if calendar and m.name in seasonal_columns:
                continue  # already said, better, as a seasonal pattern
            groups = _levels(frame, dim.name, m.name, cfg.min_group_rows)
            if not 2 <= len(groups) <= MAX_LEVELS:
                continue
            samples = list(groups.values())
            n = sum(s.size for s in samples)
            k = len(samples)
            if any(np.ptp(s) == 0 for s in samples) and all(
                np.ptp(s) == 0 for s in samples
            ):
                continue
            h, p = stats.kruskal(*samples)
            eps = max(0.0, float((h - k + 1) / (n - k)))
            cand.append((dim, m, groups, n, k, float(h), float(p), eps))
    if not cand:
        return []
    q = stats.false_discovery_control(np.array([c[6] for c in cand]), method="bh")
    out = []
    for (dim, m, groups, n, k, h, p, eps), qv in zip(cand, q, strict=True):
        if qv >= cfg.fdr or eps < cfg.min_epsilon_squared:
            continue
        med = {g: float(np.median(v)) for g, v in groups.items()}
        hi = max(med, key=lambda g: (med[g], float(np.mean(groups[g]))))
        lo = min(med, key=lambda g: (med[g], float(np.mean(groups[g]))))
        if hi == lo:
            continue
        positive = med[lo] > 0
        facts = {
            "measure": label(m.name), "dim": label(readable(dim.name)), "hi": label(hi), "lo": label(lo),
            "hi_med": num(med[hi]), "lo_med": num(med[lo]), "share": pct(eps), "k": count(k),
        }  # fmt: skip
        if positive:
            facts["gap"] = ratio(med[hi] / med[lo])
        template = ("group2" if k == 2 else "group") + (
            ".ratio" if positive and med[hi] / med[lo] >= 1.1 else ".diff"
        )
        order = sorted(med, key=med.get, reverse=True)[:12]
        out.append(
            Finding(
                id=f"group:{dim.name}:{m.name}", type="group", section="Differences between groups",
                columns=(dim.name, m.name), template=template, facts=facts,
                score=0.6 * sat(eps / 0.4) + 0.4 * sat(-math.log10(max(qv, 1e-300)) / 12),
                stats={"H": h, "p": p, "q": float(qv), "epsilon_squared": eps, "n": n, "groups": k, "min_rows": cfg.min_group_rows},
                chart={"kind": "bars", "title": f"Median {m.name} by {readable(dim.name)}", "labels": order,
                       "values": [med[g] for g in order], "percent": False},
                detail=f"Kruskal-Wallis across {k} groups ({pvalue(max(p, 1e-300)).text}); effect size epsilon-squared",
            )
        )  # fmt: skip
    return out


def identity_triples(profile: Profile) -> list[tuple[str, str, str]]:
    """(total, part, part) where total == part + part in every complete row, among additive columns."""
    frame = profile.frame
    sums = [c.name for c in profile.measures if c.agg == "sum"]
    found = []
    if len(sums) >= 3:
        for a in sums:
            for b, c2 in combinations([x for x in sums if x != a], 2):
                sub = frame.select(a, b, c2).drop_nulls()
                if sub.height >= 30 and bool((sub[a] == sub[b] + sub[c2]).all()):
                    found.append((a, b, c2))
    return found


def correlation_findings(profile: Profile, cfg: Config) -> list[Finding]:
    frame = profile.frame
    cols = [c.name for c in profile.measures]
    trivial = {
        frozenset((t, part)) for t, b, c in identity_triples(profile) for part in (b, c)
    }
    pairs = []
    for a, b in combinations(cols, 2):
        if frozenset((a, b)) in trivial:
            continue
        sub = frame.select(
            pl.col(a).cast(pl.Float64), pl.col(b).cast(pl.Float64)
        ).drop_nulls()
        if sub.height < 30 or sub[a].n_unique() < 5 or sub[b].n_unique() < 5:
            continue
        rho, p = stats.spearmanr(sub[a].to_numpy(), sub[b].to_numpy())
        if np.isnan(rho):
            continue
        pairs.append((a, b, float(rho), float(p), sub.height, sub))
    if not pairs:
        return []
    q = stats.false_discovery_control(
        np.array([max(p[3], 1e-300) for p in pairs]), method="bh"
    )
    out = []
    for (a, b, rho, p, n, sub), qv in zip(pairs, q, strict=True):
        if qv >= cfg.fdr or abs(rho) < cfg.min_abs_rho or abs(rho) >= 0.98:
            continue
        strength = (
            "very strong"
            if abs(rho) >= 0.9
            else "strong"
            if abs(rho) >= 0.7
            else "moderate"
        )
        step = max(1, n // 800)
        out.append(
            Finding(
                id=f"correlation:{a}:{b}", type="correlation", section="Relationships", columns=(a, b),
                template="correlation.pos" if rho > 0 else "correlation.neg",
                facts={"a": label(a), "b": label(b), "strength": label(strength), "rho": num(abs(rho)), "n": count(n)},
                score=0.7 * sat((abs(rho) - 0.3) / 0.6) + 0.3 * sat(-math.log10(max(qv, 1e-300)) / 20),
                stats={"rho": rho, "p": p, "q": float(qv), "n": n},
                chart={"kind": "scatter", "title": f"{a} against {b}", "x": sub[a].to_numpy()[::step].tolist(),
                       "y": sub[b].to_numpy()[::step].tolist(), "xlabel": a, "ylabel": b},
                detail=f"Spearman rank correlation, {pvalue(max(p, 1e-300)).text}; association, not cause",
            )
        )  # fmt: skip
    return out


def distribution_findings(profile: Profile, cfg: Config) -> list[Finding]:
    out = []
    for c in profile.measures:
        v = profile.frame[c.name].drop_nulls().cast(pl.Float64).to_numpy()
        if v.size < 100 or np.ptp(v) == 0:
            continue
        skew = float(stats.skew(v))
        if abs(skew) < cfg.min_skew:
            continue
        median, mean = float(np.median(v)), float(v.mean())
        right = skew > 0
        facts = {"measure": label(c.name), "median": num(median), "mean": num(mean)}
        template = "distribution.right" if right else "distribution.left"
        stats_ = {"skew": skew, "n": int(v.size)}
        if right and v.min() >= 0 and v.sum() > 0:
            top = np.sort(v)[::-1][: max(1, v.size // 100)]
            share = float(top.sum() / v.sum())
            facts["top_share"] = pct(share)
            template += ".share"
            stats_["top_1pct_share"] = share
        counts, edges = np.histogram(v, bins=30)
        out.append(
            Finding(
                id=f"distribution:{c.name}", type="distribution", section="Distributions", columns=(c.name,),
                template=template, facts=facts, score=0.5 * sat((abs(skew) - 2) / 8) + 0.2,
                stats=stats_,
                chart={"kind": "hist", "title": f"Distribution of {c.name}", "counts": counts.tolist(), "edges": edges.tolist()},
                detail="sample skewness; share = the largest 1% of rows' part of the column total",
            )
        )  # fmt: skip
    return out


def quality_findings(profile: Profile, cfg: Config) -> list[Finding]:
    frame = profile.frame
    n = frame.height
    out: list[Finding] = []
    missing = sorted(
        (
            (c.n_missing / n, c.name)
            for c in profile.columns.values()
            if c.n_missing / n >= cfg.missing_share
        ),
        reverse=True,
    )
    if missing:
        share, name = missing[0]
        out.append(
            Finding(
                id="quality:missing", type="quality", section="Data notes", columns=(name,),
                template="quality.missing" if len(missing) == 1 else "quality.missing.many",
                facts={"col": label(name), "share": pct(share), "n_cols": count(len(missing)), "limit": pct(cfg.missing_share)},
                score=0.3 + 0.7 * sat(share), stats={"columns": {m[1]: m[0] for m in missing}},
            )
        )  # fmt: skip
    dup = n - frame.unique().height  # rows that repeat an earlier row
    if dup / n >= 0.005:
        out.append(
            Finding(
                id="quality:duplicates", type="quality", section="Data notes", columns=(), template="quality.duplicates",
                facts={"n": count(dup), "share": pct(dup / n), "row_label": label(profile.row_label)},
                score=0.3 + 0.7 * sat(dup / n * 5), stats={"n": dup},
            )
        )  # fmt: skip
    if profile.time and not profile.event_table and profile.native_grain == "day":
        d = frame[profile.time].cast(pl.Date).drop_nulls()
        span = (d.max() - d.min()).days + 1
        gaps = span - d.n_unique()
        if gaps / span >= 0.05:
            out.append(
                Finding(
                    id="quality:gaps", type="quality", section="Data notes", columns=(profile.time,), template="quality.gaps",
                    facts={"n": count(gaps), "total": count(span), "share": pct(gaps / span)},
                    score=0.3 + 0.7 * sat(gaps / span * 3), stats={"gaps": gaps},
                )
            )  # fmt: skip
    for c in profile.measures:
        if c.agg != "sum" or not NEGATIVE_OK.search(c.name):
            continue
        neg = int((frame[c.name] < 0).sum())
        if 0.005 <= neg / n <= 0.5:
            out.append(
                Finding(
                    id=f"quality:negative:{c.name}", type="quality", section="Data notes", columns=(c.name,),
                    template="quality.negative",
                    facts={"col": label(c.name), "share": pct(neg / n), "n": count(neg)},
                    score=0.3 + 0.7 * sat(neg / n * 10), stats={"n": neg},
                )
            )  # fmt: skip
    for a, b, c2 in identity_triples(profile):
        rows = frame.select(a, b, c2).drop_nulls().height
        out.append(
            Finding(
                id=f"quality:identity:{a}", type="quality", section="Data notes", columns=(a, b, c2),
                template="quality.identity", facts={"total": label(a), "a": label(b), "b": label(c2), "n": count(rows)},
                score=0.35, stats={"rows": rows},
            )
        )  # fmt: skip
    return out
