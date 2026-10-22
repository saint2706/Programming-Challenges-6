"""Findings about how a measure moves over time: trend, season, shifts, outliers, latest period."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from report_generator.config import Config
from report_generator.findings import Finding, count, label, num, pct, pvalue
from report_generator.profile import Profile
from report_generator.series import ROWS, Series, build_series
from report_generator.stats_ts import (
    Decomposition,
    MannKendall,
    Repeatability,
    mann_kendall,
    pelt,
    periodic_decompose,
    repeatability,
    robust_sigma,
    stl,
)

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]  # fmt: skip
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
UNIT = {
    "day": "day",
    "week": "week",
    "month": "month",
    "quarter": "quarter",
    "year": "year",
}
YEAR_BACK = {"week": 52, "month": 12, "quarter": 4, "year": 1}
MAX_SERIES = 14


def sat(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def date_text(d: np.datetime64, grain: str) -> str:
    d = np.datetime64(d, "D")
    y, m = int(str(d)[:4]), int(str(d)[5:7])
    if grain == "month":
        return f"{MONTHS[m - 1][:3]} {y}"
    if grain == "quarter":
        return f"Q{(m - 1) // 3 + 1} {y}"
    if grain == "year":
        return str(y)
    return str(d)


def phase_name(series: Series, k: int) -> str:
    d = series.dates[k]
    if series.grain == "day":
        return DAYS[
            int((d.astype("datetime64[D]").astype(int) + 3) % 7)
        ]  # 1970-01-01 was Thursday
    if series.grain == "quarter":
        return f"Q{(int(str(d)[5:7]) - 1) // 3 + 1}"
    mid = np.datetime64(d, "D") + np.timedelta64(
        3 if series.grain == "week" else 0, "D"
    )
    return MONTHS[int(str(mid)[5:7]) - 1]


@dataclass
class Analyzed:
    series: Series
    dec: Decomposition | None
    rep: Repeatability | None
    adjusted: np.ndarray
    mk: MannKendall | None
    seasonal_adjusted: bool
    noise: float = 0.0
    p_adj: float = 1.0
    extras: dict = field(default_factory=dict)


def analyze_series(series: Series, cfg: Config) -> Analyzed | None:
    y = series.values
    if np.ptp(y) == 0 or not np.isfinite(y).all():
        return None
    if abs(float(np.mean(y))) < 0.5 * float(np.std(y)):
        return (
            None  # centred on zero: "up 40%" means nothing, so say nothing in percent
        )
    dec = rep = None
    period = series.period
    if period and len(series) >= 2 * period:
        rep = repeatability(y, period)
        # STL needs several cycles to tell a season from noise; with two, a fixed pattern is fitted
        dec = (
            stl(y, period)
            if len(series) // period >= 3
            else periodic_decompose(y, period)
        )
    real = (
        rep is not None and rep.p <= cfg.seasonal_p and rep.rho >= cfg.seasonal_min_rho
    )
    adjusted = y - dec.seasonal if real else y.copy()
    mk = mann_kendall(adjusted) if len(adjusted) >= 12 else None
    return Analyzed(series, dec, rep, adjusted, mk, bool(real), robust_sigma(adjusted))


def _chart_series(series: Series, **extra) -> dict:
    return {
        "kind": "series",
        "title": series.phrase,
        "dates": [str(d) for d in series.dates],
        "values": [float(v) for v in series.values],
        **extra,
    }


def trend_finding(a: Analyzed, cfg: Config) -> Finding | None:
    mk, s = a.mk, a.series
    if mk is None or a.p_adj >= cfg.fdr:
        return None
    n = len(a.adjusted)
    start_level = mk.intercept
    end_level = mk.intercept + mk.slope * (n - 1)
    mean_level = float(np.mean(np.abs(a.adjusted)))
    base = (
        start_level
        if abs(start_level) >= 0.25 * mean_level
        else math.copysign(max(mean_level, 1e-12), start_level or 1.0)
    )
    change = (end_level - start_level) / abs(base)
    if abs(change) < cfg.min_trend_change:
        return None
    up = change > 0
    mag = abs(change)
    if mag < 0.2:
        word = "edged up" if up else "edged down"
    elif mag >= 0.5:
        word = "climbed" if up else "dropped"
    else:
        word = "rose" if up else "fell"
    template = "trend.sa" if a.seasonal_adjusted else "trend"
    score = 0.5 * sat(mag / 0.5) + 0.5 * sat(-math.log10(max(a.p_adj, 1e-300)) / 8)
    line = (mk.intercept + mk.slope * np.arange(n)).tolist()
    return Finding(
        id=f"trend:{s.key}",
        type="trend",
        section="Trends",
        columns=(s.key,),
        template=template,
        facts={
            "what": label(s.phrase),
            "verb": label(word),
            "change": pct(mag),
            "start": label(date_text(s.dates[0], s.grain)),
            "end": label(date_text(s.dates[-1], s.grain)),
            "start_level": num(start_level),
            "end_level": num(end_level),
        },
        score=score,
        stats={
            "p": mk.p, "p_adjusted": a.p_adj, "tau": mk.tau, "sen_slope_per_step": mk.slope,
            "sen_slope_ci": [mk.slope_low, mk.slope_high], "grain": s.grain, "n": n,
            "seasonally_adjusted": a.seasonal_adjusted, "direction": "up" if up else "down",
        },
        chart=_chart_series(s, trend=line, adjusted=a.adjusted.tolist() if a.seasonal_adjusted else None),
        detail=f"Mann-Kendall with Hamed-Rao autocorrelation correction, {pvalue(max(mk.p, 1e-300)).text}; Theil-Sen slope",
    )  # fmt: skip


def _season_chart(s: Series, profile: np.ndarray, period: int) -> dict:
    """Bars per phase name (52 weekly phases are shown as the 12 months they fall in)."""
    names = [phase_name(s, k) for k in range(period)]
    order = list(dict.fromkeys(names))
    return {
        "kind": "season",
        "title": f"Seasonal pattern of {s.phrase}",
        "labels": order,
        "values": [
            float(np.mean([v for n, v in zip(names, profile, strict=True) if n == o]))
            for o in order
        ],
    }


def seasonal_finding(a: Analyzed, cfg: Config) -> Finding | None:
    dec, rep, s = a.dec, a.rep, a.series
    if dec is None or rep is None or not a.seasonal_adjusted:
        return None
    period = dec.period
    profile = np.array([dec.seasonal[k::period].mean() for k in range(period)])
    level = float(np.mean(np.abs(s.values)))
    if level == 0:
        return None
    swing = float(profile.max() - profile.min()) / level
    if swing < cfg.min_seasonal_swing:
        return None
    template = (
        "season.very_strong"
        if rep.rho >= 0.7
        else "season.strong"
        if rep.rho >= 0.5
        else "season.moderate"
    )
    peak, trough = int(profile.argmax()), int(profile.argmin())
    cycle = "weekly" if s.grain == "day" else "yearly"
    return Finding(
        id=f"season:{s.key}",
        type="seasonality",
        section="Seasonality",
        columns=(s.key,),
        template=template,
        facts={
            "what": label(s.phrase),
            "cycle": label(cycle),
            "peak": label(phase_name(s, peak)),
            "trough": label(phase_name(s, trough)),
            "swing": pct(swing),
            "rho": num(rep.rho),
            "cycles": count(rep.cycles),
        },
        score=0.5 * sat((rep.rho - 0.2) / 0.6) + 0.5 * sat(swing / 0.6),
        stats={
            "dates": [str(d) for d in s.dates], "values": [float(v) for v in s.values],
            "repeatability": rep.rho, "profile_max": float(profile.max()), "p": rep.p, "cycles": rep.cycles, "period": period,
            "swing": swing, "grain": s.grain, "method": "stl" if rep.cycles >= 3 else "periodic",
        },
        chart=_season_chart(s, profile, period),
        detail=f"cycle-to-cycle rank correlation of changes {rep.rho:.2f}, permutation {pvalue(rep.p).text}; pattern from {'STL' if rep.cycles >= 3 else 'a fixed profile around a Theil-Sen trend'}",
    )  # fmt: skip


def shift_findings(a: Analyzed, cfg: Config) -> list[Finding]:
    s, y = a.series, a.adjusted
    n = len(y)
    if n < 20 or a.noise <= 0:
        return []
    if a.rep is not None and not a.seasonal_adjusted and a.rep.rho >= 0.1:
        return []  # cycles hint at a seasonal pattern that was not established: a "step" may be a season
    x = y
    if a.mk is not None and a.p_adj < cfg.fdr:
        x = y - a.mk.slope * np.arange(
            n
        )  # a level shift is judged against the trend line
    penalty = cfg.pelt_penalty_factor * a.noise**2 * math.log(n)
    med = float(np.median(x))
    spread = 1.4826 * float(np.median(np.abs(x - med)))
    if (
        spread > 0
    ):  # a one-bucket spike is an outlier, not a new level: clip before searching
        x = np.clip(x, med - 5 * spread, med + 5 * spread)
    cuts = pelt(x, penalty, min_size=8 if n >= 40 else 5)
    if not cuts or len(cuts) > 4:  # a staircase of cuts is a trend, not a shift
        return []
    edges = [0, *cuts, n]
    mean_level = float(np.mean(np.abs(y)))
    found = []
    for i, cut in enumerate(cuts):
        before = x[edges[i] : cut]
        after = x[cut : edges[i + 2]]
        jump = float(after.mean() - before.mean())
        w = min(13, len(before), len(after))
        b_level, a_level = (
            float(y[cut - w : cut].mean()),
            float(y[cut : cut + w].mean()),
        )
        rel = (a_level - b_level) / max(abs(b_level), 0.25 * mean_level, 1e-12)
        if abs(jump) < cfg.min_shift_sigmas * a.noise or abs(rel) < cfg.min_shift:
            continue
        found.append((abs(jump) / a.noise, cut, jump, b_level, a_level, rel, w))
    found.sort(reverse=True)
    out = []
    for z, cut, jump, b_level, a_level, rel, w in found[:2]:
        up = rel > 0
        out.append(
            Finding(
                id=f"shift:{s.key}:{cut}",
                type="shift",
                section="Shifts and outliers",
                columns=(s.key,),
                template="shift.up" if up else "shift.down",
                facts={
                    "what": label(s.phrase),
                    "date": label(date_text(s.dates[cut], s.grain)),
                    "before": num(b_level),
                    "after": num(a_level),
                    "change": pct(abs(rel)),
                    "window": count(w),
                    "unit": label(UNIT[s.grain] + ("s" if w != 1 else "")),
                },
                score=0.5 * sat(abs(rel) / 0.5) + 0.5 * sat((z - 3) / 9),
                stats={"cut_index": int(cut), "jump_in_sigmas": z, "penalty": penalty, "grain": s.grain},
                chart=_chart_series(s, marks=[{"date": str(s.dates[cut]), "label": "shift"}], adjusted=y.tolist() if a.seasonal_adjusted else None),
                detail="PELT change-point search on the seasonally adjusted series, BIC-style penalty",
            )
        )  # fmt: skip
    return out


def anomaly_finding(a: Analyzed, cfg: Config) -> Finding | None:
    s = a.series
    t = np.arange(len(s))
    resid = (
        a.dec.resid
        if a.dec is not None
        else s.values - np.polyval(np.polyfit(t, s.values, 1), t)
    )
    med = float(np.median(resid))
    mad = float(stats.median_abs_deviation(resid, scale="normal"))
    mad = max(
        mad, a.noise
    )  # a robust fit shrinks its own residual spread; never trust it below the noise level
    if mad <= 0:
        return None
    z = (resid - med) / mad
    level = float(np.mean(np.abs(s.values)))
    expected = s.values - resid + med
    dev = (s.values - expected) / np.maximum(
        np.abs(expected), level
    )  # relative to a typical level
    flagged = np.flatnonzero(
        (np.abs(z) >= cfg.anomaly_z) & (np.abs(dev) >= cfg.min_anomaly_deviation)
    )
    if flagged.size == 0 or flagged.size > 0.05 * len(s):
        return None  # none, or so many that this is just a heavy-tailed series
    top = int(flagged[np.argmax(np.abs(z[flagged]))])
    many = flagged.size > 1
    above = s.values[top] > expected[top]
    return Finding(
        id=f"anomaly:{s.key}",
        type="anomaly",
        section="Shifts and outliers",
        columns=(s.key,),
        template="anomaly.many" if many else "anomaly.one",
        facts={
            "what": label(s.phrase),
            "n": count(flagged.size),
            "date": label(date_text(s.dates[top], s.grain)),
            "value": num(s.values[top]),
            "expected": num(expected[top]),
            "deviation": pct(abs(dev[top])),
            "side": label("above" if above else "below"),
        },
        score=0.5 * sat((abs(z[top]) - 4) / 8) + 0.5 * sat(abs(dev[top]) / 0.6),
        stats={"noise": a.noise, "robust_z": float(z[top]), "flagged": [int(i) for i in flagged], "grain": s.grain},
        chart=_chart_series(s, marks=[{"date": str(s.dates[i]), "label": "outlier"} for i in flagged[:6]]),
        detail="robust z-score (median and MAD) of the STL remainder (or of the residual from a straight line)",
    )  # fmt: skip


def latest_finding(a: Analyzed, cfg: Config) -> Finding | None:
    s = a.series
    y = s.values
    if len(y) < 8 or a.noise <= 0:
        return None
    v, prev = float(y[-1]), float(y[-2])
    level = float(np.mean(np.abs(y)))
    denom = max(abs(prev), 0.25 * level, 1e-12)
    d_prev = (v - prev) / denom
    back = YEAR_BACK.get(s.grain)
    d_year = None
    year_ago = None
    if back and len(y) > back:
        year_ago = float(y[-1 - back])
        d_year = (v - year_ago) / max(abs(year_ago), 0.25 * level, 1e-12)
    z_prev = abs(v - prev) / (a.noise * math.sqrt(2))
    z_year = (
        abs(v - year_ago) / (a.noise * math.sqrt(2)) if year_ago is not None else 0.0
    )
    if max(z_prev, z_year) < 3.5 or max(abs(d_prev), abs(d_year or 0)) < 0.10:
        return None

    def direction(d: float) -> str:
        return "up" if d > 0 else "down"

    unit = UNIT[s.grain]
    facts = {
        "what": label(s.phrase),
        "period": label(date_text(s.dates[-1], s.grain)),
        "value": num(v),
        "unit": label(unit),
        "dir_prev": label(direction(d_prev)),
        "chg_prev": pct(abs(d_prev)),
    }
    template = "latest.noyoy"
    if d_year is not None:
        facts.update(
            {"dir_year": label(direction(d_year)), "chg_year": pct(abs(d_year))}
        )
        template = "latest"
    return Finding(
        id=f"latest:{s.key}",
        type="latest",
        section="Latest period",
        columns=(s.key,),
        template=template,
        facts=facts,
        score=0.7 * sat(max(z_prev, z_year) / 8),
        stats={"z_prev": z_prev, "z_year": z_year, "grain": s.grain, "previous": prev, "year_ago": year_ago},
        chart=_chart_series(s),
        detail="change against the previous and the year-ago bucket; the last (partial) bucket is never used",
    )  # fmt: skip


def candidate_series(profile: Profile) -> list[Series]:
    out: list[Series] = []
    keys = [c.name for c in profile.measures]
    if profile.event_table:
        keys = [ROWS, *keys]
    for key in keys[:MAX_SERIES]:
        s = build_series(profile, key)
        if s is not None:
            out.append(s)
    return out


def time_findings(
    profile: Profile, cfg: Config
) -> tuple[list[Finding], list[Analyzed]]:
    analyzed = [
        a
        for s in candidate_series(profile)
        if (a := analyze_series(s, cfg)) is not None
    ]
    tests = [a for a in analyzed if a.mk is not None]
    if tests:
        adj = stats.false_discovery_control(
            np.array([a.mk.p for a in tests]), method="bh"
        )
        for a, q in zip(tests, adj, strict=True):
            a.p_adj = float(q)
    findings: list[Finding] = []
    for a in analyzed:
        for f in (
            trend_finding(a, cfg),
            seasonal_finding(a, cfg),
            anomaly_finding(a, cfg),
            latest_finding(a, cfg),
        ):
            if f is not None:
                findings.append(f)
        findings.extend(shift_findings(a, cfg))
    return findings, analyzed
