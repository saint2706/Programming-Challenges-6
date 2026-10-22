"""The pipeline: profile a table, find things, choose the few worth saying, write and verify them."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import stats

from report_generator.config import Config
from report_generator.detect_cross import (
    concentration_findings,
    correlation_findings,
    distribution_findings,
    group_findings,
    identity_triples,
    overview_finding,
    quality_findings,
)
from report_generator.detect_time import time_findings
from report_generator.findings import Finding
from report_generator.profile import Profile, profile_table
from report_generator.realize import audit, realize
from report_generator.verify import Check, failed, verify_findings

SECTION_ORDER = [
    "Trends",
    "Seasonality",
    "Shifts and outliers",
    "Latest period",
    "Composition",
    "Differences between groups",
    "Relationships",
    "Distributions",
]


@dataclass
class Report:
    title: str
    profile: Profile
    config: Config
    overview: Finding
    findings: list[Finding]  # the headline findings, in report order
    notes: list[Finding]  # data-quality notes
    candidates: int  # findings that passed their gates before selection
    candidate_findings: list[Finding] = field(default_factory=list)
    dropped: list[dict] = field(
        default_factory=list
    )  # {id, reason} for findings that failed audit/verify
    checks: dict[str, list[Check]] = field(default_factory=dict)
    verified: bool = False

    @property
    def takeaways(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: -f.score)[:3]


def _near_duplicate(a: Finding, b: Finding, corr: dict[tuple[str, str], float]) -> bool:
    """Two findings of one kind about measures that move together say the same thing twice."""
    if a.type != b.type:
        return False
    if a.chart and b.chart and a.chart.get("kind") in ("series", "season"):
        x, y = (
            np.asarray(a.chart["values"], float),
            np.asarray(b.chart["values"], float),
        )
        if (
            a.chart["kind"] == b.chart.get("kind")
            and x.shape == y.shape
            and x.std() > 0
            and y.std() > 0
        ):
            return abs(float(np.corrcoef(x, y)[0, 1])) >= 0.97

    def related(c1: str, c2: str) -> bool:
        return c1 == c2 or corr.get(tuple(sorted((c1, c2))), 0.0) >= 0.97

    if a.type == "group" and a.columns[0] == b.columns[0]:
        return related(a.columns[1], b.columns[1])
    if a.type == "correlation":
        (a1, a2), (b1, b2) = a.columns, b.columns
        return (
            (related(a1, b1) and a2 == b2)
            or (related(a2, b2) and a1 == b1)
            or (related(a1, b2) and a2 == b1)
            or (related(a2, b1) and a1 == b2)
        )
    return False


def select(
    findings: list[Finding],
    frame: pl.DataFrame,
    cfg: Config,
    totals: frozenset[str] = frozenset(),
) -> list[Finding]:
    """Greedy pick by score with a cap per section, skipping near-duplicates."""
    measure_cols = sorted(
        {
            c
            for f in findings
            if f.type in ("group", "correlation")
            for c in f.columns[(1 if f.type == "group" else 0) :]
        }
    )
    corr: dict[tuple[str, str], float] = {}
    for i, a in enumerate(measure_cols):
        for b in measure_cols[i + 1 :]:
            sub = frame.select(
                pl.col(a).cast(pl.Float64), pl.col(b).cast(pl.Float64)
            ).drop_nulls()
            if sub.height > 30:
                rho = stats.spearmanr(sub[a].to_numpy(), sub[b].to_numpy())[0]
                corr[(a, b)] = abs(float(rho)) if np.isfinite(rho) else 0.0
    chosen: list[Finding] = []
    per_section: dict[str, int] = {}
    latest = 0
    for f in sorted(
        findings,
        key=lambda f: (
            -(f.score + (0.02 if f.columns[:1] and f.columns[0] in totals else 0))
        ),
    ):
        if (
            per_section.get(f.section, 0) >= cfg.max_per_section
            or len(chosen) >= cfg.max_findings
        ):
            continue
        if f.type == "latest" and latest >= 1:
            continue
        if any(_near_duplicate(f, g, corr) for g in chosen):
            continue
        chosen.append(f)
        per_section[f.section] = per_section.get(f.section, 0) + 1
        latest += f.type == "latest"
    chosen.sort(key=lambda f: (SECTION_ORDER.index(f.section), -f.score))
    return chosen


def build_report(
    frame: pl.DataFrame,
    *,
    title: str = "Data report",
    config: Config | None = None,
    time: str | None = None,
    ignore: tuple[str, ...] = (),
    sum_columns: tuple[str, ...] = (),
    mean_columns: tuple[str, ...] = (),
    row_label: str = "rows",
    verify: bool = True,
) -> Report:
    cfg = config or Config()
    profile = profile_table(
        frame,
        time=time,
        ignore=ignore,
        sum_columns=sum_columns,
        mean_columns=mean_columns,
        row_label=row_label,
    )
    found: list[Finding] = []
    grain = None
    if profile.time:
        tf, analyzed = time_findings(profile, cfg)
        found += tf
        grain = analyzed[0].series.grain if analyzed else None
    seasonal_cols = {f.columns[0] for f in found if f.type == "seasonality"}
    found += concentration_findings(profile, cfg)
    found += group_findings(profile, cfg, seasonal_cols)
    found += correlation_findings(profile, cfg)
    found += distribution_findings(profile, cfg)
    notes = quality_findings(profile, cfg)
    overview = overview_finding(profile, grain)
    totals = frozenset(t for t, _, _ in identity_triples(profile))
    chosen = select(found, profile.frame, cfg, totals)
    report = Report(
        title,
        profile,
        cfg,
        overview,
        chosen,
        sorted(notes, key=lambda f: -f.score),
        len(found),
    )
    report.candidate_findings = sorted(found, key=lambda f: -f.score)
    everything = [overview, *chosen, *report.notes]
    realize(everything)
    realize([f for f in report.candidate_findings if f not in everything])
    for f in everything:
        problems = audit(f)
        if problems:
            report.dropped.append(
                {"id": f.id, "reason": "audit: " + "; ".join(problems)}
            )
    if verify:
        report.checks = verify_findings(profile, everything)
        report.verified = True
        for fid, checks in report.checks.items():
            bad = failed(checks)
            if bad:
                report.dropped.append(
                    {"id": fid, "reason": "verify: " + "; ".join(c.name for c in bad)}
                )
    gone = {d["id"] for d in report.dropped}
    report.findings = [f for f in report.findings if f.id not in gone]
    report.notes = [f for f in report.notes if f.id not in gone]
    return report
