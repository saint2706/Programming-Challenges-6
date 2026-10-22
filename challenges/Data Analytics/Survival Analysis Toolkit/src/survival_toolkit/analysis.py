"""One end-to-end analysis: curves per group, tests between them, and a Cox model with its checks.

``analyze`` is what the CLI and the HTML report both call, so the numbers on screen, in the report
and in the JSON are the same numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from survival_toolkit import cox as coxlib
from survival_toolkit import kaplan_meier as km
from survival_toolkit import logrank as lr
from survival_toolkit.data import (
    DATASETS,
    Prepared,
    Spec,
    load_builtin,
    prepare,
    read_table,
)
from survival_toolkit.design import encode
from survival_toolkit.types import Survival, SurvivalError

TESTS = ("logrank", "wilcoxon", "tarone-ware", "peto", "fh:0,1")
MAX_GROUPS = 12


@dataclass
class GroupSummary:
    label: str
    n: int
    events: int
    curve: km.KaplanMeier
    median: float
    median_ci: tuple[float, float]
    retention: list[dict]
    rmst: tuple[float, float] | None

    @property
    def censored(self) -> int:
        return self.n - self.events


@dataclass
class CoxFit:
    model: coxlib.CoxModel
    dropped: int  # rows left out for a missing covariate
    ph: dict
    strata_column: str | None
    levels: dict[str, list[str]]


@dataclass
class Analysis:
    title: str
    unit: str
    n: int
    events: int
    dropped: dict[str, int]
    ci_method: str
    group_column: str | None
    groups: list[GroupSummary]
    horizons: list[float]
    tau: float | None
    tests: dict[str, lr.LogRankResult] = field(default_factory=dict)
    pairwise: list[dict] = field(default_factory=list)
    trend: lr.LogRankResult | None = None
    rmst_pairs: list[dict] = field(default_factory=list)
    cox: CoxFit | None = None
    cox_stratified: CoxFit | None = None
    notes: list[str] = field(default_factory=list)
    follow_up_median: float | None = None


def resolve(
    dataset: str,
    *,
    time: str | None = None,
    event: str | None = None,
    entry: str | None = None,
    event_value: str | None = None,
    unit: str = "time units",
) -> tuple[Spec | None, Prepared]:
    """A bundled dataset by name, or a CSV/TSV/Parquet file with its time and event columns."""
    if dataset in DATASETS:
        spec, prepared = load_builtin(dataset)
        if entry is not None or time is not None or event is not None:
            spec_time, spec_event = time or spec.time, event or spec.event
            prepared = prepare(
                prepared.frame,
                time=spec_time,
                event=spec_event,
                entry=entry,
                event_value=event_value,
                unit=spec.unit,
            )
        return spec, prepared
    path = Path(dataset)
    if not path.exists():
        raise SurvivalError(
            f"{dataset!r} is neither a bundled dataset ({', '.join(DATASETS)}) nor a file"
        )
    if not time or not event:
        raise SurvivalError("a file needs --time and --event column names")
    return None, prepare(
        read_table(path),
        time=time,
        event=event,
        entry=entry,
        event_value=event_value,
        unit=unit,
    )


def group_labels(
    frame: pl.DataFrame, column: str, bins: int | None = None
) -> np.ndarray:
    """One label per row; a numeric column with many values is cut into quantile bins."""
    if column not in frame.columns:
        raise SurvivalError(
            f"no column {column!r}; columns are {', '.join(frame.columns)}"
        )
    s = frame[column]
    many = s.dtype.is_numeric() and s.n_unique() > MAX_GROUPS
    if bins or many:
        if not s.dtype.is_numeric():
            raise SurvivalError(f"--bins needs a numeric column; {column!r} is text")
        k = bins or 4
        values = s.cast(pl.Float64).to_numpy()
        edges = np.unique(np.nanquantile(values, np.linspace(0, 1, k + 1)))
        which = np.clip(
            np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2
        )
        names = [
            f"Q{i + 1} ({edges[i]:g} to {edges[i + 1]:g})"
            for i in range(len(edges) - 1)
        ]
        return np.array(
            [
                names[i] if np.isfinite(v) else "(missing)"
                for i, v in zip(which, values, strict=True)
            ]
        )
    labels = s.cast(pl.String).fill_null("(missing)").to_numpy()
    if np.unique(labels).size > MAX_GROUPS:
        raise SurvivalError(
            f"{column!r} has more than {MAX_GROUPS} groups; use --bins or another column"
        )
    return labels


def summarize(
    data: Survival,
    label: str,
    horizons: list[float],
    tau: float | None,
    ci: str,
    alpha: float,
) -> GroupSummary:
    curve = km.fit(data, ci=ci, alpha=alpha)  # type: ignore[arg-type]
    last = float(data.time.max())
    retention = []
    for h in horizons:
        if h > last:
            retention.append({"t": h, "s": None, "lower": None, "upper": None})
            continue
        s = float(curve.at(h))
        lo, hi = curve.ci_at(h)
        retention.append({"t": h, "s": s, "lower": float(lo), "upper": float(hi)})
    rmst = curve.restricted_mean(tau) if tau is not None and tau <= last else None
    return GroupSummary(
        label,
        len(data),
        data.n_events,
        curve,
        curve.median(),
        curve.median_ci(),
        retention,
        rmst,
    )


def _fit_cox(
    prepared: Prepared,
    covariates: list[str],
    reference: dict[str, str],
    strata_column: str | None,
    ties: str,
    penalizer: float,
    transform: str,
) -> CoxFit:
    frame = prepared.frame
    covs = [c for c in covariates if c != strata_column]
    design = encode(
        frame, covs + ([strata_column] if strata_column else []), reference=reference
    )
    data = prepared.survival.subset(design.keep)
    x, names = design.x, design.names
    strata = None
    if strata_column:
        # the strata column is encoded only to apply the same missing-row filter; use its raw labels
        strata = frame[strata_column].cast(pl.String).to_numpy()[design.keep]
        drop = [
            i
            for i, n in enumerate(names)
            if n == strata_column or n.startswith(f"{strata_column}[")
        ]
        keep_cols = [i for i in range(len(names)) if i not in drop]
        x, names = x[:, keep_cols], [names[i] for i in keep_cols]
    model = coxlib.fit(data, x, names, strata=strata, ties=ties, penalizer=penalizer)
    ph = model.proportional_hazards_test(transform)
    return CoxFit(model, int((~design.keep).sum()), ph, strata_column, design.levels)


def analyze(
    prepared: Prepared,
    *,
    title: str,
    by: str | None = None,
    bins: int | None = None,
    horizons: list[float] | None = None,
    tau: float | None = None,
    ci: str = "log-log",
    alpha: float = 0.05,
    covariates: list[str] | None = None,
    reference: dict[str, str] | None = None,
    strata: str | None = None,
    ties: str = "efron",
    penalizer: float = 0.0,
    transform: str = "km",
) -> Analysis:
    surv, frame = prepared.survival, prepared.frame
    horizons = sorted(horizons or default_horizons(surv))
    if tau is None:
        tau = float(horizons[-1]) if horizons else None
    labels = group_labels(frame, by, bins) if by else None
    if labels is None:
        groups = [summarize(surv, "all", horizons, tau, ci, alpha)]
    else:
        groups = [
            summarize(surv.subset(labels == g), g, horizons, tau, ci, alpha)
            for g in sorted(set(labels))
            if surv.subset(labels == g).n_events > 0
        ]
    out = Analysis(
        title=title,
        unit=prepared.unit,
        n=len(surv),
        events=surv.n_events,
        dropped={k: v for k, v in prepared.dropped.items() if v},
        ci_method=ci,
        group_column=by,
        groups=groups,
        horizons=horizons,
        tau=tau,
    )
    # median follow-up: the reverse Kaplan-Meier, where the "event" is being censored
    if (~surv.event).any():
        out.follow_up_median = km.fit(
            Survival(surv.time, ~surv.event, surv.entry)
        ).median()
    if labels is not None and len(groups) >= 2:
        keep = np.isin(labels, [g.label for g in groups])
        sub, sub_labels = surv.subset(keep), labels[keep]
        for weight in TESTS:
            out.tests[weight] = lr.logrank(sub, sub_labels, weight=weight)
        if len(groups) > 2:
            out.pairwise = lr.pairwise(sub, sub_labels)
            out.trend = lr.logrank(sub, sub_labels, trend=True)
        if tau is not None:
            for i, a in enumerate(groups):
                for b in groups[i + 1 :]:
                    if tau <= min(a.curve.data.time.max(), b.curve.data.time.max()):
                        row = lr.rmst_difference(a.curve.data, b.curve.data, tau)
                        out.rmst_pairs.append({"a": a.label, "b": b.label, **row})
    if covariates:
        out.cox = _fit_cox(
            prepared, covariates, reference or {}, None, ties, penalizer, transform
        )
        if strata:
            out.cox_stratified = _fit_cox(
                prepared,
                covariates,
                reference or {},
                strata,
                ties,
                penalizer,
                transform,
            )
    return out


def default_horizons(surv: Survival) -> list[float]:
    """Round horizons at a quarter, half and three quarters of the longest follow-up."""
    top = float(surv.time.max())
    magnitude = 10 ** math.floor(math.log10(top))
    return sorted(
        {float(round(top * f / magnitude) * magnitude) for f in (0.25, 0.5, 0.75)}
        - {0.0}
    )


def spec_defaults(spec: Spec | None) -> dict:
    if spec is None:
        return {}
    return {"covariates": list(spec.covariates), "reference": dict(spec.reference)}


def to_dict(a: Analysis) -> dict:
    """A JSON-safe summary (inf and NaN become null)."""

    def num(v):
        if v is None:
            return None
        v = float(v)
        return v if math.isfinite(v) else None

    def cox_dict(c: CoxFit | None):
        if c is None:
            return None
        m = c.model
        lr_stat, df, p = m.likelihood_ratio()
        return {
            "strata": c.strata_column,
            "n": m.n,
            "events": m.n_events,
            "rows_dropped_missing": c.dropped,
            "ties": m.ties,
            "converged": m.converged,
            "loglik": num(m.loglik),
            "likelihood_ratio": {"statistic": num(lr_stat), "df": df, "p": num(p)},
            "concordance": num(m.concordance()["c"]),
            "terms": [
                {k: (num(v) if k != "name" else v) for k, v in r.items()}
                for r in m.summary()
            ],
            "proportional_hazards": {
                "transform": c.ph["transform"],
                "global": {k: num(v) for k, v in c.ph["global"].items()},
                "terms": [
                    {"name": t["name"], "chi2": num(t["chi2"]), "p": num(t["p"])}
                    for t in c.ph["terms"]
                ],
            },
        }

    return {
        "title": a.title,
        "unit": a.unit,
        "n": a.n,
        "events": a.events,
        "dropped": a.dropped,
        "ci_method": a.ci_method,
        "median_follow_up": num(a.follow_up_median),
        "group_column": a.group_column,
        "horizons": a.horizons,
        "tau": a.tau,
        "groups": [
            {
                "label": g.label,
                "n": g.n,
                "events": g.events,
                "median": num(g.median),
                "median_ci": [num(g.median_ci[0]), num(g.median_ci[1])],
                "retention": [{k: num(v) for k, v in r.items()} for r in g.retention],
                "rmst": None
                if g.rmst is None
                else {"mean": num(g.rmst[0]), "se": num(g.rmst[1])},
            }
            for g in a.groups
        ],
        "tests": {
            k: {"statistic": num(v.statistic), "df": v.df, "p": num(v.p_value)}
            for k, v in a.tests.items()
        },
        "pairwise": [
            {k: (num(v) if k not in ("a", "b") else str(v)) for k, v in r.items()}
            for r in a.pairwise
        ],
        "cox": cox_dict(a.cox),
        "cox_stratified": cox_dict(a.cox_stratified),
        "notes": a.notes,
    }
