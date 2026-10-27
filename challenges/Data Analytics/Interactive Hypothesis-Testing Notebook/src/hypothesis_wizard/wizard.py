"""From a table and a research question to a recommended test, its run and a plain verdict.

``analyse`` runs the primary test and, for more than two groups, the post-hoc comparisons.
Alternatives run only when asked, through ``run_alternative``. Every test that runs is counted in
the ``Session``, and the significance threshold is ``alpha / tests_tried``: flipping through tests
raises the bar instead of giving a free look at each p-value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from hypothesis_wizard import diagnostics, effects, methods, posthoc
from hypothesis_wizard.design import Design, Plan, decide
from hypothesis_wizard.infer import level_of


@dataclass
class Session:
    alpha: float = 0.05
    tried: list[str] = field(default_factory=list)

    def register(self, test: str) -> float:
        """Count one test as run; return the threshold that now applies."""
        self.tried.append(test)
        return self.threshold

    @property
    def threshold(self) -> float:
        return self.alpha / max(1, len(self.tried))


@dataclass(frozen=True)
class Spec:
    outcome: str
    group: str | None = None
    pair: str | None = None
    paired: bool = False
    stratum: str | None = None
    kind: str | None = None
    ordinal: tuple[str, ...] = ()
    alpha: float = 0.05
    seed: int = 0


@dataclass
class Report:
    kind: str
    design: Design
    plan: Plan
    summaries: list[diagnostics.GroupSummary]
    flags: dict[str, float | None]
    data: object
    names: tuple[str, ...]
    primary: methods.TestResult
    posthoc: list[dict]
    threshold: float
    mde: float | None
    verdict: str
    alternatives: dict[str, methods.TestResult] = field(default_factory=dict)


def infer_kind(df: pl.DataFrame, spec: Spec) -> str:
    """compare (numeric outcome, categorical group), association (two categorical) or
    correlation (two numeric)."""
    if spec.kind:
        return spec.kind
    if spec.group is None:
        raise ValueError("give a group (or second) column to analyse")
    a = _level(df, spec, spec.outcome)
    b = _level(df, spec, spec.group)
    a_cat, b_cat = a in ("binary", "nominal"), b in ("binary", "nominal")
    if a_cat and b_cat:
        return "association"
    if not a_cat and not b_cat:
        return "correlation"
    return "compare"


def _level(df: pl.DataFrame, spec: Spec, column: str) -> str:
    return level_of(df[column], ordinal=column in spec.ordinal)


def _text(series: pl.Series) -> list[str]:
    return sorted(str(v) for v in series.drop_nulls().unique().to_list())


def _compare_inputs(df: pl.DataFrame, spec: Spec):
    clean = df.drop_nulls([spec.outcome, spec.group])
    names = _text(clean[spec.group])
    repeated = False
    if spec.paired:
        pairs = clean.drop_nulls([spec.pair]).select(
            [spec.pair, spec.group, spec.outcome]
        )
        if len(names) != 2:
            raise ValueError("a paired analysis needs exactly two groups")
        by_subject: dict[object, dict[str, float]] = {}
        for subject, group, value in pairs.iter_rows():
            by_subject.setdefault(subject, {}).setdefault(str(group), float(value))
        complete = [s for s in by_subject.values() if all(g in s for g in names)]
        arrays = [
            np.array([s[names[0]] for s in complete]),
            np.array([s[names[1]] for s in complete]),
        ]
    else:
        arrays = [
            clean.filter(pl.col(spec.group).cast(pl.Utf8) == g)[spec.outcome]
            .cast(pl.Float64)
            .to_numpy()
            for g in names
        ]
        if spec.pair:
            subjects = [
                set(clean.filter(pl.col(spec.group).cast(pl.Utf8) == g)[spec.pair])
                for g in names
            ]
            repeated = bool(set.intersection(*subjects)) if subjects else False
    outcome_level = _level(df, spec, spec.outcome)
    if outcome_level not in ("continuous", "ordinal"):
        raise ValueError(
            f"compare needs a continuous or ordinal outcome, got {outcome_level}"
        )
    summaries = [
        diagnostics.summarise(a, n) for n, a in zip(names, arrays, strict=True)
    ]
    design = Design(
        question="compare",
        levels=(outcome_level,),
        groups=len(names),
        paired=spec.paired,
        stratified=bool(spec.stratum),
        heavy_tails=diagnostics.heavy_tails(summaries),
        ids_repeat=repeated,
        n_min=min(len(a) for a in arrays),
    )
    flags: dict[str, float | None] = {"levene_bf_p": None, "sd_ratio": None}
    if not spec.paired and len(arrays) >= 2:
        flags["levene_bf_p"] = diagnostics.levene_brown_forsythe(arrays)
        flags["sd_ratio"] = diagnostics.sd_ratio(arrays)
    return design, arrays, names, summaries, flags


def _crosstab(
    df: pl.DataFrame, row: str, col: str
) -> tuple[list[str], list[str], np.ndarray]:
    clean = df.drop_nulls([row, col])
    rows, cols = _text(clean[row]), _text(clean[col])
    table = np.zeros((len(rows), len(cols)))
    for r, c, n in clean.group_by([row, col]).len().iter_rows():
        table[rows.index(str(r)), cols.index(str(c))] = n
    return rows, cols, table


def _association_inputs(df: pl.DataFrame, spec: Spec):
    rows, _, table = _crosstab(df, spec.outcome, spec.group)
    strata: list[np.ndarray] = []
    if spec.stratum:
        for level in _text(df[spec.stratum]):
            sub = df.filter(pl.col(spec.stratum).cast(pl.Utf8) == level)
            _, _, t = _crosstab(sub, spec.outcome, spec.group)
            if t.shape == table.shape and t.sum() > 0:
                strata.append(t)
    design = Design(
        question="association",
        levels=(_level(df, spec, spec.outcome), _level(df, spec, spec.group)),
        stratified=bool(spec.stratum) and table.shape == (2, 2),
        min_expected=_min_expected(table),
        n_min=int(table.sum(axis=1).min()),
    )
    flags: dict[str, float | None] = {
        "min_expected": design.min_expected,
        "crude_or": _crude_or(table),
        "stratified_or": _mh_or(strata) if design.stratified else None,
    }
    data = {"table": table, "strata": strata, "rows": rows}
    return design, data, flags


def _correlation_inputs(df: pl.DataFrame, spec: Spec):
    clean = df.drop_nulls([spec.outcome, spec.group])
    x = clean[spec.outcome].cast(pl.Float64).to_numpy()
    y = clean[spec.group].cast(pl.Float64).to_numpy()
    summaries = [
        diagnostics.summarise(x, spec.outcome),
        diagnostics.summarise(y, spec.group),
    ]
    design = Design(
        question="correlation",
        levels=(_level(df, spec, spec.outcome), _level(df, spec, spec.group)),
        heavy_tails=diagnostics.heavy_tails(summaries),
        n_min=len(x),
    )
    return design, (x, y), summaries


def _min_expected(table: np.ndarray) -> float:
    expected = (
        table.sum(axis=1, keepdims=True)
        @ table.sum(axis=0, keepdims=True)
        / table.sum()
    )
    return float(expected.min())


def _crude_or(table: np.ndarray) -> float | None:
    if table.shape != (2, 2) or (table == 0).any():
        return None
    return float(table[0, 0] * table[1, 1] / (table[0, 1] * table[1, 0]))


def _mh_or(strata: list[np.ndarray]) -> float | None:
    num = sum(t[0, 0] * t[1, 1] / t.sum() for t in strata)
    den = sum(t[0, 1] * t[1, 0] / t.sum() for t in strata)
    return float(num / den) if den > 0 else None


def _prepare(df: pl.DataFrame, spec: Spec, kind: str):
    """(design, data, group names, summaries, flags) for the question kind."""
    if kind == "compare":
        design, arrays, names, summaries, flags = _compare_inputs(df, spec)
        return design, arrays, names, summaries, flags
    if kind == "association":
        design, data, flags = _association_inputs(df, spec)
        return design, data, data["rows"], [], flags
    design, data, summaries = _correlation_inputs(df, spec)
    return design, data, [spec.outcome, spec.group], summaries, {}


def _execute(
    test: str, kind: str, design: Design, data, spec: Spec
) -> methods.TestResult:
    """Run one named test on the prepared data."""
    if kind == "compare" and design.groups == 2:
        x, y = data
        if test == "permutation_mean":
            return methods.permutation_mean(x, y, seed=spec.seed)
        return methods.ALL_TESTS[test](x, y)
    if kind == "compare":
        return methods.ALL_TESTS[test](data)
    if kind == "association":
        if test == "cmh":
            return methods.cmh(data["strata"])
        return methods.ALL_TESTS[test](data["table"])
    x, y = data
    return methods.ALL_TESTS[test](x, y)


def _effect_sentence(result: methods.TestResult) -> str:
    if result.effect is None:
        return ""
    text = f"{result.effect_name} = {result.effect:.3f}"
    if result.effect_name in ("Hedges g", "dz"):
        text += f" ({effects.size_label(result.effect)})"
    if result.ci is not None:
        text += f", {result.ci_name} [{result.ci[0]:.3f}, {result.ci[1]:.3f}]"
    return text


def _verdict(result: methods.TestResult, threshold: float, mde: float | None) -> str:
    p = result.p_value
    if p < threshold:
        head = (
            f"Evidence against the null: p = {p:.4g} is below the threshold "
            f"{threshold:.4g}."
        )
    else:
        head = (
            f"No evidence against the null at the threshold {threshold:.4g} "
            f"(p = {p:.4g}). That does not show the groups are equal."
        )
    parts = [head]
    effect = _effect_sentence(result)
    if effect:
        parts.append(f"Estimate: {effect}.")
    if mde is not None:
        parts.append(
            f"With these sizes, differences of about {mde:.2f} SD or larger are detectable at "
            "80% power; smaller ones may be missed."
        )
    return " ".join(parts)


def analyse(df: pl.DataFrame, spec: Spec, session: Session | None = None) -> Report:
    """Recommend, run the primary test, run post-hoc comparisons, and give a plain verdict."""
    session = session or Session(alpha=spec.alpha)
    kind = infer_kind(df, spec)
    design, data, names, summaries, flags = _prepare(df, spec, kind)
    plan = decide(design)
    threshold = session.register(plan.primary)
    primary = _execute(plan.primary, kind, design, data, spec)

    rows: list[dict] = []
    if plan.posthoc and kind == "compare":
        rows = (
            posthoc.games_howell(data, names, spec.alpha)
            if plan.posthoc == "games_howell"
            else posthoc.dunn_holm(data, names)
        )
    mde = None
    if kind == "compare" and design.groups == 2:
        n1, n2 = (len(a) for a in data)
        mde = (
            effects.mde_paired(n1, spec.alpha)
            if design.paired
            else effects.mde_two_sample(n1, n2, spec.alpha)
        )
    return Report(
        kind=kind,
        design=design,
        plan=plan,
        summaries=summaries,
        flags=flags,
        data=data,
        names=tuple(names),
        primary=primary,
        posthoc=rows,
        threshold=threshold,
        mde=mde,
        verdict=_verdict(primary, threshold, mde),
    )


def run_alternative(
    report: Report, test: str, session: Session, spec: Spec
) -> methods.TestResult:
    """Run one listed alternative; it counts toward the tests-tried budget."""
    if test not in report.plan.alternatives:
        raise ValueError(f"{test} is not an alternative for this question")
    session.register(test)
    result = _execute(test, report.kind, report.design, report.data, spec)
    report.alternatives[test] = result
    return result
