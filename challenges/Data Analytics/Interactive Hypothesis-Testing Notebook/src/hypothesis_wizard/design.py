"""The decision tree: from the shape of a question to a primary test, alternatives and reasons.

Design rules, stated once:

* The tree branches on the question (compare, association, correlation), the measurement level
  of each variable, whether observations are paired, whether a stratum is given, and expected
  cell counts. None of these depend on the outcome p-value.
* Assumption diagnostics (Levene, Shapiro-Wilk, skew, outliers) never choose the primary test.
  Their only effect is wording and ordering of the alternatives, because pre-testing an
  assumption and then choosing the test on that result inflates the error rate (see README).
* Every recommendation carries the reason for each branch it took. A plan with no steps is a bug.
* Deterministic: the same design always gives the same plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

QUESTIONS = ("compare", "association", "correlation")
LEVELS = ("binary", "nominal", "ordinal", "continuous")


@dataclass(frozen=True)
class Design:
    question: str
    levels: tuple[str, ...]
    groups: int = 0
    paired: bool = False
    stratified: bool = False
    min_expected: float = float("inf")
    heavy_tails: bool = False
    ids_repeat: bool = False
    n_min: int = 30

    def __post_init__(self) -> None:
        if self.question not in QUESTIONS:
            raise ValueError(f"question must be one of {QUESTIONS}")
        if any(level not in LEVELS for level in self.levels):
            raise ValueError(f"levels must be drawn from {LEVELS}")
        if self.question == "compare" and (len(self.levels) != 1 or self.groups < 2):
            raise ValueError("compare needs one outcome level and at least two groups")
        if self.question != "compare" and len(self.levels) != 2:
            raise ValueError(f"{self.question} needs two variable levels")
        if self.paired and (self.question != "compare" or self.groups != 2):
            raise ValueError(
                "paired designs compare exactly two measurements per subject"
            )


@dataclass(frozen=True)
class Step:
    branch: str
    reason: str


@dataclass(frozen=True)
class Plan:
    primary: str
    alternatives: tuple[str, ...]
    posthoc: str | None
    steps: tuple[Step, ...]
    warnings: tuple[str, ...] = ()
    avoid: tuple[tuple[str, str], ...] = field(default=())


def decide(d: Design) -> Plan:
    if d.question == "compare":
        return _compare(d)
    if d.question == "association":
        return _association(d)
    return _correlation(d)


def _compare(d: Design) -> Plan:
    outcome = d.levels[0]
    steps: list[Step] = [
        Step("question: compare", f"{d.groups} groups, outcome measured as {outcome}")
    ]
    warnings: list[str] = []
    if d.n_min < 5:
        warnings.append(
            f"smallest group has {d.n_min} observations: tests have little power and "
            "asymptotic p-values are rough"
        )
    if d.ids_repeat and not d.paired:
        warnings.append(
            "subject IDs repeat across groups: observations are not independent. Set the pair "
            "column to analyse them as paired"
        )

    if d.paired:
        steps.append(
            Step("paired measurements", "each subject contributes to both groups")
        )
        if outcome == "ordinal":
            steps.append(
                Step("ordinal outcome", "differences of ratings are not interval data")
            )
            return Plan(
                "wilcoxon_signed_rank",
                ("paired_t",),
                None,
                tuple(steps),
                tuple(warnings),
                (("paired_t", "a mean difference on an ordinal scale"),),
            )
        alternatives = ("wilcoxon_signed_rank",)
        if d.heavy_tails:
            steps.append(
                Step(
                    "outliers or skew",
                    "the signed-rank test is shown as the check "
                    "that does not rest on the mean difference",
                )
            )
        return Plan(
            "paired_t",
            alternatives,
            None,
            tuple(steps),
            tuple(warnings),
            (("student_t", "pairs are not independent groups; use paired_t"),),
        )

    if d.stratified:
        warnings.append(
            "a stratum column is ignored for compare with more than two groups "
            "or a continuous outcome; fit a blocked model for that analysis"
        )

    if d.groups == 2:
        if outcome == "ordinal":
            steps.append(
                Step(
                    "ordinal outcome",
                    "a mean of ratings is not meaningful; "
                    "P(X < Y) is the quantity to estimate",
                )
            )
            return Plan(
                "brunner_munzel",
                ("mann_whitney", "permutation_mean"),
                None,
                tuple(steps),
                tuple(warnings),
                (
                    ("student_t", "means of an ordinal scale"),
                    ("welch_t", "means of an ordinal scale"),
                ),
            )
        steps.append(
            Step(
                "two independent groups, continuous outcome",
                "Welch's t does not assume equal variances or equal sizes",
            )
        )
        alternatives = ("brunner_munzel", "permutation_mean")
        if d.heavy_tails:
            steps.append(
                Step(
                    "outliers or skew",
                    "Brunner-Munzel is listed first as the check that does not rest "
                    "on the mean",
                )
            )
        return Plan(
            "welch_t",
            alternatives,
            None,
            tuple(steps),
            tuple(warnings),
            (
                (
                    "student_t",
                    (
                        "assumes equal variances; choosing it after a variance test "
                        "inflates the error rate"
                    ),
                ),
                (
                    "mann_whitney",
                    (
                        "tests distributions, not means: it rejects for unequal "
                        "shapes even when means match"
                    ),
                ),
            ),
        )

    if outcome == "ordinal":
        steps.append(
            Step(
                "ordinal outcome, more than two groups",
                "ranks are the appropriate scale",
            )
        )
        return Plan(
            "kruskal_wallis",
            (),
            "dunn_holm",
            tuple(steps),
            tuple(warnings),
            (("anova", "means of an ordinal scale"),),
        )
    steps.append(
        Step(
            "more than two independent groups, continuous outcome",
            "Welch's ANOVA does not assume equal variances",
        )
    )
    return Plan(
        "welch_anova",
        ("kruskal_wallis",),
        "games_howell",
        tuple(steps),
        tuple(warnings),
        (
            (
                "anova",
                (
                    "assumes equal variances; choosing it after a variance test inflates the "
                    "error rate"
                ),
            ),
        ),
    )


def _association(d: Design) -> Plan:
    steps = [Step("question: association", "two categorical variables")]
    warnings: list[str] = []
    avoid: list[tuple[str, str]] = []
    if d.n_min < 5:
        warnings.append(
            "a cell or stratum is tiny: p-values from the chi-square approximation "
            "are unreliable"
        )

    if d.stratified:
        if d.levels == ("binary", "binary"):
            steps.append(
                Step(
                    "stratified 2x2 tables", "the Mantel-Haenszel test pools the strata"
                )
            )
            warnings.append(
                "check whether the crude and stratified associations point the same "
                "way (Simpson's paradox); the wizard reports both"
            )
            return Plan(
                "cmh",
                ("chi2",),
                None,
                tuple(steps),
                tuple(warnings),
                (("fisher_exact", "pooling strata needs the stratified test"),),
            )
        warnings.append(
            "stratified tables larger than 2x2 are not implemented; the crude "
            "table is analysed"
        )

    if d.levels == ("binary", "binary"):
        if d.min_expected < 5:
            steps.append(
                Step(
                    "expected count below 5 in a 2x2 table",
                    "the chi-square approximation is unreliable; Fisher's exact test "
                    "is the primary",
                )
            )
            return Plan(
                "fisher_exact",
                ("barnard_exact", "chi2"),
                None,
                tuple(steps),
                tuple(warnings),
                (("chi2", "expected counts below 5"),),
            )
        steps.append(
            Step(
                "2x2 table, expected counts at least 5",
                "chi-square without continuity correction is the primary; Fisher is "
                "the check",
            )
        )
        return Plan("chi2", ("fisher_exact",), None, tuple(steps), tuple(warnings), ())

    if d.min_expected < 5:
        warnings.append(
            "expected counts below 5 in some cells: chi-square p-values are "
            "approximate; an exact Fisher-Freeman-Halton test is not implemented"
        )
    if "ordinal" in d.levels:
        warnings.append(
            "an ordinal variable is treated as nominal here; a trend test would use "
            "the order and be more powerful (not implemented)"
        )
    steps.append(Step("larger table", "chi-square test of independence"))
    return Plan("chi2", (), None, tuple(steps), tuple(warnings), tuple(avoid))


def _correlation(d: Design) -> Plan:
    a, b = d.levels
    steps = [Step("question: correlation", f"both variables measured as {a} and {b}")]
    warnings: list[str] = []
    if d.n_min < 10:
        warnings.append(
            f"only {d.n_min} pairs: correlation estimates are very uncertain"
        )
    if "ordinal" in d.levels:
        steps.append(
            Step("rank-based primary", "an ordinal variable has no meaningful spacing")
        )
        return Plan(
            "spearman",
            ("kendall", "pearson"),
            None,
            tuple(steps),
            tuple(warnings),
            (("pearson", "linear correlation of ranks, not of the ordinal values"),),
        )
    steps.append(
        Step(
            "both continuous",
            "Pearson's r measures linear association with these values",
        )
    )
    if d.heavy_tails:
        # Outliers or skew change the wording only: the primary stays Pearson's r.
        steps.append(
            Step(
                "outliers or skew",
                "Spearman is the first check: a few extreme points cannot drive it",
            )
        )
    return Plan(
        "pearson", ("spearman", "kendall"), None, tuple(steps), tuple(warnings), ()
    )
