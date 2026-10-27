"""Monte Carlo study of the decision rules.

Scenarios are two-group samples with either no mean difference (type I error) or a known shift
(power). Pipelines are the wizard's real path (``design.decide`` on the data, then the primary test)
and the common pre-test habits it is meant to replace. Rates come with Monte Carlo standard errors
sqrt(p (1 - p) / R). A flip-through study checks what happens when a user tries several tests on
the same null data and keeps the best one.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from hypothesis_wizard import diagnostics, methods
from hypothesis_wizard.design import Design, decide

ALPHA = 0.05
Sampler = Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    null: bool
    sampler: Sampler
    n: tuple[int, int]


def _normal(n1, n2, sd1=1.0, sd2=1.0, shift=0.0) -> Sampler:
    def draw(rng):
        return rng.normal(shift, sd1, n1), rng.normal(0.0, sd2, n2)

    return draw


def _lognormal(n1, n2, shift=0.0) -> Sampler:
    def draw(rng):
        return rng.lognormal(0, 1, n1), rng.lognormal(0, 1, n2) + shift

    return draw


def _student_t(n1, n2, df=3) -> Sampler:
    def draw(rng):
        return rng.standard_t(df, n1), rng.standard_t(df, n2)

    return draw


def _skew_vs_normal(n1, n2) -> Sampler:
    def draw(rng):
        return rng.exponential(1.0, n1) - 1.0, rng.normal(0.0, 1.0, n2)

    return draw


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "null_normal_equal",
        "null: normal, equal SD, n 20/20",
        True,
        _normal(20, 20),
        (20, 20),
    ),
    Scenario(
        "null_normal_small_group_wide",
        "null: normal, SD 3 vs 1, n 10/30",
        True,
        _normal(10, 30, 3.0, 1.0),
        (10, 30),
    ),
    Scenario(
        "null_normal_large_group_wide",
        "null: normal, SD 3 vs 1, n 30/10",
        True,
        _normal(30, 10, 3.0, 1.0),
        (30, 10),
    ),
    Scenario(
        "null_lognormal_equal",
        "null: lognormal(0,1) both groups, n 20/20",
        True,
        _lognormal(20, 20),
        (20, 20),
    ),
    Scenario(
        "null_lognormal_unequal",
        "null: lognormal(0,1) both groups, n 8/24",
        True,
        _lognormal(8, 24),
        (8, 24),
    ),
    Scenario(
        "null_t3_heavy",
        "null: Student t (3 df), n 15/15",
        True,
        _student_t(15, 15),
        (15, 15),
    ),
    Scenario(
        "null_skew_vs_normal",
        "null: centred exponential vs normal, same mean and SD, n 15/15",
        True,
        _skew_vs_normal(15, 15),
        (15, 15),
    ),
    Scenario(
        "power_normal_shift",
        "power: normal shift 0.5 SD, n 20/20",
        False,
        _normal(20, 20, shift=0.5),
        (20, 20),
    ),
    Scenario(
        "power_normal_unequal_shift",
        "power: normal shift 0.5, SD 3 vs 1, n 10/30",
        False,
        _normal(10, 30, 3.0, 1.0, shift=0.5),
        (10, 30),
    ),
    Scenario(
        "power_lognormal_shift",
        "power: lognormal, shift +0.5 on the second group, n 20/20",
        False,
        _lognormal(20, 20, shift=0.5),
        (20, 20),
    ),
)


def wizard(x, y) -> float:
    """The real path: diagnostics feed the design, decide picks the test, that test runs."""
    summaries = [diagnostics.summarise(x, "x"), diagnostics.summarise(y, "y")]
    design = Design(
        question="compare",
        levels=("continuous",),
        groups=2,
        heavy_tails=diagnostics.heavy_tails(summaries),
        n_min=min(len(x), len(y)),
    )
    plan = decide(design)
    return methods.ALL_TESTS[plan.primary](x, y).p_value


def _p(test) -> Callable[[np.ndarray, np.ndarray], float]:
    return lambda x, y: methods.ALL_TESTS[test](x, y).p_value


def levene_then_student_or_welch(x, y) -> float:
    """Classic pre-test: Brown-Forsythe p above 0.05 means equal variances, so Student's t."""
    if diagnostics.levene_brown_forsythe([x, y]) > ALPHA:
        return methods.student_t(x, y).p_value
    return methods.welch_t(x, y).p_value


def shapiro_then_t_or_mwu(x, y) -> float:
    """Classic pre-test: if both samples pass Shapiro-Wilk, Student's t; otherwise Mann-Whitney."""
    normal = stats.shapiro(x).pvalue > ALPHA and stats.shapiro(y).pvalue > ALPHA
    if normal:
        return methods.student_t(x, y).p_value
    return methods.mann_whitney(x, y).p_value


def permutation_mean_999(x, y) -> float:
    return methods.permutation_mean(x, y, n_resamples=999).p_value


PIPELINES: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "wizard": wizard,
    "welch_always": _p("welch_t"),
    "student_always": _p("student_t"),
    "mann_whitney_always": _p("mann_whitney"),
    "brunner_munzel_always": _p("brunner_munzel"),
    "levene_then_student_or_welch": levene_then_student_or_welch,
    "shapiro_then_t_or_mwu": shapiro_then_t_or_mwu,
}
# Costly (999 relabellings per call): run on a subset of scenarios only.
EXPENSIVE: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "permutation_mean": permutation_mean_999,
}
PERMUTATION_SCENARIOS = (
    "null_normal_equal",
    "null_normal_small_group_wide",
    "null_lognormal_unequal",
    "null_skew_vs_normal",
)

FLIP_TESTS = ("welch_t", "student_t", "mann_whitney", "brunner_munzel")


def monte_carlo_se(rate: float, reps: int) -> float:
    return math.sqrt(rate * (1 - rate) / reps)


def rejection_rates(scenario: Scenario, pipelines: dict, reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    hits = {name: 0 for name in pipelines}
    for _ in range(reps):
        x, y = scenario.sampler(rng)
        for name, fn in pipelines.items():
            if fn(x, y) < ALPHA:
                hits[name] += 1
    return {
        name: {
            "rate": hits[name] / reps,
            "mc_se": monte_carlo_se(hits[name] / reps, reps),
        }
        for name in pipelines
    }


def flip_through(scenario: Scenario, reps: int, seed: int) -> dict:
    """Run every test in FLIP_TESTS on each null sample and report the best p-value.

    ``naive``: reject if any test has p < alpha (the user keeps the first hit).
    ``session``: the threshold is alpha / (number of tests tried), the wizard's counter rule,
    so the family of four tests is judged at alpha / 4 (Bonferroni over the family).
    """
    rng = np.random.default_rng(seed)
    naive = session = 0
    threshold = ALPHA / len(FLIP_TESTS)
    for _ in range(reps):
        x, y = scenario.sampler(rng)
        pvals = [methods.ALL_TESTS[t](x, y).p_value for t in FLIP_TESTS]
        if min(pvals) < ALPHA:
            naive += 1
        if min(pvals) < threshold:
            session += 1
    return {
        "tests": list(FLIP_TESTS),
        "naive_any_rejection": {
            "rate": naive / reps,
            "mc_se": monte_carlo_se(naive / reps, reps),
        },
        "alpha_over_k_rejection": {
            "rate": session / reps,
            "mc_se": monte_carlo_se(session / reps, reps),
        },
    }


def run_study(
    reps: int = 2000,
    perm_reps: int = 1000,
    seed: int = 20261027,
    scenarios: tuple[Scenario, ...] = SCENARIOS,
) -> dict:
    """Rejection rates for every scenario and pipeline, plus the flip-through study.

    The permutation test is run only on PERMUTATION_SCENARIOS and with ``perm_reps`` samples
    (its rate is then reported with its own Monte Carlo SE, using that smaller count).
    """
    results: dict[str, dict] = {}
    for i, scenario in enumerate(scenarios):
        rates = rejection_rates(scenario, PIPELINES, reps, seed + i)
        if scenario.key in PERMUTATION_SCENARIOS:
            rates.update(
                rejection_rates(scenario, EXPENSIVE, perm_reps, seed + 1000 + i)
            )
        results[scenario.key] = {
            "label": scenario.label,
            "null": scenario.null,
            "n": list(scenario.n),
            "rates": rates,
        }
    flip = None
    null_equal = next((s for s in scenarios if s.key == "null_normal_equal"), None)
    if null_equal is not None:
        flip = flip_through(null_equal, reps, seed + 99)
    return {
        "alpha": ALPHA,
        "reps": reps,
        "perm_reps": perm_reps,
        "seed": seed,
        "scenarios": results,
        "flip_through": flip,
    }
