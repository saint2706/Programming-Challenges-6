"""Properties of the decision tree, checked over the whole grid of designs."""

from __future__ import annotations

import itertools

import pytest
from hypothesis_wizard import methods
from hypothesis_wizard.design import Design, decide


def compare_grid():
    for outcome, groups, paired, heavy, ids, n_min, stratified in itertools.product(
        ("continuous", "ordinal"),
        (2, 3, 4),
        (False, True),
        (False, True),
        (False, True),
        (3, 40),
        (False, True),
    ):
        if paired and groups != 2:
            continue
        yield Design(
            question="compare",
            levels=(outcome,),
            groups=groups,
            paired=paired,
            stratified=stratified,
            heavy_tails=heavy,
            ids_repeat=ids,
            n_min=n_min,
        )


def association_grid():
    for levels, stratified, min_expected in itertools.product(
        (
            ("binary", "binary"),
            ("binary", "nominal"),
            ("nominal", "nominal"),
            ("ordinal", "nominal"),
        ),
        (False, True),
        (1.0, 4.9, 5.0, 40.0),
    ):
        yield Design(
            question="association",
            levels=levels,
            stratified=stratified,
            min_expected=min_expected,
            n_min=20,
        )


def correlation_grid():
    for levels, heavy, n_min in itertools.product(
        (("continuous", "continuous"), ("ordinal", "continuous")),
        (False, True),
        (5, 50),
    ):
        yield Design(
            question="correlation", levels=levels, heavy_tails=heavy, n_min=n_min
        )


ALL_DESIGNS = [*compare_grid(), *association_grid(), *correlation_grid()]


@pytest.mark.parametrize(
    "design", ALL_DESIGNS, ids=lambda d: f"{d.question}-{d.levels}"
)
def test_every_plan_names_a_known_primary_and_known_alternatives(design):
    plan = decide(design)
    known = {"games_howell", "dunn_holm"}
    assert plan.primary in methods.ALL_TESTS
    assert set(plan.alternatives) <= set(methods.ALL_TESTS)
    assert plan.posthoc is None or plan.posthoc in known


@pytest.mark.parametrize(
    "design", ALL_DESIGNS, ids=lambda d: f"{d.question}-{d.levels}"
)
def test_every_step_has_a_branch_and_a_reason(design):
    plan = decide(design)
    assert plan.steps, "a plan with no steps is a bug"
    for step in plan.steps:
        assert step.branch.strip()
        assert step.reason.strip()


@pytest.mark.parametrize("design", ALL_DESIGNS[:50], ids=repr)
def test_decide_is_deterministic(design):
    assert decide(design) == decide(design)


def test_every_primary_is_reachable_from_the_grid():
    primaries = {decide(d).primary for d in ALL_DESIGNS}
    expected = {
        "paired_t",
        "wilcoxon_signed_rank",
        "brunner_munzel",
        "welch_t",
        "kruskal_wallis",
        "welch_anova",
        "cmh",
        "fisher_exact",
        "chi2",
        "spearman",
        "pearson",
    }
    assert expected <= primaries
    # Nothing outside the design's leaves is ever the primary.
    assert primaries <= expected


def test_ordinal_outcomes_never_get_mean_based_primaries():
    mean_based = {"welch_t", "student_t", "paired_t", "anova", "welch_anova"}
    for design in compare_grid():
        if design.levels == ("ordinal",):
            assert decide(design).primary not in mean_based


@pytest.mark.parametrize(
    "design",
    [d for d in compare_grid() if d.levels == ("continuous",)],
    ids=repr,
)
def test_diagnostics_never_change_the_primary_in_compare(design):
    """Outliers, repeated IDs and sample size change wording and warnings only."""
    base = decide(design)
    for heavy, ids, n_min in itertools.product((False, True), (False, True), (3, 40)):
        variant = Design(
            question=design.question,
            levels=design.levels,
            groups=design.groups,
            paired=design.paired,
            stratified=design.stratified,
            heavy_tails=heavy,
            ids_repeat=ids,
            n_min=n_min,
        )
        assert decide(variant).primary == base.primary


def test_correlation_primary_depends_on_levels_not_on_outliers():
    for levels in (("continuous", "continuous"), ("ordinal", "continuous")):
        primaries = {
            decide(Design("correlation", levels, heavy_tails=h, n_min=n)).primary
            for h in (False, True)
            for n in (5, 50)
        }
        assert len(primaries) == 1


def test_stratum_ignored_outside_2x2_is_warned_not_silent():
    plan = decide(Design("compare", ("continuous",), groups=3, stratified=True))
    assert any("stratum" in w for w in plan.warnings)


def test_small_expected_counts_pick_exact_test():
    plan = decide(Design("association", ("binary", "binary"), min_expected=2.0))
    assert plan.primary == "fisher_exact"
    assert plan.alternatives


def test_large_expected_counts_pick_chi2_with_fisher_check():
    plan = decide(Design("association", ("binary", "binary"), min_expected=20.0))
    assert plan.primary == "chi2"
    assert plan.alternatives == ("fisher_exact",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"question": "regression", "levels": ("continuous",)},
        {"question": "compare", "levels": ("continuous", "continuous"), "groups": 2},
        {"question": "compare", "levels": ("continuous",), "groups": 1},
        {"question": "association", "levels": ("binary",)},
        {"question": "correlation", "levels": ("binary", "continuous", "ordinal")},
        {"question": "compare", "levels": ("unknown",), "groups": 2},
        {"question": "compare", "levels": ("continuous",), "groups": 3, "paired": True},
    ],
)
def test_invalid_designs_raise(kwargs):
    with pytest.raises(ValueError):
        Design(**kwargs)
