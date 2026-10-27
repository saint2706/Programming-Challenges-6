"""End-to-end runs of the wizard on the bundled tables, and the tests-tried counter."""

from __future__ import annotations

import pytest
from helpers import reference, table
from hypothesis_wizard import methods, wizard

REF = reference()


def test_sleep_paired_matches_r_end_to_end():
    df = table("sleep")
    spec = wizard.Spec(outcome="extra", group="group", pair="ID", paired=True)
    report = wizard.analyse(df, spec)
    assert report.kind == "compare"
    assert report.plan.primary == "paired_t"
    assert report.primary.p_value == pytest.approx(
        REF["paired_t_sleep"]["p_value"], rel=1e-6
    )
    assert report.primary.effect_name == "dz"
    assert report.mde is not None


def test_iris_three_groups_gets_welch_anova_and_games_howell():
    df = table("iris")
    spec = wizard.Spec(outcome="petal_length", group="species")
    report = wizard.analyse(df, spec)
    assert report.plan.primary == "welch_anova"
    assert report.plan.posthoc == "games_howell"
    assert len(report.posthoc) == 3
    assert report.primary.p_value < 1e-60


def test_penguins_two_groups_welch_with_hedges_g():
    df = table("penguins")
    spec = wizard.Spec(outcome="bill_length_mm", group="sex")
    report = wizard.analyse(df, spec)
    assert report.plan.primary == "welch_t"
    assert report.primary.effect_name == "Hedges g"
    assert "Estimate:" in report.verdict


def test_ordinal_wine_quality_uses_spearman():
    """Quality is whole-valued with few levels, so it is inferred as ordinal."""
    df = table("wine_red")
    spec = wizard.Spec(outcome="quality", group="alcohol")
    report = wizard.analyse(df, spec)
    assert report.kind == "correlation"
    assert report.plan.primary == "spearman"
    assert report.primary.statistic == pytest.approx(0.479, abs=0.001)


def test_ucb_stratified_association_uses_cmh_and_reports_both_odds_ratios():
    df = table("ucb_admissions")
    from hypothesis_wizard.data import expand_counts

    rows = expand_counts(df, "Freq")
    spec = wizard.Spec(outcome="Admit", group="Gender", stratum="Dept")
    report = wizard.analyse(rows, spec)
    assert report.kind == "association"
    assert report.plan.primary == "cmh"
    assert report.flags["crude_or"] is not None
    assert report.flags["stratified_or"] is not None
    assert report.primary.p_value == pytest.approx(REF["cmh_ucb"]["p_value"], rel=1e-6)


def test_session_threshold_is_alpha_over_tests_tried():
    session = wizard.Session(alpha=0.05)
    assert session.threshold == pytest.approx(0.05)
    assert session.register("welch_t") == pytest.approx(0.05)
    assert session.register("brunner_munzel") == pytest.approx(0.025)
    assert session.register("permutation_mean") == pytest.approx(0.05 / 3)


def test_alternatives_run_only_when_listed_and_count_toward_session():
    df = table("sleep")
    spec = wizard.Spec(outcome="extra", group="group", pair="ID", paired=True)
    session = wizard.Session()
    report = wizard.analyse(df, spec, session)
    assert len(session.tried) == 1
    alt = wizard.run_alternative(report, "wilcoxon_signed_rank", session, spec)
    assert isinstance(alt, methods.TestResult)
    assert len(session.tried) == 2
    assert session.threshold == pytest.approx(0.025)
    assert "wilcoxon_signed_rank" in report.alternatives


def test_unlisted_alternative_is_refused_and_not_counted():
    df = table("sleep")
    spec = wizard.Spec(outcome="extra", group="group", pair="ID", paired=True)
    session = wizard.Session()
    report = wizard.analyse(df, spec, session)
    with pytest.raises(ValueError):
        wizard.run_alternative(report, "welch_t", session, spec)
    assert len(session.tried) == 1


def test_verdict_names_the_threshold_and_the_decision():
    df = table("sleep")
    spec = wizard.Spec(outcome="extra", group="group", pair="ID", paired=True)
    report = wizard.analyse(df, spec)
    assert "threshold" in report.verdict
    assert "Evidence against the null" in report.verdict


def test_missing_group_column_is_an_error():
    df = table("iris")
    with pytest.raises(ValueError):
        wizard.analyse(df, wizard.Spec(outcome="petal_length"))
