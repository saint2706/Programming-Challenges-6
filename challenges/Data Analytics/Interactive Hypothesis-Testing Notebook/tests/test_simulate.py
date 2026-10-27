from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis_wizard import simulate


def test_monte_carlo_se_is_the_binomial_standard_error():
    assert simulate.monte_carlo_se(0.05, 2000) == pytest.approx(
        math.sqrt(0.05 * 0.95 / 2000)
    )
    assert simulate.monte_carlo_se(0.0, 100) == 0.0


def test_wizard_pipeline_runs_the_welch_primary_on_these_designs():
    """With continuous outcomes and two groups the decision tree's primary is Welch's t, so the
    wizard pipeline must reproduce welch_t exactly on the same samples."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        x, y = rng.normal(size=12), rng.lognormal(size=20)
        assert simulate.wizard(x, y) == pytest.approx(
            simulate.PIPELINES["welch_always"](x, y)
        )


def test_seeded_rejection_rates_are_reproducible():
    scenario = simulate.SCENARIOS[0]
    a = simulate.rejection_rates(scenario, simulate.PIPELINES, 60, seed=5)
    b = simulate.rejection_rates(scenario, simulate.PIPELINES, 60, seed=5)
    assert a == b


def test_flip_through_reports_both_rules_with_standard_errors():
    scenario = simulate.SCENARIOS[0]
    out = simulate.flip_through(scenario, 80, seed=3)
    assert out["tests"] == list(simulate.FLIP_TESTS)
    naive, session = out["naive_any_rejection"], out["alpha_over_k_rejection"]
    assert naive["rate"] >= session["rate"]
    assert naive["mc_se"] >= 0 and session["mc_se"] >= 0


@pytest.mark.slow
def test_null_rates_are_near_alpha_for_the_wizard_on_normal_data():
    scenario = simulate.SCENARIOS[0]
    rates = simulate.rejection_rates(
        scenario, {"wizard": simulate.wizard}, 1000, seed=8
    )
    rate = rates["wizard"]["rate"]
    assert abs(rate - 0.05) < 4 * simulate.monte_carlo_se(0.05, 1000)


@pytest.mark.slow
def test_small_study_has_every_scenario_and_the_flip_through_block():
    study = simulate.run_study(reps=60, perm_reps=30, seed=1)
    assert set(study["scenarios"]) == {s.key for s in simulate.SCENARIOS}
    assert study["flip_through"] is not None
    for block in study["scenarios"].values():
        assert "wizard" in block["rates"] and "rate" in block["rates"]["wizard"]
