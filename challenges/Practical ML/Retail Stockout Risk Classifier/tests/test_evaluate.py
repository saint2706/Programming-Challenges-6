from __future__ import annotations

import numpy as np
import pytest
from stockout_risk import evaluate


def test_scores_for_a_perfect_ranking_and_a_constant_score():
    y = np.array([0, 0, 1, 1, 0, 1])
    perfect = evaluate.scores(y, y * 0.8 + 0.1)
    assert perfect["pr_auc"] == pytest.approx(1.0) and perfect[
        "roc_auc"
    ] == pytest.approx(1.0)
    flat = evaluate.scores(y, np.full(6, 0.5))
    assert flat["pr_auc"] == pytest.approx(y.mean()) and flat["roc_auc"] == 0.5


def test_ece_is_zero_for_a_calibrated_score_and_large_for_an_inverted_one():
    rng = np.random.default_rng(1)
    p = rng.random(50_000)
    y = (rng.random(50_000) < p).astype(int)
    assert evaluate.ece(y, p) < 0.02
    assert evaluate.ece(1 - y, p) > 0.3


def test_reliability_bins_are_equal_count_and_ordered():
    p = np.linspace(0, 1, 100)
    rows = evaluate.reliability((p > 0.5).astype(int), p, bins=4)
    assert [r["n"] for r in rows] == [25, 25, 25, 25]
    assert [r["mean_p"] for r in rows] == sorted(r["mean_p"] for r in rows)
    assert rows[0]["rate"] == 0 and rows[-1]["rate"] == 1


def test_worklist_picks_the_top_of_each_day_separately():
    # two days of 10 rows; the true positives are the top-scored two rows of day 1 and one of day 2
    day = np.repeat([1, 2], 10)
    p = np.tile(np.linspace(1, 0.1, 10), 2)
    y = np.zeros(20, dtype=int)
    y[[0, 1, 10]] = 1
    (row,) = evaluate.worklist(y, p, day, fractions=(0.2,))  # top 2 of each day
    assert row["precision"] == pytest.approx(3 / 4)
    assert row["recall"] == pytest.approx(1.0)
    assert row["lift"] == pytest.approx((3 / 4) / (3 / 20))


def test_worklist_works_when_the_input_is_not_sorted_by_day():
    day = np.array([2, 1, 2, 1])
    p = np.array([0.9, 0.1, 0.2, 0.8])
    y = np.array([1, 0, 0, 1])
    (row,) = evaluate.worklist(y, p, day, fractions=(0.5,))
    assert row["precision"] == 1.0


def test_cost_matches_the_definition():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    # threshold 0.5 checks rows 0 and 2: two checks, one missed stockout of cost 10
    assert evaluate.cost(y, p, 0.5, 10) == pytest.approx((2 + 10) / 5)
    assert evaluate.cost(y, p, 0.0, 10) == pytest.approx(1.0)  # always check
    assert evaluate.cost(y, p, 2.0, 10) == pytest.approx(10 * 2 / 5)  # never check


def test_best_threshold_for_a_calibrated_score_is_near_one_over_ratio():
    rng = np.random.default_rng(2)
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)
    for ratio in (2, 5, 10):
        assert evaluate.best_threshold(y, p, ratio) == pytest.approx(
            1 / ratio, abs=0.03
        )


def test_best_threshold_can_choose_to_check_nothing_or_everything():
    rng = np.random.default_rng(2)
    p = rng.random(5000)
    y = (rng.random(5000) < 0.01).astype(int)
    assert evaluate.cost(
        y, p, evaluate.best_threshold(y, p, 0.5), 0.5
    ) == pytest.approx(0.5 * y.mean())
    assert (
        evaluate.best_threshold(y, p, 1e6) <= p[y == 1].min()
    )  # every stockout is caught


def test_cost_table_has_a_saving_for_an_informative_score():
    rng = np.random.default_rng(4)
    p = rng.random(40_000)
    y = (rng.random(40_000) < p).astype(int)
    rows = evaluate.cost_table(y[:20_000], p[:20_000], y[20_000:], p[20_000:])
    assert [r["ratio"] for r in rows] == list(evaluate.RATIOS)
    assert all(r["saving_vs_best_trivial"] > 0 for r in rows)
    assert rows[0]["saving_vs_best_trivial"] == pytest.approx(
        0.25, abs=0.03
    )  # ratio 2: 0.75 vs 1.0


def test_psi_is_zero_for_the_same_data_and_large_for_a_shift():
    rng = np.random.default_rng(5)
    a = rng.normal(size=20_000)
    assert evaluate.psi(a, rng.normal(size=20_000)) < 0.02
    assert evaluate.psi(a, rng.normal(1.5, size=20_000)) > 0.25
    assert evaluate.psi(np.zeros(10), np.zeros(10)) == 0.0
    assert evaluate.psi(np.array([np.nan, 1.0, 2.0, 3.0]), np.array([1.0, 2.0])) >= 0


def test_group_scores_skip_small_and_single_class_groups():
    rng = np.random.default_rng(6)
    groups = np.repeat([1, 2, 3], [1000, 1000, 100])
    y = rng.integers(0, 2, 2100)
    y[1000:2000] = 0  # group 2 has no positives
    rows = evaluate.group_scores(y, rng.random(2100), groups)
    assert [r["group"] for r in rows] == [1]
