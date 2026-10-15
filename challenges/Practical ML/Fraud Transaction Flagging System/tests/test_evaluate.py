from __future__ import annotations

import numpy as np
import pytest
from fraud_flagging import evaluate

# A ranking of 10 transactions, highest score first; frauds at ranks 1, 2, 4 and 7.
S = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
Y = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])
AMOUNT = np.array([50.0, 10.0, 5.0, 200.0, 1.0, 1.0, 30.0, 2.0, 2.0, 2.0])


def test_ranking_scores_for_a_perfect_ranking():
    sc = evaluate.ranking_scores(Y, Y * 0.8 + 0.1)
    assert sc["pr_auc"] == pytest.approx(1.0) and sc["roc_auc"] == pytest.approx(1.0)
    assert sc["frauds"] == 4 and sc["base_rate"] == pytest.approx(0.4)


def test_scores_add_log_loss_and_brier_and_stay_finite_at_zero_and_one():
    sc = evaluate.scores(Y, np.where(Y == 1, 1.0, 0.0))
    assert np.isfinite(sc["log_loss"]) and sc["brier"] == pytest.approx(0.0, abs=1e-12)


def test_reliability_bins_are_fixed_and_skip_empty_ones():
    p = np.array([0.0005, 0.0005, 0.05, 0.95])
    rows = evaluate.reliability(np.array([0, 1, 0, 1]), p)
    assert [(r["lo"], r["n"], r["rate"]) for r in rows] == [
        (0.0, 2, 0.5),
        (1e-2, 1, 0.0),
        (0.9, 1, 1.0),
    ]


def test_confusion_by_hand():
    c = evaluate.confusion(Y, S >= 0.7)  # flags ranks 1-4
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (3, 1, 1, 5)
    assert c["precision"] == pytest.approx(0.75) and c["recall"] == pytest.approx(0.75)
    assert c["alerts_per_10k"] == pytest.approx(4000)
    assert evaluate.confusion(Y, np.zeros(10, bool))["precision"] is None


def test_threshold_for_precision_takes_the_deepest_point_that_clears_the_bar():
    # precision by rank: 1, 1, .67, .75, .6, .5, .57, .5, .44, .4: at least 0.7 holds until rank 4
    assert evaluate.threshold_for_precision(Y, S, 0.7) == pytest.approx(0.7)
    assert evaluate.threshold_for_precision(Y, S, 1.0) == pytest.approx(0.9)
    assert evaluate.threshold_for_precision(Y, S, 0.4) == pytest.approx(
        0.1
    )  # everything is 40%
    assert evaluate.threshold_for_precision(np.zeros(10, int), S, 0.5) is None


def test_threshold_for_recall_takes_the_highest_threshold_that_reaches_it():
    assert evaluate.threshold_for_recall(Y, S, 0.5) == pytest.approx(
        0.9
    )  # 2 of 4 frauds
    assert evaluate.threshold_for_recall(Y, S, 0.75) == pytest.approx(0.7)
    assert evaluate.threshold_for_recall(Y, S, 1.0) == pytest.approx(0.4)
    assert evaluate.threshold_for_recall(np.zeros(10, int), S, 0.5) is None


def test_ties_are_flagged_together():
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    y = np.array([1, 0, 1, 1, 0])
    assert evaluate.threshold_for_precision(y, s, 0.75) == pytest.approx(
        0.5
    )  # 3 of 4 at 0.5+
    flagged = evaluate.confusion(y, s >= 0.5)
    assert (flagged["tp"], flagged["fp"]) == (3, 1)


def test_policy_cost_by_hand():
    flag = (
        S >= 0.7
    )  # flags 0-3: frauds 0, 1, 3 caught; the fraud of amount 30 is missed
    c = evaluate.policy_cost(Y, AMOUNT, flag, review_cost=10.0, friction=3.0)
    assert (
        c["missed_loss"] == 30.0
        and c["review_cost"] == 40.0
        and c["friction_cost"] == 3.0
    )
    assert c["total"] == 73.0 and c["caught_amount"] == 260.0


def test_trivial_costs_and_the_oracle_skips_frauds_cheaper_than_a_review():
    t = evaluate.trivial_costs(Y, AMOUNT, review_cost=20.0)
    assert t["flag_nothing"] == 290.0 and t["flag_everything"] == 200.0
    # frauds are 50, 10, 200 and 30: reviewing the 10 costs 20, so the oracle leaves it
    assert t["oracle"] == 3 * 20.0 + 10.0


@pytest.mark.parametrize("review_cost", [0.5, 3.0, 25.0, 400.0])
def test_best_cost_threshold_matches_a_brute_force_search(review_cost):
    rng = np.random.default_rng(int(review_cost * 10))
    s = np.round(rng.random(400), 2)  # rounding makes ties
    y = (rng.random(400) < s**3).astype(int)
    amount = rng.lognormal(3, 1, 400)
    best = evaluate.best_cost_threshold(y, amount, s, review_cost)
    candidates = [None, *np.unique(s)]
    costs = {
        t: evaluate.policy_cost(
            y, amount, np.zeros(400, bool) if t is None else s >= t, review_cost
        )["total"]
        for t in candidates
    }
    assert costs[best] == pytest.approx(min(costs.values()))


def test_amount_aware_rule_trades_probability_against_amount_and_friction():
    p = np.array([0.5, 0.01, 0.5, 0.9, 0.5])
    amount = np.array([100.0, 100.0, 10.0, 10.0, 12.0])
    # expected loss avoided is 50, 1, 5, 9, 6 against a review cost of 5 (a tie is not flagged)
    assert evaluate.amount_aware_flag(p, amount, 5.0).tolist() == [
        True,
        False,
        False,
        True,
        True,
    ]
    # friction f = 10 raises the bar by (1 - p) * f to 10, 14.9, 10, 5.1, 10: row 4 (6) drops out
    flags = evaluate.amount_aware_flag(p, amount, 5.0, friction=10.0)
    assert flags.tolist() == [True, False, False, True, False]


def test_capacity_flags_the_top_k_of_each_hour_even_when_unsorted():
    score = np.array([0.1, 0.9, 0.5, 0.8, 0.7, 0.2])
    hour = np.array([1, 0, 1, 0, 1, 0])
    assert evaluate.capacity_flag(score, hour, 1).tolist() == [
        False,
        True,
        False,
        False,
        True,
        False,
    ]
    assert evaluate.capacity_flag(score, hour, 2).sum() == 4
    assert evaluate.capacity_flag(score, hour, 99).all()


def test_precision_at_k_and_recall_at_precision():
    assert evaluate.precision_at_k(Y, S, 4) == 0.75
    assert evaluate.recall_at_precision(Y, S, 0.7) == pytest.approx(0.75)
    assert evaluate.recall_at_precision(Y, S, 1.01) == 0.0


def test_bootstrap_decision_brackets_the_estimate_and_narrows_with_data():
    rng = np.random.default_rng(0)

    def make(n):
        y = (rng.random(n) < 0.02).astype(int)
        return y, rng.random(n) < np.where(y == 1, 0.7, 0.01)

    small, large = make(5_000), make(200_000)
    for y, flag in (small, large):
        ci = evaluate.bootstrap_decision(y, flag)
        c = evaluate.confusion(y, flag)
        assert ci["precision"][0] <= c["precision"] <= ci["precision"][1]
        assert ci["recall"][0] <= c["recall"] <= ci["recall"][1]

    def width(d):
        return d["recall"][1] - d["recall"][0]

    assert width(evaluate.bootstrap_decision(*large)) < width(
        evaluate.bootstrap_decision(*small)
    )


def test_bootstrap_decision_with_no_flags_has_no_precision_interval():
    y = np.array([1, 0, 0, 1, 0] * 20)
    ci = evaluate.bootstrap_decision(y, np.zeros(100, bool))
    assert ci["precision"] == (None, None) and ci["recall"] == (0.0, 0.0)


def test_bootstrap_cost_brackets_the_cost():
    rng = np.random.default_rng(1)
    y = (rng.random(20_000) < 0.01).astype(int)
    amount = rng.lognormal(3, 1, 20_000)
    flag = rng.random(20_000) < np.where(y == 1, 0.6, 0.005)
    lo, hi = evaluate.bootstrap_cost(y, amount, flag, 10.0, n_boot=500)
    assert lo < evaluate.policy_cost(y, amount, flag, 10.0)["total"] < hi


def test_threshold_stability_spreads_with_a_small_validation_window():
    rng = np.random.default_rng(2)

    def make(n):
        y = (rng.random(n) < 0.01).astype(int)
        return y, np.clip(rng.normal(np.where(y == 1, 0.7, 0.2), 0.15), 0, 1)

    out = evaluate.threshold_stability(
        *make(3_000), *make(30_000), target=0.9, n_boot=60
    )
    assert out["resamples"] > 40
    assert (
        out["threshold"]["p5"] <= out["threshold"]["median"] <= out["threshold"]["p95"]
    )
    assert 0 <= out["test_recall"]["p5"] <= out["test_recall"]["p95"] <= 1


def test_threshold_stability_when_no_threshold_reaches_the_target():
    y = np.array([0, 1] * 50)
    s = np.tile([0.9, 0.1], 50)  # the frauds are the lowest scores
    assert (
        evaluate.threshold_stability(y, s, y, s, target=0.99, n_boot=10)["resamples"]
        == 0
    )


def test_psi_is_zero_for_the_same_data_and_large_for_a_shift():
    rng = np.random.default_rng(5)
    a = rng.normal(size=20_000)
    assert evaluate.psi(a, rng.normal(size=20_000)) < 0.02
    assert evaluate.psi(a, rng.normal(1.5, size=20_000)) > 0.25
    assert evaluate.psi(np.zeros(10), np.zeros(10)) == 0.0


def test_pr_points_keep_exactly_what_average_precision_needs():
    """Thinning the curve evenly lost the top of it; these points must reproduce the exact AP."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(3)
    y = (rng.random(50_000) < 0.003).astype(int)
    s = rng.normal(np.where(y == 1, 2.0, 0.0), 1.0)
    recall, precision = evaluate.pr_points(y, s)
    # sklearn's definition: sum over the recall steps of (recall step) x precision on arrival. At
    # equal recall the arrival point has the best precision, so ties are ordered that way.
    order = np.lexsort((-precision, recall))
    r, p = recall[order], precision[order]
    ap = float(np.sum(np.diff(r) * p[1:]))
    assert ap == pytest.approx(average_precision_score(y, s), abs=1e-9)
    assert (
        len(recall) < 2 * int(y.sum()) + 3
    )  # a few hundred points, not fifty thousand


def test_pr_points_lie_on_or_above_every_fixed_precision_operating_point():
    rng = np.random.default_rng(4)
    y = (rng.random(30_000) < 0.004).astype(int)
    s = rng.normal(np.where(y == 1, 3.0, 0.0), 1.0)
    thr = evaluate.threshold_for_precision(y, s, 0.9)
    c = evaluate.confusion(y, s >= thr)
    recall, precision = evaluate.pr_points(y, s)
    # an operating point sits on a recall plateau, at or below the best precision for that recall
    same = np.isclose(recall, c["recall"])
    assert same.any() and precision[same].max() >= c["precision"] - 1e-9
