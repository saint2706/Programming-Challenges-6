import math

import numpy as np
import pytest
from evaluate import (
    average_precision_at_k,
    bootstrap_ci,
    mrr,
    ndcg_at_k,
    precision_at_k,
    rank_metrics,
    rank_metrics_matrix,
)


def test_ndcg_matches_hand_computed_value():
    # ranked relevance [1, 0, 1]: DCG = 1 + 1/log2(4); ideal [1, 1, 0]: 1 + 1/log2(3)
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([1, 0, 1], 3) == pytest.approx(expected)


def test_ndcg_is_one_for_a_perfect_ranking_and_zero_without_relevant_items():
    assert ndcg_at_k([1, 1, 0, 0], 4) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_ndcg_ideal_dcg_counts_relevant_items_beyond_the_cutoff():
    # 5 relevant items exist but k=2: the ideal is [1, 1], so a perfect top-2 is 1.0
    assert ndcg_at_k([1, 1, 1, 1, 1], 2) == pytest.approx(1.0)


def test_mrr_is_reciprocal_rank_of_first_relevant():
    assert mrr([0, 0, 1, 1]) == pytest.approx(1 / 3)
    assert mrr([0, 0]) == 0.0
    assert mrr([]) == 0.0


def test_precision_at_k_counts_hits_in_the_top_k_only():
    assert precision_at_k([1, 0, 1, 1], 2) == 0.5
    assert precision_at_k([1], 5) == 1 / 5  # short lists are padded with misses


def test_average_precision_matches_hand_computed_value():
    # hits at ranks 1 and 3: (1/1 + 2/3) / min(k=4, 2 relevant) = 5/6
    assert average_precision_at_k([1, 0, 1, 0], 4) == pytest.approx(5 / 6)
    assert average_precision_at_k([0, 0], 2) == 0.0


def test_rank_metrics_orders_by_score_with_stable_ties():
    scores = np.array([0.1, 0.9, 0.9, 0.5])
    relevant = np.array([False, False, True, True])
    # ties between index 1 and 2 break by index, so the relevant one is second
    m = rank_metrics(scores, relevant)
    assert m["mrr"] == pytest.approx(1 / 2)


def test_rank_metrics_all_equal_scores_are_stable_not_random():
    scores = np.zeros(6)
    relevant = np.array([0, 0, 1, 0, 1, 0], dtype=bool)
    assert rank_metrics(scores, relevant) == rank_metrics(scores, relevant)
    assert rank_metrics(scores, relevant)["mrr"] == pytest.approx(1 / 3)


def test_rank_metrics_matrix_returns_one_value_per_query():
    scores = np.array([[0.9, 0.1, 0.2], [0.1, 0.9, 0.2]])
    relevant = np.array([[True, False, False], [False, True, False]])
    out = rank_metrics_matrix(scores, relevant)
    assert out["ndcg@10"].tolist() == [1.0, 1.0]
    assert set(out) == {"ndcg@10", "mrr", "p@10", "map@50"}


def test_bootstrap_ci_brackets_the_mean_and_is_seed_deterministic():
    values = np.random.default_rng(0).uniform(0.4, 0.8, size=200)
    lo, hi = bootstrap_ci(values, n=500, seed=1)
    assert lo < values.mean() < hi
    assert (lo, hi) == bootstrap_ci(values, n=500, seed=1)
    assert (hi - lo) < 0.1


def test_bootstrap_ci_of_a_constant_is_that_constant():
    assert bootstrap_ci(np.full(10, 0.3), n=100, seed=0) == (
        pytest.approx(0.3),
        pytest.approx(0.3),
    )
