"""Tests for match.py -- pure numpy/logic, no models, no index."""

import numpy as np
import pytest
from duplicate_listings import match


def test_fuse_is_convex_combination():
    t = np.array([1.0, 0.0, 0.5])
    i = np.array([0.0, 1.0, 0.5])
    np.testing.assert_allclose(match.fuse(t, i, 1.0), t)
    np.testing.assert_allclose(match.fuse(t, i, 0.0), i)
    np.testing.assert_allclose(match.fuse(t, i, 0.25), [0.25, 0.75, 0.5])


def test_fuse_rejects_weight_outside_unit_interval():
    with pytest.raises(ValueError):
        match.fuse(np.array([1.0]), np.array([1.0]), 1.5)


def test_pair_scores_are_row_dot_products():
    text = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    image = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pairs = np.array([[0, 1], [0, 2]])
    t, i = match.pair_scores(text, image, pairs)
    np.testing.assert_allclose(t, [0.0, 1.0])
    np.testing.assert_allclose(i, [1.0, 0.0])


def test_best_threshold_separates_a_clean_split():
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    labels = np.array([1, 1, 1, 0, 0, 0])
    thr, f1 = match.best_threshold(scores, labels, n_true=3)
    assert f1 == pytest.approx(1.0)
    assert 0.3 < thr <= 0.7


def test_best_threshold_counts_unretrieved_true_pairs_against_recall():
    # 2 true pairs were retrieved and scored, but 2 more were never retrieved at all.
    scores = np.array([0.9, 0.8, 0.1])
    labels = np.array([1, 1, 0])
    _, f1 = match.best_threshold(scores, labels, n_true=4)
    # precision 1.0, recall 0.5 -> F1 = 2/3
    assert f1 == pytest.approx(2 / 3)


def test_best_threshold_handles_no_positives():
    thr, f1 = match.best_threshold(np.array([0.5, 0.4]), np.array([0, 0]), n_true=0)
    assert f1 == 0.0
    assert thr > 0.5  # flag nothing


def test_pair_metrics():
    p, r, f1 = match.pair_metrics(tp=8, fp=2, n_true=16)
    assert p == pytest.approx(0.8)
    assert r == pytest.approx(0.5)
    assert f1 == pytest.approx(2 * 0.8 * 0.5 / 1.3)


def test_pair_metrics_zero_division_is_zero():
    assert match.pair_metrics(tp=0, fp=0, n_true=0) == (0.0, 0.0, 0.0)


def test_cluster_components_are_transitive():
    # 0-1 and 1-2 are flagged, 0-2 is not: union-find still joins all three.
    labels = match.cluster(5, np.array([[0, 1], [1, 2]]))
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0] and labels[4] != labels[0] and labels[3] != labels[4]


def test_cluster_with_no_pairs_is_all_singletons():
    labels = match.cluster(4, np.empty((0, 2), dtype=int))
    assert len(set(labels)) == 4


def test_cluster_f1_perfect_and_imperfect():
    truth = np.array([0, 0, 0, 1, 1])
    assert match.cluster_f1(truth, truth) == pytest.approx(1.0)
    # Everything merged into one cluster: rows of group 0 get F1 2*3/(5+3)=0.75,
    # rows of group 1 get 2*2/(5+2)=4/7.
    merged = np.zeros(5, dtype=int)
    expected = (3 * 0.75 + 2 * 4 / 7) / 5
    assert match.cluster_f1(truth, merged) == pytest.approx(expected)
    # Everything a singleton: F1 per row = 2*1/(1+size).
    single = np.arange(5)
    expected = (3 * (2 / 4) + 2 * (2 / 3)) / 5
    assert match.cluster_f1(truth, single) == pytest.approx(expected)


def test_tune_picks_the_weight_that_helps():
    rng = np.random.default_rng(0)
    n = 400
    labels = (rng.random(n) < 0.5).astype(int)
    # Text is informative, image is pure noise -> tuned weight should favor text.
    text = labels * 0.6 + rng.normal(0.2, 0.05, n)
    image = rng.normal(0.4, 0.2, n)
    tuned = match.tune(text, image, labels, n_true=int(labels.sum()))
    assert tuned.weight >= 0.7
    assert tuned.f1 > 0.95


def test_tune_prefers_image_when_image_is_informative():
    rng = np.random.default_rng(1)
    n = 400
    labels = (rng.random(n) < 0.5).astype(int)
    text = rng.normal(0.4, 0.2, n)
    image = labels * 0.6 + rng.normal(0.2, 0.05, n)
    tuned = match.tune(text, image, labels, n_true=int(labels.sum()))
    assert tuned.weight <= 0.3
