from __future__ import annotations

import numpy as np
import pytest
from comp_finder import features
from comp_finder.similarity import LeafProximity, WeightedEuclid, top_k


def test_euclid_similarity_is_negative_squared_distance(engine):
    m = WeightedEuclid(engine.z)
    q, p = np.array([5, 9]), np.array([3, 200, 700])
    s = m.sim(q, p)
    expected = -((engine.z[q][:, None, :] - engine.z[p][None, :, :]) ** 2).sum(axis=2)
    assert np.allclose(s, expected)
    assert (m.sim(q, q).diagonal() == 0).all() or np.allclose(m.sim(q, q).diagonal(), 0)


def test_weights_scale_each_feature(engine):
    w = np.zeros(engine.z.shape[1])
    w[features.COLUMNS.index("grade")] = 2.0
    m = WeightedEuclid(engine.z, w)
    q, c = 4, 40
    gi = features.COLUMNS.index("grade")
    assert -m.sim(np.array([q]), np.array([c]))[0, 0] == pytest.approx(
        4 * (engine.z[q, gi] - engine.z[c, gi]) ** 2
    )


def test_euclid_contributions_add_up_to_the_distance(engine):
    for name in ("knn", "ridge", "hedonic"):
        m = engine.metrics[name]
        for q, c in [(5, 50), (300, 12), (2000, 1500)]:
            assert m.contributions(q, c).sum() == pytest.approx(
                -m.sim(np.array([q]), np.array([c]))[0, 0]
            )
            assert m.total(q, c) == pytest.approx(m.contributions(q, c).sum())


def test_leaf_proximity_is_a_weighted_fraction_of_shared_leaves(engine):
    m: LeafProximity = engine.metrics["leaf"]
    idx = np.array([0, 10, 20, 700])
    s = m.sim(idx, idx)
    assert np.allclose(s.diagonal(), 1.0)  # tree weights sum to 1
    assert (s >= 0).all() and (s <= 1 + 1e-9).all()
    assert np.allclose(s, s.T)
    leaves = engine.gbm.booster.predict(engine.z[idx], pred_leaf=True)
    manual = np.array(
        [
            [
                (engine.gbm.tree_weight * (leaves[a] == leaves[b])).sum()
                for b in range(4)
            ]
            for a in range(4)
        ]
    )
    assert np.allclose(s, manual)


def test_leaf_contributions_add_up_to_the_proximity(engine):
    m: LeafProximity = engine.metrics["leaf"]
    for q, c in [(5, 50), (300, 12), (2000, 1500)]:
        contrib = m.contributions(q, c)
        assert contrib.shape == (len(features.COLUMNS),)
        assert (contrib >= 0).all()
        assert contrib.sum() == pytest.approx(m.total(q, c), rel=1e-6)


def test_top_k_orders_best_first_and_pushes_minus_inf_last():
    s = np.array([[0.1, 0.9, 0.5, -np.inf], [0.2, -np.inf, -np.inf, 0.1]])
    idx, sc = top_k(s, 3)
    assert idx[0].tolist() == [1, 2, 0]
    assert idx[1][:2].tolist() == [0, 3]
    assert not np.isfinite(sc[1][2])
    assert top_k(s, 10)[0].shape == (2, 4)
