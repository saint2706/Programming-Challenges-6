import numpy as np
import pytest
from helpers import make_pool
from model import C_GRID, Head, select_C


def test_proba_covers_every_class_and_unseen_classes_get_exactly_zero():
    X, y = make_pool(n=300, k=10)
    seen = np.isin(y, [0, 3, 5])
    head = Head(10, C=10.0).fit(X[seen], y[seen])
    P = head.proba(X)
    assert P.shape == (300, 10)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (P[:, [1, 2, 4, 6, 7, 8, 9]] == 0).all()
    assert set(head.predict(X)) <= {0, 3, 5}


def test_single_class_head_is_constant():
    X, y = make_pool()
    keep = np.flatnonzero(y == 2)[:5]
    P = Head(10).fit(X[keep], y[keep]).proba(X)
    assert (P[:, 2] == 1.0).all() and P.sum() == len(X)


def test_zero_labels_and_out_of_range_labels_are_rejected():
    X, y = make_pool()
    with pytest.raises(ValueError, match="zero labels"):
        Head(10).fit(X[:0], y[:0])
    with pytest.raises(ValueError, match=r"\[0, 10\)"):
        Head(10).fit(X[:5], np.array([0, 1, 2, 3, 10]))


def test_head_learns_separable_blobs():
    X, y = make_pool(n=600, k=10, spread=0.2)
    head = Head(10, C=30.0).fit(X[:200], y[:200])
    assert (head.predict(X[200:]) == y[200:]).mean() > 0.9


def test_select_C_returns_the_grid_value_with_lowest_validation_log_loss():
    X, y = make_pool(n=600, k=10, spread=0.9, seed=1)
    Xtr, ytr, Xv, yv = X[:400], y[:400], X[400:], y[400:]
    grid = (0.1, 1.0, 10.0, 100.0)
    got = select_C(Xtr, ytr, Xv, yv, 10, grid=grid, n_labels=200, seed=0)
    idx = np.random.default_rng(0).choice(400, size=200, replace=False)

    def loss(C):
        P = Head(10, C).fit(Xtr[idx], ytr[idx]).proba(Xv)
        return -np.log(np.clip(P[np.arange(len(yv)), yv], 1e-12, 1)).mean()

    losses = {C: loss(C) for C in grid}
    assert got == min(losses, key=losses.get)
    assert select_C(Xtr, ytr, Xv, yv, 10, grid=grid, n_labels=200, seed=0) == got
    assert got in grid and set(grid) != set(C_GRID)
