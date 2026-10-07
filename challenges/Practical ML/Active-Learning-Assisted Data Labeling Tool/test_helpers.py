import numpy as np
from helpers import make_pool


def test_make_pool_is_unit_norm_reproducible_and_covers_every_class():
    X, y = make_pool(n=200, k=8, seed=3)
    assert X.shape == (200, 16)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)
    assert (np.bincount(y, minlength=8) >= 3).all()
    assert (y[::3][:8] == np.arange(8)).all()
    X2, y2 = make_pool(n=200, k=8, seed=3)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
