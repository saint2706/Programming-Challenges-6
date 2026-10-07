"""Shared fixtures."""

import numpy as np
import pytest
import synth


@pytest.fixture(scope="session")
def rct():
    """``(X, t, y, tau)``: a 60k-row RCT with a strong heterogeneous effect, e = 0.85."""
    rng = np.random.default_rng(0)
    n = 60_000
    X = rng.normal(size=(n, 12)).astype(np.float32)
    t = (rng.random(n) < 0.85).astype(np.int8)
    y, tau = synth.simulate(
        X, t, seed=0, scenario="heterogeneous", base_rate=0.15, strength=2.0
    )
    return X, t, y, tau
