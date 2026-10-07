"""Test helpers: a small Criteo-shaped frame with a known heterogeneous effect."""

import numpy as np
import polars as pl
import synth

N_FEATURES = 12


def make_frame(n=20000, seed=0, confounded=False, strength=1.5) -> pl.DataFrame:
    """Frame with the real column names. ``confounded`` makes treatment depend on f0."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
    p = 1 / (1 + np.exp(-(1.7 + 1.5 * X[:, 0]))) if confounded else 0.85
    t = (rng.random(n) < p).astype(np.int8)
    visit, _ = synth.simulate(
        X, t, seed=seed, scenario="heterogeneous", base_rate=0.15, strength=strength
    )
    conv, _ = synth.simulate(
        X, t, seed=seed + 1, scenario="heterogeneous", base_rate=0.05, strength=strength
    )
    exposure = (t * (rng.random(n) < 0.7)).astype(np.int8)
    return pl.DataFrame(
        {
            **{f"f{i}": X[:, i] for i in range(N_FEATURES)},
            "treatment": t,
            "visit": visit,
            "conversion": conv,
            "exposure": exposure,
        }
    )


def confounded_rct(n=300_000, seed=0, tau=0.02):
    """Heavy users (x > 0) are both treated more often and convert more at baseline."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    e = 0.5 + 0.4 * np.tanh(2 * x)
    t = (rng.random(n) < e).astype(int)
    base = np.where(x > 0, 0.20, 0.05)
    y = (rng.random(n) < base + tau * t).astype(int)
    return x, t, y, e
