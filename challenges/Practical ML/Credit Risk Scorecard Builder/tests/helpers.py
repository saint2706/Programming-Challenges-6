"""Small synthetic datasets for the unit tests (the real data is in ``conftest.py``)."""

from __future__ import annotations

import numpy as np
import polars as pl


def synthetic_frame(n: int = 20_000, seed: int = 0) -> pl.DataFrame:
    """Two informative features, one pure-noise feature, and one near-copy of ``strong``."""
    rng = np.random.default_rng(seed)
    strong = rng.normal(size=n)
    weak = rng.normal(size=n)
    logit = -3.0 + 1.2 * strong - 0.5 * weak
    bad = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pl.DataFrame(
        {
            "strong": strong,
            "weak": weak,
            "noise": rng.normal(size=n),
            "copy_of_strong": strong + rng.normal(scale=0.05, size=n),
            "bad": bad,
        }
    )
