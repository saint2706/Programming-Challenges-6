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


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    """``(artifacts, report, results_dir)`` from a complete small run_all."""
    import pipeline
    from helpers import make_frame

    tmp = tmp_path_factory.mktemp("tiny")
    tiny_params = {"n_estimators": 40, "min_child_samples": 30, "num_leaves": 15}
    report = pipeline.run_all(
        tmp,
        tmp / "results",
        seed=0,
        n_boot=30,
        n_seeds=2,
        df=make_frame(20000, strength=2.0),
        grid=[tiny_params],
        synth_n=4000,
        synth_params=tiny_params,
    )
    art = pipeline.load_artifacts(tmp / "results", with_models=True)
    return art, report, tmp / "results"
