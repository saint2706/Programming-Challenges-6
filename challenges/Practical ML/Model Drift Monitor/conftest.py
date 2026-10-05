"""Shared fixtures: a tiny trained pipeline on synthetic drifting data, built once."""

import pytest

import data
import pipeline
from test_pipeline import W, drifting


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    """``(artifacts, report, data_dir)`` from a full small run_all."""
    tmp = tmp_path_factory.mktemp("tiny")
    report = pipeline.run_all(
        tmp,
        tmp / "results",
        seed=0,
        n_seeds=1,
        df=drifting(),
        window=W,
        bench_windows=20,
        bench_drift_window=8,
    )
    return pipeline.load_artifacts(tmp), report, tmp


@pytest.fixture(scope="session")
def tiny_context(tiny):
    """``(artifacts, live frame)`` exactly as ``pipeline.load_context`` returns them."""
    art, _report, _tmp = tiny
    _, _, live = data.split(drifting())
    return art, live
