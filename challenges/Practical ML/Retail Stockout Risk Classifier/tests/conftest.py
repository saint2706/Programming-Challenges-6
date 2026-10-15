from __future__ import annotations

import pytest
from helpers import synthetic_raw
from stockout_risk import data, features, pipeline


@pytest.fixture(scope="session")
def raw():
    return synthetic_raw()


@pytest.fixture(scope="session")
def daily(raw):
    return data.daily(raw)


@pytest.fixture(scope="session")
def feat(daily):
    return features.build(daily)


@pytest.fixture(scope="session")
def out(feat):
    """The whole experiment on the synthetic table, run once."""
    return pipeline.run(feat, rounds=150)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKOUT_HOME", str(tmp_path))
    return tmp_path
