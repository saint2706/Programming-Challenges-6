from __future__ import annotations

import pytest
from fraud_flagging import data, features, pipeline
from helpers import raw_with_labels, synthetic_published


@pytest.fixture(scope="session")
def published():
    return synthetic_published()


@pytest.fixture(scope="session")
def raw(published):
    return raw_with_labels(published)


@pytest.fixture(scope="session")
def frame(raw):
    return features.build(data.clean(raw))


@pytest.fixture(scope="session")
def out(frame):
    """The whole experiment on the synthetic table, run once."""
    return pipeline.run(frame, n_boot=20, rounds=200)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUD_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def installed(home, published):
    """FRAUD_HOME holding the synthetic data where ``data.download`` expects it."""
    (home / "data").mkdir()
    published.write_parquet(home / "data" / "creditcard.parquet")
    return home
