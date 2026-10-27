"""Pytest setup: point the challenge at the committed fixture so tests never need the download."""

from __future__ import annotations

import pytest
from dist_pipeline.engines.base import dataset, has_data
from dist_pipeline.paths import ENV_VAR
from helpers import FIXTURE_SCALE, fixture_home


@pytest.fixture
def mini_home(monkeypatch):
    """Route ``data_dir()`` (and subprocess workers, which inherit the environment) to the fixture."""
    monkeypatch.setenv(ENV_VAR, str(fixture_home()))
    ds = dataset(FIXTURE_SCALE)
    assert has_data(ds), "tests/fixtures/data is missing; run scripts/make_fixtures.py"
    return ds
