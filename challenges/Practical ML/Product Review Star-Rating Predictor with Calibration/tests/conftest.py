"""Every test runs against a throwaway challenge root: none can touch the real data/ or results/."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("REVIEW_STARS_HOME", str(tmp_path / "home"))
