from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _project_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the data lookup to this challenge folder whatever the working directory is."""
    monkeypatch.setenv("HYPOTHESIS_WIZARD_HOME", str(ROOT))
