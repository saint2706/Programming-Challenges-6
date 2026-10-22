from pathlib import Path

import pytest
from survival_toolkit.paths import ENV_VAR

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _home(monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(ROOT))
