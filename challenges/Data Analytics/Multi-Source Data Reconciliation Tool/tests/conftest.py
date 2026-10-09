import shutil
from pathlib import Path

import pytest
from data_reconciler.paths import ENV_VAR

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def home(tmp_path_factory):
    """A project root holding only the committed sample and the bundled config."""
    root = tmp_path_factory.mktemp("home")
    shutil.copytree(ROOT / "sample_data", root / "sample_data")
    shutil.copy(ROOT / "airports.toml", root / "airports.toml")
    return root


@pytest.fixture(autouse=True)
def _env(home, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(home))
