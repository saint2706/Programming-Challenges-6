import shutil
from pathlib import Path

import pytest
from sql_profiler import db
from sql_profiler.paths import ENV_VAR

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def home(tmp_path_factory):
    """A project root holding only the committed sample, so tests never depend on data/."""
    root = tmp_path_factory.mktemp("home")
    shutil.copytree(ROOT / "sample_data", root / "sample_data")
    return root


@pytest.fixture(autouse=True)
def _env(home, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(home))


@pytest.fixture(scope="session")
def con(home):
    return db.connect(home)
