from pathlib import Path

import pytest
from data_lineage.graph import build_graph
from data_lineage.paths import ENV_VAR
from data_lineage.project import load_project

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _home(monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(ROOT))


@pytest.fixture(scope="session")
def real_project():
    return load_project(ROOT / "pipeline")


@pytest.fixture(scope="session")
def real_graph(real_project):
    return build_graph(real_project)
