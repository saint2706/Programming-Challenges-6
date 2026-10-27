import os
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))  # lets test modules `import helpers`

from voicememo.asr import FakeTranscriber
from voicememo.ingest import FixedEngine
from voicememo.library import Library
from voicememo.paths import models_dir

# Read before the autouse fixture below points VOICEMEMO_MODELS at a temp folder.
REAL_MODELS = models_dir()


def pytest_collection_modifyitems(config, items):
    """`real_models` tests download nothing, but need pulled models, so they are opt-in."""
    if os.environ.get("VOICEMEMO_REAL_MODELS") == "1":
        return
    skip = pytest.mark.skip(reason="real-model test: set VOICEMEMO_REAL_MODELS=1")
    for item in items:
        if "real_models" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own library and models folder; nothing touches the real data dir."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("VOICEMEMO_HOME", str(home))
    monkeypatch.setenv("VOICEMEMO_MODELS", str(tmp_path / "models"))
    monkeypatch.delenv("VOICEMEMO_FAKE_SCRIPT", raising=False)
    return home


@pytest.fixture
def library(isolated_home):
    lib = Library(isolated_home / "data")
    yield lib
    lib.close()


@pytest.fixture
def fake():
    return FakeTranscriber()


@pytest.fixture
def engine(fake):
    return FixedEngine(fake)


@pytest.fixture
def memo_dir(tmp_path):
    folder = tmp_path / "memos"
    folder.mkdir()
    return folder
