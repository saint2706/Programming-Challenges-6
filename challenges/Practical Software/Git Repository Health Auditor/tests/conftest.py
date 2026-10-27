"""Shared fixtures. Every test that needs git is skipped cleanly when git is not installed."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from helpers import GIT, make_repo

requires_git = pytest.mark.skipif(GIT is None, reason="git is not installed")


@pytest.fixture
def repo_factory(tmp_path: Path):
    """Return a function that creates a new synthetic repository under tmp_path."""
    if GIT is None:
        pytest.skip("git is not installed")
    counter = {"n": 0}

    def _make(name: str | None = None, branch: str = "main"):
        counter["n"] += 1
        return make_repo(tmp_path / (name or f"repo{counter['n']}"), branch)

    return _make


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Strip GIT_* and GITAUDIT_* variables so the test environment is deterministic."""
    for key in list(os.environ):
        if key.startswith(("GIT_", "GITAUDIT_")):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch
