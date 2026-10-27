"""Where the project keeps its results, found from the package location or ACTIONITEMS_HOME."""

import os
from pathlib import Path


def project_root() -> Path:
    override = os.environ.get("ACTIONITEMS_HOME")
    if override:
        return Path(override).resolve()
    # src/actionitems/paths.py -> the challenge directory two levels up from src/.
    return Path(__file__).resolve().parents[2]


def corpus_dir() -> Path:
    return project_root() / "corpus"


def results_dir() -> Path:
    return project_root() / "results"
