"""Where the library and the models live. Both sit beside pyproject.toml unless overridden."""

import os
from pathlib import Path


def project_root() -> Path:
    """Walk up from this file to the folder holding pyproject.toml (fallback: current directory)."""
    override = os.environ.get("VOICEMEMO_HOME")
    if override:
        return Path(override).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def data_dir() -> Path:
    return project_root() / "data"


def models_dir() -> Path:
    override = os.environ.get("VOICEMEMO_MODELS")
    return Path(override).resolve() if override else project_root() / "models"
