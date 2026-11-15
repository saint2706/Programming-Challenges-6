"""Where the challenge folder (data/, results/) lives."""

import os
from pathlib import Path


def project_root() -> Path:
    """DATA_CATALOG_HOME if set, else the nearest folder above this file with a pyproject.toml."""
    if override := os.environ.get("DATA_CATALOG_HOME"):
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
