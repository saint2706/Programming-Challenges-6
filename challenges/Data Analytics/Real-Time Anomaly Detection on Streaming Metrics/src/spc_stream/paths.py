"""Where the challenge folder (sample_data/, results/) lives."""

import os
from pathlib import Path


def project_root() -> Path:
    """SPC_STREAM_HOME if set, else the nearest folder above this file with a pyproject.toml."""
    if override := os.environ.get("SPC_STREAM_HOME"):
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
