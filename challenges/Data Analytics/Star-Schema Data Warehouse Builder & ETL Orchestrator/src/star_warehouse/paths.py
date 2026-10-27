"""Where the challenge's runtime folders (``data/``, ``out/``, ``sample_data/``) live."""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "STAR_WAREHOUSE_HOME"


def project_root() -> Path:
    """The challenge folder.

    ``STAR_WAREHOUSE_HOME`` when it is set, otherwise the nearest folder above this file that has a
    ``pyproject.toml``, otherwise the working directory (an installed wheel has none nearby).
    """
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
