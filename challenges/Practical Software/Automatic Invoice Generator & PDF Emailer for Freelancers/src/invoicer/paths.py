"""Where the database, config and generated PDFs live."""

import os
from pathlib import Path

ENV_VAR = "INVOICER_HOME"


def project_root() -> Path:
    """``INVOICER_HOME`` when set, else the nearest folder above this file with a
    ``pyproject.toml``, else the working directory (an installed wheel has none nearby)."""
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
