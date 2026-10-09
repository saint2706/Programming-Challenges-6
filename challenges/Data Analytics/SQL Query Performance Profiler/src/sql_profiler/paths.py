"""Where the challenge's runtime folders (``data/``, ``results/``, ...) live."""

import os
from pathlib import Path

ENV_VAR = "SQL_PROFILER_HOME"


def project_root() -> Path:
    """The challenge folder.

    ``SQL_PROFILER_HOME`` when it is set, otherwise the nearest folder above this file that has a
    ``pyproject.toml``, otherwise the working directory (an installed wheel has none nearby).
    """
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
