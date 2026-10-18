"""Where the challenge's runtime folders (``sample_data/``, ``web/``, ``target/``) live."""

import os
from pathlib import Path

ENV_VAR = "PIVOT_TOOL_HOME"


def project_root() -> Path:
    """The challenge folder.

    ``PIVOT_TOOL_HOME`` when it is set, otherwise the nearest folder above this file that has a
    ``pyproject.toml``, otherwise the working directory (an installed wheel has none nearby).
    """
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
