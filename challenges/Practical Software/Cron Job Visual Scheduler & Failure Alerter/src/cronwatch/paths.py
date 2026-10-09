"""Where the challenge's runtime folders (``data/``, ``results/``, ...) live."""

import os
from pathlib import Path

ENV_VAR = "CRONWATCH_HOME"


def project_root() -> Path:
    """The folder holding ``cronwatch.db``, ``config.toml`` and ``backups/``.

    ``CRONWATCH_HOME`` when it is set, otherwise the nearest folder above this file that has a
    ``pyproject.toml``, otherwise the working directory (an installed wheel has none nearby).
    A crontab that runs the wrapper should set ``CRONWATCH_HOME`` itself: cron's environment is minimal.
    """
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
