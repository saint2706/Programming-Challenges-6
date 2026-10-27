"""Shared test helpers: the committed fixture location and the engine/oracle plumbing."""

from __future__ import annotations

import os
import sys

from dist_pipeline.engines import ENGINES, missing_imports
from dist_pipeline.paths import fixtures_dir

FIXTURE_SCALE = (
    "mini"  # tests/fixtures/data/normalized/mini: 20,000 real trips in two parts
)


def fixture_home():
    """The folder that stands in for the challenge root (its ``data/`` is the fixture)."""
    return fixtures_dir()


def installed(engine: str) -> bool:
    return not missing_imports(ENGINES[engine])


def spark_blocked() -> str | None:
    """Why Spark cannot run here, or ``None``. Windows needs Hadoop's native files for Parquet."""
    if sys.platform == "win32" and not os.environ.get("HADOOP_HOME"):
        return "Spark needs HADOOP_HOME (winutils.exe, hadoop.dll) to read Parquet on Windows"
    return None
