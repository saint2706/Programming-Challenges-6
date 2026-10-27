"""The marimo notebook runs top to bottom: export fails on any cell error (slow, runs marimo)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_notebook_exports_and_shows_the_sleep_result(tmp_path):
    out = tmp_path / "notebook.html"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            "notebook.py",
            "-o",
            str(out),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    html = out.read_text(encoding="utf-8")
    assert "paired_t" in html
    assert "Traceback" not in html
