"""Opt-in guard for the secret-scanner evaluation. Run with GITAUDIT_SLOW=1 pytest -m slow."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.slow


@pytest.mark.skipif(
    os.environ.get("GITAUDIT_SLOW") != "1", reason="set GITAUDIT_SLOW=1"
)
def test_secret_evaluation_meets_floor():
    spec = importlib.util.spec_from_file_location(
        "evaluate_secrets", ROOT / "scripts" / "evaluate_secrets.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses looks the module up here
    spec.loader.exec_module(module)
    report = module.evaluate(module.corpus())
    assert report["precision"] == 1.0, report["false_alarms"]
    assert report["recall"] >= 0.9, report["misses"]


def test_committed_results_match_schema():
    path = ROOT / "results" / "secret_eval.json"
    if not path.exists():
        pytest.skip("results not generated yet")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert {"precision", "recall", "per_rule", "samples"} <= data.keys()
