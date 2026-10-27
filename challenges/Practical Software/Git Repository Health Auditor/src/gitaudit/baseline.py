"""Baselines: accept today's findings so only new ones fail the run.

A baseline is JSON: {"version": 1, "findings": [{"fingerprint": "<64 hex>", "rule": "SEC013"}, ...]}.
It stores fingerprints and rule ids only, never evidence or secret values. Fingerprints are
unsalted hashes, so a baseline file should be treated as sensitive: it reveals which rules fire.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from gitaudit.findings import CATALOG, Finding
from gitaudit.safety import atomic_write, check_output_path, sanitize

BASELINE_VERSION = 1
MAX_BASELINE_BYTES = 8 * 1024 * 1024
MAX_BASELINE_ENTRIES = 200_000
_HEX64 = re.compile(r"[0-9a-f]{64}")


def load_baseline(path: Path) -> frozenset[str]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"baseline not found: {sanitize(path, 200)}")
    if path.stat().st_size > MAX_BASELINE_BYTES:
        raise ValueError("baseline file is too large")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"baseline is not valid JSON: {sanitize(exc, 120)}") from exc
    if not isinstance(data, dict) or data.get("version") != BASELINE_VERSION:
        raise ValueError(f"baseline must be a version {BASELINE_VERSION} object")
    entries = data.get("findings")
    if not isinstance(entries, list) or len(entries) > MAX_BASELINE_ENTRIES:
        raise ValueError("baseline 'findings' must be a list of reasonable size")
    fingerprints: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("every baseline entry must be an object")  # noqa: TRY004
        fp = entry.get("fingerprint")
        if not isinstance(fp, str) or not _HEX64.fullmatch(fp):
            raise ValueError("baseline fingerprint must be 64 lowercase hex characters")
        fingerprints.add(fp)
    return frozenset(fingerprints)


def apply_baseline(findings: list[Finding], accepted: frozenset[str]) -> list[Finding]:
    """Mark accepted findings as suppressed. Suppressed findings stay in reports but never fail a run."""
    return [
        f if f.fingerprint not in accepted else replace(f, suppressed=True)
        for f in findings
    ]


def write_baseline(
    path: Path, findings: list[Finding], forbidden_root: Path | None
) -> int:
    target = check_output_path(Path(path), forbidden_root)
    seen: dict[str, str] = {}
    for f in findings:
        if f.rule_id in CATALOG and _HEX64.fullmatch(f.fingerprint):
            seen.setdefault(f.fingerprint, f.rule_id)
    payload = {
        "version": BASELINE_VERSION,
        "findings": [
            {"fingerprint": fp, "rule": rule} for fp, rule in sorted(seen.items())
        ],
    }
    atomic_write(target, json.dumps(payload, indent=2) + "\n")
    return len(seen)
