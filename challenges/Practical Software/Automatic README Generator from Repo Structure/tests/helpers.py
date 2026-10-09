"""Build throwaway repositories from a ``{relative path: text}`` mapping."""

from pathlib import Path

from readme_generator.manifests import parse_manifests
from readme_generator.scan import RepoScan, scan_repo
from readme_generator.signals import collect_signals
from readme_generator.stack import analyze_stack


def make_repo(root: Path, files: dict[str, str]) -> Path:
    for relpath, text in files.items():
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def analyze(root: Path, files: dict[str, str]):
    """(scan, manifests, warnings, stack, signals) for a fixture repo."""
    make_repo(root, files)
    scan: RepoScan = scan_repo(root)
    manifests, warnings = parse_manifests(scan)
    return (
        scan,
        manifests,
        warnings,
        analyze_stack(scan, manifests),
        collect_signals(scan, manifests),
    )
