"""One call from a repository path to its README sections."""

from dataclasses import dataclass
from pathlib import Path

from readme_generator.manifests import parse_manifests
from readme_generator.render import Section, build_sections
from readme_generator.scan import scan_repo
from readme_generator.signals import collect_signals
from readme_generator.stack import analyze_stack


@dataclass
class Generated:
    sections: list[Section]
    warnings: list[str]


def generate_sections(
    root: Path, output: Path | None = None, tree_depth: int = 2
) -> Generated:
    """Scan ``root`` and draft its README sections.

    ``output`` is the README being (re)written: it is left out of the tree so that writing it
    cannot change what the next run sees.
    """
    scan = scan_repo(root)
    manifests, warnings = parse_manifests(scan)
    stack = analyze_stack(scan, manifests)
    signals = collect_signals(scan, manifests)
    exclude: frozenset[str] = frozenset()
    if output is not None:
        try:
            exclude = frozenset({output.resolve().relative_to(scan.root).as_posix()})
        except ValueError:
            pass
    sections = build_sections(scan, manifests, stack, signals, tree_depth, exclude)
    if scan.truncated:
        warnings.append(
            "Repository is very large; the scan stopped early and the draft may be incomplete."
        )
    return Generated(sections, warnings)
