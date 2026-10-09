"""Compare sources key by key, and lint each one, using only fingerprints and flags."""

from dataclasses import dataclass, field

from envdiff.fingerprint import is_secret_name, matches_any
from envdiff.model import Source

PARSE_ISSUE_TEXT = {
    "duplicate-key": ("warn", "defined more than once; the last definition wins"),
    "invalid-line": ("warn", "is not a KEY=value line and was skipped"),
    "unterminated-quote": ("warn", "has an opening quote that is never closed; the variable was skipped"),
    "trailing-garbage": ("warn", "has text after its closing quote that shells and dotenv libraries treat differently"),
    "bom": ("warn", "file starts with a byte-order mark, which glues itself onto the first variable name in some loaders"),
    "crlf": ("warn", "file uses Windows (CRLF) line endings; `source` keeps the \\r in every value"),
    "unquoted-space": ("info", "has whitespace in an unquoted value, which `source` or `export` would split"),
}  # fmt: skip


@dataclass
class KeyResult:
    key: str
    status: str  # same | differs | missing
    relations: list[
        tuple[int, int, str]
    ]  # (group, group, why they differ) for each explainable pair
    groups: dict[str, int]  # label -> equality group (0, 1, ...); -1 for pass-through
    absent: list[str]
    shapes: dict[str, str]
    fingerprints: dict[str, str]  # label -> short exact fingerprint


@dataclass
class Finding:
    code: str
    severity: str  # warn | info
    message: str
    key: str | None = None
    sources: list[str] = field(default_factory=list)
    line: int = 0


@dataclass
class Report:
    sources: list[Source]
    keys: list[KeyResult]
    findings: list[Finding]
    ignored: int = 0

    def count(self, status: str) -> int:
        return sum(1 for k in self.keys if k.status == status)

    @property
    def has_differences(self) -> bool:
        return any(k.status != "same" for k in self.keys)

    @property
    def has_warnings(self) -> bool:
        return any(f.severity == "warn" for f in self.findings)


def _relation(first, second) -> str | None:
    """Why two different values differ, when the reason is a benign-looking one."""
    if first.fingerprints is None or second.fingerprints is None:
        return "pass-through"
    a, b = first.fingerprints, second.fingerprints
    if a.stripped == b.stripped:
        return "whitespace"
    if a.unquoted == b.unquoted:
        return "quotes"
    if a.folded == b.folded:
        return "case"
    if first.empty or second.empty:
        return "empty"
    return None


def compare_key(key: str, sources: list[Source]) -> KeyResult:
    present = [(s.label, s.entries[key]) for s in sources if key in s.entries]
    absent = [s.label for s in sources if key not in s.entries]
    group_of: dict[str, int] = {}
    groups: dict[str, int] = {}
    for label, entry in present:
        if entry.fingerprints is None:
            groups[label] = -1
        else:
            groups[label] = group_of.setdefault(entry.fingerprints.exact, len(group_of))
    representative = {}
    for label, entry in present:
        representative.setdefault(groups[label], entry)
    relations = []
    ids = sorted(representative)
    for index, first in enumerate(ids):
        for second in ids[index + 1 :]:
            why = _relation(representative[first], representative[second])
            if why:
                relations.append((first, second, why))
    differs = len(representative) > 1
    status = "differs" if differs else ("missing" if absent else "same")
    return KeyResult(
        key=key,
        status=status,
        relations=relations,
        groups=groups,
        absent=absent,
        shapes={label: entry.shape for label, entry in present},
        fingerprints={
            label: entry.fingerprints.exact[:8]
            for label, entry in present
            if entry.fingerprints
        },
    )


def _lint_source(source: Source, ignore: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for issue in source.issues:
        if issue.key and matches_any(issue.key, ignore):
            continue
        severity, text = PARSE_ISSUE_TEXT[issue.kind]
        if issue.key:
            message = f"`{issue.key}` {text}"
        elif issue.kind in {"bom", "crlf"}:
            message = text[0].upper() + text[1:]
        else:
            message = f"A line {text}"
        where = f"line {issue.line}" if issue.line else "whole file"
        findings.append(
            Finding(
                f"parse:{issue.kind}",
                severity,
                f"{message} ({where})",
                issue.key,
                [source.label],
                issue.line,
            )
        )
    for key, entry in sorted(source.entries.items()):
        if matches_any(key, ignore):
            continue
        if entry.empty and is_secret_name(key):
            findings.append(
                Finding(
                    "empty-secret",
                    "warn",
                    f"`{key}` looks like a secret but is empty",
                    key,
                    [source.label],
                    entry.line,
                )
            )
        if entry.placeholder:
            findings.append(
                Finding(
                    "placeholder",
                    "warn",
                    f"`{key}` still holds a placeholder-looking value",
                    key,
                    [source.label],
                    entry.line,
                )
            )
        if entry.edge_whitespace:
            findings.append(
                Finding(
                    "edge-whitespace",
                    "warn",
                    f"`{key}` has leading or trailing whitespace in its value",
                    key,
                    [source.label],
                    entry.line,
                )
            )
    return findings


def _lint_across(results: list[KeyResult], sources: list[Source]) -> list[Finding]:
    findings: list[Finding] = []
    by_label = {s.label: s for s in sources}
    for result in results:
        if is_secret_name(result.key):
            members: dict[int, list[str]] = {}
            for label, group in result.groups.items():
                entry = by_label[label].entries[result.key]
                if group >= 0 and not entry.empty and not entry.placeholder:
                    members.setdefault(group, []).append(label)
            for labels in members.values():
                if len(labels) > 1:
                    findings.append(
                        Finding(
                            "shared-secret",
                            "warn",
                            f"`{result.key}` has the same value in {', '.join(labels)}; "
                            "environments that must stay isolated should not share a secret",
                            result.key,
                            labels,
                        )
                    )
        shapes = {s for s in result.shapes.values() if s not in {"redacted", "empty"}}
        if result.status == "differs" and len(shapes) > 1:
            kinds = ", ".join(
                f"{label}: {shape}"
                for label, shape in result.shapes.items()
                if shape != "redacted"
            )
            findings.append(
                Finding(
                    "shape-mismatch",
                    "info",
                    f"`{result.key}` holds different kinds of value ({kinds})",
                    result.key,
                    list(result.shapes),
                )
            )
    return findings


def compare(
    sources: list[Source], ignore: list[str] | None = None, lint: bool = True
) -> Report:
    ignore = ignore or []
    all_keys = sorted({key for s in sources for key in s.entries})
    kept = [k for k in all_keys if not matches_any(k, ignore)]
    results = [compare_key(key, sources) for key in kept]
    findings: list[Finding] = []
    if lint:
        for source in sources:
            findings += _lint_source(source, ignore)
        findings += _lint_across(results, sources)
    return Report(sources, results, findings, ignored=len(all_keys) - len(kept))
