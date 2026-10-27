"""Check 4: general repository hygiene.

Tree checks run once over the audited commit's entries. Commit checks run per commit while the
history stream is already open, so the audit reads history only once.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
import urllib.parse
from collections import Counter
from collections.abc import Iterable

from gitaudit.config import Settings
from gitaudit.findings import Finding, Severity, make_fingerprint
from gitaudit.git import GitRepo, TreeEntry
from gitaudit.safety import sanitize

IGNORABLE_GLOBS = (
    ".DS_Store", "Thumbs.db", "desktop.ini", "*.pyc", "*.pyo", "*.swp", "*.swo", "*~",
    "*.log", "*.tmp", "*.bak", "*.orig", "*.rej", "__pycache__/*", ".pytest_cache/*",
    ".mypy_cache/*", ".ruff_cache/*", "*.egg-info/*", ".coverage", "htmlcov/*",
)  # fmt: skip
DEPENDENCY_DIRS = (
    "node_modules",
    ".venv",
    "venv",
    "vendor",
    "bower_components",
    ".tox",
)
PLACEHOLDER_DOMAINS = (
    "example.com",
    "example.org",
    "example.net",
    "localhost",
    "invalid",
    "test",
)
LICENSE_NAMES = ("license", "licence", "copying", "unlicense")
README_NAMES = ("readme",)
MAX_TEXT_SCAN_FILES = 2000
# Query keys that carry a credential. Literal alternation, no nested quantifiers (linear time).
_SECRET_QUERY = re.compile(r"(?i)(?:token|secret|password|passwd|key|auth)=")


def _finding(
    rule: str,
    severity: Severity,
    title: str,
    evidence: str,
    remediation: str,
    key: str,
    path: str | None = None,
    commit: str | None = None,
) -> Finding:
    return Finding(
        rule_id=rule,
        severity=severity,
        confidence="high",
        title=sanitize(title, 200),
        category="hygiene",
        evidence=sanitize(evidence, 300),
        remediation=remediation,
        fingerprint=make_fingerprint(rule, key),
        path=path,
        commit=commit,
    )


def ignorable_path(path: str) -> bool:
    base = posixpath.basename(path)
    return any(
        fnmatch.fnmatchcase(base, g)
        or fnmatch.fnmatchcase(path, g)
        or fnmatch.fnmatchcase(path, "*/" + g)
        for g in IGNORABLE_GLOBS
    )


def _has_dependency_dir(path: str) -> str | None:
    parts = path.split("/")[:-1]
    for part in parts:
        if part in DEPENDENCY_DIRS:
            return part
    return None


def tree_findings(
    git: GitRepo, entries: list[TreeEntry], settings: Settings
) -> list[Finding]:
    blobs = [e for e in entries if e.kind == "blob"]
    paths = [e.path for e in entries]
    out: list[Finding] = []

    ignorable = [e.path for e in blobs if ignorable_path(e.path)]
    for path in ignorable:
        out.append(
            _finding(
                "HY001", Severity.LOW,
                f"Tracked file that is normally ignored: {sanitize(path, 160)}",
                "file is in the current tree",
                "git rm --cached -- <path> and add a .gitignore rule for it",
                path, path=path,
            )
        )  # fmt: skip

    seen_dep: set[str] = set()
    for path in paths:
        dep = _has_dependency_dir(path)
        if dep and dep not in seen_dep:
            seen_dep.add(dep)
            out.append(
                _finding(
                    "HY004", Severity.MEDIUM,
                    f"Dependency directory is committed: {sanitize(dep, 80)}",
                    f"first seen at {sanitize(path, 160)}",
                    f"git rm -r --cached {dep} and ignore it",
                    dep, path=path,
                )
            )  # fmt: skip

    lowered: dict[str, str] = {}
    for path in paths:
        key = path.lower()
        if key in lowered and lowered[key] != path:
            out.append(
                _finding(
                    "HY007", Severity.MEDIUM,
                    f"Paths differ only by case: {sanitize(lowered[key], 100)} and {sanitize(path, 100)}",
                    "they collide on case-insensitive filesystems",
                    "Rename one of them so the names are unique regardless of case.",
                    key, path=path,
                )
            )  # fmt: skip
        lowered.setdefault(key, path)

    out += _gitignore_findings(paths, entries)
    out += _root_file_findings(paths)
    out += _submodule_findings(git, entries)
    out += _line_ending_findings(git, blobs, settings)
    # Never truncated: the exit code comes from every finding. --top limits table rows only.
    return out


def _gitignore_findings(paths: list[str], entries: list[TreeEntry]) -> list[Finding]:
    gi = [e for e in entries if e.kind == "blob" and e.path == ".gitignore"]
    if not gi:
        return [
            _finding(
                "HY002", Severity.MEDIUM, "No .gitignore at the repository root",
                "build artefacts and secrets are easy to commit by accident",
                "Add a .gitignore for your language and tools (see github/gitignore).",
                "root", path=".gitignore",
            )
        ]  # fmt: skip
    if gi[0].size == 0:
        return [
            _finding(
                "HY003", Severity.LOW, ".gitignore exists but is empty",
                "the file has no rules", "Add rules for your language and tools.",
                "empty", path=".gitignore",
            )
        ]  # fmt: skip
    return []


def _root_file_findings(paths: list[str]) -> list[Finding]:
    roots = {p.lower() for p in paths if "/" not in p}
    out: list[Finding] = []
    if not any(
        name == n or name.startswith(n + ".") for name in roots for n in LICENSE_NAMES
    ):
        out.append(
            _finding(
                "HY009",
                Severity.LOW,
                "No LICENSE file at the repository root",
                "without a license, others have no permission to use the code",
                "Add a LICENSE file, or state explicitly that the code is all-rights-reserved.",
                "license",
            )
        )
    if not any(
        name == n or name.startswith(n + ".") for name in roots for n in README_NAMES
    ):
        out.append(
            _finding(
                "HY010", Severity.LOW, "No README file at the repository root",
                "readers have no entry point", "Add a README describing the project.",
                "readme",
            )
        )  # fmt: skip
    return out


def _read_text_blobs(
    git: GitRepo, shas: list[str], limit: int
) -> Iterable[tuple[str, bytes]]:
    for sha, data, _size in git.blob_contents(shas, limit):
        yield sha, data


def _submodule_urls(text: str) -> list[str]:
    """Values of `url =` lines in .gitmodules, split per line (no multiline regex)."""
    urls = []
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip().lower() == "url":
            urls.append(value.strip())
    return urls


def _public_url(parts: urllib.parse.SplitResult) -> str:
    """scheme://host/path with the query dropped. With userinfo present, nothing after the scheme
    is printed: a password containing '@' or '/' cannot be split from the host or path reliably."""
    if parts.username is not None or parts.password is not None:
        return f"{parts.scheme}://<credentials removed>"
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def _submodule_finding(sev: Severity, reason: str, shown: str, key: str) -> Finding:
    return _finding(
        "HY008",
        sev,
        f"Submodule URL {reason}",
        f"url {sanitize(shown, 120)}",
        "Use an https or ssh URL without credentials.",
        key,
        path=".gitmodules",
    )


def _submodule_findings(git: GitRepo, entries: list[TreeEntry]) -> list[Finding]:
    gm = [e for e in entries if e.kind == "blob" and e.path == ".gitmodules"]
    if not gm:
        return []
    out: list[Finding] = []
    for _sha, data in _read_text_blobs(git, [gm[0].sha], 64 * 1024):
        for index, url in enumerate(_submodule_urls(data.decode("utf-8", "replace"))):
            try:
                parts = urllib.parse.urlsplit(url)
            except ValueError:
                out.append(
                    _submodule_finding(
                        Severity.HIGH,
                        "could not be parsed; it may embed credentials",
                        "(unparseable URL)",
                        f"unparseable#{index}",
                    )
                )
                continue
            if (
                parts.username is not None
                or parts.password is not None
                or _SECRET_QUERY.search(parts.query)
            ):
                reason, sev = "embeds credentials", Severity.HIGH
            elif parts.scheme in ("git", "http", "ftp"):
                reason, sev = "uses an unencrypted transport", Severity.MEDIUM
            else:
                continue
            shown = _public_url(parts)
            out.append(_submodule_finding(sev, reason, shown, f"{shown}#{index}"))
    return out


def _line_ending_findings(
    git: GitRepo, blobs: list[TreeEntry], settings: Settings
) -> list[Finding]:
    candidates = [
        e
        for e in blobs
        if e.size is not None
        and 0 < e.size <= settings.max_scan_size
        and not ignorable_path(e.path)
    ][:MAX_TEXT_SCAN_FILES]
    by_sha = {e.sha: e.path for e in candidates}
    out: list[Finding] = []
    for sha, data, _size in git.blob_contents(list(by_sha), settings.max_scan_size):
        if b"\0" in data[:8000]:
            continue
        crlf = data.count(b"\r\n")
        lf_only = data.count(b"\n") - crlf
        if crlf and lf_only:
            path = by_sha[sha]
            out.append(
                _finding(
                    "HY006", Severity.LOW,
                    f"Mixed line endings in {sanitize(path, 160)}",
                    f"{crlf} CRLF and {lf_only} LF-only line endings",
                    "Normalise with a .gitattributes text=auto rule and renormalise the file.",
                    sha, path=path,
                )
            )  # fmt: skip
    return out


def commit_findings(
    commit: dict[str, object], settings: Settings, sizes: dict[str, int]
) -> list[Finding]:
    """Per-commit checks: HY005 (huge commit). Identity checks are summarised separately."""
    changes = commit["changes"]
    assert isinstance(changes, list)
    total = 0
    unknown = False
    for _status, _path, _old, new in changes:
        if new in sizes:
            total += sizes[new]
        else:
            unknown = True
    sha = str(commit["sha"])
    too_many = len(changes) > settings.huge_commit_files
    too_big = total > settings.huge_commit_bytes
    if not (too_many or too_big):
        return []
    evidence = f"{len(changes)} files"
    if not unknown or total:
        evidence += f", {total / 1024**2:.1f} MiB known"
    return [
        _finding(
            "HY005",
            Severity.LOW,
            f"Huge commit {sha[:12]}",
            evidence + f"; subject: {sanitize(commit.get('subject', ''), 80)}",
            "Split large imports into reviewable commits; consider LFS for big binaries.",
            sha,
            commit=sha,
        )
    ]


def identity_findings(identities: Counter[str], settings: Settings) -> list[Finding]:
    """HY012 per suspicious author email, most frequent first (opt-in: --check-identities)."""
    out: list[Finding] = []
    for email, count in identities.most_common():
        domain = email.rsplit("@", 1)[-1].lower() if "@" in email else ""
        suspicious = (
            not email
            or "@" not in email
            or domain in PLACEHOLDER_DOMAINS
            or domain.endswith(tuple("." + d for d in PLACEHOLDER_DOMAINS))
        )
        if not suspicious:
            continue
        out.append(
            _finding(
                "HY012", Severity.LOW,
                f"Suspicious author identity: {sanitize(email or '(empty)', 80)}",
                f"{count} commits use this address",
                "Set user.email to a real address and rewrite the author with git mailmap or filter-repo.",
                email or "(empty)",
            )
        )  # fmt: skip
    return out


def signature_findings(rows: list[tuple[str, str]]) -> list[Finding]:
    """HY011 for recent HEAD commits without a good signature (%G? codes G and U are good)."""
    bad = [(sha, code) for sha, code in rows if code not in ("G", "U")]
    if not bad:
        return []
    return [
        _finding(
            "HY011",
            Severity.LOW,
            f"{len(bad)} of the last {len(rows)} commits on HEAD are not verified",
            "signature codes: " + ", ".join(sorted({c or "?" for _s, c in bad})),
            "Sign commits (git config commit.gpgsign true) or accept that they are unverified.",
            "signatures",
        )
    ]
