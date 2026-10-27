"""Check 1: stale and abandoned branches, local and remote-tracking, plus odd tags."""

from __future__ import annotations

import time
from dataclasses import dataclass

from gitaudit.config import Settings
from gitaudit.findings import Finding, Severity, make_fingerprint
from gitaudit.git import GitError, GitRepo
from gitaudit.safety import cli_arg, is_valid_ref_name, sanitize

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class DefaultBranch:
    name: str | None
    sha: str | None


def detect_default_branch(
    git: GitRepo, refs: list[dict[str, str]], configured: str | None
) -> DefaultBranch:
    """Configured name, then origin/HEAD, then local HEAD, then main/master."""
    names: list[str] = []
    if configured:
        if not is_valid_ref_name(configured):
            raise ValueError(
                f"default_branch is not a valid branch name: {sanitize(configured, 60)!r}"
            )
        names.append(configured)
    else:
        for row in refs:
            if (
                row["name"].endswith("/HEAD")
                and row["name"].startswith("refs/remotes/")
                and row["symref"]
            ):
                names.append(row["symref"].split("/", 3)[-1])
        try:
            head = (
                git.run(["symbolic-ref", "-q", "HEAD"])
                .decode("utf-8", "backslashreplace")
                .strip()
            )
            if head.startswith("refs/heads/"):
                names.append(head[len("refs/heads/") :])
        except GitError:
            pass  # detached HEAD
        names += ["main", "master"]

    tips: dict[str, str] = {}
    for row in refs:
        tips.setdefault(row["name"], row["sha"])
    for name in names:
        if not is_valid_ref_name(name):
            continue
        for candidate in (
            f"refs/heads/{name}",
            *(f"refs/remotes/{r}/{name}" for r in _remotes(refs)),
        ):
            if candidate in tips:
                return DefaultBranch(name, tips[candidate])
    return DefaultBranch(None, None)


def _remotes(refs: list[dict[str, str]]) -> list[str]:
    seen: list[str] = []
    for row in refs:
        parts = row["name"].split("/", 3)
        if (
            row["name"].startswith("refs/remotes/")
            and len(parts) >= 4
            and parts[2] not in seen
        ):
            seen.append(parts[2])
    return seen


def _age_days(now: int, when: int) -> int:
    return max(0, (now - when) // SECONDS_PER_DAY)


def check_branches(
    git: GitRepo,
    refs: list[dict[str, str]],
    default: DefaultBranch,
    settings: Settings,
    now: int | None = None,
) -> list[Finding]:
    now = int(time.time()) if now is None else now
    findings: list[Finding] = []
    if default.name is None or default.sha is None:
        findings.append(
            Finding(
                rule_id="BR006",
                severity=Severity.INFO,
                confidence="high",
                title="Default branch not detected",
                category="branches",
                evidence="Pass --default-branch NAME to enable merged/unmerged classification.",
                remediation="Set default_branch in gitaudit.toml.",
                fingerprint=make_fingerprint("BR006", "default"),
            )
        )
        merged: set[str] = set()
    else:
        merged = git.merged_into(default.sha)

    tags = [r for r in refs if r["name"].startswith("refs/tags/")]
    for tag in tags:
        if tag["type"] not in ("commit", "tag"):
            name = tag["name"].removeprefix("refs/tags/")
            findings.append(
                Finding(
                    rule_id="BR005",
                    severity=Severity.LOW,
                    confidence="high",
                    title=f"Tag {sanitize(name, 80)} points at a {tag['type']}",
                    category="branches",
                    evidence=f"object {tag['sha'][:12]} is a {tag['type']}",
                    remediation="Re-create the tag on a commit, or delete it if it is stale.",
                    fingerprint=make_fingerprint("BR005", tag["name"]),
                    ref=tag["name"],
                    blob=tag["sha"] if tag["type"] == "blob" else None,
                )
            )

    remotes = _remotes(refs)
    default_refs = (
        {f"refs/heads/{default.name}"}
        | {f"refs/remotes/{r}/{default.name}" for r in remotes}
        if default.name
        else set()
    )
    branches = [
        r for r in refs
        if (r["name"].startswith("refs/heads/") or r["name"].startswith("refs/remotes/"))
        and not r["symref"]
        and r["name"] not in default_refs
    ]  # fmt: skip
    branches.sort(key=lambda r: int(r["date"] or 0), reverse=True)

    for index, row in enumerate(branches):
        findings += _classify(
            git,
            row,
            default,
            merged,
            settings,
            now,
            analyse=index < settings.max_branches,
        )
    return findings


def _classify(
    git: GitRepo,
    row: dict[str, str],
    default: DefaultBranch,
    merged: set[str],
    settings: Settings,
    now: int,
    analyse: bool,
) -> list[Finding]:
    ref = row["name"]
    short = ref.removeprefix("refs/heads/").removeprefix("refs/remotes/")
    tip = row["sha"]
    last_time = int(row["date"] or 0)
    age = _age_days(now, last_time)
    author = sanitize(row["author"], 60)
    is_merged = tip in merged
    out: list[Finding] = []
    ahead: int | None = None
    behind: int | None = None
    fork_age: int | None = None
    if analyse and default.sha and not is_merged:
        ahead, behind = git.ahead_behind(default.sha, tip)
        base = git.merge_base(default.sha, tip)
        if base:
            fork_age = _age_days(now, git.commit_time(base))
    detail = (
        f"tip {tip[:12]}, last commit {age} days ago by {author}"
        + (f", ahead {ahead} behind {behind}" if ahead is not None else "")
        + (f", forked {fork_age} days ago" if fork_age is not None else "")
    )
    gone = row["track"] == "[gone]"

    def add(rule: str, severity: Severity, title: str, remediation: str) -> None:
        out.append(
            Finding(
                rule_id=rule,
                severity=severity,
                confidence="high",
                title=title,
                category="branches",
                evidence=detail,
                remediation=remediation,
                fingerprint=make_fingerprint(rule, ref),
                ref=ref,
                commit=tip,
                extra={"age_days": age, "ahead": ahead, "behind": behind},
            )
        )

    if default.name is None or default.sha is None:
        return out
    if is_merged:
        add(
            "BR001",
            Severity.LOW,
            f"Branch {sanitize(short, 80)} is merged",
            "Delete it once nothing needs it: git branch -d "
            + cli_arg(short, "<branch>"),
        )
        return out
    if gone and ref.startswith("refs/heads/"):
        add(
            "BR003",
            Severity.MEDIUM,
            f"Upstream of {sanitize(short, 80)} is gone",
            "Review, then delete the local branch: git branch -D "
            + cli_arg(short, "<branch>"),
        )
    if age > settings.stale_days:
        add(
            "BR002",
            Severity.MEDIUM,
            f"Branch {sanitize(short, 80)} is stale and unmerged",
            "Merge it, rebase and open a PR, or delete it if abandoned.",
        )
    elif fork_age is not None and fork_age > settings.long_lived_days:
        add(
            "BR004",
            Severity.LOW,
            f"Branch {sanitize(short, 80)} is long-lived and unmerged",
            "Split it into smaller merges so the default branch does not drift.",
        )
    return out
