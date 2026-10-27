"""Orchestrates the four checks over one repository and returns an `AuditResult`.

Passes over git, in order (each is streamed where the output can be large):
  1. branches and the audited tree (ls-tree, for-each-ref)
  2. blob sizes for every reachable blob (rev-list + cat-file --batch-check)
  3. history commits: which blob and path each change introduced, identities, huge commits
  4. blob contents for every candidate under the scan limit (cat-file --batch), for secrets
Memory is O(distinct blobs) for the size and location maps. Blob contents are never held longer
than one object at a time.

Budgets (--time-budget, --scan-budget) and secret-hit caps stop a pass early. The result is then
marked partial, and an HY013 finding says which limit was reached. Nothing is silently dropped.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from gitaudit import blobs as blob_checks
from gitaudit import hygiene
from gitaudit.baseline import apply_baseline, load_baseline
from gitaudit.branches import check_branches, detect_default_branch
from gitaudit.config import Settings
from gitaudit.findings import CATALOG, Finding, Severity, make_fingerprint
from gitaudit.git import GitError, GitRepo
from gitaudit.safety import sanitize
from gitaudit.secrets import (
    MAX_HITS_PER_BLOB,
    Allowlist,
    content_finding,
    path_findings,
    scan_text,
)

NULL_SHA = "0" * 40
MAX_TOTAL_SECRET_HITS = 5000


@dataclass
class AuditResult:
    repo: str
    ref: str
    commit: str
    default_branch: str | None
    findings: list[Finding]
    stats: dict[str, int | float | str | None] = field(default_factory=dict)
    config_source: str | None = None
    partial: bool = False  # a budget or cap stopped a pass; see the HY013 finding

    def active(self) -> list[Finding]:
        return [f for f in self.findings if not f.suppressed]

    def failing(self, threshold: Severity | None) -> list[Finding]:
        if threshold is None:
            return []
        return [f for f in self.active() if f.severity >= threshold]


def _sort_key(f: Finding) -> tuple[int, str, str, str]:
    return (-int(f.severity), f.rule_id, f.path or f.ref or "", f.fingerprint)


def _partial_finding(reasons: list[str], allow_partial: bool) -> Finding:
    text = ", ".join(sorted(set(reasons)))
    return Finding(
        rule_id="HY013",
        # HIGH fails the run unless the operator opted out with --allow-partial.
        severity=Severity.INFO if allow_partial else CATALOG["HY013"].severity,
        confidence="high",
        title=f"Audit results are partial: {text} reached",
        category="hygiene",
        evidence="work after the stop point was not done; the exit code reflects only what ran",
        remediation="Raise --time-budget or --scan-budget, or narrow the audit with --since or --no-history.",
        fingerprint=make_fingerprint("HY013", text),
    )


def _skipped_finding(binary: int, other: int) -> Finding:
    return Finding(
        rule_id="HY014",
        severity=CATALOG["HY014"].severity,
        confidence="high",
        title="Blobs were not content-scanned for secrets",
        category="hygiene",
        evidence=(
            f"{binary} binary, {other} over the size limit, unread after a budget stop, "
            "or matched by --allow-path"
        ),
        remediation="Raise --max-scan-size to scan larger text blobs. Binary blobs are not scanned.",
        fingerprint=make_fingerprint("HY014", "skipped"),
    )


def run_audit(settings: Settings) -> AuditResult:
    started = time.perf_counter()
    deadline = started + settings.time_budget
    partial: list[str] = []
    git = GitRepo(
        settings.repo,
        trust_owner=settings.trust_owner,
        safe_directory=settings.safe_directory,
    )
    commit = git.resolve_commit(settings.ref)
    refs = git.for_each_ref()
    head_entries = git.ls_tree(commit)
    head_blobs = {e.sha: e for e in head_entries if e.kind == "blob"}
    default = detect_default_branch(git, refs, settings.default_branch)

    findings: list[Finding] = []
    findings += check_branches(git, refs, default, settings)
    findings += hygiene.tree_findings(git, head_entries, settings)

    # Pass 2: sizes of every reachable blob.
    sizes: dict[str, int] = {
        e.sha: e.size for e in head_blobs.values() if e.size is not None
    }
    if settings.history:
        missing = [sha for sha in git.object_blobs(settings.since) if sha not in sizes]
        for sha, kind, size in git.batch_check(missing):
            if kind == "blob":
                sizes[sha] = size

    # Pass 3: where each blob and path first appeared, plus commit-level checks.
    where: dict[str, tuple[str | None, str | None]] = {}
    paths: list[
        tuple[str, str | None, str | None]
    ] = []  # (path, blob, commit), deduplicated below
    seen_paths: set[tuple[str, str | None]] = set()
    identities: Counter[str] = Counter()
    commit_count = 0
    if settings.history:
        for entry in git.commits(since=settings.since):
            if time.perf_counter() > deadline:
                partial.append("time budget")
                break
            commit_count += 1
            sha = str(entry["sha"])
            changes = entry["changes"]
            assert isinstance(changes, list)
            email = str(entry["email"]).lower()
            if settings.check_identities:
                identities[email] += 1
            findings += hygiene.commit_findings(entry, settings, sizes)
            for status, path, _old, new in changes:
                blob = new if new != NULL_SHA else None
                if (path, blob) not in seen_paths:
                    seen_paths.add((path, blob))
                    paths.append((path, blob, sha))
                if blob is not None:
                    where.setdefault(blob, (path, sha))
    # Every tree entry, not one per blob: a blob under two head paths must keep both names.
    for entry in head_entries:
        if entry.kind != "blob":
            continue
        # The audited tree's name for a blob is the one to report.
        _, commit_sha = where.get(entry.sha, (None, None))
        where[entry.sha] = (entry.path, commit_sha)
        if (entry.path, entry.sha) not in seen_paths:
            seen_paths.add((entry.path, entry.sha))
            paths.append((entry.path, entry.sha, None))

    allow = Allowlist(
        settings.allow_paths,
        settings.allow_regexes,
        settings.stopwords,
        honor_inline=settings.honor_inline_allow,
    )
    for path, blob, commit_sha in paths:
        findings += path_findings(path, blob or "", commit_sha, allow)
    names: dict[
        str, set[str]
    ] = {}  # every path a blob appears under, not only the first
    for path, blob, _commit in paths:
        if blob is not None:
            names.setdefault(blob, set()).add(path)

    # Pass 4: contents of every candidate under the scan limit.
    binary_shas: set[str] = set()
    candidates = [
        sha for sha in where if sha in sizes and sizes[sha] <= settings.max_scan_size
    ]
    scanned = 0  # blobs run through the secret rules
    scanned_bytes = 0
    secret_hits = 0
    for sha, data, _size in git.blob_contents(candidates, settings.max_scan_size):
        if time.perf_counter() > deadline:
            partial.append("time budget")
            break
        if scanned_bytes + len(data) > settings.scan_budget_bytes:
            partial.append("scan budget")
            break
        scanned_bytes += len(data)
        if b"\0" in data[:8000]:
            binary_shas.add(sha)
            continue
        path, commit_sha = where.get(sha, (None, None))
        blob_names = names.get(sha, set())
        if blob_names and all(allow.path_allowed(p) for p in blob_names):
            continue  # --allow-path covers content only when every path the blob has is allowed
        scanned += 1
        hits = scan_text(data, allow)
        if len(hits) >= MAX_HITS_PER_BLOB:
            partial.append("per-blob secret hit cap")
        stop = False
        for hit in hits:
            if secret_hits >= MAX_TOTAL_SECRET_HITS:
                partial.append("total secret hit cap")
                stop = True
                break
            findings.append(
                content_finding(
                    hit.rule_id,
                    path or "(path unknown)",
                    commit_sha,
                    sha,
                    hit.line,
                    hit.shape,
                    hit.confidence,
                    hit.occurrence,
                )
            )
            secret_hits += 1
        if stop:
            break

    # Blob-size and object-store checks.
    infos = [
        blob_checks.BlobInfo(
            sha=sha,
            size=size,
            path=where.get(sha, (None, None))[0],
            commit=where.get(sha, (None, None))[1],
            in_head=sha in head_blobs,
        )
        for sha, size in sizes.items()
        if size > settings.max_blob_size
    ]
    findings += blob_checks.oversized_findings(infos, settings)
    head_infos = [
        blob_checks.BlobInfo(e.sha, e.size or 0, e.path, None, True)
        for e in head_blobs.values()
        if e.size is not None and e.size >= settings.lfs_threshold
    ]
    findings += blob_checks.lfs_findings(head_infos, settings, binary_shas)
    findings += blob_checks.gc_findings(git.count_objects())

    other_unscanned = len(where) - scanned - len(binary_shas)
    if binary_shas or other_unscanned > 0:
        findings.append(_skipped_finding(len(binary_shas), other_unscanned))

    if settings.check_identities:
        findings += hygiene.identity_findings(identities, settings)
    if settings.check_signatures:
        try:
            findings += hygiene.signature_findings(
                git.signature_status(settings.signature_limit)
            )
        except GitError:
            pass  # no commits on HEAD: nothing to verify
    if partial:
        findings.append(_partial_finding(partial, settings.allow_partial))

    if settings.baseline:
        accepted = load_baseline(Path(settings.baseline))
        findings = apply_baseline(findings, accepted)

    findings.sort(key=_sort_key)
    stats: dict[str, int | float | str | None] = {
        "commits": commit_count,
        "distinct_blobs": len(sizes),
        "blobs_content_scanned": scanned,
        "binary_blobs": len(binary_shas),
        "secret_hits": secret_hits,
        "history": "full"
        if settings.history and not settings.since
        else (settings.since or "off"),
        "seconds": round(time.perf_counter() - started, 3),
        "default_branch": default.name,
    }
    return AuditResult(
        repo=sanitize(str(git.path), 300),
        ref=sanitize(settings.ref, 120),
        commit=commit,
        default_branch=default.name,
        findings=findings,
        stats=stats,
        config_source=settings.config_source,
        partial=bool(partial),
    )
