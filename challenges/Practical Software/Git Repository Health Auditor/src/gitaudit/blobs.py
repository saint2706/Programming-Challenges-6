"""Check 2: oversized blobs (current tree and history-only), LFS candidates, gc advice."""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass

from gitaudit.config import Settings
from gitaudit.findings import Finding, Severity, make_fingerprint
from gitaudit.safety import cli_arg, sanitize

_LFS_EXT_RE = re.compile(r"\.[A-Za-z0-9_-]{1,15}")
BINARY_SNIFF_BYTES = 8000
ARCHIVE_OR_BINARY_SUFFIXES = frozenset(
    {
        ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".war", ".ear",
        ".exe", ".dll", ".so", ".dylib", ".bin", ".iso", ".dmg", ".msi", ".apk", ".aab",
        ".deb", ".rpm", ".whl", ".class", ".pyd", ".o", ".a", ".lib", ".mp4", ".mov", ".avi",
        ".mp3", ".wav", ".psd", ".sketch", ".ttf", ".otf", ".pdf", ".db", ".sqlite",
    }
)  # fmt: skip
# git's own heuristic: a NUL byte in the first 8000 bytes means binary.
GC_LOOSE_THRESHOLD = 6700  # git's gc.auto default for loose objects


@dataclass(frozen=True)
class BlobInfo:
    sha: str
    size: int
    path: str | None
    commit: str | None
    in_head: bool


def is_binary(content: bytes) -> bool:
    return b"\0" in content[:BINARY_SNIFF_BYTES]


def is_archive_or_binary_path(path: str) -> bool:
    return posixpath.splitext(path.lower())[1] in ARCHIVE_OR_BINARY_SUFFIXES


def _mib(size: int) -> str:
    return f"{size / 1024**2:.2f} MiB"


def remediation_for(path: str | None, *, in_head: bool) -> str:
    """Exact rewrite commands for the operator. Printed only; this tool never runs them."""
    # Repository-controlled names are interpolated only when inert in a shell; see cli_arg.
    target = path or ""
    lines = [
        "git filter-repo --invert-paths --path "
        + cli_arg(target, "<path>")
        + "   # rewrites every commit; force-push after",
        "bfg --delete-files "
        + cli_arg(posixpath.basename(target), "<filename>")
        + "   # BFG alternative",
    ]
    if not in_head:
        lines.append(
            "The blob is not in the current tree; deleting it from history alone reclaims space."
        )
    return " | ".join(lines)


def oversized_findings(blobs: list[BlobInfo], settings: Settings) -> list[Finding]:
    """One BL001/BL002 per oversized blob, largest first, capped at `settings.top`."""
    ranked = sorted(
        (b for b in blobs if b.size > settings.max_blob_size), key=lambda b: -b.size
    )
    out: list[Finding] = []
    for rank, blob in enumerate(ranked[: settings.top], start=1):
        rule = "BL001" if blob.in_head else "BL002"
        name = sanitize(blob.path or "(path unknown: merge-only or --no-history)", 200)
        where = "in the current tree" if blob.in_head else "history only"
        evidence = (
            f"rank {rank}: {_mib(blob.size)} at {name}; {where}; "
            f"introduced by {(blob.commit or 'unknown')[:12]}"
        )
        out.append(
            Finding(
                rule_id=rule,
                severity=Severity.HIGH
                if blob.in_head and blob.size > 10 * settings.max_blob_size
                else Severity.MEDIUM,
                confidence="high",
                title=f"Oversized blob ({_mib(blob.size)}) at {name}",
                category="blobs",
                evidence=evidence,
                remediation=remediation_for(blob.path, in_head=blob.in_head),
                fingerprint=make_fingerprint(rule, blob.path or "", blob.sha),
                path=blob.path,
                commit=blob.commit,
                blob=blob.sha,
                extra={"size": blob.size, "rank": rank},
            )
        )
    return out


def _lfs_hint(path: str | None) -> str:
    ext = posixpath.splitext(path or "")[1]
    # Only a plain extension is turned into a pattern; anything else becomes a placeholder.
    pattern = f"'*{ext}'" if _LFS_EXT_RE.fullmatch(ext) else "<pattern>"
    return (
        f"Track with Git LFS: git lfs migrate import --include={pattern} "
        "(rewrites history; coordinate first)"
    )


def lfs_findings(
    head_blobs: list[BlobInfo], settings: Settings, binary_shas: set[str]
) -> list[Finding]:
    """Binary or archive files in the current tree at or above the LFS threshold."""
    out: list[Finding] = []
    for blob in sorted(head_blobs, key=lambda b: -b.size):
        if blob.size < settings.lfs_threshold or blob.size > settings.max_blob_size:
            continue  # oversized ones are already BL001
        if not (
            blob.sha in binary_shas
            or (blob.path and is_archive_or_binary_path(blob.path))
        ):
            continue
        name = sanitize(blob.path or "", 200)
        out.append(
            Finding(
                rule_id="BL003",
                severity=Severity.LOW,
                confidence="medium",
                title=f"Binary or archive committed: {name}",
                category="blobs",
                evidence=f"{_mib(blob.size)} in the current tree",
                remediation=_lfs_hint(blob.path),
                fingerprint=make_fingerprint("BL003", blob.path or "", blob.sha),
                path=blob.path,
                blob=blob.sha,
                commit=blob.commit,
                extra={"size": blob.size},
            )
        )
    return out[: settings.top]


def gc_findings(stats: dict[str, int]) -> list[Finding]:
    """Advice only: loose objects past git's auto-gc threshold, or garbage that gc would delete."""
    reasons = []
    if stats.get("count", 0) > GC_LOOSE_THRESHOLD:
        reasons.append(
            f"{stats['count']} loose objects (gc.auto threshold is {GC_LOOSE_THRESHOLD})"
        )
    if stats.get("prune-packable", 0) or stats.get("garbage", 0):
        reasons.append(
            f"{stats.get('prune-packable', 0)} prune-packable and {stats.get('garbage', 0)} garbage objects"
        )
    if not reasons:
        return []
    return [
        Finding(
            rule_id="BL004",
            severity=Severity.LOW,
            confidence="high",
            title="Object store would benefit from git gc",
            category="blobs",
            evidence="; ".join(reasons)
            + f"; pack size {stats.get('size-pack', 0)} KiB",
            remediation="git gc --prune=now   (run on a clone you own; not run by gitaudit)",
            fingerprint=make_fingerprint("BL004", "object-store"),
            extra={"size_pack_kib": stats.get("size-pack", 0)},
        )
    ]
