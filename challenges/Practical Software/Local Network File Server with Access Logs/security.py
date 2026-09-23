"""Framework-agnostic security primitives: path-traversal-safe joining and
constant-time shared-secret verification.

Kept separate from server.py so both are unit-testable without spinning up
FastAPI/TestClient, and so the traversal-defense logic -- the one thing that
absolutely has to be right in a "serve a folder to the network" tool -- has
exactly one place to get right instead of being smeared across route
handlers.
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import unquote


class PathTraversalError(ValueError):
    """Raised when a requested path would resolve outside the served root."""


def resolve_safe_path(root: Path, rel_path: str) -> Path:
    """Resolve `rel_path` against `root`, guaranteeing containment in root.

    Defends against:
      - ``..`` segments, including percent-encoded (``%2e%2e``) and
        double-percent-encoded (``%252e%252e``) variants -- ASGI servers
        decode a URL path once before routing, so a single extra
        `unquote` here closes the double-encoding bypass a naive
        "just check the raw string" approach would miss.
      - absolute-path injection, both POSIX (``/etc/passwd``) and Windows
        (``C:\\Windows``) -- `Path.joinpath` silently *discards* everything
        before an absolute argument, so an unchecked join is itself a
        traversal bug; every segment is checked before joining.
      - null bytes.
      - symlinks that point outside the root: `.resolve()` follows
        symlinks, and the containment check runs *after* that resolution,
        so a symlink inside root pointing elsewhere is still caught.
    """
    if "\x00" in rel_path:
        raise PathTraversalError("null byte in path")

    root_resolved = root.resolve(strict=False)

    # One decode undoes normal percent-encoding; a second undoes
    # double-encoding attacks. Already-decoded input (the common case,
    # since ASGI servers decode the path once before we ever see it) is
    # unaffected -- `unquote` is a no-op on a string with no `%xx` escapes.
    decoded = unquote(unquote(rel_path))

    if "\\" in decoded:
        raise PathTraversalError("backslash not allowed in path")

    segments = [s for s in decoded.split("/") if s not in ("", ".")]
    for seg in segments:
        if seg == "..":
            raise PathTraversalError(f"traversal segment: {seg!r}")
        if PurePosixPath(seg).is_absolute() or PureWindowsPath(seg).is_absolute():
            raise PathTraversalError(f"absolute segment: {seg!r}")
        if len(seg) == 2 and seg[1] == ":" and seg[0].isalpha():
            raise PathTraversalError(f"drive-letter segment: {seg!r}")

    candidate = root_resolved.joinpath(*segments) if segments else root_resolved
    resolved = candidate.resolve(strict=False)

    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise PathTraversalError(f"escapes root: {rel_path!r} -> {resolved}")

    return resolved


def hash_token(token: str) -> bytes:
    """Digest a shared secret so it's never held or compared in plaintext."""
    return hashlib.sha256(token.encode("utf-8")).digest()


def verify_token(candidate: str, token_hash: bytes) -> bool:
    """Constant-time check that `candidate` hashes to `token_hash`."""
    candidate_hash = hashlib.sha256(candidate.encode("utf-8")).digest()
    return hmac.compare_digest(candidate_hash, token_hash)
