"""Hygiene for hostile repository content: escaping, size parsing, redaction, fingerprints.

Everything a repository controls (branch names, paths, commit subjects, author names) passes
through `sanitize` before it reaches a terminal, and through the reporters' own escaping before
it reaches HTML. Secrets never leave this module in full: `redact` keeps a short prefix/suffix.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import unicodedata
from pathlib import Path

_ESCAPED_CATEGORIES = frozenset(
    {"Cc", "Cf", "Zl", "Zp"}
)  # controls, format chars (ANSI ESC, bidi overrides), line and paragraph separators
_SIZE_RE = re.compile(r"(\d{1,15})\s*([A-Za-z]{0,3})")
_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "kb": 1000,
    "kib": 1024,
    "mb": 1000**2,
    "mib": 1024**2,
    "gb": 1000**3,
    "gib": 1024**3,
}
_REF_RE = re.compile(r"[A-Za-z0-9._/@+-]{1,250}")
MAX_OUTPUT_FILE_BYTES = 64 * 1024 * 1024


def sanitize(text: object, limit: int | None = None) -> str:
    """Make untrusted text safe to print: control and bidi characters become visible escapes."""
    raw = str(text)
    if limit is not None and len(raw) > limit * 4:
        raw = raw[: limit * 4]
    out: list[str] = []
    for ch in raw:
        if ch == "\t":
            out.append(" ")
        elif unicodedata.category(ch) in _ESCAPED_CATEGORIES:
            code = ord(ch)
            out.append(f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}")
        else:
            out.append(ch)
    result = "".join(out)
    if limit is not None and len(result) > limit:
        result = result[: max(limit - 1, 0)] + "…"
    return result


def parse_size(value: object) -> int:
    """Parse a byte count such as 5242880, "5MiB", "512KB". KB/MB/GB are decimal; KiB/MiB/GiB binary."""
    if isinstance(value, bool):
        raise ValueError("size must be a number or a string such as '5MiB'")  # noqa: TRY004
    if isinstance(value, int):
        if value < 0:
            raise ValueError("size must not be negative")
        return value
    if not isinstance(value, str) or len(value) > 32:
        raise ValueError(f"bad size: {sanitize(value, 40)!r}")
    match = _SIZE_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"bad size: {sanitize(value, 40)!r}")
    unit = match.group(2).lower()
    if unit not in _SIZE_UNITS:
        raise ValueError(f"unknown size unit in {sanitize(value, 40)!r}")
    return int(match.group(1)) * _SIZE_UNITS[unit]


def is_valid_ref_name(name: str) -> bool:
    """Accept only conservative ref names; a leading '-' would be read as an option by git."""
    return (
        bool(_REF_RE.fullmatch(name)) and not name.startswith("-") and ".." not in name
    )


_CLI_ARG_RE = re.compile(r"[A-Za-z0-9._/+@-]{1,200}")


def cli_arg(value: str, placeholder: str) -> str:
    """Return `value` for a copy-paste command only if it is inert in both POSIX shells and
    PowerShell/cmd: a conservative charset and no leading '-'. Otherwise return `placeholder`.
    The real name is still shown in the finding's evidence, so the operator can substitute it."""
    if (
        _CLI_ARG_RE.fullmatch(value)
        and not value.startswith(("-", "@"))
        and ".." not in value
    ):
        return value
    return placeholder


def redact(secret: str) -> str:
    """Short, non-reversible shape of a secret: a 4-character prefix, 2 suffix characters, length.
    Values under 16 characters are masked entirely, since a short prefix would reveal most of them."""
    length = len(secret)
    if length < 16:
        return f"{'*' * length} (len {length})"
    return f"{secret[:4]}***{secret[-2:]} (len {length})"


def fingerprint(*parts: str) -> str:
    """Unsalted SHA-256 over joined parts. Parts must never contain the secret itself."""
    return hashlib.sha256(
        "\x1f".join(parts).encode("utf-8", "surrogatepass")
    ).hexdigest()


def check_output_path(path: Path, forbidden_root: Path | None = None) -> Path:
    """Validate an output destination: no symlinks, no directories, parent must exist, not inside .git."""
    target = Path(path)
    if target.is_symlink():
        raise ValueError(
            f"refusing to write through a symlink: {sanitize(target, 200)}"
        )
    if target.is_dir():
        raise ValueError(f"output path is a directory: {sanitize(target, 200)}")
    parent = target.parent.resolve()
    if not parent.is_dir():
        raise ValueError(f"parent directory does not exist: {sanitize(parent, 200)}")
    resolved = parent / target.name
    if forbidden_root is not None:
        root = forbidden_root.resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError(
                "refusing to write inside the audited repository's .git directory"
            )
    return resolved


def atomic_write(path: Path, data: str) -> None:
    """Write text via a temp file and rename, so a crash never leaves a half-written report."""
    if len(data.encode("utf-8")) > MAX_OUTPUT_FILE_BYTES:
        raise ValueError("refusing to write more than 64 MiB")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
