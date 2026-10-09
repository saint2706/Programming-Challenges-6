"""Turn ``prod=deploy@host:/srv/app/.env`` style arguments into loaded, fingerprinted sources."""

import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from envdiff.fingerprint import Hasher
from envdiff.model import Source, build_source
from envdiff.parse import Parsed, parse
from envdiff.snapshot import SnapshotError, looks_like_snapshot, read_snapshot

MAX_BYTES = 5 * 1024 * 1024
SSH_TIMEOUT = 30
_LABEL = re.compile(r"^([A-Za-z][\w.-]*)=(.+)$", re.DOTALL)
_SSH = re.compile(
    r"^(?:(?P<user>[\w.-]+)@)?(?P<host>[\w.-]{2,}|\[[0-9A-Fa-f:]+\]):(?P<path>.+)$",
    re.DOTALL,
)

Runner = Callable[..., subprocess.CompletedProcess]


class SourceError(Exception):
    """A source could not be read. The message never contains file content."""


@dataclass(frozen=True)
class SourceSpec:
    kind: str  # file | ssh | stdin | env
    origin: str
    label: str | None = None
    path: str = ""
    destination: str = ""  # ssh: [user@]host

    @property
    def default_label(self) -> str:
        if self.kind == "file":
            return Path(self.path).name or self.path
        if self.kind == "ssh":
            return self.destination.split("@")[-1]
        return {"stdin": "stdin", "env": "env"}[self.kind]


def parse_spec(arg: str, exists: Callable[[str], bool] = os.path.exists) -> SourceSpec:
    """Classify one command-line source. A real local file always wins over a ``host:path`` reading."""
    label = None
    rest = arg
    labelled = _LABEL.match(arg)
    if labelled and not exists(arg):
        label, rest = labelled.group(1), labelled.group(2)
    if rest == "-":
        return SourceSpec("stdin", "<stdin>", label)
    if rest == "env:":
        return SourceSpec("env", "<environment>", label)
    ssh = _SSH.match(rest)
    if ssh and not exists(rest) and not ssh.group("host").startswith("-"):
        user = ssh.group("user")
        destination = f"{user}@{ssh.group('host')}" if user else ssh.group("host")
        return SourceSpec("ssh", rest, label, ssh.group("path"), destination)
    return SourceSpec("file", rest, label, rest)


def fetch_ssh(spec: SourceSpec, run: Runner = subprocess.run) -> str:
    """``cat`` a remote file over the system ``ssh``; the content stays in this process's memory.

    ``BatchMode`` stops ssh prompting (a prompt would hang a script), ``--`` and the leading-dash
    check in ``parse_spec`` stop a hostile host name being read as an ssh option, and the path is
    shell-quoted because the remote side runs it through a shell.
    """
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--",
        spec.destination, f"cat -- {shlex.quote(spec.path)}",
    ]  # fmt: skip
    try:
        result = run(command, capture_output=True, timeout=SSH_TIMEOUT, check=False)
    except FileNotFoundError as exc:
        raise SourceError("ssh is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise SourceError(f"{spec.origin}: timed out after {SSH_TIMEOUT}s") from exc
    if result.returncode != 0:
        reason = result.stderr.decode(errors="replace").strip().splitlines()
        detail = reason[-1][:200] if reason else f"exit status {result.returncode}"
        raise SourceError(f"{spec.origin}: {detail}")
    return _decode(result.stdout, spec.origin)


def _decode(data: bytes, origin: str) -> str:
    if len(data) > MAX_BYTES:
        raise SourceError(
            f"{origin}: larger than {MAX_BYTES // 1024 // 1024} MiB, not an env file"
        )
    if b"\0" in data[:8192]:
        raise SourceError(f"{origin}: looks like a binary file")
    return data.decode("utf-8", errors="replace")


def read_text(spec: SourceSpec, run: Runner = subprocess.run, stdin=None) -> str:
    if spec.kind == "ssh":
        return fetch_ssh(spec, run)
    if spec.kind == "stdin":
        stream = stdin if stdin is not None else sys.stdin.buffer
        return _decode(stream.read(MAX_BYTES + 1), spec.origin)
    try:
        with open(spec.path, "rb") as handle:
            return _decode(handle.read(MAX_BYTES + 1), spec.origin)
    except OSError as exc:
        raise SourceError(f"{spec.origin}: {exc.strerror or 'cannot be read'}") from exc


def _unique_labels(labels: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label}#{seen[label]}")
    return out


def load_sources(
    specs: list[SourceSpec],
    fmt: str = "auto",
    passphrase: str | None = None,
    reveal: frozenset[str] = frozenset(),
    run: Runner = subprocess.run,
    environ: Mapping[str, str] | None = None,
    stdin=None,
) -> list[Source]:
    """Read every source, then fingerprint under one key so equal values get equal fingerprints."""
    raw: list[tuple[SourceSpec, str | None]] = []
    for spec in specs:
        raw.append((spec, None if spec.kind == "env" else read_text(spec, run, stdin)))
    has_snapshot = any(
        text is not None and looks_like_snapshot(text) for _, text in raw
    )
    has_live = any(text is None or not looks_like_snapshot(text) for _, text in raw)
    if has_snapshot and has_live and not passphrase:
        raise SourceError(
            "comparing a snapshot with live files needs the passphrase it was made with: set ENVDIFF_PASSPHRASE"
        )
    hasher = Hasher.from_passphrase(passphrase) if passphrase else Hasher.random()

    sources: list[Source] = []
    for spec, text in raw:
        if text is not None and looks_like_snapshot(text):
            try:
                source = read_snapshot(text, spec.origin)
            except SnapshotError as exc:
                raise SourceError(str(exc)) from exc
            if spec.label:
                source.label = spec.label
            sources.append(source)
            continue
        if text is None:
            parsed = Parsed(values=dict(environ if environ is not None else os.environ))
            chosen = "environment"
        else:
            parsed, chosen = parse(text, fmt, PurePosixPath(spec.path).name)
        label = spec.label or spec.default_label
        sources.append(build_source(label, spec.origin, chosen, parsed, hasher, reveal))
    _check_keys(sources, hasher if has_live else None)
    for source, label in zip(
        sources, _unique_labels([s.label for s in sources]), strict=True
    ):
        source.label = label
    return sources


def _check_keys(sources: list[Source], hasher: Hasher | None) -> None:
    snapshots = [s for s in sources if s.fmt == "snapshot"]
    expected = hasher.key_id if hasher else (snapshots[0].key_id if snapshots else None)
    for snapshot in snapshots:
        if snapshot.key_id != expected:
            raise SourceError(
                f"{snapshot.origin}: made with a different passphrase than the other sources, "
                "so its fingerprints cannot be compared"
            )
