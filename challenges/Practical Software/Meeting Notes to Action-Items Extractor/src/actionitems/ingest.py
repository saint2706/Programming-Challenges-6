"""Read untrusted input: files, directories or stdin, with size limits and safe decoding."""

import contextlib
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from actionitems.errors import InputError
from actionitems.text import strip_controls

MAX_INPUT_BYTES = 2_000_000
MAX_LINES = 20_000
MAX_FILES = 500  # per invocation, across all directory and file inputs
MAX_TOTAL_BYTES = 20_000_000  # aggregate size of every file read in one invocation
TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".text", ".vtt", ".srt"})


@dataclass(frozen=True)
class Document:
    name: str
    text: str
    mtime: datetime | None = None  # file modification time, used for the meeting date


def decode_bytes(data: bytes) -> str:
    """Decode a transcript: UTF-8 (BOM tolerated), then cp1252, then latin-1 as a last resort."""
    if len(data) > MAX_INPUT_BYTES:
        raise InputError(f"input is larger than {MAX_INPUT_BYTES} bytes")
    if b"\x00" in data:
        raise InputError(
            "input looks binary (contains NUL bytes); export it as text first"
        )
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("latin-1")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_controls(text)
    if text.count("\n") > MAX_LINES:
        raise InputError(f"input has more than {MAX_LINES} lines")
    return text


def _read_file(path: Path) -> Document:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc.strerror or exc}") from exc
    if size > MAX_INPUT_BYTES:
        raise InputError(f"{path.name} is larger than {MAX_INPUT_BYTES} bytes")
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_INPUT_BYTES + 1)
        mtime = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc.strerror or exc}") from exc
    return Document(name=path.name, text=decode_bytes(data), mtime=mtime)


def _read_stdin() -> Document:
    data = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    return Document(name="<stdin>", text=decode_bytes(data), mtime=None)


def _directory_files(path: Path) -> list[Path]:
    children = sorted(
        child
        for child in path.iterdir()
        if child.is_file()
        and not child.is_symlink()
        and child.suffix.lower() in TEXT_SUFFIXES
    )
    if not children:
        raise InputError(f"no transcript files (.txt, .md, .vtt, .srt) in {path}")
    return children


def read_documents(inputs: list[str]) -> list[Document]:
    """Resolve ``-`` (stdin), files and directories (non-recursive, text suffixes only).

    File count and aggregate size are checked before anything is decoded, so a huge directory
    fails fast instead of being read in full.
    """
    paths: list[Path] = []
    for item in inputs:
        if item == "-":
            continue
        path = Path(item)
        if path.is_dir():
            paths.extend(_directory_files(path))
        elif path.is_file():
            paths.append(path)
        else:
            raise InputError(f"no such file or directory: {item}")
        if len(paths) > MAX_FILES:
            raise InputError(f"more than {MAX_FILES} transcript files in one run")
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError as exc:
            raise InputError(f"cannot read {path}: {exc.strerror or exc}") from exc
    if total > MAX_TOTAL_BYTES:
        raise InputError(f"inputs total more than {MAX_TOTAL_BYTES} bytes")
    documents: list[Document] = []
    for item in inputs:
        if item == "-":
            documents.append(_read_stdin())
    documents.extend(_read_file(path) for path in paths)
    return documents


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def check_output_path(out: Path, inputs: list[str], force: bool = False) -> Path:
    """Refuse output targets that would clobber an input, a directory or a symlink.

    A text-suffixed output inside an input directory is refused even when it does not exist yet,
    because the next run would read it back as a transcript. An existing file needs ``force``.
    """
    resolved = out.resolve()
    if out.is_symlink():
        raise InputError(f"refusing to write through a symbolic link: {out}")
    if out.is_dir():
        raise InputError(f"--out must be a file, not a directory: {out}")
    for item in inputs:
        if item == "-":
            continue
        path = Path(item)
        if _same_file(out, path):
            raise InputError(f"--out would overwrite an input file: {out}")
        if (
            path.is_dir()
            and out.suffix.lower() in TEXT_SUFFIXES
            and resolved.parent == path.resolve()
        ):
            raise InputError(
                f"--out would write a transcript into the input directory {path}"
            )
    if not out.parent.is_dir():
        raise InputError(f"the directory for --out does not exist: {out.parent}")
    if out.exists() and not force:
        raise InputError(f"{out} already exists; pass --force to overwrite it")
    return resolved


def write_atomic(out: Path, text: str) -> None:
    """Write through a temporary file and rename, so a failure never leaves a truncated file."""
    data = text.encode("utf-8")  # fails on lone surrogates before the target is touched
    fd, temp = tempfile.mkstemp(dir=out.parent, prefix=".actionitems-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temp, out)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp)
        raise
