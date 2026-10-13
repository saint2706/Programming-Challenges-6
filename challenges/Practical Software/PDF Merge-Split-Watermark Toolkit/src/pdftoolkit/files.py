"""Reading, writing and path planning shared by every operation."""

import glob
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.errors import PyPdfError

from pdftoolkit.errors import PdfToolError

# "report.pdf[1-3]" selects pages; the real file wins if its name really contains brackets.
_BRACKET = re.compile(r"(?P<path>.+)\[(?P<pages>[^\[\]]+)\]")
_GLOB_CHARS = ("*", "?")


@dataclass(frozen=True)
class Source:
    path: Path
    pages: str | None = None


def _natural(path: Path) -> list[object]:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(path))]


def expand_inputs(items: Iterable[str], base: Path | None = None) -> list[Source]:
    """Resolve file names, ``*``/``?`` globs (Windows shells do not expand them) and ``[pages]``."""
    sources: list[Source] = []
    for item in items:
        path_text, pages = item, None
        match = _BRACKET.fullmatch(item)
        if match and not _resolve(item, base).exists():
            path_text, pages = match["path"], match["pages"]
        if any(c in path_text for c in _GLOB_CHARS):
            pattern = str(_resolve(path_text, base))
            found = sorted((Path(p) for p in glob.glob(pattern)), key=_natural)
            if not found:
                raise PdfToolError(f"no files match {item!r}")
            sources.extend(Source(p, pages) for p in found if p.is_file())
            continue
        path = _resolve(path_text, base)
        if not path.is_file():
            raise PdfToolError(f"no such file: {path}")
        sources.append(Source(path, pages))
    if not sources:
        raise PdfToolError("no input files")
    return sources


def _resolve(text: str, base: Path | None) -> Path:
    path = Path(text).expanduser()
    return path if base is None or path.is_absolute() else base / path


def open_reader(path: Path, password: str | None = None) -> PdfReader:
    try:
        reader = PdfReader(path)
    except (PyPdfError, OSError, ValueError) as exc:
        raise PdfToolError(f"cannot read {path.name}: {exc}") from exc
    if reader.is_encrypted:
        try:
            ok = reader.decrypt(password or "")
        except (PyPdfError, ValueError, NotImplementedError) as exc:
            raise PdfToolError(f"cannot decrypt {path.name}: {exc}") from exc
        if not ok:
            hint = "wrong password" if password else "password required"
            raise PdfToolError(f"{path.name} is encrypted: {hint}")
    return reader


def write_pdf(writer: PdfWriter, dst: Path, *, force: bool = False) -> Path:
    """Write via a temp file in the destination folder so a failure never leaves half a PDF."""
    if dst.exists() and not force:
        raise PdfToolError(f"{dst} already exists (use --force to overwrite)")
    tmp_name = None
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp"
        )
        with os.fdopen(fd, "wb") as fh:
            writer.write(fh)
        os.replace(tmp_name, dst)
    except OSError as exc:
        raise PdfToolError(f"cannot write {dst}: {exc}") from exc
    finally:
        # After a successful replace the temp name no longer exists, so this is a no-op.
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
    return dst


def guard_outputs(sources: Sequence[Path], outputs: Sequence[Path]) -> None:
    """Refuse to overwrite an input, or to write two results to one file."""
    inputs = {p.resolve() for p in sources}
    seen: set[Path] = set()
    for out in outputs:
        resolved = out.resolve()
        if resolved in inputs:
            raise PdfToolError(f"{out} is also an input; choose a different output")
        if resolved in seen:
            raise PdfToolError(f"two results would be written to {out}")
        seen.add(resolved)


def plan_outputs(
    sources: Sequence[Source],
    *,
    output: Path | None,
    out_dir: Path | None,
    suffix: str,
) -> list[tuple[Source, Path]]:
    """Pair each input with its output: ``-o FILE`` for one input, else ``<stem><suffix>.pdf``."""
    if output is not None and out_dir is not None:
        raise PdfToolError("use either an output file or an output directory, not both")
    if output is not None:
        if len(sources) != 1:
            raise PdfToolError(
                "an output file needs exactly one input; use an output directory"
            )
        plan = [(sources[0], output)]
    else:
        plan = [
            (s, (out_dir or s.path.parent) / f"{s.path.stem}{suffix}.pdf")
            for s in sources
        ]
    guard_outputs([s.path for s, _ in plan], [dst for _, dst in plan])
    return plan
