"""Page-level operations. Each takes paths in, writes one or more PDFs out, returns what it wrote."""

import secrets
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pypdf import PdfReader, PdfWriter
from pypdf.constants import UserAccessPermissions

from pdftoolkit.errors import PdfToolError
from pdftoolkit.files import Source, guard_outputs, open_reader, write_pdf
from pdftoolkit.ranges import parse_groups, parse_pages

OutlineMode = Literal["nested", "files", "original", "none"]
_META_KEYS = {
    "title": "/Title",
    "author": "/Author",
    "subject": "/Subject",
    "keywords": "/Keywords",
}


def _select(reader: PdfReader, spec: str | None) -> list[int]:
    total = len(reader.pages)
    return parse_pages(spec, total) if spec else list(range(total))


def merge(
    sources: Sequence[Source],
    dst: Path,
    *,
    password: str | None = None,
    outline: OutlineMode = "nested",
    title: str | None = None,
    author: str | None = None,
    force: bool = False,
) -> int:
    """Concatenate inputs (each optionally ``[pages]``-selected). Returns the page count.

    ``outline``: ``nested`` one bookmark per input holding that file's own bookmarks; ``files``
    one bookmark per input only; ``original`` the inputs' bookmarks without a parent; ``none``.
    """
    if outline not in ("nested", "files", "original", "none"):
        raise PdfToolError(f"unknown outline mode {outline!r}")
    writer = PdfWriter()
    for src in sources:
        reader = open_reader(src.path, password)
        label = src.path.stem if outline in ("nested", "files") else None
        # Bookmarks that target pages a selection dropped are skipped by pypdf.
        writer.append(
            reader,
            outline_item=label,
            pages=_select(reader, src.pages),
            import_outline=outline in ("nested", "original"),
        )
    meta = {k: v for k, v in (("/Title", title), ("/Author", author)) if v}
    if meta:
        writer.add_metadata(meta)
    write_pdf(writer, dst, force=force)
    return len(writer.pages)


def extract(
    src: Source,
    dst: Path,
    *,
    pages: str,
    password: str | None = None,
    force: bool = False,
) -> int:
    reader = open_reader(src.path, password)
    writer = PdfWriter()
    writer.append(reader, pages=parse_pages(pages, len(reader.pages)))
    write_pdf(writer, dst, force=force)
    return len(writer.pages)


def reorder(
    src: Source,
    dst: Path,
    *,
    order: str,
    password: str | None = None,
    force: bool = False,
) -> int:
    """Rewrite pages in ``order``, which must name every page exactly once."""
    reader = open_reader(src.path, password)
    total = len(reader.pages)
    if sorted(parse_pages(order, total)) != list(range(total)):
        raise PdfToolError(f"order must list each of the {total} pages exactly once")
    return extract(src, dst, pages=order, password=password, force=force)


def split(
    src: Source,
    out_dir: Path,
    *,
    every: int | None = None,
    ranges: str | None = None,
    each: bool = False,
    password: str | None = None,
    force: bool = False,
    protect: Sequence[Path] = (),
) -> list[Path]:
    """Cut one PDF into several: ``every`` N pages, at ``ranges`` (one file per comma token),
    or ``each`` page. Files are ``<stem>_p<first>-<last>.pdf`` (``part<N>`` if not contiguous)."""
    if sum([every is not None, ranges is not None, each]) != 1:
        raise PdfToolError("choose exactly one of every, ranges, each")
    reader = open_reader(src.path, password)
    total = len(reader.pages)
    if every is not None:
        if every < 1:
            raise PdfToolError("every must be at least 1")
        groups = [list(range(i, min(i + every, total))) for i in range(0, total, every)]
    elif ranges is not None:
        groups = parse_groups(ranges, total)
    else:
        groups = [[i] for i in range(total)]
    jobs = [(_split_name(src.path.stem, n, g), g) for n, g in enumerate(groups, 1)]
    names = [name for name, _ in jobs]
    if len(set(names)) != len(names):
        raise PdfToolError("two ranges would produce the same output file")
    guard_outputs([src.path, *protect], [out_dir / n for n in names])
    # Check every target first so a clash cannot leave a half-written split.
    if not force and (
        clash := next((out_dir / n for n in names if (out_dir / n).exists()), None)
    ):
        raise PdfToolError(f"{clash} already exists (use --force to overwrite)")
    written = []
    for name, group in jobs:
        writer = PdfWriter()
        writer.append(reader, pages=group)
        written.append(write_pdf(writer, out_dir / name, force=force))
    return written


def _split_name(stem: str, n: int, group: list[int]) -> str:
    first, last = group[0] + 1, group[-1] + 1
    if group == list(range(group[0], group[0] + len(group))):
        return (
            f"{stem}_p{first}.pdf" if len(group) == 1 else f"{stem}_p{first}-{last}.pdf"
        )
    return f"{stem}_part{n:03d}.pdf"


def rotate(
    src: Source,
    dst: Path,
    *,
    degrees: int,
    pages: str | None = None,
    password: str | None = None,
    force: bool = False,
) -> int:
    """Rotate the selected pages (all by default) clockwise by a multiple of 90."""
    if degrees % 90:
        raise PdfToolError("degrees must be a multiple of 90")
    reader = open_reader(src.path, password)
    writer = PdfWriter(clone_from=reader)
    chosen = _select(reader, pages or src.pages)
    for i in set(chosen):
        writer.pages[i].rotate(degrees)
    write_pdf(writer, dst, force=force)
    return len(set(chosen))


def encrypt(
    src: Source,
    dst: Path,
    *,
    user_password: str,
    owner_password: str | None = None,
    allow_print: bool = True,
    allow_copy: bool = True,
    allow_modify: bool = True,
    password: str | None = None,
    force: bool = False,
) -> None:
    """AES-256. The user password opens the file; the owner password lifts the restrictions."""
    if not user_password:
        raise PdfToolError("a non-empty password is required")
    reader = open_reader(src.path, password)
    writer = PdfWriter(clone_from=reader)
    perms = UserAccessPermissions(0)
    if allow_print:
        perms |= (
            UserAccessPermissions.PRINT | UserAccessPermissions.PRINT_TO_REPRESENTATION
        )
    if allow_copy:
        perms |= (
            UserAccessPermissions.EXTRACT
            | UserAccessPermissions.EXTRACT_TEXT_AND_GRAPHICS
        )
    if allow_modify:
        perms |= (
            UserAccessPermissions.MODIFY
            | UserAccessPermissions.ADD_OR_MODIFY
            | UserAccessPermissions.ASSEMBLE_DOC
            | UserAccessPermissions.FILL_FORM_FIELDS
        )
    # An owner password equal to the user password would let anyone who can open the file lift
    # the restrictions, so withholding a permission without one gets a random, discarded owner key.
    restricted = not (allow_print and allow_copy and allow_modify)
    owner = owner_password or (
        secrets.token_urlsafe(24) if restricted else user_password
    )
    writer.encrypt(
        user_password,
        owner,
        permissions_flag=perms,
        algorithm="AES-256",
    )
    write_pdf(writer, dst, force=force)


def decrypt(src: Source, dst: Path, *, password: str, force: bool = False) -> None:
    reader = open_reader(src.path, password)
    if not reader.is_encrypted:
        raise PdfToolError(f"{src.path.name} is not encrypted")
    write_pdf(PdfWriter(clone_from=reader), dst, force=force)


def set_metadata(
    src: Source,
    dst: Path,
    *,
    title: str | None = None,
    author: str | None = None,
    subject: str | None = None,
    keywords: str | None = None,
    clear: bool = False,
    password: str | None = None,
    force: bool = False,
) -> None:
    """Edit the Info dictionary; ``clear`` first drops every field and any XMP packet."""
    reader = open_reader(src.path, password)
    writer = PdfWriter(clone_from=reader)
    if clear:
        writer.metadata = None
        writer._root_object.pop("/Metadata", None)
    values = {
        "title": title,
        "author": author,
        "subject": subject,
        "keywords": keywords,
    }
    updates = {_META_KEYS[k]: v for k, v in values.items() if v is not None}
    if updates:
        writer.add_metadata(updates)
    write_pdf(writer, dst, force=force)


def _count_outline(items: list) -> int:
    return sum(1 if not isinstance(i, list) else _count_outline(i) for i in items)


def info(path: Path, password: str | None = None) -> dict:
    reader = open_reader(path, password)
    pages = [
        {
            "width": round(float(p.mediabox.width), 2),
            "height": round(float(p.mediabox.height), 2),
            "rotation": p.rotation,
        }
        for p in reader.pages
    ]
    meta = {k.lstrip("/"): str(v) for k, v in (reader.metadata or {}).items()}
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "version": reader.pdf_header.removeprefix("%PDF-"),
        "pages": len(pages),
        "encrypted": bool(reader.is_encrypted),
        "bookmarks": _count_outline(reader.outline),
        "metadata": meta,
        "page_sizes": pages,
    }
