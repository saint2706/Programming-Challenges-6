"""``pdftool`` command line."""

import getpass
import json
import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from pdftoolkit import jobs, ops
from pdftoolkit.errors import PdfToolError
from pdftoolkit.files import expand_inputs
from pdftoolkit.ops import OutlineMode
from pdftoolkit.watermark import Position

app = App(
    name="pdftool",
    help="Merge, split, watermark and tidy PDFs. Page ranges look like 1-3,5,8- (1-based).",
)

Output = Annotated[
    Path | None, Parameter(name=["--output", "-o"], help="Output file (one input).")
]
OutDir = Annotated[
    Path | None,
    Parameter(name=["--out-dir", "-d"], help="Output folder (any number of inputs)."),
]
Force = Annotated[
    bool,
    Parameter(name=["--force", "-f"], negative="", help="Overwrite existing outputs."),
]
Password = Annotated[
    str | None,
    Parameter(
        name="--password",
        env_var="PDFTOOL_PASSWORD",
        help="Password for encrypted inputs; prefer the PDFTOOL_PASSWORD environment variable.",
    ),
]


def _secret(value: str | None, prompt: str) -> str:
    """Use the given secret, else ask on a terminal, so it need not sit in shell history."""
    if value:
        return value
    if sys.stdin.isatty():
        return getpass.getpass(prompt)
    raise PdfToolError(
        f"{prompt.strip(': ')} required (option, environment variable or prompt)"
    )


def _report(results: list[jobs.Result]) -> None:
    for r in results:
        src = ", ".join(p.name for p in r.sources)
        dst = ", ".join(str(p) for p in r.outputs)
        print(f"{r.op}: {src} -> {dst}" + (f" ({r.detail})" if r.detail else ""))


def _run(op, inputs, output, out_dir, force, password, **options) -> None:
    options = {k: v for k, v in options.items() if v is not None}
    if force:
        options["force"] = True
    if password:
        options["password"] = password
    _report(jobs.run_op(op, inputs, output=output, out_dir=out_dir, options=options))


@app.command
def merge(
    *inputs: str,
    output: Annotated[
        Path, Parameter(name=["--output", "-o"], help="Merged PDF to write.")
    ],
    outline: OutlineMode = "nested",
    title: str | None = None,
    author: str | None = None,
    force: Force = False,
    password: Password = None,
) -> None:
    """Concatenate PDFs. Inputs may be globs (``scans/*.pdf``) or select pages (``a.pdf[1-3]``).

    Parameters
    ----------
    outline
        Bookmarks: nested (a bookmark per input holding its own), files, original, none.
    """
    _run(
        "merge",
        inputs,
        output,
        None,
        force,
        password,
        outline=outline,
        title=title,
        author=author,
    )


@app.command
def split(
    *inputs: str,
    every: int | None = None,
    ranges: str | None = None,
    each: bool = False,
    out_dir: OutDir = None,
    force: Force = False,
    password: Password = None,
) -> None:
    """Split PDFs into several files: every N pages, at ranges (1-3,4-9) or one per page.

    Writes to ``<name>_split/`` beside each input unless --out-dir is given.
    """
    _run(
        "split",
        inputs,
        None,
        out_dir,
        force,
        password,
        every=every,
        ranges=ranges,
        each=each or None,
    )


@app.command
def extract(
    *inputs: str,
    pages: str,
    output: Output = None,
    out_dir: OutDir = None,
    force: Force = False,
    password: Password = None,
) -> None:
    """Keep only the given pages, in the order written."""
    _run("extract", inputs, output, out_dir, force, password, pages=pages)


@app.command
def reorder(
    *inputs: str,
    order: str,
    output: Output = None,
    out_dir: OutDir = None,
    force: Force = False,
    password: Password = None,
) -> None:
    """Rearrange pages; --order must name every page once, e.g. ``3,1,2`` or ``end-1``."""
    _run("reorder", inputs, output, out_dir, force, password, order=order)


@app.command
def rotate(
    *inputs: str,
    degrees: int = 90,
    pages: str | None = None,
    output: Output = None,
    out_dir: OutDir = None,
    force: Force = False,
    password: Password = None,
) -> None:
    """Rotate pages clockwise by a multiple of 90 degrees (negative turns counter-clockwise)."""
    _run(
        "rotate", inputs, output, out_dir, force, password, degrees=degrees, pages=pages
    )


@app.command
def watermark(
    *inputs: str,
    text: str | None = None,
    image: Path | None = None,
    stamp: Path | None = None,
    stamp_page: int | None = None,
    size: float | None = None,
    color: str | None = None,
    opacity: float | None = None,
    angle: float | None = None,
    position: Position | None = None,
    scale: float | None = None,
    under: bool = False,
    font: Path | None = None,
    pages: str | None = None,
    output: Output = None,
    out_dir: OutDir = None,
    force: Force = False,
    password: Password = None,
) -> None:
    """Stamp text, an image or a page of another PDF onto pages.

    Defaults: size 64, color #808080, opacity 0.3, angle 45, position center, scale 0.5.

    Parameters
    ----------
    text
        Text to draw (use --font for characters outside Windows-1252).
    image
        PNG/JPEG/etc. to draw; transparency is kept.
    stamp
        PDF whose page --stamp-page (default 1) is placed as the mark; no opacity control.
    scale
        Image/stamp width as a fraction of the page width.
    under
        Put the mark beneath the page content instead of over it.
    """
    _run(
        "watermark",
        inputs,
        output,
        out_dir,
        force,
        password,
        text=text,
        image=image,
        stamp=stamp,
        stamp_page=stamp_page,
        size=size,
        color=color,
        opacity=opacity,
        angle=angle,
        position=position,
        scale=scale,
        under=under or None,
        font=font,
        pages=pages,
    )


@app.command
def encrypt(
    *inputs: str,
    user_password: Annotated[
        str | None,
        Parameter(env_var="PDFTOOL_USER_PASSWORD", help="Password to open the file."),
    ] = None,
    owner_password: Annotated[
        str | None,
        Parameter(
            env_var="PDFTOOL_OWNER_PASSWORD", help="Password to lift restrictions."
        ),
    ] = None,
    allow_print: bool = True,
    allow_copy: bool = True,
    allow_modify: bool = True,
    output: Output = None,
    out_dir: OutDir = None,
    force: Force = False,
    password: Password = None,
) -> None:
    """Encrypt with AES-256. Prompts for the password on a terminal if none is given."""
    secret = _secret(user_password, "User password: ")
    _run(
        "encrypt",
        inputs,
        output,
        out_dir,
        force,
        password,
        user_password=secret,
        owner_password=owner_password,
        allow_print=allow_print,
        allow_copy=allow_copy,
        allow_modify=allow_modify,
    )


@app.command
def decrypt(
    *inputs: str,
    output: Output = None,
    out_dir: OutDir = None,
    force: Force = False,
    password: Password = None,
) -> None:
    """Remove encryption. Prompts for the password on a terminal if none is given."""
    secret = _secret(password, "Password: ")
    _run("decrypt", inputs, output, out_dir, force, secret)


@app.command
def meta(
    *inputs: str,
    title: str | None = None,
    author: str | None = None,
    subject: str | None = None,
    keywords: str | None = None,
    clear: bool = False,
    output: Output = None,
    out_dir: OutDir = None,
    force: Force = False,
    password: Password = None,
) -> None:
    """Edit document metadata; --clear drops all existing fields (and XMP) first."""
    _run(
        "meta",
        inputs,
        output,
        out_dir,
        force,
        password,
        title=title,
        author=author,
        subject=subject,
        keywords=keywords,
        clear=clear or None,
    )


@app.command
def info(
    *inputs: str,
    json_output: Annotated[bool, Parameter(name="--json")] = False,
    password: Password = None,
) -> None:
    """Show page count, sizes, bookmarks, encryption and metadata."""
    reports = [ops.info(s.path, password) for s in expand_inputs(inputs)]
    if json_output:
        print(json.dumps(reports, indent=2))
        return
    for r in reports:
        sizes = sorted({(p["width"], p["height"]) for p in r["page_sizes"]})
        print(
            f"{r['file']}: {r['pages']} pages, {r['bytes']:,} bytes, PDF {r['version']}"
        )
        print(f"  encrypted: {r['encrypted']}   bookmarks: {r['bookmarks']}")
        print("  page sizes (pt): " + ", ".join(f"{w:g}x{h:g}" for w, h in sizes))
        for key, value in r["metadata"].items():
            print(f"  {key}: {value}")


@app.command
def batch(
    file: Path,
    /,
    *,
    dry_run: bool = False,
    fail_fast: bool = False,
    force: Force = False,
) -> int:
    """Run the jobs in a TOML file (see README). Exit status 1 if any job failed."""
    failed = 0
    for index, label, outcome in jobs.run_batch(
        file, force=force, dry_run=dry_run, fail_fast=fail_fast
    ):
        if isinstance(outcome, PdfToolError):
            failed += 1
            print(f"[{index}] {label}: FAILED - {outcome}", file=sys.stderr)
        else:
            print(f"[{index}] {label}{' (dry run)' if dry_run else ''}")
            _report(outcome)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    try:
        result = app(argv, result_action="return_value")
    except PdfToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
