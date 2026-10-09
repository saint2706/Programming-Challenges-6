"""``envdiff``: compare env files across machines without printing a single value."""

import json
import os
import tempfile
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from envdiff import compare as comparison
from envdiff import report as rendering
from envdiff.fingerprint import is_secret_name
from envdiff.snapshot import snapshot_to_dict, write_snapshot
from envdiff.sources import SourceError, load_sources, parse_spec

PASSPHRASE_ENV = "ENVDIFF_PASSPHRASE"

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


class Dialect(str, Enum):
    auto = "auto"
    dotenv = "dotenv"
    shell = "shell"
    raw = "raw"


class Style(str, Enum):
    text = "text"
    json = "json"
    markdown = "markdown"


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(2)


def _passphrase(file: Path | None) -> str | None:
    """From a file or the environment; never from argv, where ``ps`` and shell history would keep it."""
    if file is not None:
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise _fail(f"{file}: {exc.strerror or 'cannot be read'}") from exc
        return lines[0] if lines and lines[0] else None
    return os.environ.get(PASSPHRASE_ENV) or None


FormatOption = Annotated[
    Dialect,
    typer.Option(
        "--format",
        "-f",
        help="Env-file dialect; auto picks shell for `export -p` output, else dotenv.",
    ),
]
PassphraseFileOption = Annotated[
    Path | None,
    typer.Option(
        "--passphrase-file",
        help=f"File whose first line is the shared passphrase (or set {PASSPHRASE_ENV}).",
    ),
]


@app.command()
def diff(
    sources: Annotated[
        list[str],
        typer.Argument(
            help="Env files: PATH, LABEL=PATH, [USER@]HOST:PATH (over ssh), `-` (stdin), `env:` (this process), or a snapshot .json."
        ),
    ],
    fmt: FormatOption = Dialect.auto,
    ignore: Annotated[
        list[str] | None,
        typer.Option(
            "--ignore",
            "-i",
            help="Glob of variable names to leave out (repeatable), e.g. 'PATH' or 'AWS_*'.",
        ),
    ] = None,
    reveal: Annotated[
        list[str] | None,
        typer.Option(
            "--reveal",
            "-r",
            help="Print the value of this variable (repeatable). Off by default for everything.",
        ),
    ] = None,
    show_all: Annotated[
        bool,
        typer.Option(
            "--all", "-a", help="Also list variables that are identical everywhere."
        ),
    ] = False,
    fingerprints: Annotated[
        bool,
        typer.Option(
            "--fingerprints",
            help="Show a short keyed fingerprint next to each group letter.",
        ),
    ] = False,
    style: Annotated[Style, typer.Option("--as", help="Output style.")] = Style.text,
    lint: Annotated[
        bool,
        typer.Option(
            "--lint/--no-lint",
            help="Report hygiene findings (placeholders, shared secrets, parse problems).",
        ),
    ] = True,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict", help="Exit 1 on warnings too, not only on differences."
        ),
    ] = False,
    passphrase_file: PassphraseFileOption = None,
) -> None:
    """Show which variables differ between two or more env files, hiding every value.

    Equal values share a letter in the table; whether they are equal is computed with keyed
    fingerprints, so nothing about a value, not even its length, appears in the output.
    Exit status: 0 identical, 1 differences (or warnings with --strict), 2 error.
    """
    reveal_set = frozenset(reveal or [])
    for name in sorted(reveal_set):
        if is_secret_name(name):
            typer.echo(
                f"warning: revealing {name}, which looks like a secret", err=True
            )
    specs = [parse_spec(s) for s in sources]
    if sum(spec.kind == "stdin" for spec in specs) > 1:
        raise _fail("stdin (`-`) can only be used once")
    try:
        loaded = load_sources(
            specs, fmt.value, _passphrase(passphrase_file), reveal_set
        )
    except SourceError as exc:
        raise _fail(str(exc)) from exc

    result = comparison.compare(loaded, ignore or [], lint)
    revealed: dict[str, dict[str, str]] = {}
    for key in sorted(reveal_set):
        per_source = {s.label: s.revealed[key] for s in loaded if key in s.revealed}
        if per_source:
            revealed[key] = per_source
        elif not any(key in s.entries for s in loaded):
            typer.echo(f"warning: --reveal {key}: no source defines it", err=True)
        else:
            typer.echo(
                f"warning: --reveal {key}: only snapshots define it, and they hold no values",
                err=True,
            )

    if style is Style.json:
        typer.echo(
            rendering.render_json(result, True, fingerprints, revealed), nl=False
        )
    elif style is Style.markdown:
        typer.echo(
            rendering.render_markdown(result, show_all, fingerprints, revealed),
            nl=False,
        )
    else:
        typer.echo(
            rendering.render_text(result, show_all, fingerprints, revealed), nl=False
        )
    if result.has_differences or (strict and result.has_warnings):
        raise typer.Exit(1)


@app.command()
def snapshot(
    source: Annotated[
        str,
        typer.Argument(help="Env file to fingerprint: PATH, [USER@]HOST:PATH, or `-`."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Write the snapshot here (default: stdout)."
        ),
    ] = None,
    label: Annotated[
        str | None,
        typer.Option("--label", "-l", help="Name for this host in later comparisons."),
    ] = None,
    fmt: FormatOption = Dialect.auto,
    passphrase_file: PassphraseFileOption = None,
) -> None:
    """Write names and keyed fingerprints of an env file, so a host can be compared without its secrets.

    Run it on each machine with the same passphrase, copy the small JSON files anywhere, then
    `envdiff diff a.json b.json`. A snapshot holds no values, but anyone who knows the passphrase can
    test guesses against it, so treat the passphrase like a secret and make it long.
    """
    passphrase = _passphrase(passphrase_file)
    if not passphrase:
        raise _fail(
            f"a passphrase is required: set {PASSPHRASE_ENV} or use --passphrase-file"
        )
    spec = parse_spec(source)
    if label:
        spec = replace(spec, label=label)
    try:
        (loaded,) = load_sources([spec], fmt.value, passphrase)
    except SourceError as exc:
        raise _fail(str(exc)) from exc
    if loaded.fmt == "snapshot":
        raise _fail(f"{source} is already a snapshot")
    if output is None:
        typer.echo(json.dumps(snapshot_to_dict(loaded), indent=2))
        return
    _write_atomic(output, loaded)
    typer.echo(
        f"Wrote {output} ({len(loaded.entries)} variables, key id {loaded.key_id})",
        err=True,
    )


def _write_atomic(path: Path, source) -> None:
    fd, temp = tempfile.mkstemp(
        dir=path.resolve().parent, prefix=".envdiff-", suffix=".tmp"
    )
    os.close(fd)
    try:
        write_snapshot(source, Path(temp))
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
