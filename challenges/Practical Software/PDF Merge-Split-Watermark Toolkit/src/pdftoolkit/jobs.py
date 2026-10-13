"""One entry point, ``run_op``, behind both the CLI commands and TOML batch files."""

import inspect
import os
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pdftoolkit import ops, watermark
from pdftoolkit.errors import PdfToolError
from pdftoolkit.files import expand_inputs, guard_outputs, plan_outputs

# op -> (function, output-name suffix) for operations that map one input to one output.
PER_FILE: dict[str, tuple[Callable, str]] = {
    "extract": (ops.extract, "_extract"),
    "reorder": (ops.reorder, "_reordered"),
    "rotate": (ops.rotate, "_rotated"),
    "watermark": (watermark.watermark, "_watermarked"),
    "encrypt": (ops.encrypt, "_encrypted"),
    "decrypt": (ops.decrypt, "_decrypted"),
    "meta": (ops.set_metadata, "_meta"),
}
OPS = ("merge", "split", *PER_FILE)
_PATH_OPTIONS = {"image", "stamp", "font"}
_FIXED = {"src", "dst", "sources", "out_dir", "protect"}
_SECRETS = {"password", "user_password", "owner_password"}


@dataclass(frozen=True)
class Result:
    op: str
    sources: tuple[Path, ...]
    outputs: tuple[Path, ...]
    detail: str = ""


def _resolve(text: str | Path | None, base: Path | None) -> Path | None:
    if text is None:
        return None
    path = Path(text).expanduser()
    return path if base is None or path.is_absolute() else base / path


def _check_options(fn: Callable, op: str, options: dict) -> dict:
    allowed = set(inspect.signature(fn).parameters) - _FIXED
    if unknown := sorted(set(options) - allowed):
        raise PdfToolError(
            f"{op}: unknown option(s) {', '.join(unknown)}; valid: {', '.join(sorted(allowed))}"
        )
    return options


def run_op(
    op: str,
    inputs: Sequence[str],
    *,
    output: str | Path | None = None,
    out_dir: str | Path | None = None,
    options: dict | None = None,
    base: Path | None = None,
    dry_run: bool = False,
) -> list[Result]:
    """Run ``op`` over ``inputs``. Relative paths resolve against ``base`` (a batch file's folder)."""
    if op not in OPS:
        raise PdfToolError(f"unknown op {op!r}; choose from {', '.join(OPS)}")
    options = dict(options or {})
    for key in _PATH_OPTIONS & options.keys():
        options[key] = _resolve(options[key], base)
    sources = expand_inputs(inputs, base)
    out_file, out_folder = _resolve(output, base), _resolve(out_dir, base)

    if op == "merge":
        if out_file is None or out_folder is not None:
            raise PdfToolError("merge needs an output file (not an output directory)")
        opts = _check_options(ops.merge, op, options)
        guard_outputs([s.path for s in sources], [out_file])
        pages = "" if dry_run else f"{ops.merge(sources, out_file, **opts)} pages"
        return [Result(op, tuple(s.path for s in sources), (out_file,), pages)]

    if op == "split":
        if out_file is not None:
            raise PdfToolError("split writes several files; use an output directory")
        opts = _check_options(ops.split, op, options)
        stems = [s.path.stem for s in sources]
        if out_folder is not None and len(set(stems)) != len(stems):
            raise PdfToolError(
                "inputs share a file name; their splits would collide in one folder"
            )
        results = []
        for src in sources:
            target = out_folder or src.path.parent / f"{src.path.stem}_split"
            written = (
                [target]
                if dry_run
                else ops.split(src, target, protect=[s.path for s in sources], **opts)
            )
            results.append(
                Result(op, (src.path,), tuple(written), f"{len(written)} files")
            )
        return results

    fn, suffix = PER_FILE[op]
    opts = _check_options(fn, op, options)
    results = []
    for src, dst in plan_outputs(
        sources, output=out_file, out_dir=out_folder, suffix=suffix
    ):
        if dry_run:
            results.append(Result(op, (src.path,), (dst,)))
            continue
        outcome = fn(src, dst, **opts)
        detail = f"{outcome} pages" if isinstance(outcome, int) else ""
        results.append(Result(op, (src.path,), (dst,), detail))
    return results


def load_batch(path: Path) -> tuple[list[dict], Path]:
    """Parse a TOML batch file into job dicts (``[defaults]`` merged in) and its base folder."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PdfToolError(f"cannot read batch file {path}: {exc}") from exc
    defaults = data.get("defaults", {})
    jobs = data.get("job", [])
    if not isinstance(jobs, list) or not jobs:
        raise PdfToolError(f"{path.name} has no [[job]] entries")
    return [{**defaults, **job} for job in jobs], path.resolve().parent


def run_job(job: dict, base: Path, *, force: bool, dry_run: bool) -> list[Result]:
    job = dict(job)
    op = job.pop("op", None)
    inputs = job.pop("inputs", None)
    if (
        not op
        or not isinstance(inputs, list)
        or not all(isinstance(i, str) for i in inputs)
    ):
        raise PdfToolError("each job needs op and inputs (a list of strings)")
    job.pop("name", None)
    output, out_dir = job.pop("output", None), job.pop("output_dir", None)
    options = {}
    for key, value in job.items():
        if key == "force":
            raise PdfToolError(
                "force cannot be set in a batch file; pass --force to batch"
            )
        if key in _SECRETS:
            raise PdfToolError(
                f"put {key} in the environment and use {key}_env instead"
            )
        if key.endswith("_env"):
            name = key.removesuffix("_env")
            if name not in _SECRETS:
                raise PdfToolError(f"{key} is not allowed; _env is only for passwords")
            if str(value) not in os.environ:
                raise PdfToolError(f"environment variable {value} is not set")
            key, value = name, os.environ[str(value)]
        options[key] = value
    if force:
        options["force"] = True
    try:
        return run_op(
            op,
            inputs,
            output=output,
            out_dir=out_dir,
            options=options,
            base=base,
            dry_run=dry_run,
        )
    except (TypeError, ValueError) as exc:
        raise PdfToolError(f"{op}: invalid option value ({exc})") from exc


def run_batch(
    path: Path, *, force: bool = False, dry_run: bool = False, fail_fast: bool = False
):
    """Run every job; yields ``(index, label, results | PdfToolError)``."""
    jobs, base = load_batch(path)
    for index, job in enumerate(jobs, 1):
        label = str(job.get("name") or job.get("op") or f"job {index}")
        try:
            yield index, label, run_job(job, base, force=force, dry_run=dry_run)
        except PdfToolError as exc:
            yield index, label, exc
            if fail_fast:
                return
