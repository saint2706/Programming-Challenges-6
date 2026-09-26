"""Framework-free background-removal core: file discovery, batching, and the
rembg wrapper itself. No typer/rich imports here, so this module is directly
unit-testable and its network-touching parts (``new_session``,
``remove_background_bytes``) are easy to monkeypatch from tests without a
real model download.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# rembg's general-purpose (non-specialized) matting models. Deliberately
# excludes human_seg/cloth_seg/anime (domain-specific, would mislead a
# generic CLI default) and sam/birefnet-* (extra heavy dependencies / far
# larger downloads not worth it for a Beginner-tier batch tool).
AVAILABLE_MODELS: tuple[str, ...] = ("isnet-general-use", "u2net", "u2netp", "silueta")

# isnet-general-use is rembg's newer general-purpose model with visibly
# cleaner edges than the original u2net on typical photos -- the "modern"
# pick per this repo's stack preference -- at the cost of a ~179MB one-time
# download (vs. u2netp's ~4.7MB) cached under ~/.rembg/models/ after first use.
DEFAULT_MODEL = "isnet-general-use"

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
)


@dataclass(frozen=True)
class FileResult:
    """Outcome for a single input file."""

    path: Path
    status: str  # "processed" | "skipped"
    reason: str | None = None
    output_path: Path | None = None


@dataclass
class BatchResult:
    """Outcome for a whole batch run."""

    results: list[FileResult] = field(default_factory=list)

    @property
    def processed(self) -> list[FileResult]:
        return [r for r in self.results if r.status == "processed"]

    @property
    def skipped(self) -> list[FileResult]:
        return [r for r in self.results if r.status == "skipped"]


def resolve_input_files(input_path: Path, recursive: bool) -> list[Path]:
    """Return the sorted list of files a batch run should attempt.

    A single file is returned as-is (its extension is validated later, in
    ``process_one``, so an unsupported single file still shows up as one
    explicit "skipped" result rather than silently vanishing). A directory
    is scanned for supported-extension files only, one level deep unless
    ``recursive`` is set.
    """
    if input_path.is_file():
        return [input_path]

    pattern_fn = input_path.rglob if recursive else input_path.glob
    files = [
        p
        for p in pattern_fn("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def new_session(model_name: str):
    """Create a rembg inference session for ``model_name``.

    Imports rembg lazily so simply importing this module (e.g. from a test
    that only exercises file-discovery logic) never triggers rembg's own
    heavier imports (onnxruntime, opencv, scikit-image) or a model download.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model {model_name!r}. Available models: "
            f"{', '.join(AVAILABLE_MODELS)}"
        )
    import rembg

    return rembg.new_session(model_name)


def remove_background_bytes(data: bytes, session) -> bytes:
    """Run background removal on raw image bytes, returning RGBA PNG bytes."""
    import rembg

    return rembg.remove(data, session=session)


def process_one(path: Path, output_dir: Path, base_dir: Path, session) -> FileResult:
    """Process a single file, never raising -- every failure mode becomes a
    ``FileResult(status="skipped", reason=...)`` so one bad file in a batch
    can never abort the rest of the run.
    """
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return FileResult(path=path, status="skipped", reason="unsupported_extension")

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return FileResult(path=path, status="skipped", reason=f"read_error: {exc}")

    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        return FileResult(path=path, status="skipped", reason=f"decode_error: {exc}")

    try:
        result_bytes = remove_background_bytes(raw, session)
    except Exception as exc:  # noqa: BLE001 -- any rembg/onnxruntime failure must degrade to a skip, not crash the batch
        return FileResult(path=path, status="skipped", reason=f"removal_error: {exc}")

    try:
        relative = path.relative_to(base_dir)
    except ValueError:
        relative = Path(path.name)
    output_path = (output_dir / relative).with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        output_path.write_bytes(result_bytes)
    except OSError as exc:
        return FileResult(path=path, status="skipped", reason=f"write_error: {exc}")

    return FileResult(path=path, status="processed", output_path=output_path)


def process_batch(
    input_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    recursive: bool = False,
    progress_callback: Callable[[FileResult], None] | None = None,
) -> BatchResult:
    """Run background removal over every discovered input file.

    Raises ``ValueError`` up front for an unknown ``model_name`` (a config
    mistake, not a per-file problem); every per-file failure after that is
    captured as a skipped ``FileResult`` instead of raising.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model {model_name!r}. Available models: "
            f"{', '.join(AVAILABLE_MODELS)}"
        )

    files = resolve_input_files(input_path, recursive)
    if not files:
        return BatchResult(results=[])

    base_dir = input_path if input_path.is_dir() else input_path.parent
    session = new_session(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = BatchResult()
    for path in files:
        result = process_one(path, output_dir, base_dir, session)
        batch.results.append(result)
        if progress_callback is not None:
            progress_callback(result)
    return batch
