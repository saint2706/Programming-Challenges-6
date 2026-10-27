"""Ingest pipeline: hash -> dedupe -> copy -> decode -> transcribe -> store (idempotent by content hash)."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from voicememo.asr import Segment, Transcriber, normalize_language, route_backend
from voicememo.audio import SAMPLE_RATE, DecodeError, iter_chunks
from voicememo.library import Library, file_sha256

MAX_FILE_BYTES = (
    2 * 1024**3
)  # refused before hashing or copying, so a huge file costs nothing


class Engine(Protocol):
    """Chooses the transcriber for a memo. Called once per memo with its first audio chunk."""

    def select(
        self, language: str | None, first_chunk: np.ndarray
    ) -> tuple[Transcriber, str | None]: ...


class FixedEngine:
    """Always uses one transcriber (tests and the fake backend)."""

    def __init__(self, transcriber: Transcriber) -> None:
        self.transcriber = transcriber

    def select(
        self, language: str | None, first_chunk: np.ndarray
    ) -> tuple[Transcriber, str | None]:
        return self.transcriber, language


class RoutingEngine:
    """Parakeet by default; whisper.cpp for languages Parakeet lacks or when detection says so.

    With no language given and whisper.cpp installed, the language is detected from the first
    30 seconds with whisper.cpp; otherwise the memo goes to Parakeet and its language is stored
    as unknown.
    """

    def __init__(self, whisper_model: str = "base") -> None:
        self.whisper_model = whisper_model
        self._parakeet: Transcriber | None = None
        self._whisper = None

    def _whisper_ready(self) -> bool:
        from voicememo.backends import whisper_available

        return whisper_available(self.whisper_model)

    def _get_whisper(self):
        if self._whisper is None:
            from voicememo.backends import WhisperCppTranscriber

            self._whisper = WhisperCppTranscriber(self.whisper_model)
        return self._whisper

    def _get_parakeet(self) -> Transcriber:
        if self._parakeet is None:
            from voicememo.backends import ParakeetTranscriber

            self._parakeet = ParakeetTranscriber()
        return self._parakeet

    def select(
        self, language: str | None, first_chunk: np.ndarray
    ) -> tuple[Transcriber, str | None]:
        ready = self._whisper_ready()
        lang = language
        if lang is None and ready:
            lang = self._get_whisper().detect_language(first_chunk)
        if route_backend(lang, ready) == "whisper":
            return self._get_whisper(), lang
        return self._get_parakeet(), lang


@dataclass(frozen=True)
class IngestResult:
    status: str  # "added" | "skipped" | "failed"
    memo_id: int
    message: str = ""


def _shift(seg: Segment, offset: float) -> Segment:
    words = tuple(
        replace(w, start=round(w.start + offset, 3), end=round(w.end + offset, 3))
        for w in seg.words
    )
    return replace(
        seg,
        start=round(seg.start + offset, 3),
        end=round(seg.end + offset, 3),
        words=words,
    )


def transcribe_file(
    path: Path, engine: Engine, language: str | None, source_name: str | None = None
) -> tuple[list[Segment], float, str, str]:
    """Decode and transcribe a whole file. Returns (segments, duration, language, backend name).

    ``source_name`` is the name the memo was imported under; the stored audio file is named by hash.
    """
    segments: list[Segment] = []
    transcriber: Transcriber | None = None
    used_language: str | None = language
    offset = 0.0
    name = source_name if source_name is not None else path.stem
    for chunk in iter_chunks(path):
        if transcriber is None:
            transcriber, used_language = engine.select(language, chunk)
        for seg in transcriber.transcribe(
            chunk, language=used_language, source_name=name
        ):
            segments.append(_shift(seg, offset))
        offset += len(chunk) / SAMPLE_RATE
    if (
        transcriber is None
    ):  # iter_chunks raises before yielding nothing, so this is defensive
        raise DecodeError("file contains no audio samples")
    return segments, round(offset, 3), used_language or "", transcriber.name


def ingest_file(
    library: Library,
    path: Path,
    engine: Engine,
    *,
    language: str | None = None,
    retry_failed: bool = False,
    max_bytes: int = MAX_FILE_BYTES,
) -> IngestResult:
    """Add one audio file. Re-running on the same bytes is a no-op unless the earlier attempt failed."""
    lang = normalize_language(language)
    size = path.stat().st_size
    if size > max_bytes:
        raise DecodeError(
            f"file is {size / 2**30:.2f} GiB; the limit is {max_bytes / 2**30:g} GiB"
        )
    sha = file_sha256(path)
    existing = library.find_by_sha(sha)
    if existing is not None and (existing.status == "done" or not retry_failed):
        return IngestResult(
            "skipped", existing.id, f"already in library as #{existing.id}"
        )
    memo = existing or library.create_pending(sha, path.name, path.name)
    try:
        library.store_audio(path, memo)
        segments, duration, used_language, backend = transcribe_file(path, engine, lang)
    except DecodeError as exc:
        library.fail(memo.id, str(exc))
        if existing is None:
            library.audio_path(memo).unlink(missing_ok=True)
        return IngestResult("failed", memo.id, str(exc))
    except Exception:
        # Configuration problems (missing model, unsupported language) are not the memo's fault:
        # drop the pending row so the same file can be retried once the setup is fixed.
        if existing is None:
            library.delete(memo.id)
        raise
    library.complete(
        memo.id, segments, duration=duration, language=used_language, backend=backend
    )
    return IngestResult(
        "added", memo.id, f"{len(segments)} segments, {duration:.1f}s, {backend}"
    )


def retranscribe(
    library: Library, memo_id: int, engine: Engine, language: str | None = None
) -> IngestResult:
    """Run the pipeline again on the stored audio. Tags are kept; segments are replaced."""
    memo = library.get(memo_id)
    path = library.audio_path(memo)
    if not path.is_file():
        raise DecodeError(f"stored audio for memo #{memo_id} is missing")
    segments, duration, used_language, backend = transcribe_file(
        path,
        engine,
        normalize_language(language),
        source_name=Path(memo.original_name).stem,
    )
    library.complete(
        memo.id, segments, duration=duration, language=used_language, backend=backend
    )
    return IngestResult(
        "added", memo.id, f"re-transcribed: {len(segments)} segments, {backend}"
    )
