"""Transcriber protocol, shared result types, language routing and a deterministic fake backend."""

import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

# Languages NVIDIA Parakeet TDT 0.6B v3 is trained on (25 European languages). Everything else
# is routed to whisper.cpp, which covers roughly 99 languages.
PARAKEET_LANGUAGES = frozenset(
    [
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "hr",
        "hu",
        "it",
        "lt",
        "lv",
        "mt",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "uk",
    ]
)


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = field(default=())


class Transcriber(Protocol):
    """A speech-to-text backend. Times are seconds relative to the audio passed in."""

    name: str

    def transcribe(
        self, audio: np.ndarray, *, language: str | None, source_name: str = ""
    ) -> list[Segment]: ...


def normalize_language(code: str | None) -> str | None:
    """'auto', '', None -> None (detect); otherwise a lower-case two-letter code."""
    if code is None:
        return None
    value = code.strip().lower()
    if value in {"", "auto"}:
        return None
    if not re.fullmatch(r"[a-z]{2,3}(-[a-z]{2,4})?", value):
        raise ValueError(f"not a language code: {code!r}")
    return value.split("-")[0]


def route_backend(language: str | None, whisper_available: bool) -> str:
    """Pick the backend for a language. ``language`` is already normalised (None = unknown)."""
    if language is None or language in PARAKEET_LANGUAGES:
        return "parakeet"
    if whisper_available:
        return "whisper"
    raise LanguageUnsupportedError(
        f"language {language!r} is not in Parakeet's list; install the whisper.cpp model with "
        "`voicememo models pull --whisper base`"
    )


class LanguageUnsupportedError(Exception):
    """No installed backend can transcribe the requested language."""


class FakeTranscriber:
    """Deterministic stand-in for tests and for exercising the pipeline without models.

    ``script`` maps a source name (the memo file's stem) to its transcript. Text is split into
    sentences and each sentence gets a time span proportional to its word count across the audio.
    """

    name = "fake"

    def __init__(
        self, script: dict[str, str] | None = None, default_text: str = ""
    ) -> None:
        self.script = dict(script or {})
        self.default_text = default_text
        self.calls: list[str] = []

    def transcribe(
        self, audio: np.ndarray, *, language: str | None, source_name: str = ""
    ) -> list[Segment]:
        self.calls.append(source_name)
        text = self.script.get(source_name, self.default_text).strip()
        duration = len(audio) / 16_000
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences or duration <= 0:
            return []
        total_words = sum(len(s.split()) for s in sentences) or 1
        segments: list[Segment] = []
        cursor = 0.0
        for sentence in sentences:
            words = sentence.split()
            span = duration * len(words) / total_words
            step = span / len(words)
            word_objs = tuple(
                Word(w, round(cursor + i * step, 3), round(cursor + (i + 1) * step, 3))
                for i, w in enumerate(words)
            )
            segments.append(
                Segment(round(cursor, 3), round(cursor + span, 3), sentence, word_objs)
            )
            cursor += span
        return segments
