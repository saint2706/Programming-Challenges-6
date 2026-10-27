"""Real speech backends: Parakeet TDT v3 through onnx-asr (default) and whisper.cpp (fallback).

Models are never downloaded here. ``voicememo models pull`` fetches them into the models folder
and writes a ``.ready`` marker; these constructors refuse to run without the marker, so a
transcription can never start a silent multi-hundred-megabyte download.
"""

from pathlib import Path

import numpy as np

from voicememo.asr import Segment, Word
from voicememo.paths import models_dir

PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v3"
PARAKEET_QUANTIZATION = "int8"
VAD_MODEL = "silero"
READY_MARKER = ".ready"
SENTENCEPIECE_SPACE = "▁"


class ModelMissingError(Exception):
    """A model is not pulled yet. The message names the command that fixes it."""


def parakeet_dir() -> Path:
    return models_dir() / "parakeet"


def vad_dir() -> Path:
    return models_dir() / "silero"


def whisper_dir() -> Path:
    return models_dir() / "whisper"


def whisper_available(name: str = "base") -> bool:
    return (whisper_dir() / f"ggml-{name}.bin").is_file()


def require_marker(directory: Path, what: str, command: str) -> None:
    if not (directory / READY_MARKER).is_file():
        raise ModelMissingError(
            f"{what} is not installed; run `{command}` once (it downloads the model)"
        )


def _words_from_tokens(
    seg_start: float, seg_end: float, tokens: list[str], times: list[float]
) -> tuple[Word, ...]:
    """Group SentencePiece tokens into words. A token starting with U+2581 begins a new word."""
    starts: list[tuple[str, float]] = []
    for token, t in zip(tokens, times, strict=False):
        if not token or (token.startswith("<") and token.endswith(">")):
            continue
        if token.startswith(SENTENCEPIECE_SPACE) or not starts:
            starts.append(
                (token.replace(SENTENCEPIECE_SPACE, ""), seg_start + float(t))
            )
        else:
            text, start = starts[-1]
            starts[-1] = (text + token, start)
    words: list[Word] = []
    for i, (text, start) in enumerate(starts):
        if not text:
            continue
        end = starts[i + 1][1] if i + 1 < len(starts) else seg_end
        words.append(Word(text, round(start, 3), round(max(end, start), 3)))
    return tuple(words)


class ParakeetTranscriber:
    """NVIDIA Parakeet TDT 0.6B v3 (int8 ONNX) with Silero VAD and per-token timestamps."""

    name = "parakeet"

    def __init__(self, directory: Path | None = None) -> None:
        import onnx_asr

        base = directory or parakeet_dir()
        require_marker(base, "Parakeet TDT v3", "voicememo models pull --parakeet")
        vad = onnx_asr.load_vad(VAD_MODEL, path=str(vad_dir()))
        model = onnx_asr.load_model(
            PARAKEET_MODEL, path=str(base), quantization=PARAKEET_QUANTIZATION
        )
        self._model = model.with_vad(vad).with_timestamps()

    def transcribe(
        self, audio: np.ndarray, *, language: str | None, source_name: str = ""
    ) -> list[Segment]:
        segments: list[Segment] = []
        for res in self._model.recognize(audio.astype(np.float32), sample_rate=16_000):
            text = res.text.strip()
            if not text:
                continue
            words = _words_from_tokens(
                res.start, res.end, res.tokens or [], res.timestamps or []
            )
            segments.append(
                Segment(round(res.start, 3), round(res.end, 3), text, words)
            )
        return segments


class WhisperCppTranscriber:
    """whisper.cpp through pywhispercpp. Used for languages Parakeet does not cover.

    Word timestamps are not produced: segments carry no word list, so exports use segment times.
    """

    name = "whisper"

    def __init__(self, model: str = "base", directory: Path | None = None) -> None:
        from pywhispercpp.model import Model

        base = directory or whisper_dir()
        if not (base / f"ggml-{model}.bin").is_file():
            raise ModelMissingError(
                f"whisper.cpp model {model!r} is not installed; run `voicememo models pull --whisper {model}`"
            )
        self._model = Model(
            model,
            models_dir=str(base),
            redirect_whispercpp_logs_to=None,
            print_progress=False,
            print_realtime=False,
        )

    def detect_language(self, audio: np.ndarray) -> str | None:
        """Identify the language from the first 30 seconds."""
        (code, _prob), _all = self._model.auto_detect_language(
            audio[: 30 * 16_000].astype(np.float32)
        )
        return code or None

    def transcribe(
        self, audio: np.ndarray, *, language: str | None, source_name: str = ""
    ) -> list[Segment]:
        raw = self._model.transcribe(
            audio.astype(np.float32),
            language=language or "",
            print_progress=False,
            print_realtime=False,
        )
        segments: list[Segment] = []
        for seg in raw:
            text = seg.text.strip()
            if not text:
                continue
            # whisper.cpp reports t0/t1 in 10 ms units (centiseconds).
            segments.append(
                Segment(round(seg.t0 / 100.0, 3), round(seg.t1 / 100.0, 3), text, ())
            )
        return segments
