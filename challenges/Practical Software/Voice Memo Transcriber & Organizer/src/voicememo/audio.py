"""Decode any common memo format to 16 kHz mono float32, streamed in chunks.

PyAV bundles its own FFmpeg, so no system ffmpeg is needed. Long memos are yielded in
fixed-size chunks so memory stays bounded; the transcriber runs VAD inside each chunk.
"""

from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 600.0  # 10 minutes of audio per transcription call
MAX_DECODED_SECONDS = 12 * 3600.0  # longer memos are refused, so decoding stays bounded
# Only local files; FFmpeg would otherwise open concat:, http:, pipe: and similar names.
PROTOCOL_WHITELIST = "file"


class DecodeError(Exception):
    """The file is not readable audio (corrupt, empty, too long, or has no audio stream)."""


def iter_chunks(
    path: Path,
    chunk_seconds: float = CHUNK_SECONDS,
    max_seconds: float = MAX_DECODED_SECONDS,
) -> Iterator[np.ndarray]:
    """Yield float32 mono 16 kHz chunks of at most ``chunk_seconds`` each."""
    try:
        container = av.open(
            str(path.resolve()), options={"protocol_whitelist": PROTOCOL_WHITELIST}
        )
    except (av.error.FFmpegError, OSError) as exc:
        raise DecodeError(f"cannot open audio: {exc}") from exc
    try:
        streams = container.streams.audio
        if not streams:
            raise DecodeError("no audio stream in file")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        chunk_len = int(chunk_seconds * SAMPLE_RATE)
        max_len = int(max_seconds * SAMPLE_RATE)
        pending: list[np.ndarray] = []
        pending_len = 0
        total = 0
        seen = 0  # every decoded sample, checked against the duration cap as it arrives
        try:
            for frame in container.decode(streams[0]):
                for out in resampler.resample(frame):
                    block = out.to_ndarray().reshape(-1)
                    seen += block.size
                    if seen > max_len:
                        raise DecodeError(
                            f"audio longer than {max_seconds / 3600:g} hours; refused"
                        )
                    pending.append(block)
                    pending_len += block.size
                    while pending_len >= chunk_len:
                        joined = np.concatenate(pending)
                        yield _to_float(joined[:chunk_len])
                        total += chunk_len
                        pending = [joined[chunk_len:]]
                        pending_len = pending[0].size
            for out in resampler.resample(None):
                block = out.to_ndarray().reshape(-1)
                pending.append(block)
                pending_len += block.size
        except av.error.FFmpegError as exc:
            raise DecodeError(f"corrupt audio data: {exc}") from exc
        if pending_len:
            yield _to_float(np.concatenate(pending))
            total += pending_len
        if total == 0:
            raise DecodeError("file contains no audio samples")
    finally:
        container.close()


def _to_float(block: np.ndarray) -> np.ndarray:
    return block.astype(np.float32) / 32768.0


def load_audio(path: Path, chunk_seconds: float = CHUNK_SECONDS) -> np.ndarray:
    """Whole file as one array (tests and short clips only)."""
    parts = list(iter_chunks(path, chunk_seconds))
    return np.concatenate(parts)
