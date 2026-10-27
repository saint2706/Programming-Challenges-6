"""Shared helpers: synthetic audio in many containers, and a few text builders."""

import wave
from pathlib import Path

import av
import numpy as np

RATE = 16_000


def tone(
    seconds: float, freq: float = 440.0, amplitude: float = 0.4, rate: int = RATE
) -> np.ndarray:
    """int16 mono tone. Deterministic, so the same arguments always give the same bytes."""
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def write_wav(path: Path, samples: np.ndarray, rate: int = RATE) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype("<i2").tobytes())
    return path


# (container format, codec) pairs the PyAV wheels can encode. Tests skip any that are missing.
ENCODED_FORMATS = {
    ".m4a": ("ipod", "aac"),
    ".mp3": ("mp3", "libmp3lame"),
    ".flac": ("flac", "flac"),
    ".ogg": ("ogg", "libvorbis"),
    ".opus": ("ogg", "libopus"),
}


def write_encoded(path: Path, samples: np.ndarray, rate: int = RATE) -> Path:
    """Encode int16 mono samples with PyAV into the container implied by ``path``'s suffix."""
    fmt, codec = ENCODED_FORMATS[path.suffix]
    container = av.open(str(path), mode="w", format=fmt)
    try:
        stream = container.add_stream(codec, rate=rate)
        stream.layout = "mono"
        resampler = av.AudioResampler(
            format=stream.codec_context.format.name, layout="mono", rate=rate
        )
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = rate
        frame.pts = 0
        for out in resampler.resample(frame):
            for packet in stream.encode(out):
                container.mux(packet)
        for out in resampler.resample(None):
            for packet in stream.encode(out):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    return path


def encoder_available(suffix: str) -> bool:
    _fmt, codec = ENCODED_FORMATS[suffix]
    try:
        av.codec.Codec(codec, "w")
    except Exception:  # noqa: BLE001 - PyAV raises a generic error for unknown codecs
        return False
    return True
