import numpy as np
import pytest
from helpers import ENCODED_FORMATS, encoder_available, tone, write_encoded, write_wav
from voicememo.audio import SAMPLE_RATE, DecodeError, iter_chunks, load_audio


def test_wav_roundtrip_length_and_range(tmp_path):
    samples = tone(2.0)
    path = write_wav(tmp_path / "a.wav", samples)
    audio = load_audio(path)
    assert audio.dtype == np.float32
    assert len(audio) == len(samples)
    assert np.max(np.abs(audio)) <= 1.0


def test_wav_is_resampled_to_16k(tmp_path):
    path = write_wav(tmp_path / "hi.wav", tone(1.0, rate=44_100, freq=300), rate=44_100)
    audio = load_audio(path)
    assert abs(len(audio) - SAMPLE_RATE) < 200


def test_stereo_is_downmixed(tmp_path):
    import wave

    path = tmp_path / "stereo.wav"
    left = tone(1.0)
    interleaved = np.empty(left.size * 2, dtype="<i2")
    interleaved[0::2] = left
    interleaved[1::2] = left
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(interleaved.tobytes())
    audio = load_audio(path)
    assert abs(len(audio) - SAMPLE_RATE) < 200


@pytest.mark.parametrize("suffix", sorted(ENCODED_FORMATS))
def test_compressed_formats_decode(tmp_path, suffix):
    if not encoder_available(suffix):
        pytest.skip(f"no encoder for {suffix} in this PyAV build")
    path = write_encoded(tmp_path / f"memo{suffix}", tone(1.5))
    audio = load_audio(path)
    # Lossy codecs add priming and padding, so allow a generous margin.
    assert abs(len(audio) - int(1.5 * SAMPLE_RATE)) < 0.25 * SAMPLE_RATE
    assert np.abs(audio).max() > 0.05


@pytest.mark.parametrize("chunk_seconds", [0.5, 1.0, 2.5])
def test_chunks_cover_whole_file_without_gaps(tmp_path, chunk_seconds):
    samples = tone(3.0)
    path = write_wav(tmp_path / "long.wav", samples)
    chunks = list(iter_chunks(path, chunk_seconds=chunk_seconds))
    assert sum(len(c) for c in chunks) == len(samples)
    for chunk in chunks[:-1]:
        assert len(chunk) == int(chunk_seconds * SAMPLE_RATE)
    assert np.allclose(np.concatenate(chunks), load_audio(path))


def test_chunk_boundary_on_exact_multiple_has_no_empty_tail(tmp_path):
    path = write_wav(tmp_path / "exact.wav", tone(2.0))
    chunks = list(iter_chunks(path, chunk_seconds=1.0))
    assert [len(c) for c in chunks] == [SAMPLE_RATE, SAMPLE_RATE]


def test_zero_length_wav_is_a_clean_error(tmp_path):
    path = write_wav(tmp_path / "empty.wav", np.zeros(0, dtype=np.int16))
    with pytest.raises(DecodeError):
        load_audio(path)


def test_zero_byte_file_is_a_clean_error(tmp_path):
    path = tmp_path / "zero.m4a"
    path.write_bytes(b"")
    with pytest.raises(DecodeError):
        load_audio(path)


def test_random_bytes_named_as_audio_are_rejected(tmp_path):
    path = tmp_path / "junk.mp3"
    path.write_bytes(bytes(range(256)) * 40)
    with pytest.raises(DecodeError):
        load_audio(path)


def test_text_file_is_rejected(tmp_path):
    path = tmp_path / "notes.m4a"
    path.write_text("this is not audio\n" * 20, encoding="utf-8")
    with pytest.raises(DecodeError):
        load_audio(path)


def test_missing_file_is_a_decode_error(tmp_path):
    with pytest.raises(DecodeError):
        load_audio(tmp_path / "does-not-exist.wav")


def test_truncated_wav_still_yields_what_is_readable(tmp_path):
    path = write_wav(tmp_path / "cut.wav", tone(2.0))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    audio = load_audio(path)
    assert 0 < len(audio) < 2 * SAMPLE_RATE


def test_very_long_input_streams_in_bounded_chunks(tmp_path):
    # 30 minutes of silence in 10-minute chunks: three yields, never the whole file at once.
    path = write_wav(
        tmp_path / "long.wav", np.zeros(30 * 60 * SAMPLE_RATE, dtype=np.int16)
    )
    sizes = [len(c) for c in iter_chunks(path)]
    assert sizes == [600 * SAMPLE_RATE] * 3


def test_directory_is_not_audio(tmp_path):
    with pytest.raises(DecodeError):
        load_audio(tmp_path)


def test_audio_without_stream_wrapper_names_the_cause(tmp_path):
    path = tmp_path / "x.wav"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 64)
    with pytest.raises(DecodeError, match="(cannot open|no audio|corrupt)"):
        load_audio(path)
