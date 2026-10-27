import numpy as np
import pytest
from voicememo.asr import (
    PARAKEET_LANGUAGES,
    FakeTranscriber,
    LanguageUnsupportedError,
    normalize_language,
    route_backend,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("auto", None),
        ("AUTO", None),
        ("", None),
        ("   ", None),
        (" EN ", "en"),
        ("en-US", "en"),
        ("zh-hant", "zh"),
        ("de", "de"),
        ("fil", "fil"),
    ],
)
def test_normalize_language_accepts_codes(raw, expected):
    assert normalize_language(raw) == expected


@pytest.mark.parametrize("raw", ["english", "e", "1x", "en_US", "en-", "<script>"])
def test_normalize_language_rejects_non_codes(raw):
    with pytest.raises(ValueError):
        normalize_language(raw)


def test_parakeet_covers_exactly_25_languages():
    assert len(PARAKEET_LANGUAGES) == 25
    assert {"en", "de", "fr", "uk"} <= PARAKEET_LANGUAGES
    assert "ja" not in PARAKEET_LANGUAGES


@pytest.mark.parametrize("language", ["en", "de", "uk", None])
def test_parakeet_languages_and_unknown_route_to_parakeet(language):
    assert route_backend(language, whisper_available=False) == "parakeet"
    assert route_backend(language, whisper_available=True) == "parakeet"


@pytest.mark.parametrize("language", ["ja", "zh", "ar", "hi"])
def test_other_languages_route_to_whisper_when_installed(language):
    assert route_backend(language, whisper_available=True) == "whisper"


@pytest.mark.parametrize("language", ["ja", "zh", "ar"])
def test_other_languages_fail_clearly_without_whisper(language):
    with pytest.raises(LanguageUnsupportedError, match="whisper"):
        route_backend(language, whisper_available=False)


def test_fake_transcriber_splits_sentences_with_word_times():
    fake = FakeTranscriber({"memo": "One two three. Four five."})
    audio = np.zeros(16_000 * 4, dtype=np.int16)
    segs = fake.transcribe(audio, language=None, source_name="memo")
    assert [s.text for s in segs] == ["One two three.", "Four five."]
    assert segs[0].start == 0.0
    assert segs[-1].end == pytest.approx(4.0)
    assert [w.text for w in segs[0].words] == ["One", "two", "three."]


def test_fake_word_times_are_monotone_and_inside_segment():
    fake = FakeTranscriber(
        {"m": "Alpha beta gamma delta. Epsilon zeta eta theta iota."}
    )
    segs = fake.transcribe(
        np.zeros(16_000 * 10, dtype=np.int16), language=None, source_name="m"
    )
    for seg in segs:
        starts = [w.start for w in seg.words]
        assert starts == sorted(starts)
        assert all(seg.start <= w.start <= w.end <= seg.end + 1e-6 for w in seg.words)


def test_fake_transcriber_uses_default_for_unknown_source():
    fake = FakeTranscriber(default_text="Fallback text.")
    segs = fake.transcribe(
        np.zeros(16_000, dtype=np.int16), language=None, source_name="nope"
    )
    assert segs[0].text == "Fallback text."


def test_fake_transcriber_records_calls():
    fake = FakeTranscriber({"a": "x."})
    fake.transcribe(np.zeros(16_000, dtype=np.int16), language="en", source_name="a")
    fake.transcribe(np.zeros(16_000, dtype=np.int16), language="en", source_name="b")
    assert fake.calls == ["a", "b"]


def test_fake_transcriber_empty_text_or_silent_audio_gives_no_segments():
    audio = np.zeros(16_000, dtype=np.int16)
    assert (
        FakeTranscriber({"m": "   "}).transcribe(audio, language=None, source_name="m")
        == []
    )
    assert (
        FakeTranscriber({"m": "text."}).transcribe(
            np.zeros(0, dtype=np.int16), language=None
        )
        == []
    )


def test_fake_transcriber_keeps_hostile_text_verbatim():
    # Sanitising is the export/display layer's job, not the transcriber's.
    text = "<script>alert(1)</script> and a \x1b[31m red word."
    fake = FakeTranscriber({"m": text})
    seg = fake.transcribe(
        np.zeros(16_000, dtype=np.int16), language=None, source_name="m"
    )[0]
    assert seg.text == text
