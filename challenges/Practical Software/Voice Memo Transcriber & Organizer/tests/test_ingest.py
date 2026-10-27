import pytest
import voicememo.ingest as ingest_mod
from helpers import tone, write_wav
from voicememo.asr import FakeTranscriber, LanguageUnsupportedError, Segment, Word
from voicememo.audio import DecodeError, iter_chunks
from voicememo.ingest import (
    FixedEngine,
    RoutingEngine,
    ingest_file,
    retranscribe,
    transcribe_file,
)
from voicememo.library import LibraryError


@pytest.fixture
def wav(tmp_path):
    return write_wav(tmp_path / "walk.wav", tone(2.0))


def fake_engine(script=None):
    fake = FakeTranscriber(script or {"walk": "Walked the dog. The river was cold."})
    return FixedEngine(fake), fake


def test_ingest_adds_memo_with_segments(library, wav):
    engine, _ = fake_engine()
    result = ingest_file(library, wav, engine)
    assert result.status == "added"
    memo = library.get(result.memo_id)
    assert memo.status == "done"
    assert memo.backend == "fake"
    assert memo.duration == pytest.approx(2.0, abs=0.01)
    assert [s["text"] for s in library.segments(memo.id)] == [
        "Walked the dog.",
        "The river was cold.",
    ]


def test_ingest_copies_audio_under_hash_name(library, wav):
    engine, _ = fake_engine()
    result = ingest_file(library, wav, engine)
    memo = library.get(result.memo_id)
    stored = library.audio_path(memo)
    assert stored.is_file()
    assert stored.read_bytes() == wav.read_bytes()
    assert memo.sha256 in stored.name


def test_second_ingest_of_same_bytes_is_a_noop(library, wav):
    engine, fake = fake_engine()
    first = ingest_file(library, wav, engine)
    second = ingest_file(library, wav, engine)
    assert first.status == "added"
    assert second.status == "skipped"
    assert second.memo_id == first.memo_id
    assert fake.calls == ["walk"]


def test_same_bytes_under_new_name_is_deduplicated(library, tmp_path, wav):
    engine, fake = fake_engine()
    ingest_file(library, wav, engine)
    copy = tmp_path / "renamed-copy.wav"
    copy.write_bytes(wav.read_bytes())
    result = ingest_file(library, copy, engine)
    assert result.status == "skipped"
    assert len(library.list_memos()) == 1
    assert fake.calls == ["walk"]


def test_corrupt_audio_is_recorded_as_failed_and_copy_removed(library, tmp_path):
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(b"not audio at all" * 50)
    engine, _ = fake_engine()
    result = ingest_file(library, bad, engine)
    assert result.status == "failed"
    memo = library.get(result.memo_id)
    assert memo.status == "failed"
    assert memo.error
    assert not library.audio_path(memo).exists()


def test_failed_memo_is_skipped_unless_retry_requested(library, tmp_path):
    class FlakyOnce:
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, *, language, source_name=""):
            self.calls += 1
            if self.calls == 1:
                raise DecodeError("transient decoder hiccup")
            return [Segment(0.0, 1.0, "recovered text")]

    wav = write_wav(tmp_path / "flaky.wav", tone(1.0))
    flaky = FlakyOnce()
    engine = FixedEngine(flaky)
    first = ingest_file(library, wav, engine)
    assert first.status == "failed"
    assert ingest_file(library, wav, engine).status == "skipped"
    retried = ingest_file(library, wav, engine, retry_failed=True)
    assert retried.status == "added"
    assert library.transcript(retried.memo_id) == "recovered text"


def test_configuration_errors_do_not_leave_a_pending_row(library, wav):
    class Unavailable:
        name = "none"

        def transcribe(self, audio, *, language, source_name=""):
            raise LanguageUnsupportedError("no backend for that language")

    with pytest.raises(LanguageUnsupportedError):
        ingest_file(library, wav, FixedEngine(Unavailable()))
    assert library.find_by_sha(_sha(wav)) is None
    assert library.all_ids() == []


def _sha(path):
    from voicememo.library import file_sha256

    return file_sha256(path)


def test_language_is_normalised_before_storage(library, wav):
    engine, _ = fake_engine()
    result = ingest_file(library, wav, engine, language="EN-us")
    assert library.get(result.memo_id).language == "en"


def test_invalid_language_is_rejected_before_any_row_exists(library, wav):
    engine, _ = fake_engine()
    with pytest.raises(ValueError):
        ingest_file(library, wav, engine, language="english")
    assert library.list_memos() == []


def test_missing_file_raises_file_not_found(library, tmp_path):
    engine, _ = fake_engine()
    with pytest.raises(FileNotFoundError):
        ingest_file(library, tmp_path / "gone.wav", engine)


def test_engine_is_asked_once_per_memo(library, tmp_path, monkeypatch):
    long_wav = write_wav(tmp_path / "walk.wav", tone(3.0))
    real_iter = iter_chunks
    monkeypatch.setattr(
        ingest_mod, "iter_chunks", lambda p: real_iter(p, chunk_seconds=1.0)
    )
    calls = []

    class Recording(FixedEngine):
        def select(self, language, first_chunk):
            calls.append(len(first_chunk))
            return super().select(language, first_chunk)

    ingest_file(library, long_wav, Recording(FakeTranscriber({"walk": "a b c."})))
    assert len(calls) == 1


def test_chunk_offsets_shift_later_segments(tmp_path, monkeypatch):
    path = write_wav(tmp_path / "walk.wav", tone(3.0))
    real_iter = iter_chunks
    monkeypatch.setattr(
        ingest_mod, "iter_chunks", lambda p: real_iter(p, chunk_seconds=1.0)
    )
    segments, duration, _lang, backend = transcribe_file(
        path, FixedEngine(FakeTranscriber({"walk": "one two."})), None
    )
    assert duration == pytest.approx(3.0, abs=0.01)
    assert backend == "fake"
    starts = [round(s.start) for s in segments]
    assert starts == [0, 1, 2]
    assert segments[-1].start >= 2.0


def test_chunk_offset_shifts_word_times_too(tmp_path, monkeypatch):
    path = write_wav(tmp_path / "walk.wav", tone(2.0))
    real_iter = iter_chunks
    monkeypatch.setattr(
        ingest_mod, "iter_chunks", lambda p: real_iter(p, chunk_seconds=1.0)
    )
    segments, *_ = transcribe_file(
        path, FixedEngine(FakeTranscriber({"walk": "alpha beta."})), None
    )
    second = segments[1]
    assert second.words[0].start >= 1.0 - 1e-6
    assert all(w.end <= 2.0 + 1e-6 for w in second.words)


def test_transcribe_file_rejects_zero_length_audio(tmp_path):
    path = write_wav(tmp_path / "empty.wav", tone(0.0))
    with pytest.raises(DecodeError):
        transcribe_file(path, FixedEngine(FakeTranscriber()), None)


def test_retranscribe_replaces_segments_and_keeps_tags(library, wav):
    engine, _ = fake_engine()
    memo_id = ingest_file(library, wav, engine).memo_id
    library.add_tag(memo_id, "keep")
    result = retranscribe(
        library, memo_id, FixedEngine(FakeTranscriber({"walk": "Entirely new words."}))
    )
    assert result.status == "added"
    assert library.transcript(memo_id) == "Entirely new words."
    assert library.tags(memo_id) == [("keep", "user")]


def test_retranscribe_unknown_memo_raises(library):
    with pytest.raises(LibraryError):
        retranscribe(library, 404, FixedEngine(FakeTranscriber()))


def test_retranscribe_with_missing_audio_raises_decode_error(library, wav):
    engine, _ = fake_engine()
    memo_id = ingest_file(library, wav, engine).memo_id
    library.audio_path(library.get(memo_id)).unlink()
    with pytest.raises(DecodeError, match="missing"):
        retranscribe(library, memo_id, engine)


class _Whisper:
    def __init__(self, detected=None):
        self.detected = detected

    def detect_language(self, audio):
        return self.detected


def test_routing_uses_parakeet_when_whisper_missing(monkeypatch):
    engine = RoutingEngine()
    sentinel = object()
    monkeypatch.setattr(RoutingEngine, "_whisper_ready", lambda self: False)
    monkeypatch.setattr(RoutingEngine, "_get_parakeet", lambda self: sentinel)
    chosen, lang = engine.select(None, tone(1.0))
    assert chosen is sentinel
    assert lang is None


def test_routing_sends_unlisted_language_to_whisper(monkeypatch):
    engine = RoutingEngine()
    whisper = object()
    monkeypatch.setattr(RoutingEngine, "_whisper_ready", lambda self: True)
    monkeypatch.setattr(RoutingEngine, "_get_whisper", lambda self: whisper)
    chosen, lang = engine.select("ja", tone(1.0))
    assert chosen is whisper and lang == "ja"


def test_routing_detects_language_with_whisper_when_unspecified(monkeypatch):
    engine = RoutingEngine()
    monkeypatch.setattr(RoutingEngine, "_whisper_ready", lambda self: True)
    monkeypatch.setattr(RoutingEngine, "_get_whisper", lambda self: _Whisper("ja"))
    parakeet = object()
    monkeypatch.setattr(RoutingEngine, "_get_parakeet", lambda self: parakeet)
    chosen, lang = engine.select(None, tone(1.0))
    assert lang == "ja"
    assert chosen is not parakeet


def test_routing_keeps_parakeet_for_detected_european_language(monkeypatch):
    engine = RoutingEngine()
    monkeypatch.setattr(RoutingEngine, "_whisper_ready", lambda self: True)
    monkeypatch.setattr(RoutingEngine, "_get_whisper", lambda self: _Whisper("de"))
    parakeet = object()
    monkeypatch.setattr(RoutingEngine, "_get_parakeet", lambda self: parakeet)
    chosen, lang = engine.select(None, tone(1.0))
    assert chosen is parakeet and lang == "de"


def test_word_objects_survive_shift():
    seg = Segment(0.0, 1.0, "hi", (Word("hi", 0.0, 1.0),))
    shifted = ingest_mod._shift(seg, 2.5)
    assert shifted.start == 2.5
    assert shifted.words[0].end == 3.5


def test_oversized_file_is_refused_before_hashing(library, wav):
    from voicememo.library import file_sha256

    engine, _ = fake_engine()
    with pytest.raises(DecodeError, match="limit"):
        ingest_file(library, wav, engine, max_bytes=10)
    assert library.find_by_sha(file_sha256(wav)) is None
