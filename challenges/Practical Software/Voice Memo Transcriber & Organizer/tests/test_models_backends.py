import pytest
from voicememo import models
from voicememo.backends import (
    READY_MARKER,
    ModelMissingError,
    ParakeetTranscriber,
    WhisperCppTranscriber,
    _words_from_tokens,
    require_marker,
    whisper_available,
)


def test_require_marker_names_the_fix_command(tmp_path):
    with pytest.raises(ModelMissingError, match="voicememo models pull --parakeet"):
        require_marker(tmp_path, "Parakeet TDT v3", "voicememo models pull --parakeet")


def test_require_marker_passes_once_marked(tmp_path):
    (tmp_path / READY_MARKER).write_text("ready\n", encoding="utf-8")
    require_marker(tmp_path, "x", "cmd")


def test_parakeet_refuses_to_start_without_models(tmp_path):
    with pytest.raises(ModelMissingError, match="models pull --parakeet"):
        ParakeetTranscriber(directory=tmp_path)


def test_whisper_refuses_to_start_without_model(tmp_path):
    with pytest.raises(ModelMissingError, match="models pull --whisper base"):
        WhisperCppTranscriber("base", directory=tmp_path)


def test_whisper_available_is_false_on_empty_models_dir():
    assert whisper_available("base") is False


def test_words_group_sentencepiece_tokens():
    words = _words_from_tokens(0.0, 2.0, ["▁hel", "lo", "▁world"], [0.0, 0.2, 1.0])
    assert [w.text for w in words] == ["hello", "world"]
    assert words[0].start == 0.0
    assert words[0].end == 1.0
    assert words[1].end == 2.0


def test_words_skip_special_tokens_and_empty_pieces():
    words = _words_from_tokens(0.0, 1.0, ["<blk>", "", "▁ok"], [0.0, 0.1, 0.2])
    assert [w.text for w in words] == ["ok"]


def test_words_never_end_before_they_start():
    words = _words_from_tokens(0.0, 0.5, ["▁a", "▁b"], [0.4, 0.1])
    assert all(w.end >= w.start for w in words)


def test_words_empty_input():
    assert _words_from_tokens(0.0, 1.0, [], []) == ()


def test_status_reports_every_model(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEMEMO_MODELS", str(tmp_path))
    status = models.status()
    assert set(status) == {"parakeet", "silero-vad", "whisper-base", "model2vec"}
    assert not any(status.values())


def test_is_ready_needs_marker(tmp_path):
    assert not models.is_ready(tmp_path)
    models._mark_ready(tmp_path)
    assert models.is_ready(tmp_path)


def test_fresh_dir_clears_interrupted_pull_but_keeps_ready_model(tmp_path):
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "half.onnx").write_bytes(b"x")
    models._fresh_dir(partial)
    assert not partial.exists()

    ready = tmp_path / "ready"
    models._mark_ready(ready)
    (ready / "model.onnx").write_bytes(b"x")
    models._fresh_dir(ready)
    assert (ready / "model.onnx").exists()


def test_pull_embedder_is_a_noop_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEMEMO_MODELS", str(tmp_path))
    models._mark_ready(tmp_path / "model2vec")
    lines: list[str] = []
    models.pull_embedder(log=lines.append)
    assert lines == ["model2vec already present"]


def test_pull_whisper_is_a_noop_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEMEMO_MODELS", str(tmp_path))
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / "ggml-base.bin").write_bytes(b"x")
    lines: list[str] = []
    models.pull_whisper("base", log=lines.append)
    assert lines == ["whisper base already present"]
