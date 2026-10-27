import hashlib
import json
from pathlib import Path

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
from voicememo.models import ModelIntegrityError, Pin


def _mark_ready(path: Path, files: dict[str, bytes]) -> None:
    """Write the files and the manifest the way a successful pull does."""
    path.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (path / name).write_bytes(data)
    manifest = {name: len(data) for name, data in files.items()}
    (path / READY_MARKER).write_text(json.dumps(manifest), encoding="utf-8")


def test_require_marker_names_the_fix_command(tmp_path):
    with pytest.raises(ModelMissingError, match="voicememo models pull --parakeet"):
        require_marker(tmp_path, "Parakeet TDT v3", "voicememo models pull --parakeet")


def test_require_marker_passes_once_marked(tmp_path):
    _mark_ready(tmp_path, {"a.bin": b"x"})
    require_marker(tmp_path, "x", "cmd")


def test_parakeet_refuses_to_start_without_models(tmp_path):
    with pytest.raises(ModelMissingError, match="models pull --parakeet"):
        ParakeetTranscriber(directory=tmp_path)


def test_parakeet_also_needs_the_vad_model(tmp_path):
    _mark_ready(tmp_path / "parakeet", {"a.onnx": b"x"})
    with pytest.raises(ModelMissingError, match="Silero VAD"):
        ParakeetTranscriber(directory=tmp_path / "parakeet")


def test_whisper_refuses_to_start_without_model(tmp_path):
    with pytest.raises(ModelMissingError, match="models pull --whisper base"):
        WhisperCppTranscriber("base", directory=tmp_path)


def test_whisper_refuses_names_outside_the_allow_list(tmp_path):
    with pytest.raises(ModelMissingError, match="unknown whisper.cpp model"):
        WhisperCppTranscriber("../../evil", directory=tmp_path)


def test_whisper_available_is_false_on_empty_models_dir():
    assert whisper_available("base") is False


def test_is_ready_needs_marker(tmp_path):
    assert not models.is_ready(tmp_path)
    _mark_ready(tmp_path, {"a.bin": b"x"})
    assert models.is_ready(tmp_path)


def test_is_ready_fails_when_a_listed_file_changed_size(tmp_path):
    _mark_ready(tmp_path, {"a.bin": b"hello"})
    (tmp_path / "a.bin").write_bytes(b"hel")  # truncated after the pull
    assert not models.is_ready(tmp_path)


def test_is_ready_fails_when_a_listed_file_is_missing(tmp_path):
    _mark_ready(tmp_path, {"a.bin": b"hello"})
    (tmp_path / "a.bin").unlink()
    assert not models.is_ready(tmp_path)


def test_is_ready_rejects_an_old_plain_text_marker(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x")
    (tmp_path / READY_MARKER).write_text("ready\n", encoding="utf-8")
    assert not models.is_ready(tmp_path)


def test_fresh_dir_clears_interrupted_pull_but_keeps_ready_model(tmp_path):
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "half.onnx").write_bytes(b"x")
    models._fresh_dir(partial)
    assert not partial.exists()

    ready = tmp_path / "ready"
    _mark_ready(ready, {"model.onnx": b"x"})
    models._fresh_dir(ready)
    assert (ready / "model.onnx").exists()


@pytest.fixture
def fake_hub(monkeypatch):
    """Stands in for huggingface_hub: writes the bytes in ``payloads`` and never touches the network."""
    payloads: dict[str, bytes] = {}

    def download(repo, filename, revision=None, local_dir=None, **_):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[filename])
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return payloads


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_fetch_writes_manifest_when_sha_matches(tmp_path, fake_hub):
    fake_hub["w.bin"] = b"good"
    pin = Pin("org/repo", "rev", {"w.bin": _sha(b"good")})
    models.fetch(pin, tmp_path / "dest", log=lambda _m: None)
    assert models.is_ready(tmp_path / "dest")


def test_fetch_discards_file_and_marks_nothing_on_sha_mismatch(tmp_path, fake_hub):
    fake_hub["w.bin"] = b"evil"
    pin = Pin("org/repo", "rev", {"w.bin": _sha(b"good")})
    dest = tmp_path / "dest"
    with pytest.raises(ModelIntegrityError, match="sha256"):
        models.fetch(pin, dest, log=lambda _m: None)
    assert not (dest / "w.bin").exists()
    assert not models.is_ready(dest)


def test_pull_whisper_rejects_unknown_names(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEMEMO_MODELS", str(tmp_path))
    with pytest.raises(ValueError, match="unknown whisper model"):
        models.pull_whisper("../../evil", log=lambda _m: None)


def test_status_reports_every_model(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEMEMO_MODELS", str(tmp_path))
    status = models.status()
    assert set(status) == {"parakeet", "silero-vad", "whisper-base", "model2vec"}
    assert not any(status.values())


def test_pull_embedder_is_a_noop_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEMEMO_MODELS", str(tmp_path))
    _mark_ready(tmp_path / "model2vec", {"model.safetensors": b"x"})
    lines: list[str] = []
    models.pull_embedder(log=lines.append)
    assert lines == ["model2vec already present"]


def test_pull_whisper_is_a_noop_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEMEMO_MODELS", str(tmp_path))
    _mark_ready(tmp_path / "whisper" / "base", {"ggml-base.bin": b"x"})
    lines: list[str] = []
    models.pull_whisper("base", log=lines.append)
    assert lines == ["whisper base already present"]


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
