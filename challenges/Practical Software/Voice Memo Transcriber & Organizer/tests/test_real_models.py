"""Real-model checks. Opt in with VOICEMEMO_REAL_MODELS=1 after `voicememo models pull`.

Each clip is a short Windows SAPI recording whose exact wording is known, so word error rate
(WER) is computed against that reference. RTF is wall time divided by audio duration (lower is
faster; below 1.0 is faster than real time). Numbers are printed; run with `-s` to see them.
"""

import json
import re
import time
from pathlib import Path

import pytest
from voicememo.audio import load_audio
from voicememo.backends import (
    ParakeetTranscriber,
    WhisperCppTranscriber,
    whisper_available,
)
from voicememo.models import is_ready
from voicememo.paths import models_dir
from voicememo.tagging import Model2VecEmbedder, evaluate

pytestmark = pytest.mark.real_models
# Collected before any test runs, so this is the user's models folder, not a temp one.
REAL_MODELS = models_dir()

HERE = Path(__file__).parent
SPEECH = HERE / "fixtures" / "speech"
TOPICS = HERE / "fixtures" / "topics.json"
CLIPS = {
    "reminder": "Buy milk and eggs on Thursday.",
    "river": "The river water was very cold.",
}
MAX_WER = 0.2


@pytest.fixture
def real_models(monkeypatch):
    # The autouse fixture in conftest points VOICEMEMO_MODELS at a temp folder; undo that here.
    monkeypatch.setenv("VOICEMEMO_MODELS", str(REAL_MODELS))
    return REAL_MODELS


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / max(1, len(ref))


def transcribe_clip(transcriber, name: str) -> tuple[str, float]:
    audio = load_audio(SPEECH / f"{name}.wav")
    started = time.perf_counter()
    segments = transcriber.transcribe(audio, language="en")
    elapsed = time.perf_counter() - started
    hypothesis = " ".join(s.text for s in segments)
    return hypothesis, elapsed / (len(audio) / 16_000)


def test_speech_fixtures_are_committed_and_small():
    total = sum(p.stat().st_size for p in SPEECH.glob("*.wav"))
    assert {p.stem for p in SPEECH.glob("*.wav")} == set(CLIPS)
    assert total < 200 * 1024


def test_parakeet_wer_and_rtf(real_models):
    if not is_ready(real_models / "parakeet"):
        pytest.skip("Parakeet not pulled: run `voicememo models pull --parakeet`")
    transcriber = ParakeetTranscriber()
    for name, reference in CLIPS.items():
        hypothesis, rtf = transcribe_clip(transcriber, name)
        score = wer(reference, hypothesis)
        print(f"parakeet {name}: WER {score:.3f} RTF {rtf:.3f} hyp={hypothesis!r}")
        assert score <= MAX_WER, hypothesis


def test_whisper_wer_and_rtf(real_models):
    if not whisper_available("base"):
        pytest.skip(
            "whisper base not pulled: run `voicememo models pull --whisper base`"
        )
    transcriber = WhisperCppTranscriber("base")
    for name, reference in CLIPS.items():
        hypothesis, rtf = transcribe_clip(transcriber, name)
        score = wer(reference, hypothesis)
        print(f"whisper-base {name}: WER {score:.3f} RTF {rtf:.3f} hyp={hypothesis!r}")
        assert score <= MAX_WER, hypothesis


def test_model2vec_tagging_on_labelled_set(real_models):
    if not is_ready(real_models / "model2vec"):
        pytest.skip("model2vec not pulled: run `voicememo models pull --embedder`")
    items = [
        (i["transcript"], i["topic"])
        for i in json.loads(TOPICS.read_text(encoding="utf-8"))
    ]
    result = evaluate(items, Model2VecEmbedder(real_models / "model2vec"))
    print(f"model2vec labelled set: {json.dumps(result, sort_keys=True)}")
    assert result["memos"] == 48
    assert 0.0 <= result["purity"] <= 1.0
