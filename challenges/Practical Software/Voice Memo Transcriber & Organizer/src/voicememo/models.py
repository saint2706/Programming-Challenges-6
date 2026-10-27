"""Explicit model downloads (`voicememo models pull`). Nothing else downloads anything.

onnx-asr only downloads when its target folder does not exist yet; an existing folder means
offline. So a failed pull removes its half-written folder, and the ``.ready`` marker is written
last. Runtime constructors refuse to start without the marker.
"""

import shutil
from collections.abc import Callable
from pathlib import Path

from voicememo.backends import (
    PARAKEET_MODEL,
    PARAKEET_QUANTIZATION,
    READY_MARKER,
    VAD_MODEL,
    parakeet_dir,
    vad_dir,
    whisper_dir,
)
from voicememo.paths import models_dir

EMBEDDER_REPO = "minishlab/potion-base-8M"


def _fresh_dir(path: Path) -> None:
    """Remove a folder left by an interrupted pull so onnx-asr downloads into it again."""
    if path.exists() and not (path / READY_MARKER).is_file():
        shutil.rmtree(path)


def _mark_ready(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / READY_MARKER).write_text("ready\n", encoding="utf-8")


def is_ready(path: Path) -> bool:
    return (path / READY_MARKER).is_file()


def pull_parakeet(log: Callable[[str], None] = print) -> None:
    import onnx_asr

    vad = vad_dir()
    _fresh_dir(vad)
    if not is_ready(vad):
        log(f"downloading {VAD_MODEL} VAD into {vad}")
        onnx_asr.load_vad(VAD_MODEL, path=str(vad))
        _mark_ready(vad)
    target = parakeet_dir()
    _fresh_dir(target)
    if not is_ready(target):
        log(f"downloading {PARAKEET_MODEL} ({PARAKEET_QUANTIZATION}) into {target}")
        onnx_asr.load_model(
            PARAKEET_MODEL, path=str(target), quantization=PARAKEET_QUANTIZATION
        )
        _mark_ready(target)
    log("parakeet ready")


def pull_whisper(name: str = "base", log: Callable[[str], None] = print) -> None:
    from pywhispercpp.model import Model

    target = whisper_dir()
    target.mkdir(parents=True, exist_ok=True)
    if (target / f"ggml-{name}.bin").is_file():
        log(f"whisper {name} already present")
        return
    log(f"downloading whisper.cpp model {name} into {target}")
    Model(
        name,
        models_dir=str(target),
        redirect_whispercpp_logs_to=None,
        print_progress=False,
    )
    log(f"whisper {name} ready")


def pull_embedder(log: Callable[[str], None] = print) -> None:
    from model2vec import StaticModel

    target = models_dir() / "model2vec"
    if is_ready(target):
        log("model2vec already present")
        return
    log(f"downloading {EMBEDDER_REPO} into {target}")
    StaticModel.from_pretrained(EMBEDDER_REPO).save_pretrained(str(target))
    _mark_ready(target)
    log("model2vec ready")


def status() -> dict[str, bool]:
    return {
        "parakeet": is_ready(parakeet_dir()),
        "silero-vad": is_ready(vad_dir()),
        "whisper-base": (whisper_dir() / "ggml-base.bin").is_file(),
        "model2vec": is_ready(models_dir() / "model2vec"),
    }
