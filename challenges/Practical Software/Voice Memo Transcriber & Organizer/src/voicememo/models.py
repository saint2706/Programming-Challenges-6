"""Explicit model downloads (`voicememo models pull`). Nothing else downloads anything.

Every download is pinned to a Hugging Face commit. Weight files also have a pinned sha256, checked
before the model is marked ready. The ``.ready`` marker is a JSON manifest of the files and their
sizes; a folder counts as ready only while every listed file still has its recorded size, so a
truncated or replaced file makes the model missing again instead of failing mid-transcription.
"""

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from voicememo.paths import models_dir

READY_MARKER = ".ready"
PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v3"  # onnx-asr model name
PARAKEET_QUANTIZATION = "int8"
VAD_MODEL = "silero"  # onnx-asr VAD name


class ModelIntegrityError(Exception):
    """A downloaded file does not match its pinned sha256. The file is deleted, nothing is marked ready."""


@dataclass(frozen=True)
class Pin:
    repo: str
    revision: (
        str  # Hugging Face commit; the content of every file at this commit is fixed
    )
    sha256: dict[
        str, str | None
    ]  # file -> expected sha256; None means pinned by commit only


PARAKEET = Pin(
    "istupakov/parakeet-tdt-0.6b-v3-onnx",
    "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
    {
        "config.json": None,
        "vocab.txt": None,
        "encoder-model.int8.onnx": "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09",
        "decoder_joint-model.int8.onnx": "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70",
    },
)
VAD = Pin(
    "istupakov/silero-vad-onnx",
    "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6",
    {
        "config.json": None,
        "silero_vad.onnx": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    },
)
EMBEDDER = Pin(
    "minishlab/potion-base-8M",
    "bf8b056651a2c21b8d2565580b8569da283cab23",
    {
        "config.json": None,
        "tokenizer.json": None,
        "model.safetensors": "f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2",
    },
)
_WHISPER_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"
WHISPER_PINS = {
    "tiny": Pin(
        "ggerganov/whisper.cpp",
        _WHISPER_REVISION,
        {
            "ggml-tiny.bin": "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21"
        },
    ),
    "base": Pin(
        "ggerganov/whisper.cpp",
        _WHISPER_REVISION,
        {
            "ggml-base.bin": "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"
        },
    ),
    "small": Pin(
        "ggerganov/whisper.cpp",
        _WHISPER_REVISION,
        {
            "ggml-small.bin": "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"
        },
    ),
}


def parakeet_dir() -> Path:
    return models_dir() / "parakeet"


def vad_dir() -> Path:
    return models_dir() / "silero"


def whisper_dir(name: str) -> Path:
    return models_dir() / "whisper" / name


def _fresh_dir(path: Path) -> None:
    """Remove a folder left by an interrupted or failed pull so it is downloaded again."""
    if path.exists() and not is_ready(path):
        shutil.rmtree(path)


def _size_matches(path: Path, size: int) -> bool:
    try:
        return path.stat().st_size == size
    except OSError:
        return False


def is_ready(path: Path) -> bool:
    try:
        manifest = json.loads((path / READY_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(manifest, dict) and all(
        _size_matches(path / name, size) for name, size in manifest.items()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(pin: Pin, dest: Path, log: Callable[[str], None] = print) -> None:
    """Download every file in ``pin`` at its pinned commit into ``dest`` and write the manifest."""
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, int] = {}
    for name, expected in pin.sha256.items():
        log(f"  {name}")
        local = Path(
            hf_hub_download(pin.repo, name, revision=pin.revision, local_dir=dest)
        )
        if expected is not None and _sha256(local) != expected:
            local.unlink()
            raise ModelIntegrityError(
                f"{pin.repo}/{name}: sha256 does not match the pinned value; download discarded"
            )
        manifest[name] = local.stat().st_size
    (dest / READY_MARKER).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def pull_parakeet(log: Callable[[str], None] = print) -> None:
    vad = vad_dir()
    _fresh_dir(vad)
    if not is_ready(vad):
        log(f"downloading {VAD_MODEL} VAD into {vad}")
        fetch(VAD, vad, log)
    target = parakeet_dir()
    _fresh_dir(target)
    if not is_ready(target):
        log(f"downloading {PARAKEET_MODEL} ({PARAKEET_QUANTIZATION}) into {target}")
        fetch(PARAKEET, target, log)
    log("parakeet ready")


def pull_whisper(name: str = "base", log: Callable[[str], None] = print) -> None:
    if name not in WHISPER_PINS:
        raise ValueError(
            f"unknown whisper model {name!r}; choose from {', '.join(sorted(WHISPER_PINS))}"
        )
    target = whisper_dir(name)
    _fresh_dir(target)
    if is_ready(target):
        log(f"whisper {name} already present")
        return
    log(f"downloading whisper.cpp model {name} into {target}")
    fetch(WHISPER_PINS[name], target, log)
    log(f"whisper {name} ready")


def pull_embedder(log: Callable[[str], None] = print) -> None:
    target = models_dir() / "model2vec"
    if is_ready(target):
        log("model2vec already present")
        return
    _fresh_dir(target)
    log(f"downloading {EMBEDDER.repo} into {target}")
    fetch(EMBEDDER, target, log)
    log("model2vec ready")


def status() -> dict[str, bool]:
    return {
        "parakeet": is_ready(parakeet_dir()),
        "silero-vad": is_ready(vad_dir()),
        "whisper-base": is_ready(whisper_dir("base")),
        "model2vec": is_ready(models_dir() / "model2vec"),
    }
