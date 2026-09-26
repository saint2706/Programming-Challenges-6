"""Model loading, preprocessing, and inference for the vendored SqueezeNet 1.1
ONNX classifier.

Deliberately framework-free (no FastAPI, no logging setup) so the actual
model-serving logic is unit-testable in isolation from the web layer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "squeezenet1.1-7.onnx"
LABELS_PATH = BASE_DIR / "models" / "imagenet_synset.txt"

# Pinned at vendoring time -- see README "the hard part" section. A mismatch
# means the file was corrupted or tampered with after being committed, and
# we refuse to serve predictions from it rather than fail silently.
MODEL_SHA256 = "1eeff551a67ae8d565ca33b572fc4b66e3ef357b0eb2863bb9ff47a918cc4088"
LABELS_SHA256 = "acf75ef0abe89694b19056e0796401068b459c457baa30335f240c7692857355"

IMAGE_SIZE = 224
# Standard ImageNet normalization stats that SqueezeNet1.1 was trained with.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ModelIntegrityError(RuntimeError):
    """A vendored file's sha256 doesn't match the pinned value."""


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ModelIntegrityError(
            f"{path.name} sha256 mismatch: expected {expected}, got {digest}. "
            "The vendored file may be corrupted or tampered with; refusing to load it."
        )


def load_labels(path: Path = LABELS_PATH, *, verify: bool = True) -> list[str]:
    """Parse the ImageNet synset file ("n01440764 tench, Tinca tinca" per
    line, in class-index order) into a plain list of human-readable labels."""
    if verify:
        _verify_sha256(path, LABELS_SHA256)
    labels: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        _synset_id, _, human = line.partition(" ")
        labels.append(human or line)
    return labels


@dataclass(frozen=True)
class ModelInfo:
    input_name: str
    input_shape: list
    output_name: str
    output_shape: list
    num_classes: int
    sha256: str
    onnxruntime_version: str


def load_session(
    path: Path = MODEL_PATH, *, verify: bool = True
) -> ort.InferenceSession:
    if verify:
        _verify_sha256(path, MODEL_SHA256)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def get_model_info(
    session: ort.InferenceSession, labels: list[str], sha256: str = MODEL_SHA256
) -> ModelInfo:
    """Read the model's actual graph metadata rather than hardcoding it, so
    /model/info always reflects what onnxruntime really loaded."""
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    return ModelInfo(
        input_name=inp.name,
        input_shape=list(inp.shape),
        output_name=out.name,
        output_shape=list(out.shape),
        num_classes=len(labels),
        sha256=sha256,
        onnxruntime_version=ort.__version__,
    )


def preprocess(image: Image.Image) -> np.ndarray:
    """Resize/normalize a PIL image into SqueezeNet1.1's NCHW float32 input:
    224x224 RGB, scaled to [0,1], ImageNet mean/std normalized, channel-first."""
    rgb = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    array = np.asarray(rgb, dtype=np.float32) / 255.0  # HWC in [0, 1]
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = array.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(array, axis=0).astype(np.float32)  # add batch dim -> NCHW


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / exps.sum()


def predict(
    session: ort.InferenceSession,
    array: np.ndarray,
    labels: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Run inference and return the top-k (label, confidence) pairs, most
    confident first. `array` must already be preprocessed (see `preprocess`)."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    (logits,) = session.run([output_name], {input_name: array})
    probabilities = _softmax(logits.reshape(-1))
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    return [(labels[i], float(probabilities[i])) for i in top_indices]


def warmup(session: ort.InferenceSession) -> None:
    """Run one dummy inference so the first *real* request doesn't pay
    onnxruntime's cold-start cost (graph optimization and memory-arena
    allocation are deferred until the first Run() call). The readiness probe
    only reports ready after this has succeeded once -- see app.py."""
    input_meta = session.get_inputs()[0]
    shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in input_meta.shape]
    dummy = np.zeros(shape, dtype=np.float32)
    session.run(None, {input_meta.name: dummy})
