from __future__ import annotations

from pathlib import Path

import model_loader
import numpy as np
import pytest
from PIL import Image


def test_load_labels_returns_1000_imagenet_labels() -> None:
    labels = model_loader.load_labels()
    assert len(labels) == 1000
    # First synset line is "n01440764 tench, Tinca tinca" -- human part only.
    assert labels[0] == "tench, Tinca tinca"


def test_load_session_returns_working_session() -> None:
    session = model_loader.load_session()
    assert session.get_inputs()[0].name == "data"
    assert session.get_outputs()[0].name == "squeezenet0_flatten0_reshape0"


def test_get_model_info_reflects_actual_graph_metadata() -> None:
    session = model_loader.load_session()
    labels = model_loader.load_labels()
    info = model_loader.get_model_info(session, labels)
    assert info.input_shape == [1, 3, 224, 224]
    assert info.output_shape == [1, 1000]
    assert info.num_classes == 1000
    assert info.sha256 == model_loader.MODEL_SHA256


def test_preprocess_produces_correct_nchw_shape_and_dtype() -> None:
    image = Image.new("RGB", (50, 80), color=(120, 200, 30))
    array = model_loader.preprocess(image)
    assert array.shape == (1, 3, 224, 224)
    assert array.dtype == np.float32


def test_preprocess_handles_non_rgb_input() -> None:
    # Grayscale and RGBA inputs must both be coerced to RGB before normalizing.
    grayscale = Image.new("L", (100, 100), color=128)
    rgba = Image.new("RGBA", (100, 100), color=(10, 20, 30, 255))
    for image in (grayscale, rgba):
        array = model_loader.preprocess(image)
        assert array.shape == (1, 3, 224, 224)


def test_predict_returns_top_k_sorted_by_confidence_descending() -> None:
    session = model_loader.load_session()
    labels = model_loader.load_labels()
    image = Image.new("RGB", (224, 224), color=(30, 60, 90))
    array = model_loader.preprocess(image)

    results = model_loader.predict(session, array, labels, top_k=5)

    assert len(results) == 5
    confidences = [confidence for _, confidence in results]
    assert confidences == sorted(confidences, reverse=True)
    for label, confidence in results:
        assert isinstance(label, str) and label
        assert 0.0 <= confidence <= 1.0


def test_warmup_runs_without_raising() -> None:
    session = model_loader.load_session()
    model_loader.warmup(session)  # should not raise


def test_sha256_mismatch_is_detected_without_touching_real_vendored_file(
    tmp_path: Path,
) -> None:
    corrupted = tmp_path / "squeezenet1.1-7.onnx"
    original_bytes = model_loader.MODEL_PATH.read_bytes()
    corrupted.write_bytes(original_bytes + b"\x00tampered")

    with pytest.raises(model_loader.ModelIntegrityError):
        model_loader.load_session(path=corrupted)

    # Sanity: the real vendored file is untouched and still verifies clean.
    model_loader.load_session(model_loader.MODEL_PATH)
