from __future__ import annotations

import io
import json
import uuid

from app import app
from fastapi.testclient import TestClient
from logging_config import LOG_FILE
from PIL import Image


def _synthetic_image_bytes() -> bytes:
    """A small synthetic gradient image -- the point of these tests is that
    the serving pipeline runs end-to-end, not that the classification is
    semantically meaningful for a made-up picture."""
    image = Image.new("RGB", (300, 200))
    for y in range(image.height):
        for x in range(image.width):
            image.putpixel((x, y), ((x * 3) % 256, (y * 5) % 256, (x + y) % 256))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _log_lines() -> list[str]:
    if not LOG_FILE.exists():
        return []
    return LOG_FILE.read_text(encoding="utf-8").splitlines()


def test_health_live_is_always_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_health_ready_is_ok_after_startup_warmup() -> None:
    with TestClient(app) as client:
        response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_model_info_reports_expected_fields() -> None:
    with TestClient(app) as client:
        response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["input_shape"] == [1, 3, 224, 224]
    assert body["output_shape"] == [1, 1000]
    assert body["num_classes"] == 1000
    assert len(body["sha256"]) == 64


def test_predict_happy_path_returns_top5_sorted_predictions() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("test.png", _synthetic_image_bytes(), "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    uuid.UUID(body["request_id"])  # raises if not a valid uuid4-shaped string
    assert len(body["predictions"]) == 5
    confidences = [p["confidence"] for p in body["predictions"]]
    assert confidences == sorted(confidences, reverse=True)
    assert all(isinstance(p["label"], str) and p["label"] for p in body["predictions"])
    assert body["latency_ms"] > 0
    assert response.headers["X-Request-Id"] == body["request_id"]


def test_predict_rejects_malformed_upload_with_422_not_500() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "not_an_image.txt",
                    b"this is definitely not an image",
                    "text/plain",
                )
            },
        )
    assert response.status_code == 422


def test_every_request_produces_a_well_formed_json_log_line() -> None:
    lines_before = len(_log_lines())
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("test.png", _synthetic_image_bytes(), "image/png")},
        )
    assert response.status_code == 200

    new_lines = _log_lines()[lines_before:]
    assert new_lines, "expected at least one new log line for the /predict request"

    request_log = None
    for line in new_lines:
        entry = json.loads(line)  # must be valid JSON, not just text
        if entry.get("event") == "request" and entry.get("path") == "/predict":
            request_log = entry

    assert request_log is not None, "no 'request' log line found for /predict"
    for field in ("timestamp", "request_id", "method", "status_code", "latency_ms"):
        assert field in request_log
    assert request_log["method"] == "POST"
    assert request_log["status_code"] == 200
    assert "top1_label" in request_log
    assert "model_sha256" in request_log
    assert "image_size" in request_log  # dimensions logged, not raw image bytes


def test_readiness_and_model_info_fail_before_startup_lifespan_runs() -> None:
    # Calling the route functions directly (bypassing the lifespan-managed
    # TestClient context) simulates a request arriving before startup --
    # app_state must reflect "not ready" rather than crash.
    import app as app_module
    from app import health_ready, model_info
    from fastapi import HTTPException

    original_state = dict(app_module.app_state)
    app_module.app_state["session"] = None
    app_module.app_state["ready"] = False
    try:
        try:
            health_ready()
            raise AssertionError("expected HTTPException for not-ready health check")
        except HTTPException as exc:
            assert exc.status_code == 503

        try:
            model_info()
            raise AssertionError("expected HTTPException for missing model")
        except HTTPException as exc:
            assert exc.status_code == 503
    finally:
        app_module.app_state.update(original_state)
