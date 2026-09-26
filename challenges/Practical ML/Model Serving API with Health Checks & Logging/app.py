"""Model Serving API with Health Checks & Logging.

Serves the vendored SqueezeNet 1.1 ONNX image classifier behind a small
FastAPI app: liveness/readiness health probes, a /model/info introspection
endpoint, and a structured JSON log line for every request.

Run directly:
    uv run --with fastapi --with "uvicorn[standard]" --with onnxruntime \\
        --with pillow --with structlog --with numpy --with python-multipart \\
        python app.py

Or with a reload server:
    uv run --with fastapi --with "uvicorn[standard]" --with onnxruntime \\
        --with pillow --with structlog --with numpy --with python-multipart \\
        uvicorn app:app --reload
"""

from __future__ import annotations

import io
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import model_loader
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from logging_config import configure_logging, get_logger
from PIL import Image, UnidentifiedImageError

configure_logging()
logger = get_logger()

# Mutable app-wide state (session/labels/readiness). A plain dict rather than
# globals so tests can reset it between cases without reimporting the module.
app_state: dict[str, Any] = {"session": None, "labels": None, "ready": False}


def _model_info_payload() -> dict[str, Any]:
    info = model_loader.get_model_info(app_state["session"], app_state["labels"])
    return {
        "input_name": info.input_name,
        "input_shape": info.input_shape,
        "output_name": info.output_name,
        "output_shape": info.output_shape,
        "num_classes": info.num_classes,
        "sha256": info.sha256,
        "onnxruntime_version": info.onnxruntime_version,
    }


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    app_state["session"] = model_loader.load_session()
    app_state["labels"] = model_loader.load_labels()
    app_state["ready"] = False
    try:
        model_loader.warmup(app_state["session"])
        app_state["ready"] = True
        logger.info("model_ready", **_model_info_payload())
    except Exception:
        logger.exception("warmup_failed")
    yield
    app_state["session"] = None
    app_state["labels"] = None
    app_state["ready"] = False


app = FastAPI(title="Model Serving API", lifespan=lifespan)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Times and logs every request as one structured JSON line -- not just
    /predict -- with a per-request id threaded through response headers so a
    client-reported problem can be grep'd straight out of logs/requests.jsonl."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.log_extra = {}
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            latency_ms=round(latency_ms, 2),
        )
        raise
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=round(latency_ms, 2),
        **request.state.log_extra,
    )
    response.headers["X-Request-Id"] = request_id
    return response


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008 -- FastAPI's documented DI pattern; File() is inspected at route-registration time, not called per-request
    start = time.perf_counter()
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            422, f"Could not decode uploaded file as an image: {exc}"
        ) from exc

    if not app_state["ready"]:
        raise HTTPException(503, "Model is not ready (warmup has not completed)")

    array = model_loader.preprocess(image)
    predictions = model_loader.predict(
        app_state["session"], array, app_state["labels"], top_k=5
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    # Logged (not the raw image bytes -- just its dimensions) by the
    # middleware above once this handler returns.
    request.state.log_extra = {
        "image_size": list(image.size),
        "top1_label": predictions[0][0],
        "top1_confidence": round(predictions[0][1], 4),
        "model_sha256": model_loader.MODEL_SHA256,
    }

    return {
        "request_id": request.state.request_id,
        "predictions": [
            {"label": label, "confidence": round(confidence, 6)}
            for label, confidence in predictions
        ],
        "latency_ms": latency_ms,
    }


@app.get("/health/live")
def health_live() -> dict[str, str]:
    """Always 200 once the process is up -- a liveness probe should only
    fail if the process itself is wedged, never because a dependency
    (like the model) isn't ready yet."""
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready() -> dict[str, str]:
    """200 only once the ONNX session is loaded *and* a warm-up inference
    has actually succeeded -- see model_loader.warmup and the "hard part"
    section of the README for why that distinction matters."""
    if not app_state["ready"]:
        raise HTTPException(
            503, "Model not ready: session not loaded or warmup incomplete"
        )
    return {"status": "ready"}


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    if app_state["session"] is None:
        raise HTTPException(503, "Model not loaded")
    return _model_info_payload()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)
