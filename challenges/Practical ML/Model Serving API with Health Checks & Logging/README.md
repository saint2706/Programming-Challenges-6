# Model Serving API with Health Checks & Logging

**Category:** Practical ML
**Difficulty:** B (brief: "FastAPI + ONNX Runtime, structured request/response logging.")

**Status:** Implemented (Python)

A small FastAPI service that serves a real pretrained ONNX model —
[SqueezeNet 1.1](https://github.com/onnx/models/tree/main/validated/vision/classification/squeezenet)
(ImageNet, 1000 classes, ~5 MB) — behind Kubernetes-style liveness/readiness
health probes and a structured JSON log line for every request. The brief
didn't specify which model to serve, so a small, well-known image classifier
from the official ONNX Model Zoo was vendored directly into this folder
rather than a toy model trained from scratch — the point of this challenge is
the *serving infrastructure* around a real model, not the model itself.

## Two things had to be airtight

**Cold-start latency vs. the readiness probe.** onnxruntime defers real work
(graph optimization, memory-arena allocation) until the *first* `Run()` call
on a session — so naively marking the model "ready" as soon as
`InferenceSession(...)` returns would mean the actual first request pays that
one-time cost, which in a real deployment is exactly the request a load
balancer might route traffic to right after a pod reports ready. `/health/ready`
only returns 200 after a dummy inference (`model_loader.warmup`) has
*actually succeeded* against the real model, not just after the session
object was constructed. `/health/live` deliberately does **not** check the
model at all — a liveness probe should only fail if the process itself is
wedged, never because a dependency happens to be initializing; conflating the
two is a common real-world cause of restart-loop outages. Covered by
`test_health_live_is_always_ok`, `test_health_ready_is_ok_after_startup_warmup`,
and `test_readiness_and_model_info_fail_before_startup_lifespan_runs` (which
calls the route functions directly with `app_state` reset, bypassing the
lifespan-managed `TestClient`, to prove the pre-startup code path degrades to
503 rather than crashing).

**A vendored binary you can't just eyeball for correctness.** The ONNX model
and label file are committed directly into this repo so the challenge is
network-free and reproducible — but that means a corrupted git checkout, a
bad merge, or someone accidentally re-exporting the file with different
weights would otherwise fail silently: onnxruntime will happily load a
*different* 5 MB of bytes and produce confident-looking, wrong predictions
forever. Both files are sha256-pinned at load time
(`model_loader._verify_sha256`) against the digest recorded when they were
vendored; a mismatch raises `ModelIntegrityError` and refuses to serve rather
than degrading silently. Covered by
`test_sha256_mismatch_is_detected_without_touching_real_vendored_file`, which
tampers with a *temp copy*, never the real vendored file.

## Design

- **`model_loader.py`** — framework-free (no FastAPI/logging imports) so the
  actual model logic is unit-testable in isolation. `load_session` /
  `load_labels` verify sha256 before use; `preprocess` resizes/normalizes a
  PIL image into SqueezeNet1.1's expected 224×224 NCHW float32 input
  (ImageNet mean/std, verified against the graph's real input tensor rather
  than assumed); `predict` runs inference, applies softmax (the ONNX graph's
  raw output is pre-softmax logits), and returns the top-k `(label,
  confidence)` pairs; `warmup` runs one dummy inference.
- **`logging_config.py`** — [`structlog`](https://www.structlog.org/), the
  modern structured-logging library, configured with a custom processor that
  appends every log event as one JSON line to `logs/requests.jsonl` *before*
  the final `JSONRenderer` processor turns the same event dict into the
  string written to stdout — a single source of truth so the file and
  console output can never drift apart.
- **`app.py`** — FastAPI routes plus an `@app.middleware("http")` that times
  and logs **every** request (not just `/predict`) with a per-request UUID
  threaded through the `X-Request-Id` response header, so a client-reported
  problem can be grep'd straight out of the log file. `/predict` reads the
  upload, decodes it with Pillow (a decode failure returns 422, never a raw
  500), runs the pipeline, and stashes prediction metadata on
  `request.state.log_extra` for the middleware to fold into that request's
  log line — the raw image bytes are never logged, only its dimensions.
- **Money/precision note doesn't apply here** (no currency math in this
  challenge) — the analogous "silent wrongness" risk is the vendored-binary
  integrity check above.

## What it deliberately doesn't do

- **No request batching or GPU execution provider** — `CPUExecutionProvider`
  only, one image per request. Real batching changes the API shape
  (client-side queuing or an async batching layer) and is out of scope for a
  Beginner-tier serving challenge focused on health checks and logging.
- **No model hot-swap or retraining endpoint** — the model is loaded once at
  startup and is immutable for the process's lifetime. A registry/rollback
  system is its own challenge (#17, Advanced tier) elsewhere in this repo.
- **No authentication** — matches this repo's convention of keeping each
  challenge a single self-contained script with no external credentials to
  configure; a real deployment would put this behind a gateway.
- **No Docker/Kubernetes manifests** — the health-check *semantics* (live vs.
  ready) are implemented and tested; wiring them into an actual orchestrator
  is deployment configuration, not application code, and every other
  challenge in this repo runs the same way (`uv run --with ...`), not
  containerized.
- **Single process only** — no multi-worker/replica coordination; the
  structured log file is append-only from one process, which is enough to
  demonstrate the logging story without building a log-aggregation pipeline.

## Usage

```bash
cd "challenges/Practical ML/Model Serving API with Health Checks & Logging"

uv run --with fastapi --with "uvicorn[standard]" --with onnxruntime \
    --with pillow --with structlog --with numpy --with python-multipart \
    python app.py
# -> http://127.0.0.1:8010

# or, with auto-reload during development:
uv run --with fastapi --with "uvicorn[standard]" --with onnxruntime \
    --with pillow --with structlog --with numpy --with python-multipart \
    uvicorn app:app --reload

uv run --with fastapi --with "uvicorn[standard]" --with onnxruntime \
    --with pillow --with structlog --with numpy --with python-multipart \
    --with httpx2 --with pytest pytest -q   # 15 tests
```

Try it with any image file:

```bash
curl -X POST http://127.0.0.1:8010/predict -F "file=@/path/to/some/photo.jpg"
curl http://127.0.0.1:8010/health/live
curl http://127.0.0.1:8010/health/ready
curl http://127.0.0.1:8010/model/info
```

`/predict` returns the top-5 ImageNet classes with confidences and the
request's own latency; every request (successful or not) gets one line
appended to `logs/requests.jsonl` (gitignored — a runtime artifact, not
source) and printed to stdout as JSON.

**Live-verified:** started the server, POSTed a real photo through
`/predict`, and confirmed a plausible top-1 label with a sane confidence,
then inspected `logs/requests.jsonl` to confirm the same request's
`top1_label`/`top1_confidence`/`model_sha256`/`latency_ms` were all recorded
correctly against its `request_id`, and that `X-Request-Id` in the response
headers matched the body.

## Tests

15 pytest cases across two files, run warning-free (Starlette's `TestClient`
now prefers the `httpx2` package over `httpx`, per the same upstream
migration noted in this repo's other FastAPI challenges).

- `test_model_loader.py` (8) — label parsing, session loading, model-info
  metadata matching the ONNX graph's real shapes, preprocessing shape/dtype
  correctness for RGB/grayscale/RGBA inputs, top-k ordering and confidence
  bounds, warmup, and sha256-mismatch detection against a tampered temp copy.
- `test_app.py` (7) — liveness/readiness probes, `/model/info`, a full
  `/predict` round trip on a synthetic image (valid top-5 predictions, sorted
  confidences, `X-Request-Id` header matching the body), a malformed upload
  rejected with 422 (not 500), a `logs/requests.jsonl` line schema check
  (parses the new JSON line for a real request and asserts every required
  field is present), and the pre-startup 503 code path.
