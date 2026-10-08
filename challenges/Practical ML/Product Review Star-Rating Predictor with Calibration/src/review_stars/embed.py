"""Frozen gte-modernbert-base embeddings on whichever device is *correct*, cached in shards.

* reviews are very short and very uneven (median 34 tokens, p99 about 430), but OpenVINO needs
  static shapes, so a ``BucketedEncoder`` sends each text to the smallest of four compiled shapes
  (64, 128, 256, 512 tokens) and pads only to that;
* a ``WindowRunner`` turns one fixed-shape window into CLS vectors: ``TorchRunner`` is the
  reference, ``OpenVinoRunner`` is the same model converted once per shape for a device;
* ``pick_backend`` pushes probe windows of every bucket shape through each candidate device and
  refuses one whose vectors disagree with torch or are not finite. On the dev machine the iGPU
  matches torch exactly (cosine 1.0000 at 64/128/256/512 tokens) while OpenVINO's CPU device
  returns NaN at 128+ tokens, which is why the check covers every shape. The NPU is never tried
  (fast, but numerically wrong for BERT-family fp16);
* ``embed_texts`` stores results in ``.npz`` shards keyed by model, version, buckets and text, so
  an interrupted run resumes and a stale shard from another model is never reused.
"""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

MODEL_ID = "Alibaba-NLP/gte-modernbert-base"
BUCKETS = (64, 128, 256, 512)
MAX_LEN = BUCKETS[-1]
BATCH = {64: 16, 128: 16, 256: 16, 512: 8}  # as measured on the Arc iGPU
SHARD = 5000
EMBED_VERSION = 1  # bump when pooling, normalization or bucketing changes the vectors

LONG_REVIEW = " ".join(
    ["The washer arrived on time and has run quietly every day since."] * 90
)
PROBE_TEXTS = (
    "Great",
    "Works as described, fits my fridge perfectly.",
    (
        "Stopped working after two weeks. The seller did not answer my emails, so I had to "
        "return it and wait for a refund."
    ),
    (
        "I bought this for my mother because her old one broke. She loves how quiet it is, "
        "and the installation took only ten minutes. "
    )
    * 3,
    "Très bien, merci. 非常好 \U0001f44d",
    (
        "Not what I expected: the filter is smaller than the picture shows and the manual is "
        "missing pages, but customer service sent a replacement quickly. "
    )
    * 6,
    "Five stars.",
    LONG_REVIEW,
)

Tokenize = Callable[[Sequence[str], int], list[list[int]]]


class WindowRunner(Protocol):
    seq_len: int
    batch_size: int
    dim: int

    def run(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """``[batch_size, seq_len]`` int arrays in, ``[batch_size, dim]`` CLS vectors out."""
        ...


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """``(unit-norm vectors [n, dim], token counts [n])``."""
        ...


def _l2(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm == 0, 1.0, norm)


def bucket_for(n_tokens: int) -> int:
    for b in BUCKETS:
        if n_tokens <= b:
            return b
    return BUCKETS[-1]


def run_windows(runner: WindowRunner, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Run any number of windows through a fixed-batch runner; one output row per input row."""
    n, bs = len(ids), runner.batch_size
    out = np.zeros((n, runner.dim), dtype=np.float32)
    for start in range(0, n, bs):
        stop = min(start + bs, n)
        k = stop - start
        bi = np.zeros((bs, runner.seq_len), dtype=np.int64)
        bm = np.zeros_like(bi)
        bi[:k], bm[:k] = ids[start:stop], mask[start:stop]
        # padding rows keep one live token: an all-masked row makes softmax see only -inf
        bm[k:, 0] = 1
        out[start:stop] = np.asarray(runner.run(bi, bm), dtype=np.float32)[:k]
    return out


class BucketedEncoder:
    def __init__(
        self, tokenize: Tokenize, runners: dict[int, WindowRunner], pad_id: int
    ):
        self.tokenize, self.runners, self.pad_id = tokenize, runners, pad_id
        self.dim = next(iter(runners.values())).dim

    def encode(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        n = len(texts)
        if n == 0:
            return np.zeros((0, self.dim), np.float32), np.zeros(0, np.int32)
        toks = self.tokenize(list(texts), MAX_LEN)
        n_tok = np.array([len(t) for t in toks], dtype=np.int32)
        which = np.array([bucket_for(int(k)) for k in n_tok])
        out = np.zeros((n, self.dim), dtype=np.float32)
        for b, runner in self.runners.items():
            idx = np.flatnonzero(which == b)
            if len(idx) == 0:
                continue
            ids = np.full((len(idx), b), self.pad_id, dtype=np.int64)
            mask = np.zeros((len(idx), b), dtype=np.int64)
            for r, i in enumerate(idx):
                ids[r, : len(toks[i])] = toks[i]
                mask[r, : len(toks[i])] = 1
            out[idx] = _l2(run_windows(runner, ids, mask))
        return out, n_tok


class TorchRunner:
    """Reference implementation: plain torch on CPU."""

    def __init__(self, model, seq_len: int, batch_size: int):
        self.model, self.seq_len, self.batch_size = model, seq_len, batch_size
        self.dim = int(model.config.hidden_size)

    def run(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = self.model(
                input_ids=torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention_mask),
            )
        return out.last_hidden_state[:, 0].float().numpy()


class OpenVinoRunner:
    """The same model converted once for one fixed shape and compiled for ``device``.

    Static shapes are mandatory (an NPU compile aborts the process on dynamic ones, and dynamic
    shapes are slower anyway). Inputs are fed by position: OpenVINO renames them, so feeding by
    name could silently corrupt the vectors, which the self-check in ``pick_backend`` would catch.
    """

    def __init__(self, model, seq_len: int, batch_size: int, device: str = "GPU"):
        import openvino as ov
        import torch

        shape = (batch_size, seq_len)
        example = {
            "input_ids": torch.zeros(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.long),
        }
        converted = ov.convert_model(model, example_input=example)
        converted.reshape(
            {i.get_any_name(): ov.PartialShape(list(shape)) for i in converted.inputs}
        )
        self.seq_len, self.batch_size, self.device = seq_len, batch_size, device
        self.dim = int(model.config.hidden_size)
        self._compiled = ov.Core().compile_model(converted, device)

    def run(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        return np.asarray(self._compiled([input_ids, attention_mask])[0])[:, 0]


# ---------------------------------------------------------------- choosing a backend


@dataclass
class BackendChoice:
    encoder: Encoder
    name: str
    tried: list[tuple[str, str]]
    cosines: dict[int, float] = field(default_factory=dict)


def probe_windows(
    tokenize: Tokenize, pad_id: int, texts: Sequence[str] = PROBE_TEXTS
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """One ``(ids, mask)`` window per bucket shape; the longest probe review fills every window."""
    out = {}
    for b in BUCKETS:
        toks = tokenize(list(texts), b)
        ids = np.full((len(toks), b), pad_id, dtype=np.int64)
        mask = np.zeros((len(toks), b), dtype=np.int64)
        for r, t in enumerate(toks):
            ids[r, : len(t)] = t
            mask[r, : len(t)] = 1
        out[b] = (ids, mask)
    return out


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return (a * b).sum(axis=1) / np.where(na * nb == 0, 1.0, na * nb)


def pick_backend(
    candidates: Sequence[tuple[str, Callable[[], dict[int, WindowRunner]]]],
    tokenize: Tokenize,
    pad_id: int,
    reference: dict[int, WindowRunner],
    min_cos: float = 0.99,
    reference_name: str = "torch-cpu",
) -> BackendChoice:
    """First candidate whose probe vectors match ``reference`` at *every* bucket shape.

    A candidate that cannot be built, crashes, or returns wrong or non-finite vectors at any shape
    is skipped with the reason recorded; the reference is the last resort.
    """
    windows = probe_windows(tokenize, pad_id)
    expected = {b: run_windows(reference[b], *windows[b]) for b in BUCKETS}
    tried: list[tuple[str, str]] = []
    for name, build in candidates:
        try:
            runners = build()
            got = {b: run_windows(runners[b], *windows[b]) for b in BUCKETS}
        except Exception as exc:  # noqa: BLE001 -- any failure to build or run a device means "try the next one"
            tried.append((name, f"unavailable: {type(exc).__name__}: {exc}"))
            continue
        bad_nan = [b for b in BUCKETS if not np.isfinite(got[b]).all()]
        if bad_nan:
            tried.append(
                (
                    name,
                    f"rejected: non-finite vectors at {', '.join(map(str, bad_nan))} tokens",
                )
            )
            continue
        cos = {b: float(_row_cosine(got[b], expected[b]).mean()) for b in BUCKETS}
        low = [b for b in BUCKETS if cos[b] < min_cos]
        if low:
            worst = min(cos[b] for b in low)
            tried.append(
                (
                    name,
                    f"rejected: mean cosine {worst:.3f} < {min_cos} at {', '.join(map(str, low))} tokens",
                )
            )
            continue
        detail = ", ".join(f"{b}: {cos[b]:.4f}" for b in BUCKETS)
        tried.append((name, f"ok: mean cosine {min(cos.values()):.4f} ({detail})"))
        return BackendChoice(
            BucketedEncoder(tokenize, runners, pad_id), name, tried, cos
        )
    tried.append((reference_name, "ok: reference"))
    return BackendChoice(
        BucketedEncoder(tokenize, reference, pad_id), reference_name, tried, {}
    )


def hf_tokenizer(model_id: str = MODEL_ID) -> tuple[Tokenize, int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)

    def tokenize(texts: Sequence[str], max_length: int) -> list[list[int]]:
        return tok(list(texts), truncation=True, max_length=max_length)["input_ids"]

    return tokenize, int(tok.pad_token_id)


def load_torch_model(model_id: str = MODEL_ID):
    from transformers import AutoModel

    return AutoModel.from_pretrained(model_id).eval()


def default_choice(
    devices: Sequence[str] = ("GPU",), model_id: str = MODEL_ID, min_cos: float = 0.99
) -> BackendChoice:
    """Fastest *correct* backend; torch CPU (slow) if no OpenVINO device passes the self-check."""
    if any(d.upper().startswith("NPU") for d in devices):
        raise ValueError(
            "the NPU is deliberately not used: it returns wrong embeddings for BERT-family models"
        )
    tokenize, pad_id = hf_tokenizer(model_id)
    model = load_torch_model(model_id)
    reference = {b: TorchRunner(model, b, batch_size=4) for b in BUCKETS}

    def build(device: str):
        return lambda: {b: OpenVinoRunner(model, b, BATCH[b], device) for b in BUCKETS}

    candidates = [(f"openvino-{d.lower()}", build(d)) for d in devices]
    return pick_backend(candidates, tokenize, pad_id, reference, min_cos)


# ---------------------------------------------------------------- shards


def shard_key(model_id: str, texts: Sequence[str]) -> str:
    h = hashlib.sha256(f"{model_id}|v{EMBED_VERSION}|{BUCKETS}|{MAX_LEN}".encode())
    for t in texts:
        h.update(b"\0")
        h.update(t.encode("utf-8", "surrogatepass"))
    return h.hexdigest()[:24]


def _load_shard(npz: Path, marker: Path, n: int):
    """The shard's arrays if it exists, is readable and has the expected shape; else ``None``."""
    if not (npz.exists() and marker.exists()):
        return None
    try:
        meta = json.loads(marker.read_text(encoding="utf-8"))
        with np.load(npz) as z:
            X, n_tok = z["X"], z["n_tok"]
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    if (
        X.ndim != 2
        or X.shape[0] != n
        or n_tok.shape != (n,)
        or X.shape[1] != meta.get("dim")
    ):
        return None
    return X, n_tok, meta


def embed_texts(
    texts: Sequence[str],
    cache_dir,
    get_choice: Callable[[], BackendChoice],
    model_id: str = MODEL_ID,
    shard_size: int = SHARD,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Embeddings and token counts from the shard cache, computed on a verified backend on a miss.

    A shard is written as ``.npz`` first and its ``.json`` marker last, so a crash never leaves a
    shard that looks complete. ``get_choice`` is only called when there is something to compute.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    n_shards = -(-len(texts) // shard_size)
    parts, tok_parts, backends = [], [], []
    stats = {"total": n_shards, "reused": 0, "computed": 0}
    choice: BackendChoice | None = None
    for s in range(n_shards):
        chunk = list(texts[s * shard_size : (s + 1) * shard_size])
        key = shard_key(model_id, chunk)
        npz, marker = cache_dir / f"shard-{key}.npz", cache_dir / f"shard-{key}.json"
        hit = _load_shard(npz, marker, len(chunk))
        if hit is not None:
            X, n_tok, meta = hit
            stats["reused"] += 1
            backends.append(meta.get("backend", "?"))
        else:
            if choice is None:
                choice = get_choice()
            X, n_tok = choice.encoder.encode(chunk)
            part = npz.with_name(npz.name + ".part")
            with part.open("wb") as fh:
                np.savez(fh, X=X, n_tok=n_tok)
            part.replace(npz)
            marker.write_text(
                json.dumps(
                    {"backend": choice.name, "model": model_id, "dim": int(X.shape[1])}
                ),
                encoding="utf-8",
            )  # written last: it marks the shard complete
            stats["computed"] += 1
            backends.append(choice.name)
        parts.append(X)
        tok_parts.append(n_tok)
        if progress:
            progress(s + 1, n_shards)
    X = np.concatenate(parts) if parts else np.zeros((0, 0), np.float32)
    n_tok = np.concatenate(tok_parts) if tok_parts else np.zeros(0, np.int32)
    names = sorted(set(backends))
    info = {
        "backend": names[0]
        if len(names) == 1
        else ("mixed: " + ", ".join(names) if names else None),
        "tried": choice.tried if choice else [],
        "cosines": {str(b): c for b, c in (choice.cosines if choice else {}).items()},
        "model": model_id,
        "shards": stats,
        "seconds": round(time.time() - started, 1),
    }
    return X, n_tok, info
