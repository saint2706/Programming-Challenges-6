"""Frozen sentence embeddings on whichever device is *correct*, cached on disk.

* a ``WindowRunner`` turns fixed-shape token windows into CLS vectors (torch, or an OpenVINO
  model compiled for a device);
* ``TextEncoder`` tokenizes, batches and pads, and L2-normalizes;
* ``pick_backend`` runs a probe batch through each candidate device and refuses one whose vectors
  disagree with torch. The Intel NPU on this project's dev machine runs fast and silently returns
  wrong embeddings (mean cosine 0.82 in an earlier spike), so it is never even tried;
* ``embed_cached`` keeps ``.npy`` arrays keyed by model id and a hash of the texts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

MODEL_ID = "BAAI/bge-small-en-v1.5"
SEQ_LEN = (
    128  # static so OpenVINO can compile it; Banking77 messages are well under this
)
BATCH = 32

PROBE_TEXTS = (
    "I am still waiting on my card, it has not arrived after two weeks.",
    "How do I top up my account with a transfer from another bank?",
    "Why was I charged an extra fee on my statement?",
    "My contactless payment is not working at the shop.",
    "Can I change my PIN at an ATM?",
    "I lost my phone and need to block my card immediately.",
    "What exchange rate do you use for foreign currency payments?",
    "How long does a bank transfer to another country take?",
)


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """``[len(texts), dim]`` unit-norm vectors."""
        ...


class WindowRunner(Protocol):
    batch_size: int
    seq_len: int
    dim: int

    def run(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """``[batch_size, seq_len]`` int arrays in, ``[batch_size, dim]`` CLS vectors out."""
        ...


def _l2(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm == 0, 1.0, norm)


class TextEncoder:
    def __init__(
        self,
        tokenize: Callable[[Sequence[str]], tuple[np.ndarray, np.ndarray]],
        runner: WindowRunner,
    ):
        self.tokenize, self.runner = tokenize, runner

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        r = self.runner
        out = np.zeros((len(texts), r.dim), dtype=np.float32)
        if len(texts) == 0:
            return out
        ids, mask = self.tokenize(list(texts))
        for start in range(0, len(texts), r.batch_size):
            stop = min(start + r.batch_size, len(texts))
            k = stop - start
            bi = np.zeros((r.batch_size, r.seq_len), dtype=np.int64)
            bm = np.zeros_like(bi)
            bi[:k], bm[:k] = ids[start:stop], mask[start:stop]
            # padding rows keep one live token: an all-masked row makes softmax see only -inf
            bm[k:, 0] = 1
            out[start:stop] = _l2(np.asarray(r.run(bi, bm))[:k])
        return out


class TorchRunner:
    """Reference implementation: plain torch on CPU."""

    def __init__(
        self, model_id: str = MODEL_ID, batch_size: int = BATCH, seq_len: int = SEQ_LEN
    ):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id).eval()
        self.batch_size, self.seq_len = batch_size, seq_len
        self.dim = int(self.model.config.hidden_size)

    def run(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        import torch

        ids = torch.from_numpy(input_ids)
        with torch.no_grad():
            out = self.model(
                input_ids=ids,
                attention_mask=torch.from_numpy(attention_mask),
                token_type_ids=torch.zeros_like(ids),
            )
        return out.last_hidden_state[:, 0].numpy()


class OpenVinoRunner:
    """The same model converted once, with every input shape fixed, and compiled for ``device``.

    Static shapes are mandatory. Inputs are fed by position in the model's forward-signature
    order: OpenVINO renames ``attention_mask`` to ``'11'``, so feeding by name silently corrupts
    the vectors (the cosine self-check in ``pick_backend`` is what catches a mistake here).
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "GPU",
        batch_size: int = BATCH,
        seq_len: int = SEQ_LEN,
    ):
        import openvino as ov
        import torch
        from transformers import AutoModel

        pt = AutoModel.from_pretrained(model_id).eval()
        self.batch_size, self.seq_len = batch_size, seq_len
        self.dim = int(pt.config.hidden_size)
        shape = (batch_size, seq_len)
        example = {
            "input_ids": torch.zeros(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.long),
            "token_type_ids": torch.zeros(shape, dtype=torch.long),
        }
        model = ov.convert_model(pt, example_input=example)
        model.reshape(
            {i.get_any_name(): ov.PartialShape(list(shape)) for i in model.inputs}
        )
        self.device = device
        self._compiled = ov.Core().compile_model(model, device)

    def run(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        out = self._compiled([input_ids, attention_mask, np.zeros_like(input_ids)])[0]
        return np.asarray(out)[:, 0]


def load_text_encoder(
    runner_factory: Callable[[str], WindowRunner], model_id: str = MODEL_ID
) -> TextEncoder:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)

    def tokenize(texts):
        enc = tok(
            list(texts),
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
            return_tensors="np",
        )
        return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)

    return TextEncoder(tokenize, runner_factory(model_id))


@dataclass
class BackendChoice:
    encoder: Encoder
    name: str
    tried: list[tuple[str, str]]


def pick_backend(
    candidates: Sequence[tuple[str, Callable[[], Encoder]]],
    probe: Sequence[str],
    reference: Encoder,
    min_cos: float = 0.99,
    reference_name: str = "torch-cpu",
) -> BackendChoice:
    """First candidate whose probe vectors match ``reference`` (mean cosine >= ``min_cos``).

    A candidate that cannot be built, crashes, or returns wrong vectors is skipped with the reason
    recorded; the reference is the last resort.
    """
    ref = reference.encode(probe)
    tried: list[tuple[str, str]] = []
    for name, build in candidates:
        try:
            enc = build()
            got = enc.encode(probe)
        except Exception as exc:  # noqa: BLE001 -- any failure to build or run a device means "try the next one"
            tried.append((name, f"unavailable: {type(exc).__name__}: {exc}"))
            continue
        cos = float((got * ref).sum(axis=1).mean())
        if not np.isfinite(cos) or cos < min_cos:
            tried.append((name, f"rejected: mean cosine {cos:.3f} < {min_cos}"))
            continue
        tried.append((name, f"ok: mean cosine {cos:.4f}"))
        return BackendChoice(enc, name, tried)
    tried.append((reference_name, "ok: reference"))
    return BackendChoice(reference, reference_name, tried)


def default_choice(devices: Sequence[str] = ("GPU",)) -> BackendChoice:
    """Fastest *correct* backend; torch CPU if no OpenVINO device passes the self-check."""
    if any(d.upper().startswith("NPU") for d in devices):
        raise ValueError(
            "the NPU is deliberately not used: it returns wrong embeddings (cosine ~0.82 vs torch)"
        )
    reference = load_text_encoder(TorchRunner)

    def build(device: str) -> Callable[[], Encoder]:
        return lambda: load_text_encoder(
            lambda model_id: OpenVinoRunner(model_id, device)
        )

    candidates = [(f"openvino-{d.lower()}", build(d)) for d in devices]
    return pick_backend(candidates, PROBE_TEXTS, reference)


def cache_key(model_id: str, texts: Sequence[str]) -> str:
    h = hashlib.sha256(model_id.encode("utf-8"))
    for t in texts:
        h.update(b"\0")
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:24]


def embed_cached(
    texts: Sequence[str],
    cache_dir: Path,
    get_choice: Callable[[], BackendChoice],
    model_id: str = MODEL_ID,
) -> tuple[np.ndarray, dict]:
    """Embeddings from the ``.npy`` cache, or computed on a verified backend and cached."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(model_id, texts)
    npy, meta = cache_dir / f"emb-{key}.npy", cache_dir / f"emb-{key}.json"
    if npy.exists() and meta.exists():
        return np.load(npy), json.loads(meta.read_text(encoding="utf-8"))
    choice = get_choice()
    X = np.asarray(choice.encoder.encode(list(texts)), dtype=np.float32)
    tmp = npy.with_suffix(".part")
    with tmp.open("wb") as fh:
        np.save(fh, X)
    tmp.replace(npy)
    info = {"backend": choice.name, "tried": choice.tried, "model": model_id}
    meta.write_text(
        json.dumps(info), encoding="utf-8"
    )  # written last: it marks the entry complete
    return X, info
