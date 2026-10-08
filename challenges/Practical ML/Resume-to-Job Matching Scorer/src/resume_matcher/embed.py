"""Dense scorer: chunked bge-small embeddings on whichever device is *correct*.

Layers, each testable on its own:

* a ``WindowRunner`` turns fixed-shape token windows into vectors (torch, or an
  OpenVINO model compiled for a device);
* ``ChunkedEncoder`` splits a long text into overlapping windows and
  mean-pools them (resumes are ~1.5k tokens, the model sees 256);
* ``pick_backend`` runs a probe set through each candidate device and refuses
  one whose vectors disagree with torch -- the Intel NPU on this project's
  dev machine runs fast and silently returns wrong embeddings (mean cosine
  0.82), which is exactly the failure this guards against;
* ``CachedEncoder`` keeps vectors on disk so a re-run costs nothing.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

import numpy as np

MODEL_ID = "BAAI/bge-small-en-v1.5"
SEQ_LEN = 256  # model window incl. [CLS]/[SEP]; static so OpenVINO can compile it
BATCH = 8
STRIDE = 192
MAX_CHUNKS = 8  # bounds cost per text; ~1.2k tokens, i.e. most of a resume

PROBE_TEXTS = (
    "Senior data analyst experienced in SQL, Tableau and stakeholder reporting.",
    "Registered nurse on a medical-surgical ward with ACLS certification.",
    "Executive chef managing a 40-seat kitchen, menu costing and food safety.",
    "Python developer building REST APIs and data pipelines on AWS.",
    "Accounts payable specialist reconciling invoices and month-end close.",
    "Third-grade teacher designing phonics lessons and parent communication.",
    "Construction superintendent scheduling subcontractors on commercial sites.",
    "Retail sales associate meeting daily targets and handling customer returns.",
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


def chunk_tokens(ids: list[int], window: int, stride: int) -> list[list[int]]:
    """Overlapping windows covering every token; the last one is end-aligned."""
    if len(ids) <= window:
        return [ids]
    starts = list(range(0, len(ids) - window + 1, stride))
    if starts[-1] != len(ids) - window:
        starts.append(len(ids) - window)
    return [ids[s : s + window] for s in starts]


class ChunkedEncoder:
    def __init__(
        self,
        tokenize: Callable[[str], list[int]],
        runner: WindowRunner,
        cls_id: int,
        sep_id: int,
        pad_id: int,
        stride: int = STRIDE,
        max_chunks: int = MAX_CHUNKS,
    ):
        self.tokenize = tokenize
        self.runner = runner
        self.cls_id, self.sep_id, self.pad_id = cls_id, sep_id, pad_id
        self.window = runner.seq_len - 2
        self.stride = stride
        self.max_chunks = max_chunks

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        r = self.runner
        windows: list[list[int]] = []
        owner: list[int] = []
        for i, text in enumerate(texts):
            for chunk in chunk_tokens(self.tokenize(text), self.window, self.stride)[
                : self.max_chunks
            ]:
                windows.append(chunk)
                owner.append(i)

        vecs = np.zeros((len(windows), r.dim))
        for start in range(0, len(windows), r.batch_size):
            batch = windows[start : start + r.batch_size]
            ids = np.full((r.batch_size, r.seq_len), self.pad_id, dtype=np.int64)
            mask = np.zeros_like(ids)
            for row, chunk in enumerate(batch):
                seq = [self.cls_id, *chunk, self.sep_id]
                ids[row, : len(seq)] = seq
                mask[row, : len(seq)] = 1
            # padding rows keep one live token: an all-masked row makes softmax see only -inf
            mask[len(batch) :, 0] = 1
            vecs[start : start + len(batch)] = _l2(r.run(ids, mask)[: len(batch)])

        pooled = np.zeros((len(texts), r.dim))
        np.add.at(pooled, owner, vecs)
        return _l2(pooled)


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

    Static shapes are mandatory (the NPU compiler aborts the whole process on
    dynamic ones). Inputs are fed by position in the model's forward-signature
    order -- OpenVINO renames ``attention_mask`` to ``'11'`` and the tokenizer
    returns a different order, so feeding by name or by tokenizer order
    silently corrupts the vectors.
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


def load_chunked(
    runner_factory: Callable[[str], WindowRunner], model_id: str = MODEL_ID
) -> ChunkedEncoder:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.model_max_length = 10**9  # we chunk ourselves; silence the "too long" warning
    return ChunkedEncoder(
        tokenize=lambda s: tok(s, add_special_tokens=False)["input_ids"],
        runner=runner_factory(model_id),
        cls_id=tok.cls_token_id,
        sep_id=tok.sep_token_id,
        pad_id=tok.pad_token_id,
    )


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

    A candidate that cannot be built, crashes, or returns wrong vectors is
    skipped with the reason recorded; the reference is the last resort.
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


class EmbeddingCache:
    """sqlite key -> float32 vector. ``":memory:"`` for tests."""

    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)"
        )

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            part = list(keys[i : i + 500])
            marks = ",".join("?" * len(part))
            for key, blob in self._db.execute(
                f"SELECT key, vec FROM emb WHERE key IN ({marks})", part
            ):
                found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        rows = [
            (k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()
        ]
        self._db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
        self._db.commit()


def cache_id(model_id: str = MODEL_ID) -> str:
    """Everything besides the text that changes a vector, so a config change misses the cache."""
    return f"{model_id}|L{SEQ_LEN}|s{STRIDE}|c{MAX_CHUNKS}"


class CachedEncoder:
    def __init__(self, inner: Encoder, cache: EmbeddingCache, model_id: str):
        self.inner, self.cache, self.model_id = inner, cache, model_id

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_id}\0{text}".encode()).hexdigest()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.asarray(self.inner.encode([]), dtype=np.float32)
        keys = [self._key(t) for t in texts]
        have = self.cache.get_many(list(dict.fromkeys(keys)))
        missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in have}
        if missing:
            fresh = np.asarray(
                self.inner.encode(list(missing.values())), dtype=np.float32
            )
            new = dict(zip(missing, fresh, strict=True))
            self.cache.put_many(new)
            have.update(new)
        return np.stack([have[k] for k in keys])


class EmbeddingScorer:
    """Cosine of unit vectors. Nothing to fit: the encoder is pretrained."""

    name = "embedding"

    def __init__(self, encoder: Encoder):
        self.encoder = encoder

    def fit(self, corpus: Sequence[str]) -> Self:
        return self

    def score_matrix(self, queries: Sequence[str], docs: Sequence[str]) -> np.ndarray:
        return self.encoder.encode(queries) @ self.encoder.encode(docs).T

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        return self.score_matrix([query], docs)[0]


def default_encoder(
    cache_path: Path, devices: Sequence[str] = ("GPU", "CPU")
) -> BackendChoice:
    """Fastest *correct* backend, cached on disk. Falls back to torch CPU."""
    reference = load_chunked(TorchRunner)

    def build(device: str) -> Callable[[], Encoder]:
        return lambda: load_chunked(lambda model_id: OpenVinoRunner(model_id, device))

    choice = pick_backend(
        [(f"openvino-{d.lower()}", build(d)) for d in devices], PROBE_TEXTS, reference
    )
    cached = CachedEncoder(choice.encoder, EmbeddingCache(cache_path), cache_id())
    return BackendChoice(cached, choice.name, choice.tried)
