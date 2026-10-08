"""Test helpers: tiny review archives, and synthetic stand-ins for embeddings and predictions.

Heavy package modules are imported inside functions so importing this file stays cheap.
"""

from __future__ import annotations

import gzip
import json
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def month_ms(year: int, month: int, day: int = 1) -> int:
    """Epoch milliseconds of 00:00 UTC on that day."""
    return int(datetime(year, month, day, tzinfo=UTC).timestamp() * 1000)


class FakeRunner:
    """A ``WindowRunner`` with no model: the sum of one-hot live tokens, so masks matter."""

    def __init__(self, seq_len: int, batch_size: int, dim: int):
        self.seq_len, self.batch_size, self.dim = seq_len, batch_size, dim
        self.calls, self.masks = 0, []

    def run(self, input_ids, attention_mask):
        self.calls += 1
        self.masks.append(attention_mask.copy())
        out = np.zeros((input_ids.shape[0], self.dim))
        for r, (row, live) in enumerate(zip(input_ids, attention_mask, strict=True)):
            for tok, m in zip(row, live, strict=True):
                if m:
                    out[r, tok % self.dim] += 1
        return out


def fake_tokenize(texts, max_length=512):
    """``[CLS]=1``, one id per whitespace word, ``[SEP]=2``; truncated like a real tokenizer."""
    out = []
    for t in texts:
        words = [3 + zlib.crc32(w.encode()) % 90 for w in t.split()]
        out.append([1, *words[: max_length - 2], 2])
    return out


def make_review_data(n=3000, d=16, seed=0, noise=0.6, signal=2.0):
    """Unit-norm "embeddings" with a latent sentiment direction, and J-shaped star ratings.

    Returns ``(X float32 [n, d], y int64 in 1..5)``. ``signal`` and ``noise`` control how well the
    stars can be predicted from ``X``.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    s = rng.normal(size=n)
    X = signal * s[:, None] * w[None, :] + rng.normal(size=(n, d)) * 0.5
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.clip(np.round(3.7 + 1.2 * s + noise * rng.normal(size=n)), 1, 5).astype(
        np.int64
    )
    return X.astype(np.float32), y


def make_ordinal_data(n=20_000, d=6, seed=0, theta=(-1.5, -0.3, 0.8, 2.0)):
    """Stars drawn from an exact proportional-odds model: ``P(y <= k) = sigmoid(theta_k - f)``."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    X = rng.normal(size=(n, d)).astype(np.float32)
    f = X @ w
    latent = f + rng.logistic(size=n)
    y = 1 + (latent[:, None] > np.asarray(theta)[None, :]).sum(axis=1)
    return X, y.astype(np.int64), w, np.asarray(theta)


def make_gaussian_data(n=6000, d=6, seed=0, sigma=0.5):
    """Continuous targets with a known constant noise level."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    X = rng.normal(size=(n, d)).astype(np.float32)
    return X, 3.0 + X @ w + sigma * rng.normal(size=n), w


def make_probs(n=40_000, seed=0, sharpen=1.0, scale=1.6):
    """Labels drawn from known probabilities, and a model that reports them sharpened.

    Returns ``(P_model, y)``. ``sharpen == 1`` reports the truth, so the model is calibrated by
    construction (ECE near 0). ``sharpen > 1`` is overconfident, ``< 1`` underconfident, by an
    amount that is known exactly: ``P_model = softmax(sharpen * log P_true)``.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, 5)) * scale + np.array([0.0, -0.6, -0.4, 0.2, 1.0])
    true = np.exp(logits - logits.max(axis=1, keepdims=True))
    true /= true.sum(axis=1, keepdims=True)
    cum = true.cumsum(axis=1)
    y = (rng.uniform(size=(n, 1)) > cum).sum(axis=1) + 1
    shown = np.exp(sharpen * np.log(np.clip(true, 1e-300, None)))
    return shown / shown.sum(axis=1, keepdims=True), y.astype(np.int64)


def make_classification_preds(n=20_000, seed=0, sharpen=1.0):
    """``(Predictions, y)``: a classifier whose logits are ``sharpen`` times the true log-odds."""
    from review_stars.probs import Pred, Predictions

    P, y = make_probs(n=n, seed=seed, sharpen=sharpen)
    return Predictions(
        [Pred("classification", logits=np.log(np.clip(P, 1e-300, None)))]
    ), y


def make_regression_preds(n=20_000, seed=0, sigma_true=0.8, factor=1.0):
    """Stars from ``round(N(mu, sigma_true))`` (tails to 1 and 5); the model reports
    ``sigma = factor * sigma_true``, so ``factor < 1`` is overconfident by exactly ``1 / factor``."""
    from review_stars.probs import Pred, Predictions

    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.5, 5.5, n)
    y = np.clip(np.round(mu + sigma_true * rng.normal(size=n)), 1, 5).astype(np.int64)
    sigma = np.full(n, sigma_true * factor)
    return Predictions([Pred("regression", mu=mu, sigma=sigma)]), y


def make_ordinal_preds(n=20_000, seed=0, scale_true=1.0):
    """Stars from a cumulative-link model whose latent noise has scale ``scale_true``; the model
    reports ``T = 1``, so ``scale_true > 1`` means it is overconfident by exactly that factor."""
    from review_stars.probs import Pred, Predictions

    rng = np.random.default_rng(seed)
    theta = np.array([-2.0, -0.7, 0.6, 1.9])
    f = rng.normal(size=n) * 1.5
    latent = f + scale_true * rng.logistic(size=n)
    y = 1 + (latent[:, None] > theta[None, :]).sum(axis=1)
    return Predictions([Pred("ordinal", score=f, theta=theta)]), y.astype(np.int64)


def fake_choice(dim=8):
    """A ``BackendChoice`` over fake runners: embeds without any model or device."""
    from review_stars import embed

    runners = {b: FakeRunner(b, 4, dim) for b in embed.BUCKETS}
    enc = embed.BucketedEncoder(fake_tokenize, runners, pad_id=0)
    return embed.BackendChoice(enc, "fake", [("fake", "ok")], {})


SPLIT_SIZES = {
    "train": 700,
    "val": 150,
    "cal": 300,
    "test": 400,
    "ood_test": 400,
    "ood_pool": 200,
}


def tiny_config(**kw):
    """A ``Config`` small enough that every stage runs in seconds."""
    from review_stars.config import Config

    base = {
        "l2_grid": (1e-3,),
        "mlp_hidden": 8,
        "mlp_dropouts": (0.1,),
        "mlp_seeds": 2,
        "mlp_epochs": 3,
        "mlp_patience": 2,
        "mlp_lr": 3e-3,
        "mlp_batch": 128,
        "tfidf_features": 300,
        "tfidf_c_grid": (1.0,),
        "ridge_alpha_grid": (1.0,),
        "n_boot": 12,
        "recal_ns": (30, 80),
        "recal_draws": 3,
        "ablation_train": 300,
    }
    return Config(**{**base, **kw})


def make_features(seed=0, sizes=None, d=12, ood_shift=0.6):
    """A synthetic ``pipeline.Features``: embeddings and texts that both carry the star rating.

    ``ood_shift`` makes the out-of-domain splits harder (more noise, a lower star prior), so
    the models are over-confident there.
    """
    from review_stars.pipeline import Features, SplitData

    sizes = {**SPLIT_SIZES, **(sizes or {})}
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    splits = {}
    for name, n in sizes.items():
        ood = name.startswith("ood")
        s = rng.normal(size=n)
        stars = np.clip(
            np.round(3.7 + (0.8 if ood else 1.2) * s + 0.6 * rng.normal(size=n)), 1, 5
        )
        stars = stars.astype(np.int64)
        noise = 0.5 * (1 + ood_shift * ood)
        X = (stars - 3.0)[:, None] * w[None, :] * 1.2 + rng.normal(size=(n, d)) * noise
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        texts = []
        for k in stars:
            n_pos, n_neg = max(0, int(k) - 2), max(0, 4 - int(k))
            words = (
                list(rng.choice(POS, n_pos))
                + list(rng.choice(NEG, n_neg))
                + ["the", "item"]
            )
            rng.shuffle(words)
            texts.append(" ".join(words))
        splits[name] = SplitData(
            X=X.astype(np.float32),
            y=stars,
            groups=np.array([f"B{i}" for i in rng.integers(0, max(5, n // 4), n)]),
            texts=texts,
            n_tok=rng.integers(3, 300, n).astype(np.int32),
            verified=rng.uniform(size=n) < 0.85,
            ids=[f"{name}:{i}" for i in range(n)],
        )
    return Features(splits=splits, meta={"embedding": {"backend": "synthetic"}})


POS = ("great", "love", "perfect", "works", "excellent", "happy")
NEG = ("broken", "terrible", "refund", "useless", "waste", "stopped")


def make_texts(n=800, seed=0):
    """Review-like strings whose sentiment words determine the star rating: ``(texts, stars)``."""
    rng = np.random.default_rng(seed)
    texts, stars = [], []
    for _ in range(n):
        k = int(rng.integers(1, 6))
        n_pos, n_neg = (
            max(0, k - 2) + int(rng.integers(0, 2)),
            max(0, 4 - k) + int(rng.integers(0, 2)),
        )
        words = (
            list(rng.choice(POS, n_pos))
            + list(rng.choice(NEG, n_neg))
            + ["the", "item", "it"]
        )
        rng.shuffle(words)
        texts.append(" ".join(words))
        stars.append(k)
    return texts, np.array(stars, dtype=np.int64)


def read_gz_lines(path) -> list[str]:
    """The physical lines of a ``.jsonl.gz`` (a trailing newline does not add an empty line)."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return fh.read().splitlines()


def write_reviews_gz(path, rows, blank_line_at: int | None = None) -> Path:
    """A ``.jsonl.gz`` of ``rows`` (dicts); a blank line is inserted before row ``blank_line_at``."""
    path = Path(path)
    lines = []
    for i, r in enumerate(rows):
        if blank_line_at is not None and i == blank_line_at:
            lines.append("")
        lines.append(json.dumps(r))
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
