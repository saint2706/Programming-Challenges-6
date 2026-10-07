"""Test helpers: a synthetic stand-in for an embedded text pool."""

from __future__ import annotations

import zlib

import numpy as np


def make_pool(n=600, k=10, d=16, seed=0, spread=0.35, imbalance=0.75):
    """Unit-norm Gaussian blobs with imbalanced classes: ``(X [n, d], y [n])``.

    ``spread`` is the (approximate) norm of the within-class noise against unit-norm class
    centers, so ~0.3 is cleanly separable and ~0.9 overlaps heavily. Items ``3c .. 3c+2`` are
    class ``c`` so every class is present; the rest follow geometric class shares.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    share = imbalance ** np.arange(k)
    y = rng.choice(k, size=n, p=share / share.sum())
    y[: 3 * k] = np.repeat(np.arange(k), 3)
    X = centers[y] + spread * rng.normal(size=(n, d)) / np.sqrt(d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, y.astype(np.int64)


def make_state(n=300, k=6, n_labeled=30, seed=0, spread=0.6, C=10.0, all_classes=True):
    """A fitted-head ``State`` over a synthetic pool, plus the gold labels of the whole pool."""
    from model import Head
    from strategies import State

    X, y = make_pool(n, k, seed=seed, spread=spread)
    rng = np.random.default_rng(seed + 1)
    first = np.arange(0, 3 * k, 3) if all_classes else np.array([], dtype=np.int64)
    rest = rng.choice(
        np.setdiff1d(np.arange(n), first), size=n_labeled - len(first), replace=False
    )
    labeled = np.concatenate([first, rest]).astype(np.int64)
    head = Head(k, C).fit(X[labeled], y[labeled])
    state = State(X=X, labeled=labeled, y=y[labeled], P=head.proba(X), n_classes=k, C=C)
    return state, y


def assert_valid(sel, state, b):
    """The contract every strategy must meet."""
    idx = np.asarray(sel.idx)
    assert idx.ndim == 1
    assert len(idx) == min(b, len(state.unlabeled()))
    assert len(set(idx.tolist())) == len(idx)
    assert ((idx >= 0) & (idx < len(state.X))).all()
    assert not np.isin(idx, state.labeled).any()
    assert len(sel.score) == len(idx)


def make_problem(n=400, k=6, seed=0, spread=0.6):
    """A small ``loop.Problem``: pool, validation and evaluation items from one distribution."""
    from loop import Problem

    X, y = make_pool(n + 400, k, seed=seed, spread=spread)
    return Problem(
        X=X[:n],
        y=y[:n],
        X_eval=X[n + 200 :],
        y_eval=y[n + 200 :],
        n_classes=k,
        C=10.0,
        X_val=X[n : n + 200],
        y_val=y[n : n + 200],
    )


class FakeEncoder:
    """Deterministic bag-of-words embedding: no model, no network."""

    dim = 16

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i] += np.random.default_rng(zlib.crc32(w.encode())).normal(
                    size=self.dim
                )
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1.0, norm)


def fake_choice():
    from embed import BackendChoice

    return BackendChoice(FakeEncoder(), "fake", [("fake", "ok")])


def tiny_config():
    """A ``pipeline.Config`` small enough for tests: 2 seeds, 4 rounds of 10 after 12 cold-start labels."""
    from pipeline import Config

    return Config(
        seeds=2,
        tune_seeds=2,
        budget=30,
        batch=10,
        init=12,
        batch_sizes=(5, 10),
        sens_seeds=2,
        sens_strategies=("random", "margin"),
        n_clusters=10,
        n_boot=50,
    )
