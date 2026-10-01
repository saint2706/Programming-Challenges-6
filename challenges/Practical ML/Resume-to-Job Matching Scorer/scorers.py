"""Scorers: sparse (TF-IDF, BM25) here, dense in ``embed.py``, fused below.

Every scorer exposes ``fit(corpus)`` and ``score_matrix(queries, docs)``. The
two sparse ones also expose ``query_matrix``/``doc_matrix`` such that
``score = query_matrix @ doc_matrix.T`` -- which is what makes their
explanations an exact decomposition of the score (see ``explain.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, Self

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Scorer(Protocol):
    name: str

    def fit(self, corpus: Sequence[str]) -> Self: ...

    def score_matrix(self, queries: Sequence[str], docs: Sequence[str]) -> np.ndarray:
        """``[len(queries), len(docs)]`` relevance scores, higher is better."""
        ...

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray: ...


class _SparseScorer:
    name = "sparse"
    _fitted = False

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.name} scorer is not fitted: call fit(corpus) first"
            )

    def query_matrix(self, texts: Sequence[str]) -> sparse.csr_matrix:
        raise NotImplementedError

    def doc_matrix(self, texts: Sequence[str]) -> sparse.csr_matrix:
        raise NotImplementedError

    @property
    def feature_names(self) -> np.ndarray:
        raise NotImplementedError

    def score_matrix(self, queries: Sequence[str], docs: Sequence[str]) -> np.ndarray:
        return (self.query_matrix(queries) @ self.doc_matrix(docs).T).toarray()

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        return self.score_matrix([query], docs)[0]


class TfidfScorer(_SparseScorer):
    """Cosine similarity of sublinear-tf, L2-normalised TF-IDF vectors."""

    name = "tfidf"

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), min_df: int = 2):
        self._vec = TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words="english",
            dtype=np.float64,
        )

    def fit(self, corpus: Sequence[str]) -> Self:
        self._vec.fit(corpus)
        self._fitted = True
        return self

    def _transform(self, texts: Sequence[str]) -> sparse.csr_matrix:
        self._require_fit()
        return self._vec.transform(texts).tocsr()

    query_matrix = doc_matrix = _transform

    @property
    def feature_names(self) -> np.ndarray:
        self._require_fit()
        return self._vec.get_feature_names_out()


class Bm25Scorer(_SparseScorer):
    """Lucene-style BM25.

    ``idf = ln(1 + (N - n + 0.5) / (n + 0.5))`` and the term-frequency part is
    ``tf / (tf + k1 * (1 - b + b * dl / avgdl))`` -- Lucene drops Okapi's constant
    ``(k1 + 1)`` factor, which does not change any ranking. Query terms count
    once each (presence), so a long resume does not multiply a term's weight.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._vec = CountVectorizer(stop_words="english")

    def fit(self, corpus: Sequence[str]) -> Self:
        counts = self._vec.fit_transform(corpus).tocsr()
        n_docs = counts.shape[0]
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        self._idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        self._avgdl = max(float(counts.sum(axis=1).mean()), 1.0)
        self._fitted = True
        return self

    def query_matrix(self, texts: Sequence[str]) -> sparse.csr_matrix:
        self._require_fit()
        return (self._vec.transform(texts) > 0).astype(np.float64).tocsr()

    def doc_matrix(self, texts: Sequence[str]) -> sparse.csr_matrix:
        self._require_fit()
        tf = self._vec.transform(texts).astype(np.float64).tocsr()
        dl = np.asarray(tf.sum(axis=1)).ravel()
        row = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
        norm = self.k1 * (1 - self.b + self.b * dl[row] / self._avgdl)
        weighted = tf.copy()
        weighted.data = self._idf[tf.indices] * tf.data / (tf.data + norm)
        return weighted

    @property
    def feature_names(self) -> np.ndarray:
        self._require_fit()
        return self._vec.get_feature_names_out()


def zscore(matrix: np.ndarray) -> np.ndarray:
    """Per-row z-score; a constant row (no signal) becomes zeros, never NaN."""
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return np.where(std > 0, (matrix - mean) / np.where(std == 0, 1.0, std), 0.0)


def rrf(score_matrices: Sequence[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion: ``sum 1 / (k + rank)`` per document.

    Uses ranks only, so it needs no weight and is blind to the score scales,
    which is why it is reported next to the tuned weighted fusion.
    """
    fused = np.zeros_like(score_matrices[0], dtype=np.float64)
    for m in score_matrices:
        order = np.argsort(-m, axis=1, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(
            ranks,
            order,
            np.arange(m.shape[1])[None, :].repeat(m.shape[0], axis=0),
            axis=1,
        )
        fused += 1.0 / (k + ranks + 1)
    return fused


class FusionScorer:
    """Weighted sum of per-query z-scored parts.

    Raw cosine (0..1) and BM25 (0..tens) are not on one scale, so each part is
    standardised per query before weighting.
    """

    name = "fusion"

    def __init__(self, parts: Sequence[Scorer], weights: Sequence[float]):
        if len(parts) != len(weights):
            raise ValueError(
                f"{len(parts)} parts need {len(parts)} weights, got {len(weights)}"
            )
        self.parts = list(parts)
        self.weights = list(weights)

    def fit(self, corpus: Sequence[str]) -> Self:
        for p in self.parts:
            p.fit(corpus)
        return self

    def score_matrix(self, queries: Sequence[str], docs: Sequence[str]) -> np.ndarray:
        return sum(
            w * zscore(p.score_matrix(queries, docs))
            for p, w in zip(self.parts, self.weights, strict=True)
        )

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        return self.score_matrix([query], docs)[0]
