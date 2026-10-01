"""Explanations that come from the scorer itself.

For the sparse scorers, ``score = sum_t query[t] * doc[t]``, so the per-term
products are an *exact* decomposition: they add up to the score, and the top
ones are the terms that earned it. Nothing here is a separate model.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
from scorers import _SparseScorer


def contributions(
    scorer: _SparseScorer, query: str, doc: str, k: int = 10
) -> list[tuple[str, float]]:
    """Top-``k`` ``(term, contribution)`` pairs, largest first; they sum to the score."""
    product = scorer.query_matrix([query]).multiply(scorer.doc_matrix([doc])).tocsr()
    names = scorer.feature_names
    pairs = [
        (str(names[i]), float(v))
        for i, v in zip(product.indices, product.data, strict=True)
        if v > 0
    ]
    pairs.sort(key=lambda p: (-p[1], p[0]))
    return pairs[:k]


tfidf_terms = bm25_terms = contributions


def gaps(
    scorer: _SparseScorer,
    query: str,
    doc: str,
    common_terms: Collection[str] | None,
    k: int = 10,
) -> list[str]:
    """Heavily weighted job terms the resume does not contain.

    ``common_terms`` limits the list to terms that many postings of the matched
    category share (see :func:`category_common_terms`), so a one-off phrase in a
    single posting is not reported as a skill gap.
    """
    names = scorer.feature_names
    q = scorer.query_matrix([query]).tocsr()
    d = scorer.doc_matrix([doc]).tocsr()
    in_resume = {str(names[i]) for i in q.indices[q.data > 0]}
    allowed = None if common_terms is None else set(common_terms)
    ranked = sorted(
        (
            (str(names[i]), float(v))
            for i, v in zip(d.indices, d.data, strict=True)
            if v > 0
        ),
        key=lambda p: (-p[1], p[0]),
    )
    return [
        t for t, _ in ranked if t not in in_resume and (allowed is None or t in allowed)
    ][:k]


def category_common_terms(
    scorer: _SparseScorer, docs: list[str], min_share: float = 0.02
) -> set[str]:
    """Terms present in at least ``min_share`` of ``docs`` (one category's postings)."""
    if not docs:
        return set()
    present = (scorer.doc_matrix(docs) > 0).astype(np.float64)
    share = np.asarray(present.mean(axis=0)).ravel()
    names = scorer.feature_names
    return {str(names[i]) for i in np.flatnonzero(share >= min_share)}
