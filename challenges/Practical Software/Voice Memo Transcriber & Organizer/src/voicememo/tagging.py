"""Topic tags: sentence embeddings, HDBSCAN clustering, c-TF-IDF keyword labels.

Pipeline (BERTopic-style, without the UMAP step):
1. Embed every transcript. Default: model2vec static embeddings if pulled, otherwise LSA
   (TF-IDF + truncated SVD), which needs no model download and is what the tests use.
2. Cluster with sklearn's HDBSCAN (deterministic; no random seed involved).
3. Label each cluster with its top class-based TF-IDF terms.
Memos that HDBSCAN leaves as noise, and every memo in a tiny library, get per-memo keywords.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

MIN_CLUSTER_DOCS = (
    12  # below this many memos, clustering is noise; use per-memo keywords
)
MIN_CLUSTER_SIZE = 4
KEYWORDS_PER_TAG = 2
STOP_WORDS = "english"


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray | None: ...


class LsaEmbedder:
    """TF-IDF followed by truncated SVD. Deterministic (fixed random_state)."""

    name = "lsa"

    def __init__(self, dims: int = 64) -> None:
        self.dims = dims

    def embed(self, texts: list[str]) -> np.ndarray | None:
        try:
            matrix = TfidfVectorizer(
                sublinear_tf=True, stop_words=STOP_WORDS, ngram_range=(1, 2)
            ).fit_transform(texts)
        except (
            ValueError
        ):  # empty vocabulary: every transcript is blank or stop words only
            return None
        n_components = min(self.dims, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if n_components < 1:
            return normalize(matrix.toarray())
        reduced = TruncatedSVD(n_components=n_components, random_state=0).fit_transform(
            matrix
        )
        return normalize(reduced)


class Model2VecEmbedder:
    """Static sentence embeddings from a pulled model2vec model (no torch involved)."""

    name = "model2vec"

    def __init__(self, directory: Path) -> None:
        from model2vec import StaticModel

        self._model = StaticModel.from_pretrained(str(directory))

    def embed(self, texts: list[str]) -> np.ndarray | None:
        vectors = np.asarray(self._model.encode(texts), dtype=np.float64)
        return normalize(vectors)


def ctfidf_keywords(docs: list[str], top_k: int = KEYWORDS_PER_TAG) -> list[list[str]]:
    """Class-based TF-IDF: one 'document' per class, score = in-class share * log(1 + avg/freq)."""
    if not docs:
        return []
    counts = CountVectorizer(
        stop_words=STOP_WORDS, token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b"
    )
    try:
        matrix = counts.fit_transform(docs).toarray().astype(np.float64)
    except ValueError:
        return [[] for _ in docs]
    terms = counts.get_feature_names_out()
    row_totals = matrix.sum(axis=1, keepdims=True)
    tf = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0)
    avg_words = float(row_totals.mean()) or 1.0
    freq = matrix.sum(axis=0)
    idf = np.log(1.0 + avg_words / np.maximum(freq, 1.0))
    scores = tf * idf
    out: list[list[str]] = []
    for row in scores:
        order = sorted(range(len(terms)), key=lambda j: (-row[j], terms[j]))
        out.append([terms[j] for j in order[:top_k] if row[j] > 0])
    return out


@dataclass(frozen=True)
class TagPlan:
    method: str  # "clusters" or "per-memo-keywords"
    assignments: dict[int, list[str]]
    labels: dict[int, int]  # memo id -> cluster label (-1 = noise / per-memo)


def _per_memo(texts: dict[int, str]) -> TagPlan:
    ids = sorted(texts)
    keywords = ctfidf_keywords([texts[i] for i in ids])
    assignments = {mid: kws for mid, kws in zip(ids, keywords, strict=True)}
    return TagPlan("per-memo-keywords", assignments, {mid: -1 for mid in ids})


def plan_tags(texts: dict[int, str], embedder: Embedder | None = None) -> TagPlan:
    """Decide the automatic tags for every memo in ``texts`` (memo id -> transcript)."""
    ids = sorted(texts)
    if len(ids) < MIN_CLUSTER_DOCS:
        return _per_memo(texts)
    embedder = embedder or LsaEmbedder()
    vectors = embedder.embed([texts[i] for i in ids])
    if vectors is None:
        return _per_memo(texts)
    labels = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean", copy=True
    ).fit_predict(vectors)
    if not np.any(labels >= 0):
        return _per_memo(texts)

    cluster_ids = sorted({int(lbl) for lbl in labels if lbl >= 0})
    cluster_docs = [
        " ".join(texts[ids[i]] for i in range(len(ids)) if labels[i] == c)
        for c in cluster_ids
    ]
    cluster_keywords = dict(
        zip(cluster_ids, ctfidf_keywords(cluster_docs), strict=True)
    )

    noise_ids = [ids[i] for i in range(len(ids)) if labels[i] < 0]
    noise_keywords = dict(
        zip(noise_ids, ctfidf_keywords([texts[m] for m in noise_ids]), strict=True)
    )

    assignments: dict[int, list[str]] = {}
    label_of: dict[int, int] = {}
    for i, memo_id in enumerate(ids):
        lbl = int(labels[i])
        label_of[memo_id] = lbl
        if lbl >= 0:
            name = " ".join(cluster_keywords[lbl]) or f"topic {lbl}"
            assignments[memo_id] = [name]
        else:
            assignments[memo_id] = noise_keywords[memo_id][:1]
    return TagPlan("clusters", assignments, label_of)


def evaluate(
    items: list[tuple[str, str]], embedder: Embedder | None = None
) -> dict[str, object]:
    """Score clustering against gold topics. ``items`` = (transcript, gold topic) pairs.

    Adjusted Rand index compares the whole partition with the gold partition; noise memos count
    as singletons, so leaving memos unclustered costs ARI. Purity is computed over clustered memos
    only (noise has no cluster to be pure), and ``covered`` reports how many memos were clustered.
    """
    texts = {i: text for i, (text, _gold) in enumerate(items)}
    golds = [gold for _text, gold in items]
    plan = plan_tags(texts, embedder)
    predicted: list[str] = []
    for memo_id in sorted(texts):
        lbl = plan.labels[memo_id]
        predicted.append(f"c{lbl}" if lbl >= 0 else f"solo{memo_id}")
    ari = adjusted_rand_score(golds, predicted)
    groups: dict[str, list[str]] = {}
    for pred, gold in zip(predicted, golds, strict=True):
        if pred.startswith("c"):
            groups.setdefault(pred, []).append(gold)
    covered = sum(len(g) for g in groups.values())
    majority = sum(max(_counts(g).values()) for g in groups.values())
    purity = majority / covered if covered else 0.0
    noise = sum(1 for lbl in plan.labels.values() if lbl < 0) / max(1, len(plan.labels))
    return {
        "method": plan.method,
        "embedder": embedder.name if embedder else LsaEmbedder.name,
        "memos": len(golds),
        "clusters": len({lbl for lbl in plan.labels.values() if lbl >= 0}),
        "covered": covered,
        "noise_share": round(noise, 3),
        "adjusted_rand": round(float(ari), 3),
        "purity": round(float(purity), 3),
    }


def _counts(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
