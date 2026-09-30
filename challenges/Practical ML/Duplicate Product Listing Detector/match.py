"""Score candidate pairs, tune the fusion weight + threshold, cluster, and measure.

Everything here is plain numpy over pair arrays of row indices, so it is
testable without models or an index. A "pair" is always ``[a, b]`` with
``a < b``, row indices into the embedding matrices.
"""

from dataclasses import dataclass

import numpy as np


def fuse(text_score: np.ndarray, image_score: np.ndarray, weight: float) -> np.ndarray:
    """``weight`` * text + (1 - weight) * image. weight=1 is text-only, 0 image-only."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    return weight * text_score + (1.0 - weight) * image_score


def pair_scores(
    text: np.ndarray, image: np.ndarray, pairs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity per pair for each modality (inputs are L2-normalized)."""
    a, b = pairs[:, 0], pairs[:, 1]
    return (text[a] * text[b]).sum(axis=1), (image[a] * image[b]).sum(axis=1)


def pair_metrics(tp: int, fp: int, n_true: int) -> tuple[float, float, float]:
    """(precision, recall, F1). ``n_true`` counts *all* true pairs, retrieved or not."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def best_threshold(
    scores: np.ndarray, labels: np.ndarray, n_true: int
) -> tuple[float, float]:
    """Threshold maximizing pair F1, scanning every distinct observed score.

    Recall is measured against ``n_true`` -- every true pair in the split -- not
    just the ones candidate retrieval surfaced, so a retrieval miss costs recall
    instead of being invisible. Pairs flagged are those with ``score >= threshold``.
    Returns ``(threshold, f1)``; with no positives the threshold flags nothing.
    """
    if n_true == 0 or len(scores) == 0:
        return (float(scores.max()) + 1e-6 if len(scores) else 1.0), 0.0
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    tp = np.cumsum(y)
    flagged = np.arange(1, len(s) + 1)
    precision = tp / flagged
    recall = tp / n_true
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    # Only cut between distinct scores: flagging half of a tie block is meaningless.
    is_block_end = np.append(s[1:] != s[:-1], True)
    f1 = np.where(is_block_end, f1, -1.0)
    k = int(np.argmax(f1))
    return float(s[k]), float(f1[k])


@dataclass
class Tuned:
    weight: float
    threshold: float
    f1: float


def tune(
    text_score: np.ndarray,
    image_score: np.ndarray,
    labels: np.ndarray,
    n_true: int,
    weights: np.ndarray | None = None,
) -> Tuned:
    """Grid-search the fusion weight; each weight gets its own best threshold."""
    if weights is None:
        weights = np.linspace(0.0, 1.0, 21)
    best = Tuned(float(weights[0]), 1.0, -1.0)
    for w in weights:
        thr, f1 = best_threshold(
            fuse(text_score, image_score, float(w)), labels, n_true
        )
        if f1 > best.f1:
            best = Tuned(float(w), thr, f1)
    return best


def cluster(n: int, flagged_pairs: np.ndarray) -> np.ndarray:
    """Connected components over the flagged pairs (union-find); returns a label per row."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    for a, b in flagged_pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def cluster_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Mean per-listing F1 between its predicted cluster and its true group.

    For one listing, F1 = 2|P ∩ T| / (|P| + |T|) where P is its predicted
    cluster and T its true duplicate group (both include the listing itself).
    """
    sizes_t = {g: c for g, c in zip(*np.unique(truth, return_counts=True), strict=True)}
    sizes_p = {
        g: c for g, c in zip(*np.unique(predicted, return_counts=True), strict=True)
    }
    joint: dict[tuple[int, int], int] = {}
    for t, p in zip(truth.tolist(), predicted.tolist(), strict=True):
        joint[(t, p)] = joint.get((t, p), 0) + 1
    total = 0.0
    for t, p in zip(truth.tolist(), predicted.tolist(), strict=True):
        total += 2 * joint[(t, p)] / (sizes_p[p] + sizes_t[t])
    return total / len(truth)
