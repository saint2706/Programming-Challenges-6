"""Prepare a split for scoring, tune on validation, report on test.

The protocol that keeps the numbers honest:

* validation and test are disjoint sets of *groups*, each with its own index, so
  retrieval only ever searches within the split being evaluated (as a real
  "dedupe this catalog" run would);
* the fusion weight and decision threshold of every mode are chosen on
  validation labels only, then applied unchanged to test;
* recall is always measured against **all** true pairs in the split, so pairs
  that candidate retrieval never surfaced still count against the score.
"""

from dataclasses import dataclass
from pathlib import Path

import index
import match
import numpy as np
import polars as pl
from embed import Embeddings

MODES = ("text", "image", "fused")


@dataclass
class SplitData:
    df: pl.DataFrame
    emb: Embeddings
    pairs: np.ndarray  # (m, 2) candidate pairs, row indices, a < b
    text_score: np.ndarray  # cosine per candidate pair
    image_score: np.ndarray
    labels: np.ndarray  # 1 if the pair is a true duplicate
    n_true: int  # true pairs in the whole split, retrieved or not
    candidate_recall: float
    group_codes: np.ndarray  # label_group per row


@dataclass
class ModeResult:
    weight: float
    threshold: float
    precision: float
    recall: float
    f1: float
    cluster_f1: float


@dataclass
class EvalReport:
    modes: dict[str, ModeResult]
    candidate_recall: float
    n_listings: int
    n_true_pairs: int
    n_candidates: int

    def to_markdown(self) -> str:
        lines = [
            "| scoring | text weight | threshold | precision | recall | pair F1 | cluster F1 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name in MODES:
            m = self.modes[name]
            lines.append(
                f"| {name} | {m.weight:.2f} | {m.threshold:.3f} | {m.precision:.3f} | {m.recall:.3f} "
                f"| {m.f1:.3f} | {m.cluster_f1:.3f} |"
            )
        return "\n".join(lines)


def _true_index_pairs(group_codes: np.ndarray) -> set[tuple[int, int]]:
    by_group: dict[int, list[int]] = {}
    for row, g in enumerate(group_codes.tolist()):
        by_group.setdefault(g, []).append(row)
    return {
        (a, b)
        for rows in by_group.values()
        for i, a in enumerate(rows)
        for b in rows[i + 1 :]
    }


def prepare_split(
    df: pl.DataFrame, emb: Embeddings, db_dir: Path, name: str, k: int = 10
) -> SplitData:
    """Index the split in LanceDB, retrieve candidate pairs, score and label them."""
    table = index.build_index(db_dir, name, df, emb)
    pairs = index.candidate_pairs(table, emb, k=k)
    group_codes = df["label_group"].to_numpy()
    true_pairs = _true_index_pairs(group_codes)
    text_score, image_score = match.pair_scores(emb.text, emb.image, pairs)
    labels = (group_codes[pairs[:, 0]] == group_codes[pairs[:, 1]]).astype(int)
    return SplitData(
        df=df,
        emb=emb,
        pairs=pairs,
        text_score=text_score,
        image_score=image_score,
        labels=labels,
        n_true=len(true_pairs),
        candidate_recall=index.candidate_recall(pairs, true_pairs),
        group_codes=group_codes,
    )


def flag_pairs(split: SplitData, weight: float, threshold: float) -> np.ndarray:
    """Candidate pairs whose fused score reaches `threshold`."""
    score = match.fuse(split.text_score, split.image_score, weight)
    return split.pairs[score >= threshold]


@dataclass
class PairLevel:
    precision: float
    recall: float
    f1: float
    n_flagged: int


def pair_level(split: SplitData, weight: float, threshold: float) -> PairLevel:
    """Pair precision/recall/F1 at an arbitrary operating point (what the UI sliders drive)."""
    flagged = match.fuse(split.text_score, split.image_score, weight) >= threshold
    tp = int(split.labels[flagged].sum())
    n_flagged = int(flagged.sum())
    precision, recall, f1 = match.pair_metrics(tp, n_flagged - tp, split.n_true)
    return PairLevel(precision, recall, f1, n_flagged)


def _apply(split: SplitData, weight: float, threshold: float) -> ModeResult:
    live = pair_level(split, weight, threshold)
    flagged = flag_pairs(split, weight, threshold)
    clusters = match.cluster(split.df.height, flagged)
    return ModeResult(
        weight,
        threshold,
        live.precision,
        live.recall,
        live.f1,
        match.cluster_f1(split.group_codes, clusters),
    )


def evaluate(val: SplitData, test: SplitData) -> EvalReport:
    """Tune each mode on `val`, then measure it on `test` with those settings frozen."""
    modes: dict[str, ModeResult] = {}
    for name, fixed_weight in (("text", 1.0), ("image", 0.0)):
        threshold, _ = match.best_threshold(
            match.fuse(val.text_score, val.image_score, fixed_weight),
            val.labels,
            val.n_true,
        )
        modes[name] = _apply(test, fixed_weight, threshold)
    tuned = match.tune(val.text_score, val.image_score, val.labels, val.n_true)
    modes["fused"] = _apply(test, tuned.weight, tuned.threshold)
    return EvalReport(
        modes=modes,
        candidate_recall=test.candidate_recall,
        n_listings=test.df.height,
        n_true_pairs=test.n_true,
        n_candidates=len(test.pairs),
    )


def duplicate_clusters(
    split: SplitData, weight: float, threshold: float
) -> pl.DataFrame:
    """Listings grouped into predicted duplicate clusters (size >= 2 only)."""
    labels = match.cluster(split.df.height, flag_pairs(split, weight, threshold))
    sizes = np.bincount(labels, minlength=split.df.height)
    keep = sizes[labels] >= 2
    return (
        split.df.with_columns(pl.Series("cluster_id", labels))
        .filter(pl.Series(keep))
        .select(["cluster_id", "posting_id", "title", "image"])
        .sort(["cluster_id", "posting_id"])
    )


def review_pairs(
    split: SplitData,
    weight: float,
    threshold: float,
    band: float | None = None,
    limit: int = 50,
) -> pl.DataFrame:
    """Candidate pairs for human review, highest fused score first.

    ``band`` keeps only pairs within +-band of the threshold -- the borderline
    cases where a reviewer's judgement is actually needed.
    """
    fused = match.fuse(split.text_score, split.image_score, weight)
    keep = (
        np.ones(len(fused), dtype=bool)
        if band is None
        else np.abs(fused - threshold) <= band
    )
    order = np.flatnonzero(keep)
    order = order[np.argsort(-fused[order], kind="stable")][:limit]
    a, b = split.pairs[order, 0], split.pairs[order, 1]
    df = split.df
    return pl.DataFrame(
        {
            "title_a": df["title"].gather(a),
            "title_b": df["title"].gather(b),
            "image_a": df["image"].gather(a),
            "image_b": df["image"].gather(b),
            "text": split.text_score[order],
            "image": split.image_score[order],
            "fused": fused[order],
            "flagged": fused[order] >= threshold,
            "is_duplicate": split.labels[order].astype(bool),
        }
    )
