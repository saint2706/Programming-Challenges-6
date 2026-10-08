"""Glue: prepare splits, evaluate every scorer, write the report.

Protocol (the order matters for honesty):

1. Resumes and balanced job postings are each split by id into disjoint
   validation and test halves, stratified by category.
2. For each variant (``full`` resume text, ``stripped`` of its headline) and
   each direction, sparse-scorer statistics are fit on that split's own texts
   only (no labels involved).
3. The dense/sparse fusion weight is tuned on *validation* and frozen.
4. Everything is reported on *test*.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from resume_matcher import data, labels
from resume_matcher.embed import (
    BackendChoice,
    EmbeddingScorer,
    Encoder,
    default_encoder,
)
from resume_matcher.evaluate import random_scores, summarise, tune_fusion_weight
from resume_matcher.scorers import Bm25Scorer, TfidfScorer, rrf, zscore

GRID = np.round(np.linspace(0.0, 1.0, 11), 2)
SCORERS = ("random", "tfidf", "bm25", "embedding", "fusion", "rrf")


@dataclass
class Prepared:
    resumes_val: pl.DataFrame
    resumes_test: pl.DataFrame
    jobs_val: pl.DataFrame
    jobs_test: pl.DataFrame
    skipped: dict[str, int] = field(default_factory=dict)


def prepare(
    resumes: pl.DataFrame,
    jobs: pl.DataFrame,
    per_category: int = 200,
    min_pool: int = 20,
    seed: int = 0,
) -> Prepared:
    """Balanced job pool, disjoint val/test halves, and the categories that had to be dropped.

    ``jobs`` must already carry ``category`` (see ``labels.label_postings``).
    A category with fewer than ``min_pool`` postings cannot be ranked against
    meaningfully, so it is skipped on both sides and reported, not hidden.
    """
    pool = labels.balanced_pool(jobs, per_category, seed)
    counts = dict(pool.group_by("category").len().iter_rows())
    resume_cats = set(resumes["category"])
    usable = {c for c, n in counts.items() if n >= min_pool} & resume_cats
    skipped = {c: counts.get(c, 0) for c in (set(counts) | resume_cats) - usable}
    pool = pool.filter(pl.col("category").is_in(usable))
    resumes = resumes.filter(pl.col("category").is_in(usable))

    def halves(df: pl.DataFrame, key: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        val, test = data.split_ids(
            df[key].to_list(), df["category"].to_list(), 0.5, seed
        )
        data.check_split_disjoint(val, test)
        return df.filter(pl.col(key).is_in(val)), df.filter(pl.col(key).is_in(test))

    resumes_val, resumes_test = halves(resumes, "id")
    jobs_val, jobs_test = halves(pool, "job_id")
    return Prepared(resumes_val, resumes_test, jobs_val, jobs_test, skipped)


def _matrices(
    queries: Sequence[str], docs: Sequence[str], encoder: Encoder
) -> dict[str, np.ndarray]:
    fit_corpus = [*queries, *docs]
    return {
        "tfidf": TfidfScorer().fit(fit_corpus).score_matrix(queries, docs),
        "bm25": Bm25Scorer().fit(fit_corpus).score_matrix(queries, docs),
        "embedding": EmbeddingScorer(encoder).score_matrix(queries, docs),
    }


def _relevance(query_cats: Sequence[str], doc_cats: Sequence[str]) -> np.ndarray:
    return np.array(query_cats)[:, None] == np.array(doc_cats)[None, :]


def evaluate_all(
    p: Prepared, encoder: Encoder, seed: int = 0, grid: Sequence[float] = GRID
) -> dict:
    results: dict = {}
    fusion_weight: dict[str, float] = {}
    for variant, transform in (
        ("full", lambda t: t),
        ("stripped", data.strip_headline),
    ):
        r_val = [transform(t) for t in p.resumes_val["text"]]
        val = _matrices(r_val, p.jobs_val["text"].to_list(), encoder)
        w = tune_fusion_weight(
            val["embedding"],
            val["bm25"],
            _relevance(p.resumes_val["category"], p.jobs_val["category"]),
            grid,
        )
        fusion_weight[variant] = w

        r_test = [transform(t) for t in p.resumes_test["text"]]
        j_test = p.jobs_test["text"].to_list()
        results[variant] = {}
        for direction, queries, docs, q_cats, d_cats in (
            (
                "resume->jobs",
                r_test,
                j_test,
                p.resumes_test["category"],
                p.jobs_test["category"],
            ),
            (
                "jobs->resumes",
                j_test,
                r_test,
                p.jobs_test["category"],
                p.resumes_test["category"],
            ),
        ):
            m = _matrices(queries, docs, encoder)
            rel = _relevance(q_cats, d_cats)
            scores = {
                "random": random_scores(rel.shape, seed),
                **m,
                "fusion": w * zscore(m["embedding"]) + (1 - w) * zscore(m["bm25"]),
                "rrf": rrf([m["embedding"], m["bm25"]]),
            }
            results[variant][direction] = {
                name: summarise(scores[name], rel, list(q_cats), seed)
                for name in SCORERS
            }
    return {
        "results": results,
        "fusion_weight": fusion_weight,
        "skipped_categories": p.skipped,
        "n": {
            "val_resumes": p.resumes_val.height,
            "test_resumes": p.resumes_test.height,
            "val_jobs": p.jobs_val.height,
            "test_jobs": p.jobs_test.height,
        },
    }


def run_all(
    data_dir: Path,
    out_dir: Path,
    per_category: int = 200,
    min_pool: int = 20,
    seed: int = 0,
    encoder_factory: Callable[[Path], BackendChoice] = default_encoder,
) -> dict:
    """Load, label, split, embed, evaluate, and write ``report.json`` to ``out_dir``."""
    resumes = data.load_resumes(data_dir / "Resume.csv")
    jobs = labels.label_postings(data.load_postings(data_dir / "postings.csv"))
    prepared = prepare(resumes, jobs, per_category, min_pool, seed)
    choice = encoder_factory(data_dir / "embed_cache.sqlite")
    report = evaluate_all(prepared, choice.encoder, seed)
    report["backend"] = {"name": choice.name, "tried": choice.tried}
    report["config"] = {
        "per_category": per_category,
        "min_pool": min_pool,
        "seed": seed,
        "labelled_postings": jobs.height,
        "resumes": resumes.height,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
