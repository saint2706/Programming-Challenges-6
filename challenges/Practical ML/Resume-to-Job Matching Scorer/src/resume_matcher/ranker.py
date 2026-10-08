"""One ranking service behind both the CLI and the Streamlit page.

Built once from a labelled job pool and the resumes: sparse scorers are fit
and their document matrices cached, job vectors go into a LanceDB table, and
every result carries the evidence for its score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from resume_matcher import data, explain, labels
from resume_matcher.embed import Encoder, default_encoder
from resume_matcher.index import JobIndex
from resume_matcher.paths import project_root
from resume_matcher.scorers import Bm25Scorer, TfidfScorer, zscore

SCORER_NAMES = ("fusion", "embedding", "bm25", "tfidf")
RESULTS = project_root() / "results" / "report.json"
COMMON_MIN_SHARE = 0.1


@dataclass
class Match:
    id: str
    title: str
    category: str
    score: float
    terms: list[tuple[str, float]] = field(default_factory=list)
    # terms beyond the listed ones: for an "exact" explanation,
    # sum(terms) + terms_rest == score
    terms_rest: float = 0.0
    terms_total: int = 0
    gaps: list[str] = field(default_factory=list)
    # "exact": the terms sum to the score. "lexical-overlap": the score came from
    # embeddings, and the terms are only the words the two texts share.
    explanation: str = "exact"

    def evidence_note(self) -> str:
        """What the listed terms do and do not account for, in one line."""
        if self.explanation != "exact":
            return "lexical overlap only; the score came from embeddings"
        listed = len(self.terms)
        if self.terms_total > listed:
            return (
                f"top {listed} of {self.terms_total} terms; the other "
                f"{self.terms_total - listed} add {self.terms_rest:.3f}, "
                "so all of them sum to the score"
            )
        noun = "term" if self.terms_total == 1 else "terms"
        return f"all {self.terms_total} {noun}; they sum to the score"


class Ranker:
    def __init__(
        self,
        jobs: pl.DataFrame,
        resumes: pl.DataFrame,
        encoder: Encoder,
        index: JobIndex,
        fusion_weight: float,
    ):
        self.jobs, self.resumes, self.encoder = jobs, resumes, encoder
        self.index, self.fusion_weight = index, fusion_weight
        self.job_texts = jobs["text"].to_list()
        self.resume_texts = resumes["text"].to_list()
        corpus = [*self.job_texts, *self.resume_texts]
        self.sparse = {
            "tfidf": TfidfScorer().fit(corpus),
            "bm25": Bm25Scorer().fit(corpus),
        }
        self._docs = {
            name: {
                "jobs": s.doc_matrix(self.job_texts),
                "resumes": s.doc_matrix(self.resume_texts),
            }
            for name, s in self.sparse.items()
        }
        self.job_vecs = encoder.encode(self.job_texts)
        self.resume_vecs = encoder.encode(self.resume_texts)
        job_cats = jobs["category"].to_list()
        self.common = {
            cat: explain.category_common_terms(
                self.sparse["tfidf"],
                [t for t, c in zip(self.job_texts, job_cats, strict=True) if c == cat],
                COMMON_MIN_SHARE,
            )
            for cat in set(job_cats)
        }

    @classmethod
    def build(
        cls,
        jobs: pl.DataFrame,
        resumes: pl.DataFrame,
        encoder: Encoder,
        index_dir: Path,
        fusion_weight: float = 0.5,
    ) -> Ranker:
        vecs = encoder.encode(jobs["text"].to_list())
        index = JobIndex.build(index_dir, jobs["job_id"].to_list(), vecs)
        return cls(jobs, resumes, encoder, index, fusion_weight)

    # -- scoring ---------------------------------------------------------------

    def _sparse_scores(self, name: str, text: str, direction: str) -> np.ndarray:
        s = self.sparse[name]
        docs = self._docs[name]["jobs" if direction == "jobs" else "resumes"]
        return (s.query_matrix([text]) @ docs.T).toarray()[0]

    def _scores(self, text: str, scorer: str, direction: str) -> np.ndarray:
        if scorer in ("tfidf", "bm25"):
            return self._sparse_scores(scorer, text, direction)
        vecs = self.job_vecs if direction == "jobs" else self.resume_vecs
        dense = vecs @ self.encoder.encode([text])[0]
        if scorer == "embedding":
            return dense
        w = self.fusion_weight
        return (
            w * zscore(dense[None, :])[0]
            + (1 - w) * zscore(self._sparse_scores("bm25", text, direction)[None, :])[0]
        )

    def _check(self, text: str, scorer: str, what: str) -> str:
        if scorer not in SCORER_NAMES:
            raise ValueError(
                f"unknown scorer {scorer!r}; choose from {', '.join(SCORER_NAMES)}"
            )
        text = text.strip()
        if not text:
            raise ValueError(f"empty {what} text")
        return text

    def _evidence(self, scorer: str, query: str, doc: str, k: int = 8) -> dict:
        terms, rest, total = explain.breakdown(self.sparse[scorer], query, doc, k)
        return {"terms": terms, "terms_rest": rest, "terms_total": total}

    # -- public ----------------------------------------------------------------

    def rank_jobs(
        self, resume_text: str, top: int = 10, scorer: str = "fusion"
    ) -> list[Match]:
        text = self._check(resume_text, scorer, "resume")
        if scorer == "embedding":
            # dense retrieval straight from the LanceDB table
            hits = self.index.search(self.encoder.encode([text]), k=top)[0]
            position = {j: i for i, j in enumerate(self.jobs["job_id"].to_list())}
            ranked = [(position[job_id], sim) for job_id, sim in hits]
        else:
            scores = self._scores(text, scorer, "jobs")
            ranked = [
                (int(i), float(scores[i]))
                for i in np.argsort(-scores, kind="stable")[:top]
            ]
        evidence = scorer if scorer in ("tfidf", "bm25") else "tfidf"
        out = []
        for i, score in ranked:
            row = self.jobs.row(i, named=True)
            out.append(
                Match(
                    id=row["job_id"],
                    title=row["title"],
                    category=row["category"],
                    score=score,
                    **self._evidence(evidence, text, row["text"]),
                    gaps=explain.gaps(
                        self.sparse["tfidf"],
                        text,
                        row["text"],
                        self.common.get(row["category"]),
                        k=6,
                    ),
                    explanation="exact"
                    if scorer in ("tfidf", "bm25")
                    else "lexical-overlap",
                )
            )
        return out

    def rank_resumes(
        self, job_text: str, top: int = 10, scorer: str = "fusion"
    ) -> list[Match]:
        text = self._check(job_text, scorer, "job")
        scores = self._scores(text, scorer, "resumes")
        order = np.argsort(-scores, kind="stable")[:top]
        evidence = scorer if scorer in ("tfidf", "bm25") else "tfidf"
        out = []
        for i in order:
            row = self.resumes.row(int(i), named=True)
            out.append(
                Match(
                    id=row["id"],
                    title=row["text"].split("\n", 1)[0][:60],
                    category=row["category"],
                    score=float(scores[i]),
                    **self._evidence(evidence, text, row["text"]),
                    explanation="exact"
                    if scorer in ("tfidf", "bm25")
                    else "lexical-overlap",
                )
            )
        return out

    def explain_dense(
        self, resume_text: str, job_id: str, k: int = 5
    ) -> list[tuple[str, float]]:
        """Post-hoc: which resume sentences the embedding match to ``job_id`` depends on."""
        row = self.jobs.filter(pl.col("job_id") == job_id).row(0, named=True)
        return explain.occlusion(self.encoder, resume_text, row["text"], k=k)


def load_default_ranker(
    data_dir: Path = data.DATA_DIR, per_category: int = 200, seed: int = 0
) -> Ranker:
    """The ranker over the same pool the evaluation used (so cached vectors are reused)."""
    resumes = data.load_resumes(data_dir / "Resume.csv")
    jobs = labels.balanced_pool(
        labels.label_postings(data.load_postings(data_dir / "postings.csv")),
        per_category,
        seed,
    )
    jobs = jobs.filter(pl.col("category").is_in(set(resumes["category"])))
    choice = default_encoder(data_dir / "embed_cache.sqlite")
    weight = 0.5
    if RESULTS.exists():
        weight = float(
            json.loads(RESULTS.read_text(encoding="utf-8"))["fusion_weight"]["full"]
        )
    return Ranker.build(jobs, resumes, choice.encoder, data_dir / "lancedb", weight)
