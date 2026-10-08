import random
import zlib

import numpy as np
import polars as pl
import pytest
from resume_matcher.evaluate import random_scores, summarise, tune_fusion_weight
from resume_matcher.pipeline import Prepared, evaluate_all, prepare, run_all

VOCAB = {
    "HEALTHCARE": ["nurse", "patient", "ward", "clinic", "dosage", "triage"],
    "CHEF": ["kitchen", "menu", "sauce", "saute", "pastry", "plating"],
    "TEACHER": ["classroom", "lesson", "pupils", "grading", "curriculum", "phonics"],
}
COMMON = [
    "team",
    "worked",
    "years",
    "managed",
    "responsible",
    "experience",
    "skills",
    "daily",
]


class HashEncoder:
    """Deterministic bag-of-words embedding; stands in for the real model."""

    def encode(self, texts):
        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.encode()) % 64] += 1
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1, norm)


def _resumes(n_per_cat, seed, headline_only=False):
    rng = random.Random(seed)
    rows = []
    for cat, words in VOCAB.items():
        for i in range(n_per_cat):
            body = (
                rng.choices(COMMON, k=60)
                if headline_only
                else rng.choices(words, k=40) + rng.choices(COMMON, k=20)
            )
            rows.append(
                (
                    f"{cat}-{seed}-{i}",
                    cat,
                    f"{words[0].upper()} {words[1].upper()}\n" + " ".join(body),
                )
            )
    return pl.DataFrame(rows, schema=["id", "category", "text"], orient="row")


def _jobs(n_per_cat, seed):
    rng = random.Random(seed)
    rows = []
    for cat, words in VOCAB.items():
        for i in range(n_per_cat):
            body = rng.choices(words, k=40) + rng.choices(COMMON, k=20)
            rows.append((f"j-{cat}-{seed}-{i}", f"{cat} role", " ".join(body), cat))
    return pl.DataFrame(
        rows, schema=["job_id", "title", "text", "category"], orient="row"
    )


def _prepared(headline_only=False):
    return Prepared(
        resumes_val=_resumes(8, 1, headline_only),
        resumes_test=_resumes(10, 2, headline_only),
        jobs_val=_jobs(10, 3),
        jobs_test=_jobs(12, 4),
        skipped={},
    )


def test_summarise_a_perfect_ranking_scores_one():
    cats = np.array(["a", "a", "b", "b"])
    scores = (cats[:, None] == cats[None, :]).astype(float)
    s = summarise(scores, scores > 0, list(cats), seed=0)
    assert s["ndcg@10"]["mean"] == pytest.approx(1.0)
    assert s["per_category_ndcg@10"] == {
        "a": pytest.approx(1.0),
        "b": pytest.approx(1.0),
    }
    assert s["ndcg@10"]["lo"] <= s["ndcg@10"]["mean"] <= s["ndcg@10"]["hi"]


def test_random_baseline_is_near_the_chance_precision():
    cats = np.array(["a", "b", "c"] * 100)
    rel = cats[:, None] == cats[None, :]
    s = summarise(random_scores(rel.shape, seed=0), rel, list(cats), seed=0)
    assert 0.2 < s["p@10"]["mean"] < 0.47  # chance is 1/3 for three equal categories


def test_tune_fusion_weight_picks_the_part_that_ranks_better():
    cats = np.array(["a", "b"] * 20)
    rel = cats[:, None] == cats[None, :]
    good = rel.astype(float) + np.random.default_rng(0).normal(0, 0.01, rel.shape)
    bad = random_scores(rel.shape, seed=1)
    grid = np.linspace(0, 1, 11)
    assert tune_fusion_weight(good, bad, rel, grid) >= 0.5
    assert tune_fusion_weight(bad, good, rel, grid) <= 0.5


def test_evaluate_all_reports_every_scorer_variant_and_direction():
    r = evaluate_all(_prepared(), HashEncoder(), seed=0)
    assert set(r["results"]) == {"full", "stripped"}
    for variant in r["results"].values():
        assert set(variant) == {"resume->jobs", "jobs->resumes"}
        assert set(variant["resume->jobs"]) == {
            "random",
            "tfidf",
            "bm25",
            "embedding",
            "fusion",
            "rrf",
        }
    for direction in r["results"]["full"].values():
        assert (
            direction["tfidf"]["ndcg@10"]["mean"]
            > direction["random"]["ndcg@10"]["mean"] + 0.3
        )


def test_fusion_weight_is_tuned_on_validation_from_the_grid_and_recorded():
    r = evaluate_all(_prepared(), HashEncoder(), seed=0, grid=np.array([0.0, 0.5, 1.0]))
    assert set(r["fusion_weight"]) == {"full", "stripped"}
    assert all(w in (0.0, 0.5, 1.0) for w in r["fusion_weight"].values())


def test_stripped_variant_loses_signal_that_only_the_headline_carried():
    r = evaluate_all(_prepared(headline_only=True), HashEncoder(), seed=0)
    full = r["results"]["full"]["resume->jobs"]["tfidf"]["ndcg@10"]["mean"]
    stripped = r["results"]["stripped"]["resume->jobs"]["tfidf"]["ndcg@10"]["mean"]
    assert full > 0.9
    assert stripped < full - 0.3


def test_evaluate_all_is_deterministic_for_a_seed():
    a = evaluate_all(_prepared(), HashEncoder(), seed=3)
    b = evaluate_all(_prepared(), HashEncoder(), seed=3)
    assert a == b


def test_prepare_caps_pool_splits_disjointly_and_reports_skipped_categories(tmp_path):
    resumes = pl.concat(
        [_resumes(20, 5), _resumes(5, 6).with_columns(pl.lit("RARE").alias("category"))]
    )
    jobs = pl.concat(
        [_jobs(30, 7), _jobs(1, 8).with_columns(pl.lit("RARE").alias("category"))]
    )
    p = prepare(resumes, jobs, per_category=10, min_pool=6, seed=0)
    assert set(p.skipped) == {"RARE"}
    assert "RARE" not in set(p.resumes_test["category"]) | set(p.jobs_test["category"])
    assert set(p.jobs_val["job_id"]).isdisjoint(p.jobs_test["job_id"])
    assert set(p.resumes_val["id"]).isdisjoint(p.resumes_test["id"])
    assert max(dict(p.jobs_val.group_by("category").len().iter_rows()).values()) <= 5


def test_run_all_writes_a_report_from_kaggle_shaped_csvs(tmp_path):
    from resume_matcher.embed import BackendChoice

    titles = {
        "HEALTHCARE": "Registered Nurse",
        "CHEF": "Executive Chef",
        "TEACHER": "Middle School Teacher",
    }
    resumes = _resumes(14, 1)
    pl.DataFrame(
        {
            "ID": list(range(resumes.height)),
            "Resume_str": [t + " " + "filler " * 20 for t in resumes["text"]],
            "Resume_html": ["x"] * resumes.height,
            "Category": resumes["category"],
        }
    ).write_csv(tmp_path / "Resume.csv")
    jobs = _jobs(20, 2)
    pl.DataFrame(
        {
            "job_id": jobs["job_id"],
            "title": [titles[c] for c in jobs["category"]],
            "description": [t + " filler" * 20 for t in jobs["text"]],
        }
    ).write_csv(tmp_path / "postings.csv")
    out = tmp_path / "out"
    report = run_all(
        tmp_path,
        out,
        per_category=10,
        min_pool=6,
        seed=0,
        encoder_factory=lambda cache: BackendChoice(
            HashEncoder(), "hash", [("hash", "ok")]
        ),
    )
    assert (out / "report.json").exists()
    assert report["backend"]["name"] == "hash"
    assert report["n"]["test_resumes"] > 0 and report["n"]["test_jobs"] > 0
