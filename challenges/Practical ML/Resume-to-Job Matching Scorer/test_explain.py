import random

import explain
import pytest
from scorers import Bm25Scorer, TfidfScorer

CORPUS = [
    "python sql tableau dashboards analyst reporting team",
    "registered nurse patient care hospital ward team",
    "chef kitchen menu cooking restaurant team",
    "python machine learning model training team",
    "analyst reporting finance budget forecast team",
]
QUERY = "python sql tableau analyst reporting team player excellent"


def test_tfidf_term_contributions_sum_exactly_to_the_score():
    s = TfidfScorer(ngram_range=(1, 1), min_df=1).fit(CORPUS)
    terms = explain.tfidf_terms(s, QUERY, CORPUS[0], k=1000)
    assert sum(w for _, w in terms) == pytest.approx(s.score(QUERY, [CORPUS[0]])[0])
    assert [w for _, w in terms] == sorted((w for _, w in terms), reverse=True)


def test_bm25_term_contributions_sum_exactly_to_the_score():
    s = Bm25Scorer().fit(CORPUS)
    terms = explain.bm25_terms(s, QUERY, CORPUS[0], k=1000)
    assert sum(w for _, w in terms) == pytest.approx(s.score(QUERY, [CORPUS[0]])[0])


def test_top_k_truncates_and_unmatched_pair_explains_nothing():
    s = TfidfScorer(ngram_range=(1, 1), min_df=1).fit(CORPUS)
    assert len(explain.tfidf_terms(s, QUERY, CORPUS[0], k=2)) == 2
    assert explain.tfidf_terms(s, "chef kitchen", CORPUS[1], k=5) == []
    assert explain.tfidf_terms(s, "", CORPUS[0], k=5) == []


def _drop(text: str, terms: set[str]) -> str:
    return " ".join(w for w in text.split() if w not in terms)


def test_removing_top_explained_terms_hurts_more_than_removing_random_shared_terms():
    s = TfidfScorer(ngram_range=(1, 1), min_df=1).fit(CORPUS)
    doc = CORPUS[0]
    terms = explain.tfidf_terms(s, QUERY, doc, k=1000)
    top = {t for t, _ in terms[:3]}
    base = s.score(QUERY, [doc])[0]
    drop_top = base - s.score(_drop(QUERY, top), [doc])[0]
    rng = random.Random(0)
    drops = []
    for _ in range(20):
        pick = set(rng.sample([t for t, _ in terms], 3))
        drops.append(base - s.score(_drop(QUERY, pick), [doc])[0])
    assert drop_top >= sum(drops) / len(drops)


def test_gaps_are_job_terms_the_resume_lacks_and_exclude_shared_terms():
    job = "python sql kubernetes airflow dashboards"
    s = TfidfScorer(ngram_range=(1, 1), min_df=1).fit([*CORPUS, job])
    common = {"kubernetes", "airflow", "python", "sql"}
    gaps = explain.gaps(s, "python sql analyst", job, common, k=5)
    assert set(gaps) == {"kubernetes", "airflow"}
    assert "dashboards" not in gaps  # not common in the category
    assert "python" not in gaps  # resume has it


def test_category_common_terms_keeps_terms_shared_by_enough_postings():
    docs = [
        "kubernetes docker python",
        "kubernetes terraform",
        "kubernetes aws",
        "rareword",
    ]
    s = TfidfScorer(ngram_range=(1, 1), min_df=1).fit([*CORPUS, *docs])
    assert explain.category_common_terms(s, docs, min_share=0.5) == {"kubernetes"}
