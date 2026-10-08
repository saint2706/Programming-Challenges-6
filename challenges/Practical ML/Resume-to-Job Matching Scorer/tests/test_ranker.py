import re

import pytest
from helpers import HashEncoder, _frames
from resume_matcher.ranker import Ranker


@pytest.fixture
def ranker(tmp_path):
    jobs, resumes = _frames()
    return Ranker.build(
        jobs, resumes, HashEncoder(), index_dir=tmp_path / "idx", fusion_weight=0.5
    )


@pytest.mark.parametrize("scorer", ["tfidf", "bm25", "embedding", "fusion"])
def test_every_scorer_puts_same_category_jobs_on_top(ranker, scorer):
    resume = "nurse patient ward triage dosage clinic nurse patient team worked"
    top = ranker.rank_jobs(resume, top=5, scorer=scorer)
    assert len(top) == 5
    assert {m.category for m in top} == {"HEALTHCARE"}
    assert [m.score for m in top] == sorted((m.score for m in top), reverse=True)


def test_sparse_matches_carry_exact_evidence_terms_that_overlap_the_resume(ranker):
    resume = "kitchen menu sauce pastry plating team"
    m = ranker.rank_jobs(resume, top=1, scorer="tfidf")[0]
    assert m.explanation == "exact"
    assert m.terms and all(t in resume.split() or " " in t for t, _ in m.terms)


def test_dense_matches_are_labelled_as_lexical_overlap_not_exact(ranker):
    m = ranker.rank_jobs("classroom lesson pupils grading", top=1, scorer="embedding")[
        0
    ]
    assert m.explanation == "lexical-overlap"


def test_gaps_name_category_terms_the_resume_lacks(ranker):
    m = ranker.rank_jobs("nurse patient", top=1, scorer="tfidf")[0]
    assert m.category == "HEALTHCARE"
    assert set(m.gaps) & {"ward", "clinic", "dosage", "triage"}
    assert "nurse" not in m.gaps


def test_rank_resumes_is_the_reverse_direction(ranker):
    job = "kitchen menu sauce saute pastry plating kitchen menu"
    top = ranker.rank_resumes(job, top=4, scorer="bm25")
    assert {m.category for m in top} == {"CHEF"}


@pytest.mark.parametrize("text", ["", "   \n "])
def test_empty_query_is_a_clear_error_not_a_crash(ranker, text):
    with pytest.raises(ValueError, match="empty"):
        ranker.rank_jobs(text, top=3)


def test_unknown_scorer_is_rejected(ranker):
    with pytest.raises(ValueError, match="unknown scorer"):
        ranker.rank_jobs("nurse", top=3, scorer="magic")


def test_top_larger_than_the_pool_returns_the_whole_pool(ranker):
    assert len(ranker.rank_jobs("nurse patient", top=1000, scorer="tfidf")) == 18


def test_explain_dense_names_the_resume_sentence_that_drives_the_match(ranker):
    resume = "Led nurse patient ward triage work. Collects stamps on weekends. Enjoys long walks."
    out = ranker.explain_dense(resume, "HEALTHCARE-0", k=2)
    assert out[0][0].startswith("Led nurse patient")


@pytest.mark.parametrize("scorer", ["tfidf", "bm25"])
def test_listed_terms_plus_the_remainder_equal_the_score_exactly(ranker, scorer):
    resume = "nurse patient ward clinic dosage triage nurse patient team worked years managed responsible"
    for m in ranker.rank_jobs(resume, top=3, scorer=scorer):
        assert m.terms_total > len(m.terms)  # more terms matched than are listed
        assert sum(w for _, w in m.terms) + m.terms_rest == pytest.approx(m.score)


def test_when_every_matching_term_is_listed_there_is_no_remainder(ranker):
    m = ranker.rank_jobs("nurse", top=1, scorer="tfidf")[0]
    assert m.terms_total == len(m.terms) and m.terms_rest == pytest.approx(0.0)


def test_evidence_note_says_how_much_of_the_score_the_listed_terms_explain(ranker):
    resume = "nurse patient ward clinic dosage triage nurse patient team worked years managed responsible"
    note = ranker.rank_jobs(resume, top=1, scorer="tfidf")[0].evidence_note()
    assert re.fullmatch(
        r"top 8 of \d+ terms; the other \d+ add \d\.\d{3}, so all of them sum to the score",
        note,
    )
    short = ranker.rank_jobs("nurse", top=1, scorer="tfidf")[0].evidence_note()
    assert short == "all 1 terms; they sum to the score" or short.startswith("all ")
    dense = ranker.rank_jobs(resume, top=1, scorer="embedding")[0].evidence_note()
    assert "lexical overlap only" in dense and "sum to the score" not in dense
