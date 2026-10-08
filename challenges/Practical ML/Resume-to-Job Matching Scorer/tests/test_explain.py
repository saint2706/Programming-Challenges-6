import random

import numpy as np
import pytest
from resume_matcher import explain
from resume_matcher.scorers import Bm25Scorer, TfidfScorer

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


# --- dense (embedding) explanations: occlusion ---------------------------------


class _BagEncoder:
    def encode(self, texts):
        import zlib

        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in t.lower().replace(".", " ").split():
                out[i, zlib.crc32(w.encode()) % 64] += 1
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1, norm)


def test_split_units_splits_sentences_and_lines_and_drops_fragments():
    units = explain.split_units(
        "Built dashboards in Tableau. Led a team of five\n- SQL reporting\nok"
    )
    assert units == [
        "Built dashboards in Tableau.",
        "Led a team of five",
        "- SQL reporting",
    ]


def test_occlusion_ranks_the_unit_that_carries_the_match_first():
    resume = "Led python and sql dashboard work. Enjoys hiking and pottery on weekends. Likes baking bread at home."
    job = "python sql dashboard analytics role"
    out = explain.occlusion(_BagEncoder(), resume, job, k=3)
    assert out[0][0].startswith("Led python and sql")
    assert out[0][1] > 0
    assert [d for _, d in out] == sorted((d for _, d in out), reverse=True)


def test_occlusion_of_text_with_no_units_is_empty():
    assert explain.occlusion(_BagEncoder(), "", "python", k=3) == []


def test_occlusion_drops_are_measured_against_the_same_truncated_text_they_perturb():
    # 60 sentences: only the first carries the match. Units beyond MAX_UNITS are
    # ignored by the probe, so the baseline must ignore them too.
    units = ["Led python and sql dashboard work."] + [
        f"Filler sentence number {i} about nothing." for i in range(59)
    ]
    resume = " ".join(units)
    job = "python sql dashboard analytics role"
    enc = _BagEncoder()
    kept = explain.split_units(resume)[: explain.MAX_UNITS]
    d = enc.encode([job])[0]
    base = float(enc.encode([" ".join(kept)])[0] @ d)
    out = explain.occlusion(enc, resume, job, k=3)
    top_unit, top_drop = out[0]
    without = " ".join(u for u in kept if u != top_unit)
    assert top_unit.startswith("Led python")
    assert top_drop == pytest.approx(base - float(enc.encode([without])[0] @ d))
