import numpy as np
import pytest
from resume_matcher.scorers import Bm25Scorer, FusionScorer, TfidfScorer, rrf, zscore

CORPUS = [
    "python sql tableau dashboards analyst reporting",
    "registered nurse patient care hospital ward",
    "chef kitchen menu cooking restaurant",
    "python machine learning model training",
    "analyst reporting finance budget forecast",
]


def _scorers():
    return [
        TfidfScorer(ngram_range=(1, 1), min_df=1).fit(CORPUS),
        Bm25Scorer().fit(CORPUS),
    ]


@pytest.mark.parametrize("scorer", _scorers(), ids=lambda s: s.name)
def test_topical_document_ranks_first(scorer):
    assert scorer.score("python sql analyst", CORPUS).argmax() == 0
    assert scorer.score("nurse patient hospital", CORPUS).argmax() == 1


@pytest.mark.parametrize("scorer", _scorers(), ids=lambda s: s.name)
def test_score_matrix_rows_equal_single_query_scores(scorer):
    queries = ["python sql analyst", "chef kitchen menu"]
    m = scorer.score_matrix(queries, CORPUS)
    assert m.shape == (2, len(CORPUS))
    for i, q in enumerate(queries):
        np.testing.assert_allclose(m[i], scorer.score(q, CORPUS))


@pytest.mark.parametrize("scorer", _scorers(), ids=lambda s: s.name)
@pytest.mark.parametrize("query", ["", "   ", "zzzz qqqq"])
def test_degenerate_queries_give_finite_equal_scores(scorer, query):
    out = scorer.score(query, CORPUS)
    assert np.isfinite(out).all()
    assert (out == out[0]).all()


@pytest.mark.parametrize("scorer", _scorers(), ids=lambda s: s.name)
def test_document_with_no_known_terms_scores_zero_not_nan(scorer):
    out = scorer.score("python", ["qqqq zzzz", ""])
    assert out.tolist() == [0.0, 0.0]


def test_unfitted_scorer_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="fit"):
        TfidfScorer().score("python", CORPUS)
    with pytest.raises(RuntimeError, match="fit"):
        Bm25Scorer().score("python", CORPUS)


def test_bm25_matches_the_independent_bm25s_implementation():
    bm25s = pytest.importorskip("bm25s")
    scorer = Bm25Scorer(k1=1.5, b=0.75).fit(CORPUS)
    tokenized = [d.split() for d in CORPUS]
    ref = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    ref.index(bm25s.tokenization.Tokenized(*_to_ids(tokenized)), show_progress=False)
    query = ["python", "analyst", "reporting"]
    vocab = _to_ids(tokenized)[1]
    expected = ref.get_scores([vocab[t] for t in query])
    np.testing.assert_allclose(
        scorer.score(" ".join(query), CORPUS), expected, rtol=1e-6
    )


def _to_ids(tokenized):
    vocab: dict[str, int] = {}
    ids = [[vocab.setdefault(t, len(vocab)) for t in doc] for doc in tokenized]
    return ids, vocab


# --- fusion -----------------------------------------------------------------


def test_zscore_normalises_each_row_and_constant_rows_become_zeros():
    out = zscore(np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]))
    assert out[0].mean() == pytest.approx(0.0)
    assert out[0].std() == pytest.approx(1.0)
    assert out[1].tolist() == [0.0, 0.0, 0.0]


def test_fusion_with_full_weight_on_one_part_ranks_like_that_part():
    tfidf = TfidfScorer(ngram_range=(1, 1), min_df=1)
    bm25 = Bm25Scorer()
    q = ["python sql analyst", "nurse hospital"]
    only_tfidf = FusionScorer([tfidf, bm25], [1.0, 0.0]).fit(CORPUS)
    ref = tfidf.score_matrix(q, CORPUS)
    got = only_tfidf.score_matrix(q, CORPUS)
    assert (
        np.argsort(-got, axis=1, kind="stable")
        == np.argsort(-ref, axis=1, kind="stable")
    ).all()


def test_fusion_combines_parts_so_a_doc_both_like_beats_one_only_one_likes():
    class Fixed:
        def __init__(self, m):
            self.m, self.name = np.array(m, dtype=float), "fixed"

        def fit(self, corpus):
            return self

        def score_matrix(self, queries, docs):
            return self.m

    a = Fixed([[3.0, 2.0, 0.0]])
    b = Fixed([[0.0, 2.0, 3.0]])
    out = FusionScorer([a, b], [0.5, 0.5]).score_matrix(["q"], ["d0", "d1", "d2"])
    assert out.argmax() == 1


def test_fusion_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="weights"):
        FusionScorer([TfidfScorer()], [0.5, 0.5])


def test_rrf_of_identical_rankings_keeps_the_order_and_is_scale_free():
    a = np.array([[0.9, 0.5, 0.1]])
    fused = rrf([a, a * 1000])
    assert np.argsort(-fused, axis=1).tolist() == [[0, 1, 2]]


def test_rrf_rewards_documents_ranked_well_by_both_over_one_list_only():
    # doc1 is 2nd and 1st; doc0 is 1st and 4th; doc2 is 3rd and 2nd
    a = np.array([[4.0, 3.0, 2.0, 1.0]])
    b = np.array([[1.0, 4.0, 3.0, 2.0]])
    assert rrf([a, b]).argmax() == 1
