import numpy as np
import pytest
from scorers import Bm25Scorer, TfidfScorer

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
