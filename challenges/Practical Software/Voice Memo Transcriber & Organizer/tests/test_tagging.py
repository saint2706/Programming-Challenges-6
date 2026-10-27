import json
from pathlib import Path

import numpy as np
import pytest
from voicememo.tagging import (
    MIN_CLUSTER_DOCS,
    LsaEmbedder,
    ctfidf_keywords,
    evaluate,
    plan_tags,
)

FIXTURE = Path(__file__).parent / "fixtures" / "topics.json"
TOPICS = ["cooking", "travel", "work", "fitness", "gardening", "music"]


def load_topics() -> list[tuple[str, str]]:
    items = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [(i["transcript"], i["topic"]) for i in items]


class OneHotEmbedder:
    """Perfect embedder for tests: each gold topic becomes its own orthogonal direction."""

    name = "onehot"

    def __init__(self, items: list[tuple[str, str]]) -> None:
        self.topic_of = dict(items)
        self.index = {t: i for i, t in enumerate(sorted(set(self.topic_of.values())))}

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(self.index)))
        for row, text in enumerate(texts):
            out[row, self.index[self.topic_of[text]]] = 1.0
        return out


class NoneEmbedder:
    name = "none"

    def embed(self, texts):
        return None


class NoiseEmbedder:
    """Points far apart on a grid, so HDBSCAN finds no dense cluster at all."""

    name = "noise"

    def embed(self, texts):
        n = len(texts)
        out = np.zeros((n, n))
        np.fill_diagonal(out, 1.0)
        return out


def test_ctfidf_returns_one_list_per_doc_and_respects_top_k():
    docs = ["the oven bakes bread", "the guitar plays chords", "the oven heats soup"]
    keywords = ctfidf_keywords(docs, top_k=2)
    assert len(keywords) == 3
    assert all(len(k) <= 2 for k in keywords)


def test_ctfidf_drops_english_stop_words():
    keywords = ctfidf_keywords(["the and of to in", "garlic onions garlic"])
    assert "the" not in keywords[0] and "and" not in keywords[0]
    assert keywords[1][0] == "garlic"


def test_ctfidf_empty_input_and_blank_docs():
    assert ctfidf_keywords([]) == []
    assert ctfidf_keywords(["", "   "]) == [[], []]


def test_ctfidf_ignores_digits_and_single_letters():
    keywords = ctfidf_keywords(["x1 y2 42 a b", "pancakes syrup"])
    assert keywords[0] == []


def test_ctfidf_is_deterministic_on_ties():
    docs = ["alpha beta", "gamma delta"]
    assert ctfidf_keywords(docs) == ctfidf_keywords(docs)


def test_lsa_returns_unit_rows():
    vectors = LsaEmbedder().embed([t for t, _ in load_topics()])
    assert vectors is not None
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6) or np.allclose(norms, 0.0)


def test_lsa_returns_none_for_stop_words_only():
    assert LsaEmbedder().embed(["the and of", "to in on"]) is None


def test_lsa_dims_shrink_for_tiny_inputs():
    vectors = LsaEmbedder(dims=64).embed(["alpha beta", "gamma delta", "epsilon zeta"])
    assert vectors is not None
    assert vectors.shape[0] == 3


def test_small_library_uses_per_memo_keywords():
    texts = {i: f"garlic onions pasta number {i}" for i in range(MIN_CLUSTER_DOCS - 1)}
    plan = plan_tags(texts)
    assert plan.method == "per-memo-keywords"
    assert set(plan.labels.values()) == {-1}


def test_per_memo_assignments_cover_every_memo():
    texts = {10: "pasta sauce basil", 3: "flight train hotel"}
    plan = plan_tags(texts)
    assert set(plan.assignments) == {3, 10}


def test_clusters_are_named_by_their_keywords():
    items = load_topics()
    plan = plan_tags(dict(enumerate(t for t, _ in items)), OneHotEmbedder(items))
    assert plan.method == "clusters"
    cooking_ids = [i for i, (_, g) in enumerate(items) if g == "cooking"]
    names = {" ".join(plan.assignments[i]) for i in cooking_ids}
    assert len(names) == 1
    assert names.pop() != ""


def test_perfect_embedder_gives_perfect_scores():
    items = load_topics()
    result = evaluate(items, OneHotEmbedder(items))
    assert result["clusters"] == len(TOPICS)
    assert result["adjusted_rand"] == 1.0
    assert result["purity"] == 1.0
    assert result["covered"] == len(items)
    assert result["noise_share"] == 0.0


def test_pure_noise_falls_back_to_per_memo_and_scores_zero_covered():
    items = load_topics()[:16]
    result = evaluate(items, NoiseEmbedder())
    assert result["method"] == "per-memo-keywords"
    assert result["covered"] == 0
    assert result["purity"] == 0.0


def test_failed_embedding_falls_back_to_per_memo():
    texts = {i: f"word{i} sauce" for i in range(MIN_CLUSTER_DOCS + 2)}
    assert plan_tags(texts, NoneEmbedder()).method == "per-memo-keywords"


def test_embedder_name_is_reported():
    items = load_topics()[:6]
    assert evaluate(items)["embedder"] == "lsa"
    assert evaluate(items, NoiseEmbedder())["embedder"] == "noise"


def test_lsa_plan_is_deterministic_across_runs():
    texts = {i: t for i, (t, _) in enumerate(load_topics())}
    first = plan_tags(texts)
    second = plan_tags(texts)
    assert first == second


def test_labelled_set_has_six_topics_of_eight_memos():
    items = load_topics()
    assert len(items) == 48
    counts = {t: sum(1 for _, g in items if g == t) for t in TOPICS}
    assert counts == {t: 8 for t in TOPICS}


def test_lsa_evaluation_on_labelled_set_is_reproducible():
    items = load_topics()
    first = evaluate(items)
    second = evaluate(items)
    assert first == second
    assert first["memos"] == 48
    assert -1.0 <= first["adjusted_rand"] <= 1.0
    assert 0.0 <= first["purity"] <= 1.0


def test_evaluate_counts_noise_against_coverage():
    items = load_topics()
    result = evaluate(items, NoiseEmbedder())
    assert result["noise_share"] == 1.0
    assert result["covered"] == 0


def test_evaluate_handles_empty_input():
    result = evaluate([])
    assert result["memos"] == 0
    assert result["purity"] == 0.0


@pytest.mark.parametrize("count", [1, 5, MIN_CLUSTER_DOCS - 1])
def test_below_threshold_never_clusters(count):
    texts = {i: f"topic words {i}" for i in range(count)}
    assert plan_tags(texts).method == "per-memo-keywords"
