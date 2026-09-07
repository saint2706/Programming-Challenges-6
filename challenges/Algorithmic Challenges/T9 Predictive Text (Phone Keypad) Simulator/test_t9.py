"""Tests for t9.py: cross-verification, edge cases, incremental consistency.

Run with:  uv run --with wordfreq --with pytest pytest -q
Skip the slow full-vocabulary tests with:  -m "not slow"
"""

from __future__ import annotations

import itertools
import random

import pytest
from t9 import (
    KEYPAD,
    AdaptiveT9,
    HashmapT9,
    TrieT9,
    Vocabulary,
    build_vocabulary,
    naive_lookup,
    naive_prefix_lookup,
    verify,
    word_to_digits,
)

# A small, hand-picked vocabulary that deliberately contains real multi-tap
# collisions, so most tests run in milliseconds without needing wordfreq.
#   good/home/goof/hoof all -> 4663 (the textbook T9 example)
#   cat/act/bat           all -> 228
SMALL_VOCAB: Vocabulary = [
    ("good", 6.12),
    ("home", 5.81),
    ("goof", 2.99),
    ("hoof", 3.06),
    ("gone", 5.00),
    ("hello", 6.50),
    ("cat", 5.40),
    ("act", 3.10),
    ("bat", 4.00),
    ("a", 6.00),
    ("i", 5.50),
]


# ---------------------------------------------------------------------------
# word_to_digits
# ---------------------------------------------------------------------------


def test_word_to_digits_classic_collision():
    # The reason 4663 is the canonical T9 example: four common words share it.
    assert word_to_digits("good") == "4663"
    assert word_to_digits("home") == "4663"
    assert word_to_digits("goof") == "4663"
    assert word_to_digits("hoof") == "4663"


def test_word_to_digits_case_insensitive():
    assert word_to_digits("GOOD") == word_to_digits("good")
    assert word_to_digits("GoOd") == word_to_digits("good")


def test_word_to_digits_covers_every_letter():
    for digit, letters in KEYPAD.items():
        for letter in letters:
            assert word_to_digits(letter) == digit


def test_word_to_digits_rejects_non_alpha():
    for bad in ("don't", "well-known", "café", "3rd", "a b"):
        with pytest.raises(ValueError):
            word_to_digits(bad)


def test_word_to_digits_empty_word_is_empty_signature():
    assert word_to_digits("") == ""


# ---------------------------------------------------------------------------
# Edge cases: empty / invalid digit strings
# ---------------------------------------------------------------------------


def test_empty_digit_string_rejected():
    hashmap = HashmapT9(SMALL_VOCAB)
    trie = TrieT9(SMALL_VOCAB)
    with pytest.raises(ValueError):
        naive_lookup(SMALL_VOCAB, "")
    with pytest.raises(ValueError):
        hashmap.lookup("")
    with pytest.raises(ValueError):
        trie.lookup_exact("")
    with pytest.raises(ValueError):
        trie.predict("")


@pytest.mark.parametrize("digits", ["0", "1", "01", "410", "9991"])
def test_invalid_digits_0_and_1_rejected(digits):
    hashmap = HashmapT9(SMALL_VOCAB)
    trie = TrieT9(SMALL_VOCAB)
    with pytest.raises(ValueError):
        naive_lookup(SMALL_VOCAB, digits)
    with pytest.raises(ValueError):
        hashmap.lookup(digits)
    with pytest.raises(ValueError):
        trie.lookup_exact(digits)


def test_single_digit_lookup():
    # "a" -> "2", nothing else in SMALL_VOCAB is a single letter on digit 2.
    assert naive_lookup(SMALL_VOCAB, "2") == ["a"]


def test_digit_sequence_with_no_matches():
    assert naive_lookup(SMALL_VOCAB, "22222222") == []
    assert HashmapT9(SMALL_VOCAB).lookup("22222222") == []
    assert TrieT9(SMALL_VOCAB).lookup_exact("22222222") == []


def test_repeated_digits():
    # "228" matches cat / act / bat -- three different letter orderings that
    # happen to collapse onto the same digit sequence.
    expected = sorted(["cat", "act", "bat"])
    assert sorted(naive_lookup(SMALL_VOCAB, "228")) == expected


# ---------------------------------------------------------------------------
# Cross-verification: naive vs hashmap vs trie, on the small vocabulary
# ---------------------------------------------------------------------------


def test_hashmap_matches_naive_on_small_vocab():
    hashmap = HashmapT9(SMALL_VOCAB)
    for digits in ("4663", "228", "43556", "2", "56663", "9999"):
        assert hashmap.lookup(digits) == naive_lookup(SMALL_VOCAB, digits)


def test_trie_exact_matches_naive_on_small_vocab():
    trie = TrieT9(SMALL_VOCAB)
    for digits in ("4663", "228", "43556", "2", "56663", "9999"):
        assert trie.lookup_exact(digits) == naive_lookup(SMALL_VOCAB, digits)


def test_multi_tap_ranking_is_by_frequency_descending():
    # "gone" (g-o-n-e) also lands on 4663 (n is on the same key as m), so the
    # collision is five-way: good (6.12) > home (5.81) > gone (5.00) >
    # hoof (3.06) > goof (2.99).
    expected = ["good", "home", "gone", "hoof", "goof"]
    assert naive_lookup(SMALL_VOCAB, "4663") == expected
    assert HashmapT9(SMALL_VOCAB).lookup("4663") == expected
    assert TrieT9(SMALL_VOCAB).lookup_exact("4663") == expected


def test_limit_truncates_but_preserves_order():
    full = naive_lookup(SMALL_VOCAB, "4663")
    assert naive_lookup(SMALL_VOCAB, "4663", limit=2) == full[:2]
    assert HashmapT9(SMALL_VOCAB).lookup("4663", limit=2) == full[:2]
    assert TrieT9(SMALL_VOCAB).lookup_exact("4663", limit=2) == full[:2]


# ---------------------------------------------------------------------------
# Prefix / predictive queries
# ---------------------------------------------------------------------------


def test_predict_matches_naive_prefix_oracle():
    trie = TrieT9(SMALL_VOCAB, top_k=10)
    for prefix in ("4", "46", "466", "4663", "2", "22", "9"):
        expected = naive_prefix_lookup(SMALL_VOCAB, prefix, limit=10)
        assert trie.predict(prefix, limit=10) == expected


def test_predict_prefix_includes_completions_beyond_current_length():
    # "43" is a strict prefix of hello's "43556" and nothing else's full
    # signature -- predict should surface it as a completion.
    trie = TrieT9(SMALL_VOCAB, top_k=10)
    assert "hello" in trie.predict("43")
    assert "hello" not in trie.lookup_exact("43")  # exact requires equality


def test_predict_no_match_returns_empty():
    trie = TrieT9(SMALL_VOCAB)
    assert trie.predict("99999999") == []


# ---------------------------------------------------------------------------
# Incremental typing consistency
# ---------------------------------------------------------------------------


def test_incremental_prefix_consistency():
    """Typing digit-by-digit must agree with a fresh predict() at every length."""
    trie = TrieT9(SMALL_VOCAB, top_k=10)
    signature = word_to_digits("hello")  # "43556"
    session = trie.session()
    for i, digit in enumerate(signature, 1):
        got = session.type_digit(digit)
        expected = trie.predict(signature[:i], limit=10)
        assert got == expected, f"mismatch after typing {signature[:i]!r}"


def test_typing_session_reset():
    trie = TrieT9(SMALL_VOCAB)
    session = trie.session()
    session.type_digit("4")
    session.type_digit("6")
    assert session.digits == "46"
    session.reset()
    assert session.digits == ""
    # Before any digit is typed the session sits at the root; predict("")
    # itself is invalid (an empty query is refused everywhere else too), but
    # the session's suggestions() should read the root's cached top-k intact.
    assert session.suggestions() == [w for w, _f in trie.root.top]


def test_typing_session_dead_end_then_recovers_state():
    trie = TrieT9(SMALL_VOCAB)
    session = trie.session()
    # "9999" has no matches in SMALL_VOCAB.
    for d in "9999":
        result = session.type_digit(d)
    assert result == []
    assert session.node is None


def test_typing_session_rejects_invalid_digit():
    trie = TrieT9(SMALL_VOCAB)
    session = trie.session()
    with pytest.raises(ValueError):
        session.type_digit("1")
    with pytest.raises(ValueError):
        session.type_digit("0")


# ---------------------------------------------------------------------------
# Adaptive re-ranking
# ---------------------------------------------------------------------------


def test_adaptive_default_order_matches_backend_before_any_selection():
    hashmap = HashmapT9(SMALL_VOCAB)
    adaptive = AdaptiveT9(hashmap)
    assert adaptive.lookup("4663") == hashmap.lookup("4663")


def test_adaptive_selection_moves_word_to_front():
    # 4663 -> good, home, gone, hoof, goof (frequency order). "goof" is last
    # by corpus frequency, so selecting it once is a genuine non-default pick.
    hashmap = HashmapT9(SMALL_VOCAB)
    adaptive = AdaptiveT9(hashmap)
    before = adaptive.lookup("4663")
    assert before == ["good", "home", "gone", "hoof", "goof"]
    assert before[0] != "goof"

    adaptive.select("4663", "goof")
    after = adaptive.lookup("4663")
    assert after[0] == "goof"
    # Everything else keeps its relative (frequency) order behind the pick.
    assert after[1:] == ["good", "home", "gone", "hoof"]


def test_adaptive_selection_persists_across_repeated_queries():
    """Type the same digits twice, select a non-default word the first time."""
    hashmap = HashmapT9(SMALL_VOCAB)
    adaptive = AdaptiveT9(hashmap)

    first_query = adaptive.lookup("4663")
    default = first_query[0]
    assert default == "good"

    adaptive.select("4663", "home")  # user picks the 2nd-ranked word instead
    second_query = adaptive.lookup("4663")
    assert second_query[0] == "home"
    assert second_query[0] != default


def test_adaptive_repeated_selection_outranks_single_selection():
    hashmap = HashmapT9(SMALL_VOCAB)
    adaptive = AdaptiveT9(hashmap)
    adaptive.select("4663", "hoof")
    adaptive.select("4663", "goof")
    adaptive.select("4663", "goof")  # picked twice -> should rank above hoof
    ranked = adaptive.lookup("4663")
    assert ranked.index("goof") < ranked.index("hoof")


def test_adaptive_select_unknown_word_raises():
    hashmap = HashmapT9(SMALL_VOCAB)
    adaptive = AdaptiveT9(hashmap)
    with pytest.raises(ValueError):
        adaptive.select("4663", "notaword")  # not a candidate for this signature
    with pytest.raises(ValueError):
        adaptive.select("4663", "cat")  # a real word, but wrong signature
    # A rejected selection must not be recorded.
    assert adaptive.selection_counts("4663") == {}


def test_adaptive_select_validates_digits():
    adaptive = AdaptiveT9(HashmapT9(SMALL_VOCAB))
    with pytest.raises(ValueError):
        adaptive.select("019", "good")
    with pytest.raises(ValueError):
        adaptive.lookup("")


def test_adaptive_selection_counts_reflect_choices():
    adaptive = AdaptiveT9(HashmapT9(SMALL_VOCAB))
    adaptive.select("4663", "goof")
    adaptive.select("4663", "goof")
    adaptive.select("4663", "hoof")
    counts = adaptive.selection_counts("4663")
    assert counts["goof"] == 2
    assert counts["hoof"] == 1
    assert counts["good"] == 0


def test_adaptive_selection_scoped_to_exact_signature():
    # Selecting under the full signature "4663" must not leak into a query
    # for a *different* signature that happens to share candidates.
    adaptive = AdaptiveT9(HashmapT9(SMALL_VOCAB))
    adaptive.select("4663", "goof")
    assert adaptive.lookup("228") == naive_lookup(SMALL_VOCAB, "228")


def test_adaptive_wraps_trie_lookup_and_predict():
    trie = TrieT9(SMALL_VOCAB, top_k=10)
    adaptive = AdaptiveT9(trie)
    assert adaptive.lookup("4663") == trie.lookup_exact("4663")
    before = adaptive.predict("466")
    assert before == trie.predict("466")

    adaptive.select("4663", "goof")
    after_exact = adaptive.lookup("4663")
    assert after_exact[0] == "goof"
    # predict() on the exact signature itself picks up the same boost.
    after_predict = adaptive.predict("4663")
    assert after_predict[0] == "goof"


def test_adaptive_predict_requires_trie_backend():
    adaptive = AdaptiveT9(HashmapT9(SMALL_VOCAB))
    with pytest.raises(TypeError):
        adaptive.predict("4663")


def test_adaptive_limit_applies_after_reranking():
    adaptive = AdaptiveT9(HashmapT9(SMALL_VOCAB))
    adaptive.select("4663", "goof")
    assert adaptive.lookup("4663", limit=2) == ["goof", "good"]


# ---------------------------------------------------------------------------
# Randomized cross-verification against the naive oracle
# ---------------------------------------------------------------------------


def test_verify_passes_on_small_vocab():
    problems = verify(SMALL_VOCAB, trials=50, seed=1)
    assert problems == []


def test_random_synthetic_vocab_agreement():
    """Build random synthetic vocabularies and check all methods agree."""
    rng = random.Random(42)
    letters = "abcdefghijklmnopqrstuvwxyz"
    for trial in range(20):
        vocab: Vocabulary = []
        for _ in range(rng.randint(5, 60)):
            length = rng.randint(1, 6)
            word = "".join(rng.choice(letters) for _ in range(length))
            vocab.append((word, rng.uniform(0.0, 7.0)))
        hashmap = HashmapT9(vocab)
        trie = TrieT9(vocab, top_k=5)
        words_only = [w for w, _f in vocab]
        for _ in range(10):
            probe = rng.choice(words_only)
            digits = word_to_digits(probe)
            expected = naive_lookup(vocab, digits)
            assert hashmap.lookup(digits) == expected
            assert trie.lookup_exact(digits) == expected


# ---------------------------------------------------------------------------
# Vocabulary construction (wordfreq-backed) and full-scale verification
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def small_real_vocab() -> Vocabulary:
    return build_vocabulary(2000)


def test_build_vocabulary_shape(small_real_vocab):
    assert len(small_real_vocab) == 2000
    words = [w for w, _f in small_real_vocab]
    assert len(set(words)) == len(words)  # no duplicates
    assert all(w.isalpha() and w.isascii() for w in words)
    freqs = [f for _w, f in small_real_vocab]
    for a, b in itertools.pairwise(freqs):
        assert a >= b  # sorted descending by frequency


def test_build_vocabulary_contains_common_words(small_real_vocab):
    words = {w for w, _f in small_real_vocab}
    for common in ("the", "and", "you", "good", "home"):
        assert common in words


def test_verify_passes_on_real_vocab(small_real_vocab):
    problems = verify(small_real_vocab, trials=100, seed=2, top_k=8)
    assert problems == []


@pytest.mark.slow
def test_verify_passes_on_full_default_vocab():
    vocab = build_vocabulary(30_000)
    problems = verify(vocab, trials=200, seed=3, top_k=10)
    assert problems == []


@pytest.mark.slow
def test_default_demo_words_collide_in_full_vocab():
    # Confirms the textbook example is real, not contrived, at full scale.
    vocab = build_vocabulary(30_000)
    hashmap = HashmapT9(vocab)
    matches = hashmap.lookup("4663")
    for word in ("good", "home", "goof", "hoof"):
        if word in [w for w, _f in vocab]:
            assert word in matches
