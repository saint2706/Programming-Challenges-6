"""Tests for lexicon.py.

Two tiers, matching this repo's convention of testing business logic in
isolation from any real data dependency:

- Mocked-`wn`-API tests (`Fake*` classes below) exercise `_senses_for`,
  `define`, `synonyms`, `antonyms`, and the `suggest`/`search` fuzzy-ranking
  logic without touching the real Open English WordNet data at all, so they
  run in any environment.
- Real-data tests (marked `real_lexicon`) exercise the actual `wn` package
  against the genuine OEWN database, and are skipped (not failed) if it
  hasn't been downloaded in the current environment via
  `wn.download("oewn:2021")`.
"""

from __future__ import annotations

import lexicon
import pytest

real_lexicon = pytest.mark.skipif(
    not lexicon.is_installed(),
    reason='Open English WordNet not downloaded; run wn.download("oewn:2021") first',
)


# --- Fakes standing in for the real `wn` object graph -----------------------


class FakeSynset:
    def __init__(
        self, definition: str, examples: tuple[str, ...], lemmas: tuple[str, ...]
    ):
        self._definition = definition
        self._examples = examples
        self._lemmas = lemmas

    def definition(self) -> str:
        return self._definition

    def examples(self) -> tuple[str, ...]:
        return self._examples

    def lemmas(self) -> tuple[str, ...]:
        return self._lemmas


class FakeAntonymRelation:
    """Stands in for a `wn.Sense` returned by `get_related('antonym')`."""

    def __init__(self, lemma: str):
        self._lemma = lemma

    def word(self) -> FakeAntonymRelation:
        return self

    def lemma(self) -> str:
        return self._lemma


class FakeSense:
    def __init__(self, synset: FakeSynset, antonyms: tuple[str, ...] = ()):
        self._synset = synset
        self._antonyms = antonyms

    def synset(self) -> FakeSynset:
        return self._synset

    def get_related(self, name: str) -> list[FakeAntonymRelation]:
        if name == "antonym":
            return [FakeAntonymRelation(a) for a in self._antonyms]
        return []


class FakeWord:
    def __init__(self, pos: str, senses: list[FakeSense]):
        self.pos = pos
        self._senses = senses

    def senses(self) -> list[FakeSense]:
        return self._senses


class FakeWordnet:
    def __init__(self, words_by_lemma: dict[str, list[FakeWord]]):
        self._words_by_lemma = words_by_lemma

    def words(self, lemma: str, pos: str | None = None) -> list[FakeWord]:
        words = self._words_by_lemma.get(lemma, [])
        if pos is not None:
            words = [w for w in words if w.pos == pos]
        return words


@pytest.fixture
def fake_wordnet(monkeypatch: pytest.MonkeyPatch) -> FakeWordnet:
    hot_synset = FakeSynset(
        "having a high degree of heat",
        ("a hot stove",),
        ("hot", "warm"),
    )
    fw = FakeWordnet(
        {
            "hot": [FakeWord("a", [FakeSense(hot_synset, antonyms=("cold",))])],
            "cold": [
                FakeWord(
                    "a",
                    [
                        FakeSense(
                            FakeSynset(
                                "having a low temperature", (), ("cold", "chilly")
                            )
                        )
                    ],
                )
            ],
        }
    )
    monkeypatch.setattr(lexicon, "_wordnet", lambda: fw)
    return fw


@pytest.fixture
def fake_lemmas(monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    words = (
        "happy",
        "happily",
        "unhappy",
        "apply",
        "harpy",
        "cat",
        "dog",
        "a",
        "h",
        "p",
    )
    monkeypatch.setattr(lexicon, "all_lemmas", lambda: words)
    return words


# --- define / synonyms / antonyms -------------------------------------------


def test_define_returns_definition_and_examples(fake_wordnet: FakeWordnet) -> None:
    senses = lexicon.define("hot")
    assert len(senses) == 1
    assert senses[0].definition == "having a high degree of heat"
    assert senses[0].examples == ("a hot stove",)
    assert senses[0].pos == "a"


def test_define_excludes_queried_word_from_its_own_synonym_list(
    fake_wordnet: FakeWordnet,
) -> None:
    senses = lexicon.define("hot")
    assert senses[0].synonyms == ("warm",)


def test_synonyms_pools_across_senses(fake_wordnet: FakeWordnet) -> None:
    assert lexicon.synonyms("hot") == ["warm"]


def test_antonyms_reads_sense_relation(fake_wordnet: FakeWordnet) -> None:
    assert lexicon.antonyms("hot") == ["cold"]


def test_antonyms_empty_when_none_recorded(fake_wordnet: FakeWordnet) -> None:
    assert lexicon.antonyms("cold") == []


def test_exists_true_for_known_word(fake_wordnet: FakeWordnet) -> None:
    assert lexicon.exists("hot") is True


def test_exists_false_for_unknown_word(fake_wordnet: FakeWordnet) -> None:
    assert lexicon.exists("zzzznotaword") is False


def test_define_unknown_word_returns_empty_list(fake_wordnet: FakeWordnet) -> None:
    assert lexicon.define("zzzznotaword") == []


def test_pos_name_maps_known_codes() -> None:
    assert lexicon.pos_name("n") == "noun"
    assert lexicon.pos_name("v") == "verb"
    assert lexicon.pos_name("s") == "adjective satellite"


def test_pos_name_falls_back_to_raw_code_for_unknown() -> None:
    assert lexicon.pos_name("zzz") == "zzz"


# --- fuzzy suggestion / search ranking --------------------------------------


def test_suggest_ranks_close_typo_above_unrelated_words(
    fake_lemmas: tuple[str, ...],
) -> None:
    hits = lexicon.suggest("happpy")
    assert hits[0] == "happy"


def test_suggest_does_not_return_short_substring_noise(
    fake_lemmas: tuple[str, ...],
) -> None:
    # A naive substring/partial-ratio scorer would rank single-letter words
    # like "a"/"h"/"p" highly just because they're trivially contained in
    # "happpy" -- this is the regression test for that bug.
    hits = lexicon.suggest("happpy")
    assert "a" not in hits
    assert "h" not in hits
    assert "p" not in hits


def test_suggest_respects_limit(fake_lemmas: tuple[str, ...]) -> None:
    hits = lexicon.suggest("happpy", limit=2)
    assert len(hits) <= 2


def test_suggest_empty_for_wildly_different_word(fake_lemmas: tuple[str, ...]) -> None:
    assert lexicon.suggest("xyzxyzxyz") == []


def test_search_prioritizes_substring_hits(fake_lemmas: tuple[str, ...]) -> None:
    hits = lexicon.search("app")
    assert "happy" in hits
    assert "apply" in hits


def test_search_falls_back_to_fuzzy_when_no_substring_hits(
    fake_lemmas: tuple[str, ...],
) -> None:
    hits = lexicon.search("happpy")
    assert "happy" in hits


def test_search_respects_limit(fake_lemmas: tuple[str, ...]) -> None:
    hits = lexicon.search("a", limit=1)
    assert len(hits) == 1


def test_search_ranks_prefix_matches_above_other_substring_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "chappy" would sort before "happy" alphabetically despite "happy"
    # being a prefix match and "chappy" only a contains-match.
    monkeypatch.setattr(lexicon, "all_lemmas", lambda: ("chappy", "happy", "happier"))
    hits = lexicon.search("happ", limit=3)
    assert hits[0] == "happy"


# --- real Open English WordNet data (skipped if not downloaded) -------------


@real_lexicon
def test_real_define_run_has_many_senses() -> None:
    senses = lexicon.define("run")
    assert len(senses) > 10
    assert any(s.pos == "v" for s in senses)
    assert any(s.pos == "n" for s in senses)


@real_lexicon
def test_real_synonyms_of_happy_includes_glad() -> None:
    assert "glad" in lexicon.synonyms("happy")


@real_lexicon
def test_real_antonyms_of_hot_includes_cold() -> None:
    assert "cold" in lexicon.antonyms("hot")


@real_lexicon
def test_real_suggest_corrects_typo() -> None:
    assert "happy" in lexicon.suggest("happpy")


@real_lexicon
def test_real_all_lemmas_is_large() -> None:
    assert len(lexicon.all_lemmas()) > 100_000


@real_lexicon
def test_real_exists_false_for_garbage() -> None:
    assert lexicon.exists("zzzqqqxxxnotaword") is False
