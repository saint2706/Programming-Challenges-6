"""Offline dictionary/thesaurus lookups against a local Open English WordNet copy.

Zero web-framework dependency, so this is testable and reusable on its own
(the CLI in cli.py is a thin rendering layer on top of it), per this repo's
convention.

Word/definition/relation lookups go through `wn`'s public API (Wordnet.words),
which is fast (<1ms) because it only touches the one requested lemma. Fuzzy
"did you mean" suggestions need the *entire* vocabulary to rank against, and
`wn`'s object API (`Wordnet.words()` with no lemma filter) takes ~30s to
materialize all 163k Word objects for that -- so `all_lemmas()` instead reads
the `forms` table directly out of wn's own SQLite database (the same file
`wn`'s public API reads from), which does the equivalent full-vocabulary scan
in well under a second.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import wn
from rapidfuzz import fuzz
from rapidfuzz import process as fuzz_process

LEXICON_SPEC = "oewn:2021"
FUZZY_SUGGESTION_COUNT = 5
FUZZY_MIN_SCORE = 60.0


class LexiconNotInstalledError(RuntimeError):
    """Raised when the Open English WordNet data hasn't been downloaded yet."""

    def __init__(self) -> None:
        super().__init__(
            "The Open English WordNet lexicon isn't installed yet.\n"
            "Run this once (requires internet access, ~13MB download):\n\n"
            f'    uv run --with wn python -c "import wn; wn.download({LEXICON_SPEC!r})"\n'
        )


@dataclass(frozen=True, slots=True)
class Sense:
    """One synset a word belongs to, flattened to what the CLI renders."""

    pos: str
    definition: str
    examples: tuple[str, ...]
    synonyms: tuple[str, ...]
    antonyms: tuple[str, ...]


POS_NAMES = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "s": "adjective satellite",
    "r": "adverb",
}


def pos_name(pos: str) -> str:
    return POS_NAMES.get(pos, pos)


@lru_cache(maxsize=1)
def _wordnet() -> wn.Wordnet:
    """The shared Wordnet handle, constructed lazily so import doesn't need data."""
    try:
        return wn.Wordnet(LEXICON_SPEC)
    except wn.Error as exc:
        raise LexiconNotInstalledError() from exc


def is_installed() -> bool:
    try:
        _wordnet()
    except LexiconNotInstalledError:
        return False
    return True


def _senses_for(lemma: str, *, pos: str | None = None) -> list[Sense]:
    en = _wordnet()
    out: list[Sense] = []
    for word in en.words(lemma, pos=pos):
        for sense in word.senses():
            synset = sense.synset()
            antonym_senses = sense.get_related("antonym")
            antonyms = tuple(dict.fromkeys(r.word().lemma() for r in antonym_senses))
            synonyms = tuple(
                lemma_ for lemma_ in synset.lemmas() if lemma_.lower() != lemma.lower()
            )
            out.append(
                Sense(
                    pos=word.pos,
                    definition=synset.definition() or "",
                    examples=tuple(synset.examples()),
                    synonyms=synonyms,
                    antonyms=antonyms,
                )
            )
    return out


def define(word: str) -> list[Sense]:
    """All senses (definitions + examples) for `word`, across every part of speech."""
    return _senses_for(word)


def synonyms(word: str) -> list[str]:
    """Deduplicated synonyms for `word`, pooled across every sense/synset it's in."""
    seen: dict[str, None] = {}
    for sense in _senses_for(word):
        for syn in sense.synonyms:
            seen.setdefault(syn, None)
    return list(seen)


def antonyms(word: str) -> list[str]:
    """Deduplicated antonyms for `word`, pooled across every sense it's in."""
    seen: dict[str, None] = {}
    for sense in _senses_for(word):
        for ant in sense.antonyms:
            seen.setdefault(ant, None)
    return list(seen)


def exists(word: str) -> bool:
    return len(_wordnet().words(word)) > 0


def _wn_db_path() -> Path:
    return wn.config.data_directory / "wn.db"


@lru_cache(maxsize=1)
def all_lemmas() -> tuple[str, ...]:
    """Every distinct lemma form in the lexicon, for fuzzy ranking.

    See the module docstring for why this bypasses `wn`'s object API.
    """
    if not is_installed():
        raise LexiconNotInstalledError()
    conn = sqlite3.connect(f"file:{_wn_db_path()}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT DISTINCT form FROM forms").fetchall()
    finally:
        conn.close()
    return tuple(sorted(row[0] for row in rows))


def suggest(
    word: str,
    *,
    limit: int = FUZZY_SUGGESTION_COUNT,
    min_score: float = FUZZY_MIN_SCORE,
) -> list[str]:
    """Fuzzy "did you mean" suggestions for a word that isn't in the lexicon."""
    matches = fuzz_process.extract(
        word, all_lemmas(), scorer=fuzz.ratio, limit=limit, score_cutoff=min_score
    )
    return [match for match, _score, _index in matches]


def _search_relevance(lemma_lower: str, query_lower: str) -> tuple[int, int, str]:
    """Sort key: prefix matches first, then other substring hits, shortest first.

    Plain alphabetical order buries the most relevant hit -- e.g. searching
    "happ" would rank "Chappe telegraph"/"chapped"/"chapping" ahead of
    "happy" purely because 'C'/'c' precedes 'h'. Ranking prefix matches
    first, then shorter words, fixes that without needing a fuzzy scorer for
    what's already an exact substring match.
    """
    if lemma_lower.startswith(query_lower):
        tier = 0
    elif query_lower in lemma_lower:
        tier = 1
    else:
        tier = 2
    return (tier, len(lemma_lower), lemma_lower)


def search(query: str, *, limit: int = 20) -> list[str]:
    """Words whose spelling is close to `query` (substring hits ranked first)."""
    lemmas = all_lemmas()
    query_lower = query.lower()
    substring_hits = sorted(
        (lem for lem in lemmas if query_lower in lem.lower()),
        key=lambda lem: _search_relevance(lem.lower(), query_lower),
    )
    if len(substring_hits) >= limit:
        return substring_hits[:limit]
    fuzzy_hits = suggest(query, limit=limit, min_score=45.0)
    merged = list(dict.fromkeys([*substring_hits, *fuzzy_hits]))
    return merged[:limit]
