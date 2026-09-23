"""Tests for cli.py.

Most tests monkeypatch the `lexicon` functions cli.py imported (`cli.exists`,
`cli.define`, etc.) with canned return values, so the CLI's argument parsing,
exit codes, and output formatting are tested without needing the real Open
English WordNet data. A small set of `real_lexicon`-marked tests drive the
actual command line against the genuine dataset end-to-end, skipped (not
failed) if it hasn't been downloaded in the current environment.
"""

from __future__ import annotations

import cli
import lexicon
import pytest
from cli import app
from lexicon import Sense
from typer.testing import CliRunner

runner = CliRunner()

real_lexicon = pytest.mark.skipif(
    not lexicon.is_installed(),
    reason='Open English WordNet not downloaded; run wn.download("oewn:2021") first',
)


def _output(result) -> str:
    return result.stdout + (result.stderr if result.stderr_bytes is not None else "")


# --- define -------------------------------------------------------------


def test_define_known_word(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "exists", lambda w: True)
    monkeypatch.setattr(
        cli,
        "define",
        lambda w: [
            Sense(
                pos="a",
                definition="having a high degree of heat",
                examples=("a hot stove",),
                synonyms=(),
                antonyms=(),
            ),
        ],
    )
    result = runner.invoke(app, ["define", "hot"])
    assert result.exit_code == 0, result.output
    assert "having a high degree of heat" in result.output
    assert "a hot stove" in result.output
    assert "adjective" in result.output


def test_define_groups_multiple_parts_of_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "exists", lambda w: True)
    monkeypatch.setattr(
        cli,
        "define",
        lambda w: [
            Sense(
                pos="n",
                definition="a score in baseball",
                examples=(),
                synonyms=(),
                antonyms=(),
            ),
            Sense(
                pos="v",
                definition="move fast on foot",
                examples=(),
                synonyms=(),
                antonyms=(),
            ),
        ],
    )
    result = runner.invoke(app, ["define", "run"])
    assert result.exit_code == 0
    assert "noun" in result.output
    assert "verb" in result.output


def test_define_unknown_word_suggests_alternatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "exists", lambda w: False)
    monkeypatch.setattr(cli, "suggest", lambda w: ["happy", "happily"])
    result = runner.invoke(app, ["define", "happpy"])
    assert result.exit_code == 1
    assert "Did you mean" in _output(result)
    assert "happy" in _output(result)


def test_define_unknown_word_no_suggestions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "exists", lambda w: False)
    monkeypatch.setattr(cli, "suggest", lambda w: [])
    result = runner.invoke(app, ["define", "zzzqqqxxx"])
    assert result.exit_code == 1
    assert "no close matches" in _output(result)


def test_define_lexicon_missing_exits_with_friendly_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(word: str) -> bool:
        raise lexicon.LexiconNotInstalledError()

    monkeypatch.setattr(cli, "exists", _raise)
    result = runner.invoke(app, ["define", "hot"])
    assert result.exit_code == 2
    assert "wn.download" in _output(result)


# --- synonyms / antonyms -------------------------------------------------


def test_synonyms_known_word(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "exists", lambda w: True)
    monkeypatch.setattr(cli, "synonyms", lambda w: ["glad", "felicitous"])
    result = runner.invoke(app, ["synonyms", "happy"])
    assert result.exit_code == 0
    assert "glad" in result.output
    assert "felicitous" in result.output


def test_synonyms_none_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "exists", lambda w: True)
    monkeypatch.setattr(cli, "synonyms", lambda w: [])
    result = runner.invoke(app, ["synonyms", "xylophone"])
    assert result.exit_code == 0
    assert "No synonyms recorded" in result.output


def test_antonyms_known_word(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "exists", lambda w: True)
    monkeypatch.setattr(cli, "antonyms", lambda w: ["cold"])
    result = runner.invoke(app, ["antonyms", "hot"])
    assert result.exit_code == 0
    assert "cold" in result.output


def test_antonyms_unknown_word(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "exists", lambda w: False)
    monkeypatch.setattr(cli, "suggest", lambda w: [])
    result = runner.invoke(app, ["antonyms", "zzzqqqxxx"])
    assert result.exit_code == 1


# --- search ---------------------------------------------------------------


def test_search_returns_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli, "search", lambda q, limit=20: ["happy", "happily", "unhappy"]
    )
    result = runner.invoke(app, ["search", "happ"])
    assert result.exit_code == 0
    for word in ("happy", "happily", "unhappy"):
        assert word in result.output


def test_search_respects_limit_option(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_limit = {}

    def fake_search(q: str, limit: int = 20) -> list[str]:
        seen_limit["limit"] = limit
        return ["happy"]

    monkeypatch.setattr(cli, "search", fake_search)
    result = runner.invoke(app, ["search", "happ", "--limit", "3"])
    assert result.exit_code == 0
    assert seen_limit["limit"] == 3


def test_search_no_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "search", lambda q, limit=20: [])
    result = runner.invoke(app, ["search", "zzzqqqxxx"])
    assert result.exit_code == 0
    assert "No matches" in result.output


def test_search_lexicon_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(q: str, limit: int = 20):
        raise lexicon.LexiconNotInstalledError()

    monkeypatch.setattr(cli, "search", _raise)
    result = runner.invoke(app, ["search", "happy"])
    assert result.exit_code == 2


# --- help -------------------------------------------------------------------


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("define", "synonyms", "antonyms", "search"):
        assert command in result.output


# --- real Open English WordNet data, end to end (skipped if not downloaded) -


@real_lexicon
def test_real_define_run_end_to_end() -> None:
    result = runner.invoke(app, ["define", "run"])
    assert result.exit_code == 0
    assert "verb" in result.output


@real_lexicon
def test_real_typo_suggests_correction_end_to_end() -> None:
    result = runner.invoke(app, ["define", "happpy"])
    assert result.exit_code == 1
    assert "happy" in _output(result)


@real_lexicon
def test_real_synonyms_end_to_end() -> None:
    result = runner.invoke(app, ["synonyms", "happy"])
    assert result.exit_code == 0
    assert "glad" in result.output


@real_lexicon
def test_real_search_end_to_end() -> None:
    result = runner.invoke(app, ["search", "happ", "--limit", "5"])
    assert result.exit_code == 0
    assert "happy" in result.output
