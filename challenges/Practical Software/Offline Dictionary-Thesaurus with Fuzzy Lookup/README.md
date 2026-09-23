# Offline Dictionary/Thesaurus with Fuzzy Lookup

**Category:** Practical Software
**Difficulty:** B (brief: "Bundle a wordlist, use edit-distance for 'did you mean'.")

**Status:** Implemented (Python)

An offline dictionary and thesaurus: real definitions, example sentences,
synonyms, and antonyms, looked up from a local copy of the Open English
WordNet -- no network calls at lookup time, no API keys, and typo-tolerant
via fuzzy "did you mean" suggestions.

## Why "bundle a wordlist" undersells the actual problem

A flat wordlist with short definitions gets you spelling and glosses, but
not a *thesaurus* -- "did you mean" and "synonyms of" are different
problems. A real thesaurus needs a graph of word senses: "run" the verb
(travel on foot) and "run" the noun (a score in baseball) don't share
synonyms, so synonym lookup has to be sense-aware, not just a flat
word-to-word mapping. That graph is exactly what WordNet-family databases
are built around (`synsets` -- sets of synonymous senses -- linked by
relations like antonymy and hypernymy), so this reuses one instead of
hand-rolling a weaker version of the same thing.

The other real problem: naive fuzzy matching over a 150k+ word vocabulary
is *worse* than no fuzzy matching if you pick the wrong scorer -- see
[Fuzzy suggestions: the scorer choice actually matters](#fuzzy-suggestions-the-scorer-choice-actually-matters)
below for a bug this project hit and fixed.

## Design

- **`wn` + Open English WordNet (OEWN).** [`wn`](https://pypi.org/project/wn/)
  is a modern, actively maintained Python package for WordNet-family
  lexicons (the older, more commonly reached-for `nltk.corpus.wordnet` is
  effectively unmaintained and ships an outdated Princeton WordNet copy).
  OEWN is a free, open-license, actively updated fork of the original
  Princeton WordNet, downloaded once via `wn.download("oewn:2021")` into
  `wn`'s own local SQLite cache (`~/.wn_data/wn.db` by default, ~13MB
  download, ~40MB unpacked). After that one-time step, every lookup is
  fully offline -- `lexicon.py` never makes a network call.
- **`lexicon.py`** -- all WordNet access and fuzzy-matching logic, zero CLI
  dependency (so it's testable and reusable on its own, per this repo's
  convention). `define`/`synonyms`/`antonyms` wrap `wn.Wordnet.words()` and
  flatten the sense/synset graph into a simple `Sense` dataclass the CLI
  renders. `LexiconNotInstalledError` catches `wn.Error` at the one place
  `wn` raises it (constructing the `Wordnet` handle) and turns it into a
  message that tells the user the exact command to run, instead of leaking
  a bare traceback the first time someone runs this without having
  downloaded the data yet.
- **`cli.py`** -- a Typer CLI (`define`, `synonyms`, `antonyms`, `search`)
  with `rich`-formatted output, matching this repo's established CLI style.

### The `all_lemmas()` performance trap

Fuzzy suggestions need to rank the *entire* vocabulary against the typo, not
just one word. The obvious way to get "every word in the lexicon" through
`wn`'s public API is `Wordnet.words()` with no lemma filter -- but that
constructs all ~163k `Word` objects, which took **~30 seconds** measured
against this exact dataset. That's fine once at startup for a long-running
server, but unacceptable for a CLI command that should feel instant.

`wn` stores its data in a plain SQLite file at `wn.config.data_directory /
"wn.db"` -- the same file its own public API reads from. `all_lemmas()`
queries `SELECT DISTINCT form FROM forms` against that file directly instead
of going through `wn`'s object layer, which does the same full-vocabulary
scan in **~0.08 seconds** (measured on the same dataset -- roughly 400x
faster). Single-word lookups (`define`/`synonyms`/`antonyms`, which only
need to construct objects for the one queried lemma) stay on `wn`'s public
API, since those are already sub-millisecond and there's no reason to
bypass it there.

### Fuzzy suggestions: the scorer choice actually matters

The first version of `suggest()` used `rapidfuzz.process.extract()` with its
default scorer (`WRatio`, which leans on partial-ratio matching). Typing
`happpy` (typo) suggested `happy` correctly -- but *also* ranked `a`, `h`,
and `p` above real words, at a 90% "match" score. `WRatio`'s partial-ratio
component finds the best-aligned *substring* of the longer string, so a
single-character word trivially scores near-perfect against any query that
happens to contain that character anywhere -- exactly the wrong behavior
for spelling correction, where the two strings should be compared as
wholes. Switching to `fuzz.ratio` (whole-string Levenshtein-based
similarity) fixed it: `happpy` now ranks `happy`, `happily`, `unhappy` at
the top and never surfaces single-letter noise.
`test_suggest_does_not_return_short_substring_noise` in `test_lexicon.py` is
the regression test for this.

The `search` command's plain substring matching had a related, subtler bug:
sorting substring hits alphabetically buried the most relevant result --
searching `happ` ranked `Chappe telegraph` and `chapped` above `happy`,
purely because `C`/`c` precedes `h`. `search()` now ranks prefix matches
above other substring hits, and shorter words above longer ones within each
tier, so `happy` (a 5-letter prefix match) outranks `chapped` (a longer,
non-prefix contains-match). See `_search_relevance()` in `lexicon.py`.

## Usage

```bash
cd "challenges/Practical Software/Offline Dictionary-Thesaurus with Fuzzy Lookup"

# One-time setup (needs internet, ~13MB download):
uv run --with wn python -c "import wn; wn.download('oewn:2021')"

# Then everything below is fully offline:
uv run --with wn --with rapidfuzz --with typer --with rich python cli.py define run
uv run --with wn --with rapidfuzz --with typer --with rich python cli.py synonyms happy
uv run --with wn --with rapidfuzz --with typer --with rich python cli.py antonyms hot
uv run --with wn --with rapidfuzz --with typer --with rich python cli.py search happ --limit 5

uv run --with wn --with rapidfuzz --with typer --with rich --with pytest pytest -q   # 42 tests
```

```
$ python cli.py define happy
happy

adjective
  1. enjoying or showing or marked by joy or pleasure
     "a happy smile"
     "spent many happy days on the beach"
     "a happy marriage"

adjective satellite
  1. marked by good fortune
     "a felicitous life"
     "a happy outcome"
  2. eagerly disposed to act or to be of service
     "glad to help"
  3. well expressed and to the point
     "a happy turn of phrase"

$ python cli.py define happpy
'happpy' not found. Did you mean:
  happy
  happily
  unhappy
  apply
  haply
```

If the lexicon hasn't been downloaded yet, every command fails with a
friendly message instead of a traceback, telling you the exact command to
run:

```
$ python cli.py define run
The Open English WordNet lexicon isn't installed yet.
Run this once (requires internet access, ~13MB download):

    uv run --with wn python -c "import wn; wn.download('oewn:2021')"
```

## What it deliberately doesn't do

- No custom word list editing (add/remove your own words/definitions) --
  the brief asked for lookup, not a dictionary *authoring* tool.
- No pronunciation/audio -- OEWN doesn't ship audio, and text-to-speech is a
  different problem than fuzzy lookup.
- No web UI -- a CLI is the right interface for a fast, offline reference
  tool you'd actually reach for mid-terminal-session; the other three
  challenges built alongside this one use a web UI or TUI where that fits
  their brief better.

## Tests

42 pytest cases across two files, split into two tiers per this repo's
convention of testing business logic independent of any real-data
dependency:

- **Mocked-`wn` tests** (24 in `test_lexicon.py`, 14 of 18 in `test_cli.py`)
  stand in for the real `wn` object graph with small `Fake*` classes (a fake
  `Wordnet`/`Word`/`Sense`/`Synset`) or monkeypatch `lexicon`'s functions
  directly, and run in any environment with no OEWN download required --
  including the regression tests for both scorer bugs described above.
- **Real-data tests** (6 in `test_lexicon.py`, 4 in `test_cli.py`, marked
  `real_lexicon`) drive the actual `wn` package and the real CLI against
  the genuine OEWN dataset end-to-end, and are *skipped* (not failed) via
  `pytest.mark.skipif` if the lexicon hasn't been downloaded in the current
  environment -- so the suite passes cleanly either way, but running it
  after the one-time `wn.download` step (as this project's tests were)
  proves the real pipeline actually works, not just the mocked path.
