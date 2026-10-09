# Environment Variable / Secrets Diff Tool Across Machines

**Category:** Practical Software
**Difficulty:** Intermediate
**Status:** Implemented (Python)

Source modules live in `src/envdiff/`; the tests are in `tests/`.

`envdiff` compares `.env` files, `export -p` dumps and Docker env-files across hosts and answers "which variables differ?" without ever printing a value. Whether two values are equal is computed with keyed fingerprints; the output shows only a group letter per cell (same letter, same value), so a screenshot, a CI log, or a pasted PR comment cannot leak a secret.

## Usage

```bash
cd "challenges/Practical Software/Environment Variable - Secrets Diff Tool Across Machines"

uv run envdiff diff prod.env staging.env dev.env
uv run envdiff diff prod=deploy@web1:/srv/app/.env staging=deploy@web2:/srv/app/.env
uv run envdiff diff .env env:                 # this file vs the current shell environment
uv run envdiff diff a.env b.env --as json     # machine-readable; also --as markdown for PR comments
uv run envdiff diff a.env b.env -i 'AWS_*' -i PATH   # leave variables out
uv run envdiff diff a.env b.env --strict      # warnings fail too, for CI
```

A source is any of: `PATH`, `LABEL=PATH`, `[USER@]HOST:PATH` (fetched with the system `ssh`), `-` (stdin), `env:` (this process), or a snapshot `.json`. Exit status: `0` identical, `1` differences (or warnings with `--strict`), `2` error, as with `diff(1)`.

```text
VARIABLE       ┃ prod ┃ staging ┃ dev ┃ VERDICT
━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APP_ENV        │  A   │    B    │  C  │ differs
DEBUG          │  A   │    B    │  C  │ differs (A/B: only whitespace differs; A/C: only letter case differs)
NEW_FEATURE    │  —   │    A    │  —  │ missing in prod, dev
SECRET_KEY     │  A   │    A    │  B  │ differs (A/B: one is empty)
STRIPE_API_KEY │  A   │    B    │  —  │ differs; missing in dev

Findings
  warn  `DEBUG` has leading or trailing whitespace in its value  [staging]
  warn  `STRIPE_API_KEY` still holds a placeholder-looking value  [staging]
  warn  `SECRET_KEY` has the same value in prod, staging; environments that must stay isolated should not share a secret  [prod, staging]
```

### Snapshots: compare hosts that cannot see each other's secrets

```bash
export ENVDIFF_PASSPHRASE='a long shared passphrase'     # same on every host
envdiff snapshot /srv/app/.env --label prod -o prod.json  # on prod
envdiff snapshot /srv/app/.env --label staging -o staging.json
envdiff diff prod.json staging.json                       # anywhere, later, with no secret in sight
```

A snapshot holds names, keyed fingerprints and coarse flags, no values. It records an id derived from the key, so mixing snapshots made with different passphrases is an error rather than a wall of false differences. The passphrase is read from `ENVDIFF_PASSPHRASE` or `--passphrase-file`, never from argv, where `ps` and shell history would keep it.

### Other options

| Option                                  | Effect                                                                                                                      |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `--format auto\|dotenv\|shell\|raw`     | Dialect (see below). `auto` picks `shell` for `declare -x` output or `*.sh`, else `dotenv`.                                |
| `--all`                                 | Also list variables identical everywhere.                                                                                   |
| `--fingerprints`                        | Show a short keyed fingerprint beside each group letter.                                                                    |
| `--reveal KEY`                          | Print that variable's value (repeatable). The only way any value is ever shown; warns when the name looks secret.          |
| `--lint / --no-lint`                    | Hygiene findings: placeholders, empty secrets, shared secrets, whitespace, parse problems.                                  |

## How "without leaking values" is enforced

1. **Equality by keyed fingerprint.** `HMAC-SHA256(key, variant | name | value)`. The key is random per run, so fingerprints printed by `--fingerprints` mean nothing outside that run and cannot be tested against a dictionary of common values. For snapshots the key is `scrypt(passphrase)`, so a weak passphrase is still costly to guess.
2. **Bound to the variable name.** Two different variables holding the same secret do not reveal that fact.
3. **Raw values do not survive loading.** `build_source` turns each value into fingerprints and flags, then the strings go out of scope. Only variables named by `--reveal` are retained. Every renderer works from the fingerprinted model, so no output path has a value to leak.
4. **Nothing derived from a secret's content.** Not its length, not a prefix. Types are reported as a coarse `shape` (`url`, `bool`, `integer`, ...) and as `redacted` for any secret-looking name.
5. **Errors never quote input.** Parse problems carry a line number and, at most, a syntactically valid variable name.

The one place this is not absolute: for `user@host:path` sources the file travels over ssh into this process's memory. It is never printed, logged, or written to disk, but the machine running `envdiff` does see it. If even that is too much, use snapshots and run `envdiff snapshot` on the host itself.

### The leak test found a real leak

`tests/test_no_leak.py` plants unique strings in every awkward place (quoted and multi-line values, a malformed line, an unterminated quote, text after a closing quote, a BOM and CRLF endings, a password inside a URL, a duplicate definition) and asserts none appears in any output mode or error path. It failed on the first run: in `raw` mode, the continuation line of a multi-line value has no `=`, so it was parsed as a *pass-through variable name* and printed. A PEM body is a column of alphanumeric lines, so this was a realistic leak. The fix: a line without `=` only counts as a variable when it looks like a variable name (`UPPER_SNAKE`); otherwise it is reported as a skipped line, by number. In shell format a bare name needs `export` or `declare`.

## What a difference means

Values are compared exactly, but each differing pair is also explained from four fingerprint variants (exact, trimmed, unquoted, case-folded), so the output can say *why* without showing anything:

| Verdict                          | Meaning                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------ |
| `only whitespace differs`        | The classic invisible bug: a trailing space or `\r`.                           |
| `only quoting differs`           | `KEY="abc"` vs `KEY=abc` when read by a dialect that keeps quotes.            |
| `only letter case differs`       | `FALSE` vs `false`.                                                            |
| `one is empty`                   | Set in one place, blank in another.                                            |
| `missing in X`                   | Defined elsewhere, absent here.                                                |

### Dialects

The same line means different things to different loaders, and that is a frequent cause of "works on my machine":

* `dotenv`: optional `export`, quotes with `\n \t \" \\ \$` escapes, multi-line quoted values, ` #` inline comments.
* `shell`: `export -p` / `declare -x` output parsed as shell words, including `$'...'` and the `'\''` idiom. `$VAR` is left unexpanded.
* `raw`: Docker `--env-file` and `env` output. Everything after the first `=` is the value, verbatim, so `KEY="x"` keeps its quote characters.

A bare `KEY` (no value) is a pass-through: it takes its value from the runtime environment, so it forms its own group and never equals an explicit value.

## Findings (lint)

| Finding                         | Why it matters                                                                                       |
| ------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `shared-secret`                 | A secret-named variable identical in two sources. Environments that must be isolated should not share one. Reported only for non-empty, non-placeholder values. |
| `empty-secret`, `placeholder`   | `API_TOKEN=` or `changeme` shipped to a real host.                                                   |
| `edge-whitespace`               | Leading/trailing whitespace inside a value.                                                          |
| `shape-mismatch` (info)         | `true` here, `maybe` there.                                                                          |
| parse findings                  | Duplicate keys (last wins), skipped lines, unterminated quotes, trailing text after a quote, BOM, CRLF, unquoted whitespace. |

## Architecture

| Module           | Responsibility                                                                                  |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| `parse.py`       | The three dialect parsers; issues carry line numbers, never content                             |
| `fingerprint.py` | `Hasher` (HMAC, scrypt), variant fingerprints, secret-name and placeholder detection, shapes    |
| `model.py`       | `Entry`/`Source`: what is kept after loading (fingerprints and flags only)                      |
| `sources.py`     | Argument classification, ssh fetch, one shared hasher per run, snapshot key checks              |
| `snapshot.py`    | Snapshot JSON read/write                                                                        |
| `compare.py`     | Per-key grouping, pairwise explanations, lint findings                                          |
| `report.py`      | Text (rich), JSON and Markdown renderers                                                        |
| `cli.py`         | Typer commands `diff` and `snapshot`, atomic snapshot write                                     |

### ssh details

`ssh -o BatchMode=yes -o ConnectTimeout=10 -- HOST "cat -- 'PATH'"`. `BatchMode` stops ssh prompting (a prompt would hang a script); `--` plus a leading-dash check on the host stop a hostile host name (`-oProxyCommand=...`) being read as an option; the path is shell-quoted because the remote side runs it through a shell. A local file always wins over a `host:path` reading, and `C:\` paths are not mistaken for hosts. Output over 5 MiB or containing NUL bytes is rejected as not being an env file.

## Testing

```bash
uv run pytest -q # 109 tests
```

- **test_parse.py** (21): every dialect, escapes, multi-line values, duplicates, BOM/CRLF, issues carry no content, bare-line handling, the same line read differently per dialect.
- **test_fingerprint.py** (12): keyed, name-bound, variants collapse exactly what they should, passphrase derivation, secret-name/placeholder/shape detection.
- **test_compare.py** (19): grouping, missing vs differs, pairwise relations, pass-through, ignore globs, each lint finding.
- **test_sources.py** (25): spec classification (labels, ssh, IPv6, Windows drives, option smuggling), ssh command construction and quoting, failures, snapshot/passphrase rules.
- **test_cli.py** (21): exit codes, every output style, `--reveal`, snapshots end to end, and **real `ssh` subprocess calls against a fake `ssh` executable** on `PATH`.
- **test_no_leak.py** (11 with parametrisation): the planted-secret check across 7 command variants, snapshot output, error paths, and `--reveal` showing exactly what was requested.

## Limitations

- **Equality is visible by design.** Group letters reveal which sources share a value, and `shape` reveals a value's coarse type for non-secret names. With a known passphrase, snapshot fingerprints of low-entropy values (`DEBUG=true`) can be guessed; use a long passphrase and treat snapshots as sensitive-ish.
- **Heuristics.** "Looks like a secret" is a name pattern; `placeholder` is a pattern list. Both have false positives and negatives, and `--ignore` exists for the former.
- **No variable expansion.** `A=$B` is compared as the literal text `$B`.
- **Not a secrets manager.** It reads files; it does not talk to Vault, AWS SSM, or Kubernetes.
- **Lowercase bare names** (`http_proxy` with no `=`) in dotenv/raw files are reported as skipped lines rather than pass-throughs, as the price of the leak fix above.
