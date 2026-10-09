# Automatic README Generator from Repo Structure

**Category:** Practical Software
**Difficulty:** Intermediate
**Status:** Implemented (Python)

Source modules live in `src/readme_generator/`; the tests are in `tests/`.

`readme-gen` walks a repository, reads its manifests, and drafts a README from what is actually there: tech stack, install/run/test commands, environment variables, Docker, CI, project tree, license. It is deterministic and offline, with no LLM and no network, so the same repo always yields the same README and the output can be checked in CI.

## Usage

```bash
cd "challenges/Practical Software/Automatic README Generator from Repo Structure"
uv run readme-gen /path/to/repo              # writes /path/to/repo/README.md
uv run readme-gen /path/to/repo --stdout     # preview, writes nothing
uv run readme-gen /path/to/repo --check      # exit 1 if README.md is missing or stale (CI)
uv run readme-gen /path/to/repo -o DOCS.md   # write somewhere else
uv run readme-gen /path/to/repo --force      # replace a hand-written README
uv run readme-gen /path/to/repo --depth 3    # directory levels in the project tree
```

Exit codes: `0` success, `1` `--check` found drift, `2` usage error (not a directory, README exists and would be lost).

### Example

Run against the sibling Multi-Timezone Meeting Scheduler:

````markdown
<!-- readme-gen:begin stack -->
## Tech stack

- **Languages:** Python (100%)
- **Web framework:** FastAPI, Uvicorn
- **CLI:** Rich, Typer
- **Testing:** pytest
- **Package manager:** uv
<!-- readme-gen:end stack -->
````

The same run also produces a Getting started section whose usage table lists `uv run meeting-scheduler` (from `[project.scripts]`) and `uv run uvicorn meeting_scheduler.web:app --reload` (found by locating the `app = FastAPI()` object in the source).

## What it infers, and from where

| Section         | Source                                                                                                                                                                  |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Header          | Manifest `name` and `description`; failing that, the first line of the top-level package docstring; failing that, a visible `_TODO_` (never an invented description).   |
| Tech stack      | Byte share per language (programming languages only, so Markdown never outweighs the code), plus dependencies grouped by category from per-ecosystem lookup tables.      |
| Getting started | Package manager from lockfiles (`uv.lock`, `poetry.lock`, `pnpm-lock.yaml`, `yarn.lock`, ...) or the `packageManager` field; entry points; test and lint tools.         |
| Make targets    | Top-level `Makefile` targets, with `## doc` comments as descriptions.                                                                                                   |
| Packages        | Manifests in subdirectories (monorepos), up to three levels deep.                                                                                                       |
| Configuration   | `.env.example` / `.env.sample` / `.env.template`; comments directly above a variable become its description.                                                            |
| Docker          | `Dockerfile` and compose files (service names read by indentation, so no YAML dependency).                                                                              |
| Project tree    | The scan, with one-line annotations for well-known paths.                                                                                                               |
| License         | Manifest field, else recognised from the `LICENSE` file text.                                                                                                           |

Ecosystems: Python (`pyproject.toml` PEP 621 and Poetry, `requirements*.txt`), Node (`package.json`), Rust (`Cargo.toml`), Go (`go.mod`), Ruby (`Gemfile`), PHP (`composer.json`), Java (`pom.xml`, Gradle detection).

## Design decisions

### Regenerating must not destroy prose

A generator you can run only once is a toy. Every generated section is wrapped in markers:

```markdown
<!-- readme-gen:begin stack -->
...
<!-- readme-gen:end stack -->
```

When the target README already contains markers, only those blocks are rewritten, and everything outside them is preserved byte for byte. Blocks that no longer apply (the `Dockerfile` was deleted) are removed rather than left lying; new sections are inserted after the nearest earlier section in canonical order. Delete a block's markers to adopt it as hand-written text.

A README *without* markers is never touched unless `--force` is passed, since there would be no way to tell your prose from generated text. That is why `--check` also fails on such a file instead of silently passing.

### Regeneration must be idempotent

The generated README sits in the repository it describes, so a naive project tree lists `README.md` on the second run and not the first, which would make `--check` fail forever. The output file is excluded from the tree; a test asserts that two consecutive runs produce identical files.

### Secrets in `.env.example` files

Templates are supposed to hold placeholders, but real keys get pasted into them. Values are shown as defaults only when the variable name is not secret-looking (`SECRET`, `TOKEN`, `PASSWORD`, `API_KEY`, `DSN`, ...) and the value is short; otherwise only the variable name appears. A test plants `SECRET_KEY=hunter2` and asserts it never reaches the README.

### Walking the tree like git does

`scan.py` applies each directory's own `.gitignore` to its subtree (matched relative to that directory, via `pathspec`'s `GitIgnoreSpec`, so directory patterns and `**` behave as in git). Known junk (`node_modules`, `.venv`, `target`, caches) is skipped regardless. Symlinked directories are listed but not entered, so a link cycle cannot hang the walk, and the walk stops at 50,000 files with a warning.

### Fixtures are not the project

Manifests under `tests/`, `fixtures/`, `examples/`, `docs/` and the like describe sample projects, so they are ignored; otherwise a test fixture's `package.json` would make a Python library claim to be a Node app.

### Broken input degrades, never crashes

A malformed `package.json` becomes a `warning:` on stderr and the rest of the README is still drafted.

## Architecture

| Module         | Responsibility                                                                                           |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| `scan.py`      | gitignore-aware walk, language byte counts                                                               |
| `manifests.py` | Seven manifest parsers into one `Manifest` shape                                                         |
| `stack.py`     | Technology tables, package-manager detection, install/run/test commands, FastAPI/Flask/Streamlit hints   |
| `signals.py`   | Env templates, Makefile, compose, CI, license, description fallback                                      |
| `tree.py`      | Annotated tree rendering, per-directory truncation                                                       |
| `render.py`    | Facts into `Section`s                                                                                    |
| `document.py`  | Marker wrapping and the merge algorithm                                                                  |
| `generate.py`  | Path in, sections out                                                                                    |
| `cli.py`       | Typer command, atomic write (temp file + `os.replace`)                                                   |

Everything except `cli.py` is pure, so the whole pipeline is tested against throwaway repositories built in `tmp_path`.

## Testing

```bash
uv run pytest -q # 64 tests
```

- **test_scan.py** (8): nested `.gitignore` scoping, ignored directories, byte-share languages, symlink cycle, file cap.
- **test_manifests.py** (9): all seven formats, PEP 621 vs Poetry, malformed input, fixture and depth filtering, ordering.
- **test_stack.py** (15): uv/Poetry/pip commands, npm/pnpm/yarn from lockfile and `packageManager`, Rust and Go, FastAPI and Streamlit app discovery, category grouping, Go module-prefix matching.
- **test_signals.py** (6): env template parsing and secret hiding, Makefile, compose, license, description fallback, CI.
- **test_tree.py** (4): ordering, annotations, depth, lockfile and output exclusion, truncation.
- **test_document.py** (7): in-place update, no-op, stale removal, ordered insertion.
- **test_cli.py** (15): writes, idempotence, refusal to overwrite, `--force`, refresh preserving prose, `--check`, `--stdout`, atomic-write cleanup.

Mutating the secret-hiding check makes two tests fail, so they do test it.

## Limitations

- **Inference, not execution.** Commands come from manifests and conventions. A repo that builds with a bespoke script gets only what its manifests reveal; the description falls back to a `_TODO_` rather than a guess.
- **Commands come from the root manifest** of each ecosystem. Monorepo packages are listed in a table, but their individual install/run commands are not.
- **Cross-level `.gitignore` negation** (a child directory re-including something a parent ignored) is not modelled; each rule is applied independently.
- **Gradle** is detected but not parsed; only the `./gradlew build` / `test` commands are suggested.
- **Titles use the manifest name verbatim** (`my-package`), not a prettified version.
