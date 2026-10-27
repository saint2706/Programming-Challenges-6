# Git Repository Health Auditor

**Category:** Practical Software
**Difficulty:** A
**Status:** Complete

A read-only health audit for a git repository. It finds stale and abandoned branches, oversized
blobs and Git LFS candidates, secrets in the full history and the working tree, and general hygiene
problems. Output is a Rich table, JSON, SARIF 2.1.0, or a self-contained HTML report. The git
history is read through the `git` CLI; nothing is written to the repository.

## Quick start

Run from this challenge directory:

```bash
uv run --locked gitaudit /path/to/repo                       # table on stdout
uv run --locked gitaudit . --format json -o report.json      # JSON file
uv run --locked gitaudit . --format sarif -o report.sarif    # SARIF 2.1.0 for code scanning
uv run --locked gitaudit . --format html -o report.html      # self-contained HTML
uv run --locked gitaudit . --baseline audit-baseline.json    # accept known findings
uv run --locked gitaudit . --write-baseline audit-baseline.json
uv run --locked pytest -q                                    # tests, offline
```

## Command table

| Flag                    | Default    | Purpose                                                                                    |
| ----------------------- | ---------- | ------------------------------------------------------------------------------------------ |
| `repo` (positional)     | `.`        | Repository to audit                                                                        |
| `--config FILE`         | discovered | Explicit `gitaudit.toml` or `pyproject.toml` (`[tool.gitaudit]`)                           |
| `--ref REF`             | `HEAD`     | Commit-ish to audit                                                                        |
| `--default-branch NAME` | auto       | Name of the default branch, used for merge checks                                          |
| `--stale-days N`        | config     | Branch idle time before BR001 fires; must be positive                                      |
| `--long-lived-days N`   | config     | Age before a non-default branch counts as long-lived                                       |
| `--max-blob-size SIZE`  | config     | Blob size limit, e.g. `5MiB` (BL001/BL002)                                                 |
| `--lfs-threshold SIZE`  | config     | Size above which a blob is an LFS candidate, e.g. `1MiB`                                   |
| `--max-scan-size SIZE`  | config     | Per-blob content scan limit, e.g. `2MiB`                                                   |
| `--top N`               | config     | Rows per list in table output                                                              |
| `--fail-on LEVEL`       | `high`     | Lowest severity that fails the run: `info`, `low`, `medium`, `high`, `critical`, or `none` |
| `--no-history`          | off        | Audit only the tree at `--ref`, skipping history                                           |
| `--since DATE`          | none       | Only history newer than this git date, e.g. `'2 years ago'`                                |
| `--check-signatures`    | off        | Verify recent commit signatures (runs `gpg`)                                               |
| `--check-identities`    | off        | Flag placeholder author emails                                                             |
| `--baseline FILE`       | none       | Accept findings whose fingerprints are listed in the file                                  |
| `--write-baseline FILE` | none       | Write current fingerprints to the file and exit 0                                          |
| `--allow-path GLOB`     | none       | Skip paths for secret checks (repeatable)                                                  |
| `--allow-regex REGEX`   | none       | Ignore matching secret values (repeatable)                                                 |
| `--allow-partial`       | off        | Do not fail when a budget or cap stopped the audit early (default: it fails)               |
| `--format`              | `table`    | `table`, `json`, `sarif`, or `html`                                                        |
| `--output`, `-o FILE`   | stdout     | Write JSON, SARIF or HTML to a file                                                        |
| `--quiet`, `-q`         | off        | Table only: summary without per-row output                                                 |

Exit codes: `0` no finding at or above `--fail-on`; `1` such a finding exists, or the audit was
partial (a `--time-budget`, `--scan-budget` or hit cap stopped it early) unless `--allow-partial`
is given; `2` usage or configuration error (printed to stderr as `gitaudit: error: ...`).

When `GITHUB_ACTIONS=true`, the run also prints workflow annotations.

Configuration can come from `gitaudit.toml` or `[tool.gitaudit]` in `pyproject.toml`. CLI flags
override file values. Unknown keys and out-of-range values are rejected, and CLI overrides get the
same validation as file values.

## Method

The audit runs in four check families. Each finding has a stable rule ID from a single catalog.

- **Branches (BR).** Last-commit age per branch, merged-into-default checks, and long-lived branch
  checks. Remediation hints name the branch only when it passes a strict charset check.
- **Blobs (BL).** `rev-list --objects --filter=object:type=blob` lists every distinct blob without
  reading content. `cat-file --batch-check` gives sizes. Oversized blobs (BL001, BL002) are
  HIGH when still in HEAD and more than 10x the limit, otherwise MEDIUM. LFS candidates (BL003) and
  the `git gc` advice (BL004, using a loose-object threshold of 6700) come from `count-objects`.
- **Secrets (SEC) and paths (PTH).** Blob content is streamed with `cat-file --batch` and scanned
  in overlapping 4 KiB windows. Rules are fixed-prefix provider tokens (SEC001 to SEC014), keyword
  assignments, an entropy rule (SEC015, requires digit, lower and upper characters), and path
  rules (PTH001 to PTH006). `DATA_URI` payloads are stripped first. Lines containing
  `gitaudit:allow` are skipped. The allowlist covers path globs, value regexes, and stopwords such as
  `example` and `placeholder`. Output shows a short prefix and suffix plus the length, and
  fingerprints are unsalted SHA-256.
- **Hygiene (HY).** Repository-level checks, including placeholder identities (`--check-identities`)
  and signature status (`--check-signatures`).

History is streamed from `git log --raw -z` and `ls-tree -r -l -z`. Git is called with argv lists
only, `GIT_CONFIG_GLOBAL` set to the null device, `GIT_OPTIONAL_LOCKS=0`, `GIT_TERMINAL_PROMPT=0`,
and `LC_ALL=C`.

## Results

Measured on the development machine (Windows 11, Python 3.14). Files are in `results/`.

**Secret evaluation** (`results/secret_eval.md`): 72 labelled samples, 55 positive and 17 negative.
Precision 1.0000, recall 0.9636. The two misses are `aws_secret/2` and `aws_secret/3`, the AWS
secret key with no `aws` prefix or key name in the context. Negatives include hashes, UUIDs,
placeholders, lorem text, data URIs, integrity strings, and the AWS documentation example.

**Scaling** (`results/scaling.md`): synthetic linear histories built with `git fast-import`.

| Commits | Audit (s) | Seconds per 1000 commits | Findings |
| ------- | --------- | ------------------------ | -------- |
| 250     | 0.38      | 1.51                     | 6        |
| 1000    | 0.41      | 0.41                     | 9        |
| 4000    | 0.82      | 0.21                     | 19       |

The time grows roughly linearly, with a fixed startup cost that dominates small histories. The
counts rose by one per size after the security review: each history has a 2 MiB random blob that is
now reported as a single INFO HY014 (skipped blob) finding.

**Monorepo** (`results/monorepo.md`): this challenge monorepo with full history. 458 commits,
3793 distinct blobs, 3615 content-scanned, 173 binary. Audit time 17.5 s. 119 findings: SEC015 50,
SEC013 61, BL003 4, BL004 1, HY004 1, HY014 1, SEC008 1. SEC013 rose from 41 because the blanket
`password` stopword was removed. Peak memory was not measured, because the standard library has no
portable counter on Windows. The triage notes in the file explain the noisy rules.

## Security review

An independent review found items A1 to A8 and several smaller nits. Each has a regression test in
`tests/test_hardening.py` (or `tests/test_units.py` where noted).

- **A1, git gpg and editor execution.** Git subprocesses run with argv lists, `-c` pins that
  outrank repository config (`gpg.program`, `gpg.openpgp.program`, `gpg.ssh.program`,
  `gpg.x509.program`, `gpg.format`, `core.sshCommand`, `core.editor`, `sequence.editor`,
  `credential.helper`), and a global config of `os.devnull`. Pagers, prompts and lazy fetch are off.
- **A2, hostile names in commit fields.** Author and subject are read with `log -z` and NUL
  separators, parsed by token count. Control and format characters (Cc, Cf, Zl, Zp, including
  U+2028 and U+2029) are escaped before output. `cli_arg` also rejects a leading `@`.
- **A3, quadratic parsing, budgets and timeouts.** Record parsing is linear and each record is
  capped at 4 MiB. Git streams are killed on a timer. Two new budgets, `time_budget` (default 600 s)
  and `scan_budget_bytes` (default 2 GiB), stop the audit early and mark the result partial. Secret
  hits are capped per blob (200) and in total (5000).
- **A4, ReDoS in hygiene.** The submodule URL check no longer uses a multiline regex. Parsing is
  per line, and the query-secret check uses a literal alternation. Long URL lines and long colon runs
  are timed in tests (under 2 s).
- **A5, credentials echoed as evidence.** A submodule URL with userinfo prints only
  `scheme://<credentials removed>`. A secret-looking query string is reported HIGH without its
  value. Plain `https` URLs without credentials are not reported.
- **A6, truncation hid HIGH findings.** `--top` now limits table rows only. Findings are never
  truncated, so the exit code reflects every finding (a test with 300 files and a HIGH finding exits 1).
- **A7, nonzero git exit misclassified.** A nonzero git exit raises `GitError` unless the consumer
  stopped reading early. Early stops are not errors.
- **A8, hostile storage refused.** A UNC or `//` gitfile or commondir, alternates outside the
  repository or on UNC, and `[include]`/`[includeIf]` in config are refused. Exit 2.
- **Nits.** Inline `gitaudit:allow` is opt-in (`--honor-inline-allow`). Skipped blobs are an INFO
  finding (HY014). `--allow-path` skips a blob's content only when every path it appears under is
  allowed. Stopwords no longer include
  `password`, `passwd`, `todo` or `fixme`. `redact` masks anything under 16 characters completely.
  Fingerprints include an occurrence index. Output is written atomically with `mkstemp`. Git runs
  with `--no-ext-diff --no-textconv` and `i18n.logOutputEncoding=UTF-8`. Stdout is UTF-8. An
  unexpected exception exits 2 with `internal error`. SARIF `executionSuccessful` is false when the
  run is partial. An incomplete audit adds HY013 at HIGH, so it fails the run (see below).

### Second review (items B1 to B4)

A post-commit review found four more issues. Each has a regression test.

- **B1, quoted alternates.** Git C-unquotes an alternates line that starts with `"`, which the
  plain-text check could not judge. Such a line is now refused. Git splits alternates on newlines
  only (`a:b` is one path), so no colon splitting is needed.
- **B2, include keys.** The regex that spotted `[include` is gone. Git lists each config file's keys
  with `--no-includes`, and any `include.*` or `includeif.*` key is refused. This covers
  `config`, `config.worktree` and the common `config`, plus every spelling git accepts. A file git
  cannot parse fails closed.
- **B3, storage discovery.** The audit collects every storage directory that could apply (gitfile
  targets, `.git` directories and bare repositories above the path) and checks each one. It then
  asks git for `rev-parse --absolute-git-dir`. If git's answer is not one of the checked
  directories, the audit refuses to run. Bare repositories, gitfiles and nested repositories are
  tested. Bare repositories keep git's default `safe.bareRepository`, so they stay auditable.
- **B4, partial audits fail open.** A budget or cap now makes the run fail (exit 1) even with
  `--fail-on none`, unless `--allow-partial` is given. With that flag the HY013 finding drops to
  INFO. Partial results are never written as a clean pass.

### Third review (items C1 to C3)

A third review found parser differences in the storage code. Re-implementing git's parsing kept
leaving gaps, so the rules are now simple and fail closed. Each has regression tests.

- **C1, check, then use.** The nearest storage directory is validated locally, then pinned with
  `GIT_DIR` (and `GIT_WORK_TREE` unless bare) in the environment of every later git call. The
  `rev-parse --absolute-git-dir` answer is only a post-check and must match the pinned directory.
  A `.git` file must be exactly one line that starts with `gitdir: `, ends in a single newline, and
  holds non-empty UTF-8 with no control characters (CRLF, a missing space, or a second line are
  refused). A `.git` file that is neither this nor a directory fails closed, even beside a
  bare-looking directory.
- **C2, nested alternates.** Git follows `objects/info/alternates` for every object read, so a
  finding alone cannot keep the audit honest. Any alternates entry is refused, including ones
  inside the repository and ones whose target has its own alternates. No `--allow-alternates`
  flag was added. To audit such a repository, run `git repack -a -d` and remove the file.
- **C3, alternates bytes.** The file is read as bytes and split on `\n` only. Anything else (a
  carriage return, NUL, non-UTF-8 bytes, leading or trailing spaces) is refused, and a file with no
  entries (empty or only newlines) is accepted.

### Fourth review (items D1 and D2)

- **D1, core.worktree.** `GIT_WORK_TREE` is now always set, so a `core.worktree` in any config
  cannot pick the work tree. A bare repository gets an empty scratch directory. Every checked
  config file's `core.worktree` values are also read, and a network path is refused. Tests cover a
  UNC value (refused) and a normal path (accepted, and ignored).
- **D2, ownership.** Superseded by E1 in the fifth review. The unpinned `rev-parse` it added ran
  repo-scoped git before validation, so it was removed.

### Fifth review (items E1 to E3)

The unpinned `rev-parse` from D2 let git read repository config before the candidate was checked.
It is removed, which keeps one rule: no git process runs unless `GIT_DIR` and `GIT_WORK_TREE` pin
it to the validated candidate.

- **E1, pinned git only.** The gitfile, commondir and owner checks run in Python first. Then every
  `git config --file` read and the final `rev-parse` run with `GIT_DIR` and `GIT_WORK_TREE` set.
- **E2, ownership.** Before any git process, the audit compares the owner of the storage directory
  (and of the worktree, if not bare) with the current user. On POSIX this is `st_uid` against
  `geteuid()`, so root-owned files are refused unless we run as root. On Windows it compares the
  owner SID with the current user's SID through `advapi32` (ctypes). If the owner cannot be read,
  the audit fails closed. Two CLI flags are the only way past the check: `--safe-directory PATH`
  exempts one repository, and `--trust-owner` skips the check. Neither is read from config.
- **E3, invalid `.git`.** A `.git` directory without a `HEAD` file and an `objects` directory is
  refused. It no longer falls through to treating the parent as a bare repository.

### Threat model

Defended: the audited repository is untrusted data. Its config, gitfile, commondir, alternates,
`core.worktree` and the storage directory's owner are checked before any git process reads it. Git
is pinned to the nearest storage directory, with `GIT_WORK_TREE` always set, so a `core.worktree`
never picks the work tree. A network path is refused in any of these.

Not defended:

- A hostile `git` binary on `PATH`.
- A hostile parent directory tree that changes discovery beyond the nearest candidate. Only the
  nearest candidate is pinned. The others are checked but not used.
- Windows junctions or symlinks used as storage directories. They are not resolved specially.
- A race between the owner check and git's later reads. The owner is checked by path, not by an
  open file handle, so a repository swapped in between is not caught.
- A `.git` directory with a valid `HEAD` and `objects` but corrupt contents. Git reads it as it
  would, and a crafted file can still make git misbehave beyond the checks above.
- Repositories owned by another user are refused by default. Auditing one needs `--safe-directory`
  or `--trust-owner`, which accept that risk.

Deviations from the review's suggestions:

- Gitfile, alternates and `[include]` problems are refused, not reported as findings. Alternates
  refusal in C2 is a stricter rule than the review's "refuse or flag" option, because a flagged
  alternates file still lets git read the other object store.
- Credentialed URLs print no host or path, because a password can contain `@` or `/`.
- The `password` stopword was removed, so SEC013 rises on test fixtures (see the monorepo results).
- The fingerprint format changed (it now includes the occurrence). Baselines written before this
  change must be regenerated.

Earlier findings from the first review, still in place: `cli_arg(value, placeholder)` in
`safety.py` emits a branch or path only when it matches a conservative charset, so copy-paste hints
cannot inject commands. Config discovery is skipped inside the audited worktree and never searches
upwards.

## Where this is actually used

Git history auditing is standard practice in two forms. Gitleaks, a secret scanner, runs in two
places: as a pre-commit hook configured through `.pre-commit-config.yaml`, and in CI through its
GitHub Action. For git scans it runs `git log -p` under the hood, so it reads the commit history
rather than only the working tree. Its README notes that a baseline helps with repositories that
have a long history. The project is now feature complete and its maintainer has shifted focus to
Betterleaks, which means new users should check the project's current status.

History rewriting is the remediation step when a secret has been committed. git-filter-repo is
described on its project page as a tool for rewriting history that serves a similar purpose to
`git filter-branch`, but is much faster and has more capabilities. The page states that the git
project now recommends it over `git filter-branch`.

This challenge follows the same split. It only reports findings and never rewrites history. The
baseline lets a team accept known findings in a CI gate while credentials are rotated.

## Layout

```text
src/gitaudit/          package: git wrapper, branches, blobs, secrets, hygiene, baseline,
                       reporters (table, JSON, SARIF, HTML), config, safety, cli
tests/                 pytest suite: units, integration (real git repos), security, eval guard
scripts/               evaluate_secrets.py (precision and recall), scaling.py (fast-import timing)
results/               secret_eval.{json,md}, scaling.{json,md}, monorepo.md
data/                  git-ignored output of the monorepo run
```

## Tests

```bash
uv run --locked pytest -q                            # 234 passed, 3 skipped (slow guard, 2 symlink tests)
GITAUDIT_SLOW=1 uv run --locked pytest -q -m slow    # secret-evaluation guard (2 tests)
uv run scripts/evaluate_secrets.py                   # regenerate results/secret_eval.*
uv run scripts/scaling.py                            # regenerate results/scaling.*
```

The default run is hermetic: tests build their own repositories with `git fast-import` and temporary
directories, so it passes offline and on a shallow clone. Git-backed tests are marked `requires_git`
and skip when git is missing. The `slow` marker is opt-in through `GITAUDIT_SLOW=1`.

## Stack

- **rich** for the table output. It handles column layout and terminal width.
- **pytest** and **pytest-timeout** for the tests. The timeout is set per test.
- **tomllib** from the standard library for config files, so no TOML dependency is needed.
- **git CLI** for all repository reads. Its plumbing commands are streamable and need no extra
  dependency.
- Not used: pygit2 or dulwich (an extra object reader to keep in step with git), and gitleaks (the
  point is to build the scanner).

## Limits

- SEC002 (AWS secret key) only fires when the value is next to an `aws` prefix or a key name.
- The `example` stopword suppresses any value containing `EXAMPLE`. This is intentional for the AWS
  documentation key, and it can also hide a real key that contains that word.
- SEC013 (keyword rule) is noisy on code identifiers. On this monorepo, 41 hits are almost all false
  positives.
- SEC015 (entropy) still flags about 50 CamelCase identifiers in JSON metrics files.
- Peak memory was not measured.
- `--check-signatures` needs `gpg` on PATH.
- The shallow fetch of a large public repository was not run, because it needs network access.
