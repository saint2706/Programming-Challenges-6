# Batch File Renamer with Regex Preview

**Category:** Practical Software
**Difficulty:** B (brief: "Live-preview renames before committing; support undo.")

**Status:** Implemented (Python)

A CLI tool that renames many files at once using a regular expression, always
showing exactly what will happen before anything is touched, refusing to run
if the plan would collide, and recording every batch so it can be undone
later.

## Why this is harder than `for f in *; do mv ...`

Three things break a naive bulk-rename script, and this tool handles all
three:

1. **Case-only renames on a case-insensitive filesystem.** `readme.txt ->
   README.txt` on NTFS or APFS looks like "rename onto an existing file" if
   you just check `dst.exists()`. A naive tool either refuses a perfectly
   valid rename or silently no-ops it.
2. **Chains and swaps.** If file `a` should become `b` *and* `b` should
   become `c`, renaming in the wrong order clobbers `b`'s original content
   before it's read. A two-file swap (`a<->b`) can't be done with a single
   `os.rename` at all without a temp name.
3. **Partial failure.** If the 47th of 200 renames fails (permissions, a
   locked file), the other 46 shouldn't be silently lost, and the failure
   needs to say exactly which file and why.

## Design

**Plan, then apply.** `build_plans` turns a directory + regex + replacement
into a list of `(src, dst)` pairs — files that don't match the pattern at all
are skipped, not shown as no-ops. `find_conflicts` checks the *plan*, not the
filesystem naively: a destination that collides with another file's
**source** in the same batch is a chain/swap (fine), a destination that's a
case-only variant of its own source is fine, and everything else that
already exists is a real, blocking conflict.

**Two-phase temp rename.** `apply_renames` never renames straight from `src`
to `dst`. It first moves every source to a unique temp name
(`.__rename_tmp_<uuid>`) in the same directory, then moves every temp name to
its final destination. This one trick makes case-only renames, chains, and
full swaps all correct without any special-case ordering logic — a temp name
can never collide with an original name. If the backup phase fails partway,
everything already moved is rolled back. If the second phase fails for some
files, the successful ones stay renamed (best-effort), the failing ones are
left at their temp name for manual recovery, and the error names exactly
which files failed and why.

**Undo is a log-replayed reverse batch, not a separate code path.** Every
successful `apply` appends a batch (id, timestamp, pattern, and every
`(src, dst)` pair actually applied) to `.batch_rename_log.json` in the target
directory. `undo` builds the *reverse* plan (`dst -> src` for every
operation) and runs it through the exact same `find_conflicts` /
`apply_renames` machinery — so undoing a swap is just applying the same swap
again, and undoing a chain reverses it correctly, for free. Before touching
anything, undo verifies every renamed file is still where the batch left it
and refuses with a clear message listing what's missing (e.g. the user
already renamed or deleted it since).

## Usage

```bash
cd "challenges/Practical Software/Batch File Renamer with Regex Preview"

# Preview only -- never touches the filesystem
uv run --with typer --with rich python renamer.py preview . '^IMG_(\d+)\.jpg$' 'photo_\1.jpg'

# Apply (asks for confirmation unless --yes); recursive, case-insensitive
uv run --with typer --with rich python renamer.py apply ./photos '^img_(\d+)\.jpg$' 'photo_\1.jpg' \
    --recursive --ignore-case --include '*.jpg'

# See past batches for a directory
uv run --with typer --with rich python renamer.py history ./photos

# Undo the most recent batch, or a specific one by id
uv run --with typer --with rich python renamer.py undo ./photos
uv run --with typer --with rich python renamer.py undo ./photos --batch-id a1b2c3d4e5f6

uv run --with typer --with rich --with pytest pytest -q      # 22 tests
```

The replacement string is a normal Python `re.sub` replacement -- it supports
numbered (`\1`) and named (`\g<name>`) backreferences to groups in the
pattern, so `(?P<date>\d{4}-\d{2}-\d{2})_(?P<name>.+)\.csv` /
`\g<name>_\g<date>.csv` reorders a filename around a captured date.

## What it deliberately doesn't do

- **Directories are not renamed**, only files -- keeping the rename/undo
  semantics unambiguous. Point it at a subdirectory to affect just its files.
- **No cross-directory moves.** The replacement changes a filename, not a
  path with separators in it (`with_name` rejects a slash) -- this is a
  renamer, not a mover.
- **Undo needs the renamed files to still exist under their post-rename
  name.** If you rename, then rename again, then try to undo the *first*
  batch, it will (correctly) refuse -- the test suite covers exactly this
  case (`test_undo_specific_batch_by_id`).

## Tests

22 pytest cases covering: basic and named-group regex substitution,
non-matching files being skipped, case-insensitive matching, duplicate-target
and target-exists conflict detection, case-only renames and same-batch
swaps/chains *not* being flagged as conflicts, a real 3-cycle rotation
(`a->b->c->a`), a simulated partial-failure mid-apply (via monkeypatching
`Path.rename`) verifying the successful half survives and the failure detail
names the right file, full apply/undo round trips, undo refusing on a stale
or already-superseded batch, and CLI smoke tests through Typer's
`CliRunner` (including that `preview` truly never touches disk).
