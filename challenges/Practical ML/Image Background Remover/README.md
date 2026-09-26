# Image Background Remover

**Category:** Practical ML
**Difficulty:** B (brief: "Pretrained segmentation model wrapped as a batch CLI tool.")

**Status:** Implemented (Python)

A Typer CLI that removes the background from every supported image under an
input file or directory, writing a transparent-background PNG per input into
an output directory. The segmentation itself is delegated to
[`rembg`](https://github.com/danielgatis/rembg), a purpose-built background-
removal library that wraps several pretrained ONNX matting models
(`u2net`/`u2netp`/`isnet-general-use`/`silueta`/...) behind one `remove(...)`
call, with its own session-based model caching (`~/.rembg/models/`, populated
on first use per model). This was chosen over this repo's other pattern —
hand-rolling `onnxruntime` plus a manually vendored, sha256-pinned model file
(as in the *Model Serving API* sibling challenge) — because for this specific
challenge the point is the *batch-CLI wrapper*, and `rembg` is the modern,
least-code way to get a real pretrained segmentation model behind that
wrapper without reinventing session/model management by hand.

The default model is **`isnet-general-use`**, rembg's newer general-purpose
model (noticeably cleaner edges on typical photos than the original `u2net`)
— chosen over `u2net`/`u2netp` for the CLI's own default despite a much
larger one-time download (~179MB vs. ~4.7MB for `u2netp`), because output
*quality* is the more important default for a tool whose entire job is
"looks right when you open the PNG." `--model` lets you switch to `u2net`,
`u2netp`, or `silueta` if you'd rather trade quality for a smaller/faster
model.

## The hard parts

**One bad file must never abort the batch.** A batch CLI processing an
arbitrary folder will eventually hit a file with the wrong extension, a
corrupted image, or a model failure on a pathological input — silently
crashing the whole run because of one bad file (or, worse, writing a
half-written/garbage output and calling it a success) is the actual failure
mode worth designing against here. `remove_bg.process_one` **never raises**:
every failure path (unsupported extension, unreadable file, `PIL` decode
failure, an exception from `rembg.remove` itself) returns a
`FileResult(status="skipped", reason=...)` instead, and the output file is
only written — via `output_path.write_bytes(...)`, after `process_batch`'s
directory-creation step — once removal has actually succeeded, so a failure
never leaves a partial/corrupt file behind. Covered by
`test_removal_error_is_caught_and_leaves_no_partial_output` and
`test_corrupted_file_with_image_extension_is_skipped_not_crashed`.

**Verifying decodability without trusting the file extension.** A `.png`
extension doesn't mean the bytes are actually a valid PNG (or any image at
all) — `process_one` opens every file with Pillow and calls `.verify()`
*before* handing its raw bytes to `rembg`, inside a `with Image.open(path)`
block (Windows keeps a file handle open on a bare `Image.open(...).verify()`
call otherwise, which showed up as intermittent file-lock issues against the
same path in earlier test runs on this machine). Conversely, the
`test_successful_removal_...` test deliberately names its fixture file
`photo.jpg` while writing real PNG bytes into it — proving decodability is
checked by real content sniffing, not by trusting the extension, in either
direction.

**Testing a network-downloaded ML model without making every test run
depend on the network.** `rembg.remove(...)` and `rembg.new_session(...)`
are thin, explicitly-named module-level functions
(`remove_background_bytes`/`new_session`) precisely so tests can
monkeypatch them — 26 of this suite's 27 tests never touch the network or a
real model at all, exercising file-discovery, batch looping, per-file error
handling, and the CLI's argument/summary logic entirely against a fake
"remove" function. The one real end-to-end test
(`test_remove_background_real_model_produces_varying_alpha`) deliberately
uses `u2netp` (~4.7MB) instead of the CLI's actual `isnet-general-use`
default, to keep that one network-dependent test's download small; it draws
a synthetic ellipse-on-background image (a flat solid-color image gives a
real model nothing to segment) and asserts the output's alpha channel has a
real `min != max` spread — proving the model actually ran and produced a
matte, without asserting exact pixel values a model version bump could
break. It's marked `@pytest.mark.network` and calls `pytest.skip(...)`
(never fails) if the model can't be downloaded or run in this environment.

## Design

- **`remove_bg.py`** — framework-free (no `typer`/`rich` imports), so the
  actual batching/error-handling logic is directly unit-testable and its
  network-touching calls are easy to monkeypatch. `resolve_input_files`
  turns an input path into the list of files a batch run should attempt (a
  single file is returned as-is, unfiltered, so an unsupported single file
  still surfaces as one explicit skipped result rather than vanishing; a
  directory is scanned — one level deep, or recursively with
  `recursive=True` — for `SUPPORTED_EXTENSIONS` only).
  `process_one`/`process_batch` do the actual work described in "the hard
  parts" above, preserving each file's path relative to the input root
  (`sub/nested.jpg` → `<output_dir>/sub/nested.png`) so a recursive batch's
  directory structure survives into the output.
- **`cli.py`** — the Typer entrypoint. Validates `--model` against
  `AVAILABLE_MODELS` up front (a clear "Unknown model, available models
  are..." message and exit code 2, rather than surfacing whatever internal
  error `rembg` would raise for a bad model name), shows a `rich` progress
  bar advanced by a `progress_callback` from `process_batch`, prints each
  skip as it happens (so a long batch's problems are visible immediately,
  not only in a final summary), and ends with a summary table plus a
  skip-reason breakdown. Exit code is `1` if literally nothing was
  successfully processed (an empty input, or a batch where every file
  failed) and `0` otherwise — a batch with some skips alongside real
  output is still a successful run.

## What it deliberately doesn't do

- **No GPU execution provider** — `rembg[cpu]`/`onnxruntime` CPU inference
  only; a batch of many large images will be slow (isnet-general-use took
  roughly 15-20s per 200×200 test image on this machine's CPU — see
  "Usage" below). GPU support is a `pip install "rembg[gpu]"` + `providers=`
  change away, out of scope for a Beginner-tier CLI wrapper.
- **No alpha-matting refinement pass** — `rembg` exposes an
  `alpha_matting=True` option (edge refinement via `pymatting`, slower) that
  this CLI doesn't expose as a flag; the base matte from each model is used
  as-is.
- **No human/cloth/anime-specialized models** — `AVAILABLE_MODELS` is
  restricted to rembg's general-purpose models (`isnet-general-use`,
  `u2net`, `u2netp`, `silueta`). Domain-specific models
  (`u2net_human_seg`, `u2net_cloth_seg`, `isnet-anime`) and the much
  heavier `sam`/`birefnet-*` variants are deliberately excluded from a
  generic batch tool's default option set.
- **No output-format choice** — every output is always an RGBA PNG (the
  only format that actually carries the transparency this challenge is
  about); JPEG has no alpha channel, so offering it as an output format
  would silently discard the entire point of the tool.
- **No parallel/multiprocess batch execution** — files are processed
  sequentially. onnxruntime's CPU provider already uses multiple threads
  internally per inference; adding process-level parallelism on top would
  mostly contend for the same CPU cores rather than genuinely speed things
  up, and isn't worth the added complexity for a Beginner-tier tool.

## Usage

```bash
cd "challenges/Practical ML/Image Background Remover"

uv run --with "rembg[cpu]" --with typer --with pillow \
    python cli.py path/to/photos/ path/to/output/ --recursive
# First run downloads and caches the isnet-general-use model (~179MB) to
# ~/.rembg/models/ -- subsequent runs (any model) reuse the cache.

# Use a smaller/faster model instead:
uv run --with "rembg[cpu]" --with typer --with pillow \
    python cli.py path/to/photos/ path/to/output/ --model u2netp

# A single file works too, no --recursive needed:
uv run --with "rembg[cpu]" --with typer --with pillow \
    python cli.py photo.jpg out/

uv run --with "rembg[cpu]" --with typer --with pillow --with pytest \
    pytest -q                        # 26 tests, no network required
uv run --with "rembg[cpu]" --with typer --with pillow --with pytest \
    pytest -q -m network             # +1 real-model end-to-end test
```

**Live-verified:** ran the CLI for real (default `isnet-general-use` model,
`--recursive`) against a two-file sample (a 200×200 synthetic "portrait" —
a solid ellipse on a dark background — and a 100×100 flat solid-color
image nested one directory deep), and confirmed both outputs are real RGBA
PNGs, the flat-color image (nothing for a segmentation model to find) came
back mostly transparent (alpha extrema `(0, 33)`), the ellipse-on-background
image came back with a genuine full-range matte (alpha extrema `(0, 255)`),
and the nested file's `sub/` subdirectory was preserved in the output. Each
image took roughly 15-20s of CPU inference time with `isnet-general-use` on
this machine.

## Tests

27 pytest cases across two files.

- `test_remove_bg.py` (17, +1 marked `network`) — `resolve_input_files`
  behavior for single files, directories (recursive vs. non-recursive), and
  mixed/unsupported extensions; `process_one`'s skip-vs-process outcomes
  (unsupported extension, corrupted file, a working removal, a mocked
  removal failure leaving no partial output, nested-path preservation); and
  `process_batch`'s model validation, empty-input short-circuit (never
  creates a session with nothing to do), mixed valid/invalid batches, the
  progress callback firing once per discovered file, and the
  recursive/non-recursive directory-scan difference. The final test is the
  real, network-dependent, `u2netp`-based end-to-end check described above.
- `test_cli.py` (9) — via Typer's `CliRunner`, all mocking the underlying
  `rembg` calls: a nonexistent input path's usage error (exit 2), an
  invalid `--model` choice (exit 2), an empty input directory (exit 1), a
  fully successful batch (exit 0, output files exist), a batch with one
  corrupted file that still exits 0 overall (skip is reported, not fatal),
  a batch where *everything* is skipped (exit 1), the `--recursive`
  flag actually reaching nested files, and its absence correctly not
  finding them, and single-file (non-directory) input.
