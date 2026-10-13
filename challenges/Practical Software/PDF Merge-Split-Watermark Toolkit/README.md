# PDF Merge/Split/Watermark Toolkit

**Category:** Practical Software
**Difficulty:** Intermediate
**Status:** Implemented (Python)

Source modules live in `src/pdftoolkit/`; the tests are in `tests/`.

`pdftool` is a command-line toolkit for the PDF jobs that come up every week: merge, split, extract and reorder pages, rotate, watermark (text, image or another PDF), encrypt and decrypt, edit metadata, and inspect. Every command takes several inputs, so the same invocation works on one file or a folder of them, and a TOML **batch file** runs a whole list of jobs in one go.

![A report page with a tiled "CONFIDENTIAL" text mark and a corner logo](docs/watermark.png)

## Quick start

```bash
cd "challenges/Practical Software/PDF Merge-Split-Watermark Toolkit"
uv run pdftool --help

uv run pdftool merge "scans/*.pdf" cover.pdf[1] -o book.pdf
uv run pdftool split report.pdf --every 10 --out-dir chunks
uv run pdftool watermark report.pdf --text DRAFT --position tile --opacity 0.2 -o draft.pdf
uv run pdftool encrypt report.pdf -o locked.pdf        # prompts for the password
uv run pdftool batch jobs.toml --dry-run
```

| Command                                                 | What it does                                                                                                                          |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `merge INPUTS -o OUT`                                   | Concatenate. Inputs can be globs and can select pages: `a.pdf[1-3,9-]`. `--outline nested\|files\|original\|none` controls bookmarks. |
| `split INPUTS --every N \| --ranges 1-3,4-9 \| --each`  | One output per chunk, named `<stem>_p1-3.pdf` (`_part003` if the pages are not contiguous).                                           |
| `extract INPUTS --pages SPEC`                           | Keep only those pages, in the order written.                                                                                          |
| `reorder INPUTS --order SPEC`                           | Rearrange; the order must name every page exactly once.                                                                               |
| `rotate INPUTS --degrees 90 [--pages SPEC]`             | Clockwise by a multiple of 90; negative turns the other way.                                                                          |
| `watermark INPUTS --text T \| --image F \| --stamp PDF` | Mark all or some pages; see below.                                                                                                    |
| `encrypt` / `decrypt INPUTS`                            | AES-256. Print, copy and modify permissions can each be withheld.                                                                     |
| `meta INPUTS --title ... --author ... [--clear]`        | Edit the Info dictionary; `--clear` also drops the XMP packet.                                                                        |
| `info INPUTS [--json]`                                  | Pages, sizes, rotation, bookmarks, encryption, metadata.                                                                              |
| `batch FILE [--dry-run] [--fail-fast]`                  | Run the jobs in a TOML file; exit status 1 if any failed.                                                                             |

Every per-file command accepts `-o FILE` (exactly one input) or `--out-dir DIR`; with neither, results go beside the input as `<stem>_extract.pdf`, `_watermarked.pdf`, and so on. **Nothing is overwritten without `--force`, an output can never be one of its own inputs, and two results can never land on the same file**, which matters most in batch runs where a typo would otherwise destroy a source.

### Page ranges

One small grammar everywhere (1-based, comma separated): `3`, `2-5`, `7-` (to the end), `end`, `odd`, `even`, `all`, and a reversed range such as `5-2` for descending order. Parsing returns indices in the order written with duplicates kept, so the same spec means "select" for `extract`, "permutation" for `reorder` and "one file per token" for `split --ranges`. Out-of-range pages and malformed specs are refused before anything is written.

## Watermarks

Text and images are drawn with reportlab into a one-page overlay per distinct page size and merged with pypdf; a PDF stamp is placed by pypdf directly. All three share one placement routine, so they support the same options:

- `--position center | tile | top-left | top-right | bottom-left | bottom-right`, `--angle`, `--opacity`, `--under` (behind the page content instead of over it), `--pages` to mark only some pages.
- Text: `--size`, `--color #rrggbb`, and `--font file.ttf` for characters outside Windows-1252 (a corrupt font is reported, not a traceback). Without a font, text the built-in face cannot draw is refused instead of printing boxes.
- Images (PNG with transparency, JPEG, ...): `--scale` is the width as a fraction of the page width. Images over 50 megapixels are refused (a 24 KB PNG can decode to hundreds of MB).
- `--stamp other.pdf --stamp-page N` places a page of another PDF, for letterheads, "APPROVED" stamps and so on. A stamp keeps its own transparency; there is no `--opacity` for it.
- Tiling lays the grid out in the rotated frame over the page diagonal, so a 45 degree watermark leaves no bare corner, and refuses a mark so small that it would need thousands of tiles.

Pages with a `/Rotate` flag have the rotation baked into their content first, so the mark lands upright on the page as displayed; pages whose media box does not start at the origin and documents with mixed page sizes are handled (each size gets its own overlay).

### A bug the pixel tests caught

The tests render pages with pypdfium2 and count ink, rather than just checking that a file was written. That found **opacity being silently ignored**: reportlab's `setFillColor` resets the fill alpha to the colour's own (opaque) alpha, so `setFillAlpha(0.2)` called *before* the colour was overwritten, and a 0.2 and a 1.0 mark rendered with identical darkness. The alpha is now set after the colour.

## Bookmarks and metadata

`merge` keeps document structure. The default `--outline nested` adds one bookmark per input (named after the file) with that file's own bookmarks nested inside; `files` keeps only the per-file bookmarks, `original` keeps the inputs' bookmarks without a parent, `none` drops them. When you select pages with `a.pdf[2]`, bookmarks pointing at dropped pages are left out. `--title` and `--author` set the merged document's metadata; `meta` edits it on any file.

## Encryption

`encrypt` writes AES-256 (the `cryptography` package does the work). The user password opens the file; an optional owner password lifts the restrictions chosen with `--no-allow-print`, `--no-allow-copy`, `--no-allow-modify`. Passwords are never required on the command line: `--user-password` and `--password` read `PDFTOOL_USER_PASSWORD` and `PDFTOOL_PASSWORD`, and `encrypt`/`decrypt` prompt on a terminal if none is given, so a secret need not sit in shell history. Results of other commands on an encrypted input are written unencrypted; re-encrypt them if you want that. If you withhold a permission and give no owner password, a random one is generated and discarded: with the same password for both, anyone who can open the file could lift the restrictions. (Passwords given on the command line are visible in the process list; use the environment variables or the prompt for anything real.) Permissions are advisory: a PDF reader decides whether to honour them.

## Batch files

A TOML file lists jobs; `[defaults]` applies to every job. **A batch file is a trusted script**: it can name any path, so only run files you wrote or read. It still cannot do two dangerous things: set `force` (only `batch --force` can overwrite) or carry a password in plain text. Relative paths resolve against **the batch file's folder**, not the current directory, so the same file works from anywhere.

```toml
[[job]]
name = "quarterly pack"
op = "merge"
inputs = ["reports/q*.pdf", "appendix.pdf[2-]"]
output = "out/pack.pdf"
title = "Quarterly pack"

[[job]]
op = "watermark"
inputs = ["reports/*.pdf"]
output_dir = "out/marked"
text = "CONFIDENTIAL"
position = "tile"
opacity = 0.2

[[job]]
op = "encrypt"
inputs = ["out/pack.pdf"]
output = "out/pack.locked.pdf"
user_password_env = "PACK_PASSWORD"     # read from the environment; never put secrets in the file
```

Any option of the command is a key (`every`, `ranges`, `degrees`, `text`, ...); `password`, `user_password` and `owner_password` must be given as `password_env = "VAR"` etc., which reads that environment variable; `_env` is refused for every other option, so a job cannot copy an arbitrary variable into a watermark or a title. Unknown keys and unknown ops are job failures, so a typo like `pagez` cannot silently do nothing. Jobs run in order and a failure does not stop the rest (`--fail-fast` does); `--dry-run` plans every job, including overwrite and input-equals-output checks, without writing anything. The CLI commands and batch jobs go through the same `run_op`, so they cannot drift apart.

## Layout

| File           | Role                                                                                                      |
| -------------- | --------------------------------------------------------------------------------------------------------- |
| `ranges.py`    | Page-range grammar.                                                                                       |
| `files.py`     | Input expansion (globs, `[pages]`), decrypt-on-open, atomic writes, output planning and overwrite guards. |
| `ops.py`       | merge, split, extract, reorder, rotate, encrypt, decrypt, metadata, info.                                 |
| `watermark.py` | Text, image and PDF-stamp watermarks.                                                                     |
| `jobs.py`      | `run_op` shared by the CLI and batch files; TOML loading.                                                 |
| `cli.py`       | The Cyclopts command line.                                                                                |

Writes go to a temp file in the destination folder and are moved into place, so a failure never leaves half a PDF, and `split` checks every target before writing the first.

## Performance

Measured on this machine (Windows 11, Python 3.14), a 500-page letter-size document of plain text pages (220 KB):

| Operation                        | Time   |
| -------------------------------- | ------ |
| Text watermark, centred          | 0.55 s |
| Text watermark, tiled (36 pt)    | 3.3 s  |
| Image watermark                  | 0.42 s |
| Merge two 500-page files         | 0.47 s |
| Split into 500 single-page files | 0.93 s |
| Encrypt (AES-256)                | 0.15 s |
| Extract odd pages                | 0.12 s |

Tiling is the only slow path: each page's overlay draws a whole grid of tiles. Real documents with heavy images or fonts will be slower than this synthetic one.

## Tests

```bash
uv run pytest -q
```

137 tests: the range grammar (including failures), file expansion and output guards, every operation (page order, bookmark modes, selection dropping bookmarks, split naming and clashes, AES-256 and permissions, wrong or missing passwords, metadata), pixel-level watermark checks (position, colour, opacity, tiling, `--under`, rotated pages, offset media boxes, mixed sizes, transparent images, PDF stamps), the CLI end to end (exit statuses, prompts, environment passwords, a `python -m` smoke test) batch files (relative paths, failures, `--fail-fast`, `--dry-run`, `_env` secrets) and a hardening suite from a security review of the first version: restricted encryption without a distinct owner key, non-finite sizes, decompression-bomb images, split targets colliding with other inputs, unwritable destinations, and batch files trying to force overwrites or copy environment variables.

## Stack

Python 3.14, `uv`, [pypdf](https://pypdf.readthedocs.io/) 6 for page operations and encryption, [reportlab](https://docs.reportlab.com/) 5 and Pillow for overlays, [Cyclopts](https://cyclopts.readthedocs.io/) for the CLI, `cryptography` for AES-256, and pypdfium2 (dev only) to render pages in tests. Chosen as the current releases of the libraries the challenge suggests (pypdf, reportlab), with a type-hint-driven CLI framework in place of the Typer used by earlier challenges.

## Limits

- Watermark `--opacity` does not apply to `--stamp`; the stamp keeps its own transparency.
- Margins for corner positions use the unrotated mark, so a rotated corner mark can sit slightly closer to the edge.
- Text watermarks use one line and one font; there is no multi-line text.
- Form fields, annotations and other document-level structure beyond bookmarks and metadata have not been tested through any operation.
