# Voice Memo Transcriber & Organizer

**Category:** Practical Software
**Difficulty:** A (brief: "Local speech-to-text (e.g. Whisper), auto-tag by topic.")

**Status:** Implemented (Python)

A local command-line tool and loopback web UI that transcribes voice memos on your own machine,
tags each memo by topic without a hosted model, and makes the transcripts searchable. Audio never
leaves the machine. Models are downloaded only when you ask for them.

## Quick start

```bash
cd "challenges/Practical Software/Voice Memo Transcriber & Organizer"

uv run --locked voicememo models pull --parakeet --whisper base --embedder   # one-time downloads
uv run --locked voicememo ingest ~/VoiceMemos                                # transcribe a folder
uv run --locked voicememo search "grocery list"
uv run --locked voicememo serve                                              # http://127.0.0.1:8000
```

## Features

| Feature        | Command or screen                   | Notes                                                                  |
| -------------- | ----------------------------------- | ---------------------------------------------------------------------- |
| Transcribe     | `ingest`, `retranscribe`            | Parakeet TDT 0.6B v3 (int8) with word timestamps; whisper.cpp fallback |
| Watch a folder | `watch`                             | Ingests a memo only after its size and mtime stop changing             |
| Topic tags     | `retag`, `tag`, automatic on ingest | model2vec embeddings, HDBSCAN clusters, c-TF-IDF labels                |
| Search         | `search`, web search box            | SQLite FTS5 full text, quoted terms, highlighted snippets              |
| Export         | `export`, `export-library`          | Markdown, SRT, VTT, JSON                                               |
| Web UI         | `serve`                             | Litestar + HTMX, loopback only, CSRF-protected writes                  |
| Tag evaluation | `eval-tags`                         | Adjusted Rand index and purity on a labelled JSON file                 |

## Method

1. **Decode.** PyAV reads any container FFmpeg can and resamples to 16 kHz mono float32.
2. **Transcribe.** The router sends English and other supported languages to onnx-asr with
   Parakeet TDT 0.6B v3 (Silero VAD for long-form audio, token-level timestamps). When Parakeet is
   not pulled it falls back to whisper.cpp base, which returns segment times but no word times.
3. **Store.** The library is a SQLite database. Each memo is keyed by a SHA-256 of its audio, so
   re-ingesting the same file is a no-op. Transcript segments and tags live in the same database,
   and an FTS5 index covers the transcripts.
4. **Tag.** model2vec embeds each transcript, HDBSCAN groups them, and each cluster is named by
   its top c-TF-IDF terms. Memos that fall into no cluster are left untagged rather than forced
   into one. If model2vec is missing, an LSA (TF-IDF plus truncated SVD) fallback is used.
5. **Export and serve.** Exports are plain files. The web UI streams audio by memo id with HTTP
   Range support.

## Results

All numbers below come from the commands in this folder on this machine (Windows 11, CPU only,
Intel Core Ultra 7 255H). They are a small check, not a benchmark suite.

**Speech recognition, two clips.** Each clip is a short Windows SAPI recording of a known sentence
(`tests/fixtures/speech/`). WER is word error rate against that sentence. RTF is wall time divided
by audio length, so below 1.0 is faster than real time.

| Engine           | Clip     | WER   | RTF   | Hypothesis                     |
| ---------------- | -------- | ----- | ----- | ------------------------------ |
| Parakeet v3 int8 | reminder | 0.000 | 1.511 | Buy milk and eggs on Thursday. |
| Parakeet v3 int8 | river    | 0.000 | 0.735 | The river water was very cold. |
| whisper.cpp base | reminder | 0.000 | 0.258 | Buy milk and eggs on Thursday. |
| whisper.cpp base | river    | 0.000 | 0.240 | The river water was very cold. |

Parakeet's first clip includes one-time model loading in the timing, so its RTF is high. Two
sentences cannot show how these engines do on real, noisy, accented speech; the WER of 0.000 only
shows that both engines get clean synthetic speech right.

**Topic tagging, labelled set.** `tests/fixtures/topics.json` holds 48 synthetic transcripts in
six topics of eight each. I wrote these and chose the topic labels myself, so this measures
agreement with my labels, not with real memos.

| Method                       | Clusters | Covered | Noise share | ARI   | Purity |
| ---------------------------- | -------- | ------- | ----------- | ----- | ------ |
| model2vec (default settings) | 3        | 31      | 0.354       | 0.319 | 0.548  |
| LSA (default settings)       | 0        | 0       | --          | 0.000 | --     |

The defaults were kept as they were. A sweep over LSA settings reached an ARI of about 0.18, but
tuning settings against the labels this set was built with would overfit it, so that sweep is not
reported as a result. The model2vec figures are the ones to trust most: 3 clusters for 6 true topics
means it merges topics and leaves a third of the memos untagged.

## Security review

The web UI is built for a single user on loopback. Each measure has a regression test. A second
review (findings verified and fixed) added the items marked *review fix*.

- **DNS rebinding.** A Host header must name `127.0.0.1`, `localhost` or `::1`; everything else
  gets a 400.
- **Cross-site writes.** A POST with an Origin header that does not match the Host is refused with
  a 403. Writes also need a double-submit CSRF token: an HttpOnly, SameSite=strict cookie that must
  equal the form field, compared with `hmac.compare_digest` on bytes, so a non-ASCII form field gets
  a 403 instead of a 500 (*review fix*). A cookie that is not a 43-character token is replaced
  (*review fix*).
- **Stored XSS.** Titles, transcripts, tags and file names are escaped by Jinja autoescape.
  Search snippets are escaped before the highlight markers are turned into `<mark>` tags.
  Control characters, including the C1 range U+0080 to U+009F, are stripped before display
  (*review fix*).
- **Path traversal.** Audio is served by memo id. The file path comes from a hash-derived name
  stored in the database, never from the request. Stored names containing `/`, `\`, `..` or `:`
  (Windows drive-relative names such as `C:x.m4a`) are refused (*review fix*). Non-numeric ids do
  not route. A tag filter that cannot be a valid tag returns an empty list, not an error
  (*review fix*).
- **Headers.** CSP with `default-src 'self'` and no `unsafe-inline` scripts, `frame-ancestors
  'none'`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`.
- **Network.** `serve` refuses `0.0.0.0` and every other non-loopback address. The former
  `--allow-remote` override is gone (*review fix*).
- **Downloads.** Model downloads happen only under `models pull`, and every download is pinned to a
  Hugging Face commit. Weight files are checked against a pinned sha256 before the model is marked
  ready; a mismatch deletes the file and fails the pull (*review fix*). The `.ready` marker is a JSON
  manifest of file sizes, so a truncated or replaced file makes the model missing again (*review
  fix*). The runtime never fetches a model. A folder without a valid manifest is treated as missing.
  Parakeet also needs its Silero VAD model. Whisper names are an allow-list (`tiny`, `base`,
  `small`), and the whisper.cpp model is loaded by absolute path, so no other folder is searched
  (*review fix*).
- **Untrusted audio.** FFmpeg may open only local files (`protocol_whitelist=file`), so a memo
  cannot make decoding fetch a URL or read a concat list (*review fix*). Files over 2 GiB are
  refused before they are hashed or copied, and decoded audio longer than 12 hours is refused
  while it streams (*review fix*).
- **Watch folder.** Symlinks are skipped, so a link cannot pull in a file from outside the folder.
  A file that cannot be stat'ed, locked or refused by the library is logged and skipped; the rest of
  the scan continues (*review fix*).
- **CLI errors.** Unreadable input files and a model integrity failure are reported as errors with
  exit code 1, not as tracebacks (*review fix*). `--whisper` accepts only the allow-listed names.

## Where this is actually used

Local speech-to-text is a well-trodden path, and two of the projects behind this tool are in
wide use.

**whisper.cpp** describes itself as a "Port of OpenAI's Whisper model in C/C++", a
"High-performance inference" engine that runs fully on-device. Its README lists bindings for
Rust, JavaScript, Go, Java, Ruby, Swift, .NET, Python and R, and it names downstream apps, such as
a React Native binding and a Neovim plugin, that build on it. This tool uses it as the fallback
engine, the one that runs when Parakeet is not installed.

**onnx-asr** is a "lightweight, fast, and easy-to-use pure Python package" that runs ONNX speech
models without PyTorch, Transformers or FFmpeg. Its README lists NeMo Parakeet TDT v3 among the
supported models and says it can return token-level timestamps. This tool uses it for the main
engine, which is why it can show word-level timing in the transcript.

The gap between these projects and this one is the workflow. Both are engines. A memo app also
needs a dedupe step, folder watching that waits for a file to finish copying, tags for finding
memos again, and an export path. Those are the parts this challenge adds.

## Layout

```text
src/voicememo/
  asr.py          Segment type, FakeTranscriber for tests, language normalisation
  audio.py        PyAV decode to 16 kHz mono float32, chunking
  backends.py     Parakeet (onnx-asr) and whisper.cpp wrappers, RoutingEngine
  cli.py          argparse commands
  export.py       Markdown, SRT, VTT and JSON writers
  ingest.py       transcribe_file, ingest_file, retranscribe
  library.py      SQLite schema, FTS5, tags, tag normalisation
  models.py       pinned model pulls, sha256 checks, .ready size manifests, status
  paths.py        data and models folders (env overrides)
  tagging.py      model2vec and LSA embedders, HDBSCAN, c-TF-IDF, evaluate
  watch.py        settle-window folder watcher with an injected clock
  web.py          Litestar app: guards, CSRF, Range audio, HTMX fragments
  templates/      Jinja2 pages (autoescaped)
  static/         htmx and CSS, served from /static
tests/            pytest suite; fixtures/ holds topics.json and speech/*.wav
```

Data and models live beside `pyproject.toml` in `data/` and `models/`. Both are git-ignored.

## Tests

```bash
uv run --locked pytest -q                                     # 316 pass, 5 skip
VOICEMEMO_REAL_MODELS=1 uv run --locked pytest -q -s          # 320 pass, 1 skip (needs models)
```

The suite covers ingest (success, dedupe, corrupt audio, retries, language normalisation, chunk
offsets, retranscribe), tagging (determinism, the labelled set, LSA and model2vec paths), export
in all four formats, the watch folder (settle window, growing files, temp and hidden names, restart
without re-transcribing), the web app (Host and Origin guards, CSRF, XSS escaping, Range requests,
traversal probes, security headers) and the CLI end to end.

The `real_models` marker is opt-in because it needs pulled models. It also checks the WER and RTF
numbers above. One test is skipped by default and runs on any machine: the `.ogg` round trip, which
needs a PyAV build with the Vorbis encoder.

## Stack

Python 3.12+, uv (uv_build backend), PyAV, onnx-asr (Parakeet TDT v3, int8), pywhispercpp
(whisper.cpp), model2vec, scikit-learn (HDBSCAN, TruncatedSVD), SQLite FTS5, Litestar, Jinja2,
HTMX, uvicorn, pytest with pytest-timeout.

## Limits

- Two clips are not a real accuracy test. The WER column shows the pipeline works on clean speech,
  not how it does on real recordings with noise, accents or cross-talk.
- The labelled topic set is synthetic and was labelled by me. The ARI and purity numbers say how
  well the method matches those labels, nothing more.
- LSA with default settings finds no clusters on this set. Model2vec finds three of six topics.
  Tagging is a starting point for browsing, not a classifier to trust unattended.
- whisper.cpp base gives segment timestamps only, so word-level highlighting is available only
  with Parakeet.
- The web UI is for one user on one machine. It has no login and must not be exposed beyond
  loopback.
- Model downloads are large and need network access once. Nothing else needs the network.
