# Resume-to-Job Matching Scorer — design

Practical ML #6 (B). Brief: "TF-IDF or embedding similarity ranking, explain top matches."

## Understanding (confirmed with the user)

Rank job postings for a resume (and resumes for a posting), benchmark sparse
vs dense scorers honestly, and explain each match from the scorer itself.
Success = a ranked table of scorers on real data with stated proxy-label
caveats, explanations that provably account for the score, a CLI and a
Streamlit page. Decisions already made: Kaggle data, TF-IDF vs BM25 vs
embeddings benchmark, OpenVINO iGPU (CPU fallback) for embeddings, Typer CLI +
Streamlit, same layout as Practical ML #5, solo build.

## Data

- `snehaanbhawal/resume-dataset` `Resume/Resume.csv`: 2,484 resumes, 24
  categories (`Resume_str` plain text). Mean ~6.3k chars.
- `arshkon/linkedin-job-postings` `postings.csv`: 123,849 postings
  (`title`, `description`, mean ~3.8k chars).
- Both live in `data/` (gitignored), fetched by `data.py fetch` with the Kaggle
  API (single-file dataset downloads; no competition rules needed).

## Ground truth is a proxy — and the design says so

Postings carry no resume-style category. `labels.py` maps a posting **title** to
one of the 24 resume categories with an ordered table of anchored regexes
(`nurse|physician|...` -> HEALTHCARE, etc.); unmapped and ambiguous titles are
dropped, not guessed. Relevance = same category (binary).

- The mapping is an *evaluation* device, not a model input: no scorer ever sees
  a label or the regex table.
- A fixed-seed random sample of 100 mapped postings is hand-audited; the
  README reports the agreement and the audited sample is committed as a CSV so
  anyone can re-judge it.
- The pool is balanced: at most N postings per category (default 200) so a
  frequent category (sales) cannot dominate; every one of the 24 categories
  has enough postings or is reported as skipped.
- **Headline leakage.** Resume category was derived from the job title that
  starts each resume. Every scorer is evaluated twice: `full` text and
  `headline-stripped` (first line removed), and both are reported, because
  stripped is the honest "skills-only" number and full is what a real user
  would paste.

## Splits

Resumes split by resume (val/test, stratified by category); postings split by
posting into disjoint val/test pools. Corpus statistics (IDF, BM25 stats,
vocabulary) are fit on that split's *own side only* (train-side resumes +
the val postings for tuning; frozen before test). Fusion weight tuned on val,
reported on test.

## Scorers (all behind one `Scorer` protocol: `fit(corpus)`, `score(query, docs)`)

1. `TfidfScorer`: sklearn `TfidfVectorizer` (sublinear tf, 1-2 grams,
   `min_df`), cosine.
2. `Bm25Scorer`: `bm25s`.
3. `EmbeddingScorer`: `BAAI/bge-small-en-v1.5` (English docs; CLS pooling,
   normalised), documents chunked to fit a 256-token window with
   mean-of-chunks (long resumes are ~1.5k tokens). Backend selection:
   OpenVINO `GPU` with static [B,256] shapes -> OpenVINO `CPU` -> torch CPU.
   A load-time cosine self-check against torch on a fixed probe set refuses
   a backend whose mean cosine is < 0.99 (this is exactly why the NPU is not
   a candidate: measured mean cos 0.82 on bge-small). Embeddings cached on
   disk keyed by (model, text hash).
4. `FusionScorer`: `w * z(embedding) + (1 - w) * z(bm25)` on per-query
   z-normalised scores (fusion of raw cosine and BM25 is meaningless), `w`
   tuned on val. Reciprocal-rank-fusion reported alongside as a
   parameter-free reference.

Candidate retrieval for the app: LanceDB table of job vectors (as in #5) for
top-k dense retrieval, sparse scorers re-rank/fuse that candidate set.

## Metrics

Per scorer, per direction (resume->jobs primary), `full` and `stripped`:
nDCG@10, MRR, P@10, MAP@50, plus per-category nDCG@10 so the hard categories
(CONSULTANT, ARTS, BPO, BUSINESS-DEVELOPMENT) are visible. Bootstrap 95% CI
over resumes. A random-ranking baseline anchors the table.

## Explanations (come from the scorer, not bolted on)

- **TF-IDF:** cosine = sum over shared terms of `w_resume(t) * w_job(t)`, so
  per-term contributions are an *exact* decomposition. `explain()` returns the
  top contributing terms and a test asserts they sum to the score.
- **BM25:** per-term BM25 contribution (exact as well).
- **Embedding:** no exact decomposition exists; uses occlusion (re-score with a
  sentence/phrase removed, rank by score drop) and is labelled post-hoc in the
  UI and README.
- **Gaps:** job terms with high weight in that posting that the resume lacks,
  restricted to terms common in the matched category's postings (data-driven,
  no hand-written skill list).
- Faithfulness check: removing the top-k explained terms from the resume must
  drop the TF-IDF score more than removing k random shared terms (tested).

## Components

| File | Role |
|---|---|
| `data.py` | fetch, load, clean (HTML/whitespace), headline strip, splits |
| `labels.py` | title->category table, balanced pool, audit sample writer |
| `scorers.py` | `Scorer` protocol, TF-IDF, BM25, fusion |
| `embed.py` | backend choice, static-shape OpenVINO export, chunking, cache, self-check |
| `index.py` | LanceDB job-vector table + top-k retrieval |
| `explain.py` | exact decompositions, occlusion, gaps |
| `evaluate.py` | metrics, bootstrap CI, tuning on val, frozen test report |
| `pipeline.py` | stage glue + caching |
| `cli.py`, `app.py` | Typer CLI (`fetch`, `evaluate`, `rank-jobs`, `rank-resumes`), Streamlit page |

## Testing

Network-free tests on tiny fixtures: label table (positive/negative/ambiguous
titles), splits are disjoint by id, metric functions against hand-computed
values, TF-IDF/BM25 decomposition sums to the score, faithfulness test,
embedding chunking, backend fallback order and the self-check rejecting a
corrupted backend (monkeypatched), LanceDB retrieval round trip, CLI via
Typer's runner, Streamlit via `AppTest`. One end-to-end test on the real
`Resume.csv` slice that skips if `data/` is absent; the embedding smoke test
skips if the model can't be fetched.

## Non-goals

No fine-tuning, no cross-encoder reranker (that is #29's territory), no
parsing of PDF resumes, no claim of real hiring performance — the README
states relevance is a category proxy.
