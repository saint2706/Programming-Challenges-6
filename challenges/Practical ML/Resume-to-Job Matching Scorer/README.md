# Resume-to-Job Matching Scorer

**Category:** Practical ML
**Difficulty:** B (brief: "TF-IDF or embedding similarity ranking, explain top matches.")

**Status:** Implemented (Python)

Source modules live in `src/resume_matcher/`; the tests are in `tests/`.

Ranks job postings for a resume (and resumes for a posting) with four scorers --
TF-IDF, BM25, bge-small embeddings and a tuned fusion of the last two --
benchmarks them against each other on real data, and explains every match from
the scorer itself. A Typer CLI runs it; a Streamlit page lets you paste a resume
and see the evidence.

## Data

- [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
  (Kaggle, 2,484 resumes, 24 categories; 2,483 usable after dropping stubs).
- [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
  (Kaggle, 123,849 postings; 123,782 with a usable description).

Both are fetched by `resume-matcher fetch` into a gitignored `data/` folder using your
Kaggle token in `~/.kaggle/kaggle.json` (dataset downloads, so no competition
rules to accept).

## The ground truth is a proxy -- read this before the numbers

Postings have no resume-style category, so a posting's category is inferred from
its **title** with one regex per category (`labels.py`). A title that matches no
category, or more than one, is dropped rather than guessed. Relevance is binary:
same category. No scorer ever sees the table.

How good is it? I judged random samples by hand (me, not an independent
annotator) and fixed what the samples exposed:

| round | sample                                             | correct | what it exposed                                                                                                                  |
| ----- | -------------------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 1     | 100 random mapped titles                           | 94/100  | bare "auditor", "finance", "instructional designer" over-reach                                                                   |
| 2     | fresh 100                                          | 95/100  | (after round-1 fixes)                                                                                                            |
| 3     | fresh 100, after scanning for noise in real titles | 98/100  | `hr` matching hourly rates (`$40/hr`: ~10% of HR postings), bare `mechanic`, `farm*` matching "Farmbrook", "retail merchandiser" |

The round-3 sample is committed as `audit/labels_audit.csv` (with my `ok` column)
and a test fails if the label table stops agreeing with it. Round 2's 95% was
real but a random sample of 100 cannot see a bug that hits 10% of one of 24
categories -- the `/hr` bug only showed up when I read the CLI's output. Treat
98% as "no remaining *gross* error found", not as a measured precision.

**The resume side is the bigger problem.** The Kaggle resume "categories" are
search keywords, not job families. I read the headlines of the worst-scoring
ones: ADVOCATE is mostly *patient / customer / family / health* advocates (not
attorneys, which is what the postings are); ARTS is "language-arts teacher",
"culinary-arts instructor"; HEALTHCARE is healthcare *marketing*, *recruiter* and
*administrator*; BPO is a grab-bag of managers and engineers. Scores for those
categories measure the label, not the scorer, so they are reported but should not
be read as model failures. I did not relabel anything to improve the numbers.

**Headline leakage.** Each resume begins with the job title its category came
from (and usually repeats it as the first words of the body). A scorer that sees
it is partly matching the label. Every result below is therefore given twice:
`full` (what a user would paste) and `stripped` (headline and its repeat
removed: the skills-only view).

## How it works

```
fetch -> label postings -> split -> {tfidf, bm25, embeddings} -> tune fusion on val -> report on test
```

| File               | Role                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------ |
| `data.py`          | Fetch, clean, headline stripping, stratified id splits                               |
| `labels.py`        | Title-to-category table, balanced pool (200 per category), audit sample              |
| `scorers.py`       | TF-IDF, BM25 (own implementation), z-score fusion, reciprocal-rank fusion            |
| `embed.py`         | bge-small windows, OpenVINO/torch backends with a correctness self-check, disk cache |
| `index.py`         | LanceDB table of job vectors                                                         |
| `explain.py`       | Exact term contributions, skill gaps, occlusion for the dense scorer                 |
| `evaluate.py`      | nDCG/MRR/P/MAP, bootstrap CIs, fusion-weight tuning                                  |
| `pipeline.py`      | Splits, the tune-on-val / report-on-test protocol, `report.json`                     |
| `ranker.py`        | The service behind both the CLI and the app                                          |
| `cli.py`, `app.py` | Typer CLI, Streamlit page                                                            |

**Protocol.** Resumes and the balanced job pool are each split by id into
disjoint validation and test halves, stratified by category (test: 1,239 resumes
x 2,183 postings). Sparse statistics are fit on that split's own texts only, with
no labels. The fusion weight is the only thing tuned, on validation, then frozen.
Everything reported is on test, with 95% bootstrap intervals over queries.

**The scorers.**

- **TF-IDF**: sublinear tf, unigrams + bigrams, stop words removed, cosine.
- **BM25**: Lucene-style, written out in `scorers.py` and checked against the
  independent `bm25s` package to 1e-6 in a test. Query terms count once each
  (presence), so a long resume does not multiply a term's weight; I did not test
  the query-term-frequency variant.
- **Embedding**: `BAAI/bge-small-en-v1.5`, 256-token windows with overlap,
  mean-pooled over at most 8 windows per text (about 1.2k tokens -- the tail of
  the longest resumes is ignored by this scorer only).
- **Fusion**: `w * z(embedding) + (1 - w) * z(bm25)`, each z-scored per query
  (raw cosine and BM25 are not on one scale). Tuned `w`: **0.8** on full text,
  **0.7** on stripped. Reciprocal-rank fusion is reported beside it as the
  parameter-free reference.

## Results

Held-out test split, 95% bootstrap interval in brackets. Random is 0.043, which
is 1/24, as it should be.

**Resume to jobs, full resume text**

| scorer    | nDCG@10 [95% CI]     | MRR   | P@10  | MAP@50 |
| --------- | -------------------- | ----- | ----- | ------ |
| random    | 0.043 [0.039, 0.047] | 0.143 | 0.042 | 0.005  |
| tfidf     | 0.430 [0.407, 0.452] | 0.553 | 0.422 | 0.256  |
| bm25      | 0.367 [0.346, 0.387] | 0.505 | 0.357 | 0.166  |
| embedding | 0.413 [0.391, 0.434] | 0.542 | 0.405 | 0.226  |
| fusion    | 0.416 [0.395, 0.438] | 0.533 | 0.410 | 0.230  |
| rrf       | 0.407 [0.385, 0.428] | 0.534 | 0.398 | 0.217  |

**Resume to jobs, headline stripped (skills only)**

| scorer    | nDCG@10 [95% CI]     | MRR   | P@10  | MAP@50 |
| --------- | -------------------- | ----- | ----- | ------ |
| random    | 0.043 [0.039, 0.047] | 0.143 | 0.042 | 0.005  |
| tfidf     | 0.206 [0.188, 0.223] | 0.298 | 0.206 | 0.109  |
| bm25      | 0.193 [0.176, 0.210] | 0.297 | 0.190 | 0.086  |
| embedding | 0.190 [0.174, 0.206] | 0.295 | 0.187 | 0.090  |
| fusion    | 0.209 [0.191, 0.226] | 0.314 | 0.205 | 0.103  |
| rrf       | 0.203 [0.186, 0.221] | 0.314 | 0.199 | 0.100  |

**Jobs to resumes (the reverse direction)**

| scorer    | full nDCG@10 [95% CI] | stripped nDCG@10 [95% CI] |
| --------- | --------------------- | ------------------------- |
| random    | 0.041 [0.038, 0.044]  | 0.041 [0.038, 0.044]      |
| tfidf     | 0.411 [0.398, 0.424]  | 0.344 [0.333, 0.355]      |
| bm25      | 0.343 [0.331, 0.355]  | 0.294 [0.283, 0.305]      |
| embedding | 0.370 [0.357, 0.382]  | 0.302 [0.292, 0.313]      |
| fusion    | 0.387 [0.375, 0.399]  | 0.323 [0.312, 0.334]      |
| rrf       | 0.384 [0.371, 0.395]  | 0.317 [0.306, 0.328]      |

**What the numbers support:**

- On full text every real scorer is about ten times better than chance; on headline-stripped text, about 4.5 times.
- On full text TF-IDF is nominally best, but its interval overlaps embedding's and
  fusion's, and I did not run a paired test, so I am not claiming it beats them.
  BM25 is clearly below TF-IDF (intervals do not overlap).
- The headline carries a lot of the signal: removing it roughly halves resume to
  jobs nDCG for every scorer (TF-IDF 0.430 to 0.206). The drop is much smaller in
  the reverse direction (0.411 to 0.344). I did not investigate why.
- Dense and lexical scorers fail in different places, which is why fusion is a
  hedge rather than a winner. Stripped, embeddings are far better on DESIGNER
  (0.396 vs TF-IDF 0.191) and better on CHEF, AVIATION and AUTOMOBILE; TF-IDF is
  far better on DIGITAL-MEDIA (0.333 vs 0.115) and INFORMATION-TECHNOLOGY
  (0.302 vs 0.183). Fusion is within noise of the best single scorer overall.
- The best categories (full text, TF-IDF): ACCOUNTANT 0.874, CHEF 0.827,
  HR 0.820, TEACHER 0.813. The worst are the keyword-bucket categories above
  (ARTS 0.036, AUTOMOBILE 0.064, BPO 0.066, ADVOCATE 0.099). The full 24-way
  breakdown for every scorer is in `results/report.json`.

Absolute values (about 0.4 nDCG@10) are low because relevance is a coarse proxy
across 24 categories; compare scorers with each other, not with other benchmarks.

## Explanations

| Scorer              | What you get                                                                                                                                                           |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| tfidf, bm25         | **Exact**: score = sum over terms of query weight x document weight. The CLI/app list the top 8 terms *and* what the rest add, so listed + remainder = score (tested). |
| embedding, fusion   | The terms shown are only words both texts share, labelled **lexical overlap**. `--explain-dense` adds a **post-hoc** probe.                                            |
| all, resume to jobs | **Gaps**: heavily weighted job terms the resume lacks, restricted to terms common in that category's postings.                                                         |

The post-hoc probe re-scores the pair with each resume sentence removed and lists
the biggest drops; it probes the model, it does not decompose the score. A test
checks that removing the top explained terms hurts the TF-IDF score more than
removing random shared terms. The gap list is a heuristic and its output is
imperfect (it shows words like "modern" and "degree computer").

```
$ uv run resume-matcher rank-jobs examples/sample_resume.txt --top 3 --scorer tfidf
 1.   0.142  Sr Data engineer with AWS, PYTHON (W2) [INFORMATION-TECHNOLOGY]
      evidence: aws 0.031, big data 0.018, python 0.017, certifications aws 0.017, ...   (top 8 of 19 terms; the other 11 add 0.021, so all of them sum to the score)
      missing:  frameworks, degree computer, pipelines, engineering, java, modern
 2.   0.138  Sr Data scientist [INFORMATION-TECHNOLOGY]
      evidence: tensorflow pytorch 0.019, pytorch 0.018, aws 0.015, tensorflow 0.015, ...

$ uv run resume-matcher rank-jobs examples/sample_resume.txt --top 1 --scorer embedding --explain-dense
post-hoc: resume sentences the embedding match to the top job depends on
  +0.0309  Hands-on with Python, SQL, R, Tableau and Power BI for dashboards, reporting ...
  +0.0201  Python, SQL, R, Pandas, NumPy, Tableau, Power BI, scikit-learn, TensorFlow, ...
  +0.0011  Education
```

`examples/sample_resume.txt` is a short resume written from the author's public
portfolio fields, without contact details.

## Hardware: why the GPU and not the NPU

Measured on this machine (Intel Core Ultra 7 255H, no NVIDIA GPU) with
bge-small, 128 texts, fixed `[8, 256]` shapes, through OpenVINO 2026.4:

| device                      | texts/s | cosine vs torch |
| --------------------------- | ------- | --------------- |
| torch CPU                   | 17      | reference       |
| OpenVINO CPU                | 26      | 1.0000          |
| **OpenVINO GPU (Arc iGPU)** | **321** | **1.0000**      |
| OpenVINO NPU (AI Boost)     | 61      | **0.82**        |

The NPU is fast but **wrong**: its fp16 numerics shift the embeddings (mean cosine
0.82 for bge-small, 0.90 for e5-small-v2, 0.45 for MiniLM-L6). Patching the
attention-mask constant that overflows fp16 moved it from 0.59 to 0.82 and no
further; I did not try int8 quantization. So `embed.pick_backend` runs a probe
set through each candidate device and **refuses any whose vectors disagree with
torch** (mean cosine < 0.99): it picks the iGPU here and would fall back to
OpenVINO CPU, then torch CPU. The benchmark's backend and its self-check result
are recorded in `results/report.json`. A cold end-to-end run (model load, ~6.8k
texts embedded, all scoring and bootstrap intervals) took 239 s.

Gotchas that cost time: the NPU compiler aborts the whole process on dynamic
shapes (so every input is fixed), OpenVINO renames `attention_mask` to `'11'`,
and the model's input order differs from the tokenizer's -- feeding by tokenizer
order silently gives cosine ~0.5 even on CPU.

## Usage

```bash
uv sync                                                # dependencies from pyproject.toml / uv.lock
uv run resume-matcher fetch                             # Kaggle token in ~/.kaggle/kaggle.json
uv run resume-matcher evaluate                          # full benchmark -> results/report.json
uv run resume-matcher report                            # print the saved benchmark
uv run resume-matcher rank-jobs examples/sample_resume.txt --top 5 --scorer fusion --explain-dense
uv run resume-matcher rank-resumes some_job.txt --top 5 --scorer bm25
uv run streamlit run src/resume_matcher/app.py
uv run pytest -q                                       # 153 tests, no network except two model smoke tests
```

## Tests

153 tests. Everything except two model smoke tests runs with no network or
accelerator, using small fixtures and a hashing encoder that stands in for the
model. They pin: the label table (including regressions found on real data and
sync with the committed audit), the splits being disjoint, metrics against
hand-computed values, TF-IDF/BM25 contributions summing exactly to the score,
BM25 against `bm25s`, chunking and padding, the backend self-check rejecting a
corrupted device and surviving one that crashes, the cache, fusion, LanceDB
round trips, the evaluation protocol, the CLI (Typer runner) and the Streamlit
page (`AppTest`).

## Limitations

- A query that shares no vocabulary with any posting scores 0 everywhere under tfidf/bm25, and the
  CLI then lists the first postings in pool order with "evidence: none" -- they are not matches.
- The post-hoc probe looks at a resume's first 40 sentence units only.
- Relevance is a category proxy, and 4+ of 24 resume categories are keyword
  buckets; absolute numbers are not comparable to real hiring outcomes.
- The 98% label audit is my own judgment on 100 titles, not an independent
  annotator.
- One seed, one split. Intervals are over queries, not over resamplings of the
  split.
- The fusion weight is tuned in one direction (resume to jobs) and reused for the
  reverse.
- English-only model; the postings are LinkedIn US-centric.
- No cross-encoder re-ranker (that is the Hybrid Retrieval challenge's job).
