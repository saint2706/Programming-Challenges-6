# Resume-to-Job Matching Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rank job postings for a resume (and the reverse) with TF-IDF, BM25, embedding and fused scorers, benchmark them on real data, and explain every match.

**Architecture:** One Python package-less folder (flat modules, like Practical ML #5). Pure-logic modules (`labels`, `scorers`, `explain`, `evaluate`) are separate from IO (`data`, `embed`, `index`) so they test without network. A `Scorer` protocol lets evaluation treat sparse, dense and fused scorers uniformly.

**Tech Stack:** Python >=3.12, uv, polars, scikit-learn, bm25s, sentence-transformers/transformers/torch, openvino, lancedb, typer, streamlit, kaggle, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-resume-job-matching-scorer-design.md`

## Global Constraints

- Folder: `challenges/Practical ML/Resume-to-Job Matching Scorer/`, own `pyproject.toml` + `uv.lock`, `requires-python = ">=3.12"`, `>=` lower bounds, pytest in `[dependency-groups] dev`.
- Embedding model `BAAI/bge-small-en-v1.5`, window 256 tokens, static `[B,256]` OpenVINO shapes; backend order OpenVINO GPU -> OpenVINO CPU -> torch CPU; self-check rejects mean cosine < 0.99.
- Resume classes: 24 Kaggle categories; relevance = same category; pool capped at 200 postings per category.
- Splits by resume id / posting id, never by row; stats fit on one side only.
- Data in gitignored `data/`; Kaggle token only from `~/.kaggle/`; no network in the default test run.
- Comments/README style matches Practical ML #5; ruff (`ruff.toml` at repo root, not editable) and dprint must pass.

## Review Focus

- Empty / whitespace-only / 21-char resume (the dataset's min length) -> scorers return finite scores, never NaN or crash.
- Query sharing no vocabulary with any posting -> all-zero scores rank stably (no random order, no exception).
- Duplicate resume text across splits -> flagged by `check_split_disjoint`, not silently leaked.
- Postings with null/very short description (2 chars exist) -> dropped by `load_postings`, counted in the report.
- Title that matches two categories (e.g. "Sales Engineer") -> dropped as ambiguous, not assigned to the first.
- Corrupted accelerator output -> embedding backend refuses it and falls back instead of serving silently wrong vectors.

---

### Task 1: Scaffold and data loading

**Files:** Create `pyproject.toml`, `pytest.ini`, `data.py`, `test_data.py`; modify root `.gitignore` (add `challenges/Practical ML/Resume-to-Job Matching Scorer/data/`).

**Interfaces:**
- Produces: `clean_text(s: str) -> str`; `strip_headline(s: str) -> str`; `load_resumes(path) -> pl.DataFrame[id:str, category:str, text:str]`; `load_postings(path) -> pl.DataFrame[job_id:str, title:str, text:str]`; `split_ids(ids: list[str], strata: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]`; `check_split_disjoint(a: list[str], b: list[str]) -> None` (raises `ValueError`); `fetch(data_dir) -> None`.

- [ ] **Step 1: Failing tests** (`test_data.py`)

```python
def test_clean_text_collapses_whitespace_and_html_entities():
    assert clean_text("  A&amp;B \n\n C\t") == "A&B C"

def test_strip_headline_drops_first_line_only():
    assert strip_headline("HR ADMINISTRATOR\n\nSummary\nBody") == "Summary\nBody"

def test_strip_headline_single_line_returns_empty_not_crash():
    assert strip_headline("ONLY TITLE") == ""

def test_split_is_stratified_and_disjoint():
    ids = [f"r{i}" for i in range(100)]; strata = ["a"] * 50 + ["b"] * 50
    val, test = split_ids(ids, strata, 0.5, seed=0)
    assert set(val).isdisjoint(test) and len(val) == 50
    assert sum(i < "r50" or int(i[1:]) < 50 for i in val) == 25

def test_check_split_disjoint_raises_on_overlap():
    with pytest.raises(ValueError): check_split_disjoint(["a", "b"], ["b"])

def test_load_postings_drops_null_and_tiny_descriptions(tmp_path):
    # csv with job_id,title,description rows incl. null and 2-char description
    ...assert out.height == 1
```

- [ ] **Step 2:** `uv run pytest test_data.py -q` -> FAIL (module missing).
- [ ] **Step 3:** Implement `data.py` (polars `read_csv`, `html.unescape`, regex whitespace collapse, per-stratum seeded shuffle split, Kaggle `datasets_download_file` per file with zip sniffing as in Practical ML #5 `data.py`).
- [ ] **Step 4:** run -> PASS.
- [ ] **Step 5:** `git add` folder + `.gitignore`; commit `Scaffold Resume-to-Job Matching Scorer and data loading`.

### Task 2: Title -> category labels and audit sample

**Files:** Create `labels.py`, `test_labels.py`.

**Interfaces:**
- Consumes: `load_postings` output.
- Produces: `CATEGORY_PATTERNS: list[tuple[str, re.Pattern]]` (24 categories); `title_category(title: str) -> str | None`; `label_postings(df) -> pl.DataFrame` adding `category`, dropping unmapped/ambiguous; `balanced_pool(df, per_category: int, seed: int) -> pl.DataFrame`; `audit_sample(df, n: int, seed: int) -> pl.DataFrame`.

- [ ] **Step 1: Failing tests**

```python
@pytest.mark.parametrize("title,cat", [
    ("Registered Nurse - RN - LTAC", "HEALTHCARE"), ("Staff Accountant", "ACCOUNTANT"),
    ("Software Engineer", "INFORMATION-TECHNOLOGY"), ("Executive Chef", "CHEF"),
    ("HR Generalist", "HR"), ("Middle School Teacher", "TEACHER")])
def test_title_maps_to_expected_category(title, cat):
    assert title_category(title) == cat

def test_ambiguous_title_is_dropped():
    assert title_category("Sales Engineer") is None   # SALES and ENGINEERING both match

def test_unmapped_title_is_none():
    assert title_category("Xylophone Wrangler") is None

def test_balanced_pool_caps_each_category():
    ...assert all(c <= 3 for c in pool["category"].value_counts()["count"])

def test_every_resume_category_has_a_pattern():
    assert {c for c, _ in CATEGORY_PATTERNS} == RESUME_CATEGORIES
```

- [ ] **Step 2:** run -> FAIL. **Step 3:** implement (ordered regex table, word-boundary anchored; title matching >1 category returns None). **Step 4:** PASS.
- [ ] **Step 5:** run `label_postings` on the real file, hand-read the 100-row audit sample, write `audit/labels_audit.csv` with an `ok` column, fix patterns that fail, record final agreement. Commit `Add posting title-to-category labels with audited sample`.

### Task 3: Sparse scorers with exact explanations

**Files:** Create `scorers.py`, `explain.py`, `test_scorers.py`, `test_explain.py`.

**Interfaces:**
- Produces: `class Scorer(Protocol): name: str; def fit(self, corpus: Sequence[str]) -> Self; def score(self, query: str, docs: Sequence[str]) -> np.ndarray`; `TfidfScorer(ngram_range=(1,2), min_df=2)`; `Bm25Scorer(k1=1.5, b=0.75)`; `explain.tfidf_terms(scorer, query, doc, k=10) -> list[tuple[str, float]]` with `sum(all contributions) == score`; `explain.bm25_terms(...)`; `explain.gaps(scorer, query, doc, category_df_terms, k) -> list[str]`.

- [ ] **Step 1: Failing tests**

```python
CORPUS = ["python sql tableau analyst", "nurse patient care hospital", "chef kitchen menu", "python machine learning"]

def test_tfidf_ranks_topical_doc_first():
    s = TfidfScorer(min_df=1).fit(CORPUS)
    assert s.score("python analyst", CORPUS).argmax() == 0

def test_tfidf_contributions_sum_to_score():
    s = TfidfScorer(min_df=1).fit(CORPUS)
    terms = explain.tfidf_terms(s, "python sql analyst", CORPUS[0], k=100)
    assert sum(w for _, w in terms) == pytest.approx(s.score("python sql analyst", [CORPUS[0]])[0])

def test_bm25_contributions_sum_to_score(): ...same shape...

@pytest.mark.parametrize("q", ["", "   ", "zzzz qqqq"])
def test_degenerate_queries_give_finite_stable_scores(q):
    for s in (TfidfScorer(min_df=1).fit(CORPUS), Bm25Scorer().fit(CORPUS)):
        out = s.score(q, CORPUS); assert np.isfinite(out).all() and (out == out[0]).all()

def test_removing_top_explained_terms_hurts_more_than_random_terms():
    ...faithfulness: rescored text with top-3 terms deleted < rescored with 3 random shared terms deleted
```

- [ ] **Step 2:** FAIL. **Step 3:** implement (sklearn `TfidfVectorizer(sublinear_tf=True)` — contributions from the sparse row product; `bm25s` index per `score` call via stored tokenizer + corpus stats, contributions computed from the same formula and asserted against bm25s in a test). **Step 4:** PASS. **Step 5:** commit `Add TF-IDF and BM25 scorers with exact term explanations`.

### Task 4: Ranking metrics

**Files:** Create `evaluate.py` (metrics part), `test_evaluate.py`.

**Interfaces:** `ndcg_at_k(rels: Sequence[int], k: int) -> float`; `mrr(rels) -> float`; `precision_at_k(rels, k)`; `average_precision_at_k(rels, k)`; `rank_metrics(scores: np.ndarray, relevant: np.ndarray) -> dict[str,float]`; `bootstrap_ci(values: np.ndarray, n=1000, seed=0) -> tuple[float,float]`.

- [ ] **Step 1: Failing tests** with hand-computed values: `ndcg_at_k([1,0,1],3) == (1 + 1/log2(4)) / (1 + 1/log2(3))`; `mrr([0,0,1]) == 1/3`; all-irrelevant -> 0 (no NaN); `bootstrap_ci` brackets the mean and is seed-deterministic; tie-breaking in `rank_metrics` is stable by index.
- [ ] **Step 2-4:** FAIL, implement, PASS. **Step 5:** commit `Add ranking metrics`.

### Task 5: Embedding backend

**Files:** Create `embed.py`, `test_embed.py`.

**Interfaces:** `chunk_tokens(ids: list[int], window: int, stride: int) -> list[list[int]]`; `class EmbeddingScorer` (`name="embedding"`, `fit` is a no-op, `score(query, docs)` = cosine of mean-of-chunks vectors); `class Encoder(Protocol): def encode(self, texts: Sequence[str]) -> np.ndarray`; `pick_backend(candidates: Sequence[Backend], probe: Sequence[str], reference: Encoder, min_cos=0.99) -> Backend`; `OpenVinoEncoder(device)`, `TorchEncoder()`; `EmbeddingCache(path)` keyed by `sha256(model + text)`.

- [ ] **Step 1: Failing tests**

```python
def test_chunk_tokens_covers_everything_with_overlap():
    chunks = chunk_tokens(list(range(600)), window=256, stride=192)
    assert chunks[0][0] == 0 and chunks[-1][-1] == 599 and all(len(c) <= 256 for c in chunks)

def test_empty_text_gives_one_zero_safe_vector(): ...finite, unit-norm or zeros handled

def test_pick_backend_rejects_corrupted_and_falls_back():
    good, bad = FakeEncoder(noise=0.0), FakeEncoder(noise=1.0)
    assert pick_backend([bad, good], probe, reference=TorchLike()) is good

def test_pick_backend_skips_backend_that_raises(): ...
def test_cache_roundtrip_and_key_changes_with_model(tmp_path): ...
def test_real_model_smoke(): pytest.importorskip + skip if model not downloadable
```

- [ ] **Step 2-4:** FAIL, implement (static-shape export exactly as in the NPU probe: `ov.convert_model`, `reshape` every input to `[B,256]`, feed inputs positionally in `input_ids, attention_mask, token_type_ids` order; CLS pooling + L2 norm), PASS.
- [ ] **Step 5:** on the real machine confirm GPU backend selected and cosine >= 0.99 vs torch; commit `Add embedding backend with accelerator self-check`.

### Task 6: Fusion and LanceDB index

**Files:** Create `index.py`, modify `scorers.py` (add `FusionScorer`, `rrf`), `test_index.py`, extend `test_scorers.py`.

**Interfaces:** `zscore(x: np.ndarray) -> np.ndarray` (constant vector -> zeros); `FusionScorer(parts: Sequence[Scorer], weights: Sequence[float])`; `rrf(rankings: Sequence[np.ndarray], k=60) -> np.ndarray`; `JobIndex.build(dir, ids, vectors)`; `JobIndex.search(vectors: np.ndarray, k: int) -> list[list[tuple[str,float]]]`.

- [ ] **Step 1: Failing tests:** z-score of a constant vector is zeros (no NaN); fusion with weight 1/0 reproduces each part; RRF of identical rankings preserves order; index round trip returns the nearest vector id first and batch search equals per-query search.
- [ ] **Step 2-4:** FAIL, implement, PASS. **Step 5:** commit `Add score fusion and LanceDB job index`.

### Task 7: Evaluation pipeline

**Files:** Modify `evaluate.py`; create `pipeline.py`, `test_pipeline.py`.

**Interfaces:** `evaluate_scorer(scorer, resumes: DataFrame, jobs: DataFrame, direction: str, stripped: bool) -> ScorerReport`; `tune_fusion_weight(embedding_scores, bm25_scores, relevant, grid) -> float` (validation only); `run_all(data_dir, out_dir, per_category=200, seed=0) -> dict` writing `report.json`.

- [ ] **Step 1: Failing tests** on a 3-category toy corpus: perfect scorer -> nDCG 1.0; random baseline near expected; tuned weight is chosen on val and the test report records it unchanged; `full` vs `stripped` differ when the headline carries the signal; a category with too few postings appears in `skipped_categories`.
- [ ] **Step 2-4:** FAIL, implement, PASS. **Step 5:** commit `Add evaluation pipeline`.

### Task 8: CLI and Streamlit app

**Files:** Create `cli.py`, `app.py`, `test_cli.py`, `test_app.py`.

**Interfaces:** Typer commands `fetch`, `evaluate`, `rank-jobs RESUME_FILE --top 10 --scorer fusion`, `rank-resumes JOB_FILE`; Streamlit page: paste/upload resume -> top jobs with highlighted evidence terms, missing-terms list, scorer selector.

- [ ] **Step 1: Failing tests:** `CliRunner` `rank-jobs` on a tiny fixture prints top-N with explanation terms and exits 0; empty file -> exit code 2 with a clear message; `AppTest.from_file("app.py")` renders without exception and shows rows after pasting text (scorers monkeypatched to the fixture).
- [ ] **Step 2-4:** FAIL, implement, PASS. **Step 5:** commit `Add Typer CLI and Streamlit page`.

### Task 9: Real run, docs, repo bookkeeping

**Files:** Create/overwrite `README.md`; modify root `README.md` (row 6 -> Implemented (Python) with a notes cell, Practical ML 6/30 = 20%, Total 61/150 = 41%); `.gitignore`.

- [ ] **Step 1:** run `uv run python cli.py evaluate` on real data (long embedding step: if it exceeds the 10-minute tool cap, run in background and poll).
- [ ] **Step 2:** write README from the measured numbers only (scorer table full vs stripped, per-category worst cases, fusion weight, audit agreement, hardware/backend numbers incl. the NPU finding, honest proxy-label caveat).
- [ ] **Step 3:** `uv run pytest -q`, `ruff check`, `ruff format --check`, dprint check on the folder and root README.
- [ ] **Step 4:** update `memory/project_programming_challenges_6_overview.md`.
- [ ] **Step 5:** commit `Implement Practical ML challenge 6: Resume-to-Job Matching Scorer`; push.
