# Duplicate Product Listing Detector

**Category:** Practical ML
**Difficulty:** B (brief: "Text + image embedding similarity for e-commerce catalogs.")

**Status:** Implemented (Python)

Finds listings in an e-commerce catalog that are really the same product.
Titles are embedded with `multilingual-e5-small`, photos with SigLIP 2, both
go into one [LanceDB](https://lancedb.com) table, and candidate pairs are
retrieved by nearest neighbour in each modality, scored by a tuned text/image
fusion, and clustered. A Typer CLI runs the pipeline; a Streamlit page lets a
reviewer look at the pairs side by side and move the threshold live.

## Data

[Shopee Product Matching](https://www.kaggle.com/competitions/shopee-product-matching)
(Kaggle, 34,250 listings, 11,014 duplicate groups). Listings sharing a
`label_group` are duplicates -- real labels, not synthetic ones. Titles are a
mix of Indonesian, Malay and English, which drove the text-model choice.

Things worth knowing about it:

- **Every group has 2+ listings; there are no unique products.** Other groups
  act as the negatives, but a real catalog is mostly non-duplicates, so the
  precision figures here are friendlier than production would be.
- The slice is **whole groups** (default 1,400 groups, about 4.4k listings and
  10k true pairs). Sampling listings instead would cut real duplicate pairs in
  half and quietly turn them into false negatives.
- Validation and test are **disjoint sets of groups**, never of listings, so
  no product appears on both sides of the split.

Downloading needs a Kaggle API token in `~/.kaggle/kaggle.json` **and** the
competition rules accepted on your account (otherwise `403`).

## How it works

```
fetch -> embed -> evaluate -> dedupe            (cli.py)
           |         |
    embeddings.npz   LanceDB (val / test tables) + report.json
```

| File               | Role                                                                          |
| ------------------ | ----------------------------------------------------------------------------- |
| `data.py`          | Fetch the catalog/archive, slice and split by whole group, ground-truth pairs |
| `embed.py`         | `MultilingualTextEncoder`, `SigLIP2ImageEncoder`, caching, bad-image handling |
| `index.py`         | LanceDB table with both vector columns; top-k candidate retrieval             |
| `match.py`         | Score fusion, threshold search, union-find clustering, pair/cluster metrics   |
| `evaluate.py`      | Tune on validation, freeze, report on test; data for the review UI            |
| `pipeline.py`      | Stage glue and on-disk caching shared by the CLI and the app                  |
| `cli.py`, `app.py` | Typer CLI, Streamlit review page                                              |

**Retrieval is recall-oriented, scoring is precision-oriented.** LanceDB
returns the top-k cosine neighbours per listing in text space and in image
space, unioned. Then `w * text_cos + (1 - w) * image_cos >= threshold` decides
which candidate pairs are duplicates, and connected components (union-find)
turn flagged pairs into clusters, so `A~B` and `B~C` put all three together
even when `A~C` scored low.

**Honest evaluation protocol.**

- The fusion weight and threshold are chosen on the **validation** groups only,
  then applied unchanged to the **test** groups (there is a test that corrupts
  the test labels and checks the operating point doesn't move).
- Recall is measured against **every** true pair, not just the ones retrieval
  surfaced, so a retrieval miss costs recall instead of hiding. Candidate
  recall is reported separately so you can tell retrieval misses from scoring
  misses.
- Text-only, image-only and fused scoring are each tuned the same way and
  reported side by side, so the fused row has to earn its place.
- Cluster F1 is the mean per-listing F1 between a listing's predicted cluster
  and its true group (the flavour of metric the Shopee competition used).

**A missing image is not an error.** Unreadable or absent images become an
all-zero vector (cosine 0, never NaN), are skipped as search queries, and are
listed in the embedding result; the listing still matches on text.

## Usage

```bash
uv sync
uv run python cli.py fetch --groups 1400     # catalog + slice + images
uv run python cli.py embed                   # SigLIP 2 + multilingual-e5, cached
uv run python cli.py evaluate                # tune on val, report on test
uv run python cli.py dedupe --split test --out duplicates.csv
uv run streamlit run app.py                  # review UI
```

`fetch` downloads the competition archive once (about 1.8 GB) and extracts
only the slice's images from it. Fetching images one at a time looks cheaper
but Kaggle rate-limits it (HTTP 429 after a few hundred requests), so don't.
A 429 is retried with backoff and the server's `Retry-After` is honoured; the
archive is kept so re-slicing with another `--seed` needs no new download.

Embedding runs on CPU (no CUDA needed); expect it to take a while for a few
thousand images. Everything is cached under `data/` (gitignored).

## Tests

```bash
uv run pytest -q
```

Network-free except one smoke test that loads the real models and
`pytest.skip`s (never fails) if they can't be downloaded. The suite covers
group-aware slicing and splitting without leakage, archive extraction
(including path-traversal member names), 429/`Retry-After` handling, the
embedding cache and bad-image path, LanceDB candidate retrieval (including
zero vectors), threshold search and clustering maths, the tune-on-val-only
guarantee, the CLI end to end, and the Streamlit page headless with
`AppTest`.

## Results

Real Shopee data: 1,400 whole groups (4,404 listings), split 60/40 by group
into test (2,701 listings, 6,593 true duplicate pairs) and validation. Each
row's weight and threshold were tuned on validation only, then frozen for test.
`k = 10`, seed 0.

| scoring    | text weight | threshold | precision | recall    | pair F1   | cluster F1 |
| ---------- | ----------- | --------- | --------- | --------- | --------- | ---------- |
| text only  | 1.00        | 0.900     | 0.764     | 0.597     | 0.670     | 0.551      |
| image only | 0.00        | 0.816     | 0.812     | 0.618     | 0.702     | 0.773      |
| **fused**  | 0.65        | 0.857     | **0.867** | **0.728** | **0.792** | **0.849**  |

Candidate recall at `k = 10` is **0.883**: retrieval never surfaces 11.7% of
the true pairs, so recall cannot exceed 0.883 whatever the scoring does.

What the numbers say:

- **Fusing beats either modality on every column.** Pair F1 0.792 against
  0.702 (image) and 0.670 (text); cluster F1 0.849 against 0.773 and 0.551.
  The tuned weight leans text (0.65) even though image alone scores higher,
  which fits the two modalities making different mistakes -- an inference, not
  something tested directly.
- **Image beats text on this data.** That is not explained by identical image
  files: only about 5% of true pairs (326 of 6,593 on test) are the same file.
- **Text-only cluster F1 (0.551) is far below its pair F1 (0.670).** Clustering
  takes connected components, so a few wrong text links can merge unrelated
  groups. That is my hypothesis for the gap; I have not diagnosed it.

### Retrieval depth is a trade-off

Raising `k` fixes candidate recall but not for free (fused, test split, each
`k` re-tuned on validation):

| k  | candidate recall | candidates | precision | recall | pair F1 | cluster F1 |
| -- | ---------------- | ---------- | --------- | ------ | ------- | ---------- |
| 10 | 0.883            | 32,825     | 0.867     | 0.728  | 0.792   | 0.849      |
| 20 | 0.968            | 69,081     | 0.861     | 0.761  | 0.808   | 0.849      |
| 40 | 0.985            | 144,228    | 0.845     | 0.795  | 0.819   | 0.837      |

Pair F1 keeps climbing with `k`, but cluster F1 peaks at 10-20 and drops at 40,
because more candidates also means more false links to chain together. The
default stays at `k = 10`: it ties `k = 20` on cluster F1 with half the
candidates. Pick by which metric you care about.

### Caveats

- One slice (1,400 groups, seed 0) and one validation/test split, with no
  confidence intervals. Differences of a point or two between rows are within
  what another seed could move.
- Friendlier than production: every group has 2+ listings, so there are no
  unique products to wrongly flag (see [Data](#data)).
- 4,404 listings map to 4,131 image files, because some duplicate listings
  reuse the exact same image. No image was missing or unreadable.
