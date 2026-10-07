# Active-Learning-Assisted Data Labeling Tool

**Category:** Practical ML
**Difficulty:** I

Suggest which unlabeled samples to label next, and why.

**Status:** Implemented (Python)

A labeling tool for text classification. Given a pool of unlabeled messages it suggests which ones to label next and says why, retrains as labels come in, and shows how much labeling effort that saves. It is also a benchmark: a simulated annotator with hidden gold labels measures eight query strategies against random sampling, with paired confidence intervals over seeds. The pool is Banking77 (77 customer-service intents).

## Run it

```bash
uv sync
uv run python cli.py fetch            # Banking77 CSVs from PolyAI's GitHub into data/ (about 1 MB)
uv run python cli.py embed            # embed pool/validation/test once; prints the verified backend
uv run python cli.py benchmark        # 8 strategies x 10 seeds + batch-size and stopping stages (resumable)

uv run python cli.py init projects/demo --demo        # a labeling project on the Banking77 pool
uv run python cli.py suggest projects/demo --strategy margin --batch 5
uv run python cli.py label projects/demo 2517 card_acceptance
uv run python cli.py status projects/demo
uv run python cli.py export projects/demo labels.csv
AL_PROJECT=projects/demo uv run streamlit run app.py   # the labeling UI

uv run python cli.py init projects/mine --csv my.csv --classes classes.txt   # your own texts
uv run pytest -q                      # 164 tests, no network, no model download
```

`benchmark` is resumable: each (strategy, seed, batch size) run is cached as one JSON file keyed by the data, the head's `C`, the budget and the run's own settings, so an interrupted run continues, a changed `Config`, dataset or head `C` never reuses stale curves, and adding seeds computes only the new ones. The key does not include the strategy code itself: after changing a strategy or the loop, bump `STAGE_VERSION` in `pipeline.py` (or run `benchmark --fresh`).

## Data and protocol

- **Data:** Banking77 (Casanueva et al. 2020, CC BY 4.0), fetched from PolyAI's GitHub because the HuggingFace copy is a loading script. Train has 10,003 messages, test 3,080, 77 intents.
- **Splits:** train is de-duplicated on exact text, then split stratified by intent into a **pool of 8,000** (what gets labeled) and a **validation set of 2,003** (picks the head's `C` once and tunes the stopping rule). **Test** (3,080) is only ever scored. In this dataset the exact-duplicate step removed nothing and no test text occurs in train, so the guard matters for your own CSV pools, not here; near-paraphrases are plentiful (see the limits).
- **Model:** frozen `BAAI/bge-small-en-v1.5` embeddings (384-d, CLS pooling, L2-normalized) and a multinomial logistic-regression head retrained every round. Probabilities cover all 77 intents, with exactly 0 for intents that have no label yet. Labeling all 8,000 pool items gives the **ceiling: 0.928 test accuracy**. `C = 300` was picked on validation with a 500-label subset; that is the edge of the grid, and checking larger values showed the best `C` moves between 1,000 (500 labels) and 300 (1,500 labels), so one fixed value is a compromise that is the same for every strategy.
- **Loop:** 77 random cold-start labels per seed (not stratified, so some intents start unseen), then batches of 50 up to 1,500 queried labels (30 rounds; curves run from 77 to 1,577 labels). All strategies share a seed's cold-start set, so every comparison is paired. 10 seeds. Test accuracy is scored every round and never used to select.
- **Embedding device:** the OpenVINO build on the Arc iGPU was checked against torch on a probe batch before use (mean cosine **1.0000**, threshold 0.99) and was used for all embeddings. The NPU is never tried: in an earlier spike it was fast and numerically wrong.

| Strategy           | Rule                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------ |
| `random`           | uniform                                                                                    |
| `least-confidence` | 1 - max p                                                                                  |
| `margin`           | smallest gap between the top two probabilities                                             |
| `entropy`          | largest predictive entropy                                                                 |
| `k-center`         | farthest-first on embeddings, seeded with the labeled set                                  |
| `badge`            | k-means++ over gradient embeddings; distances via the Kronecker identity (never built)     |
| `qbc`              | 5 bootstrap heads, vote entropy                                                            |
| `cluster-margin`   | the 5b smallest margins, then round-robin over Ward clusters (150), smallest cluster first |

## Results (test split, 10 paired seeds, batch 50)

Intervals are 95% bootstrap CIs over seeds; "vs random" columns are paired differences. A target of 90% / 95% means 90% / 95% of the 0.928 ceiling, i.e. accuracy 0.835 / 0.881.

| Strategy         | Accuracy at 1,577 labels | ALC       | ALC minus random         | Labels to 90% | Saving vs random (90%) | Labels to 95% | Saving vs random (95%) |
| ---------------- | ------------------------ | --------- | ------------------------ | ------------- | ---------------------- | ------------- | ---------------------- |
| random           | 0.890 [0.888, 0.893]     | 0.807     | -                        | 690           | -                      | 1,277         | -                      |
| least-confidence | 0.920 [0.919, 0.921]     | 0.819     | 0.011 [0.004, 0.018]     | 626           | 64 [-9, 131]           | 908           | 369 [292, 430]         |
| **margin**       | 0.920 [0.918, 0.921]     | **0.850** | **0.042 [0.035, 0.048]** | **429**       | **262 [198, 320]**     | **682**       | **595 [479, 693]**     |
| entropy          | 0.911 [0.910, 0.912]     | 0.801     | -0.007 [-0.015, -0.000]  | 724           | -33 [-108, 35]         | 1,020         | 257 [162, 348]         |
| k-center         | 0.901 [0.898, 0.903]     | 0.810     | 0.002 [-0.004, 0.008]    | 661           | 29 [-23, 83]           | 1,107         | 170 [61, 270]          |
| badge            | 0.916 [0.915, 0.917]     | 0.842     | 0.035 [0.026, 0.043]     | 464           | 226 [159, 287]         | 746           | 530 [421, 621]         |
| qbc              | 0.909 [0.906, 0.912]     | 0.821     | 0.013 [0.005, 0.021]     | 604           | 86 [10, 154]           | 942           | 334 [223, 433]         |
| cluster-margin   | 0.916 [0.915, 0.917]     | 0.833     | 0.025 [0.018, 0.032]     | 550           | 140 [68, 209]          | 876           | 401 [292, 498]         |

(ALC is the normalized area under the learning curve: mean test accuracy over the labeling effort. Every strategy reached both targets in all 10 seeds. Macro-F1 at the end tracks accuracy within 0.003.) Test accuracy at checkpoints, mean over seeds:

| Labels | random | least-confidence | margin | entropy | k-center | badge | qbc   | cluster-margin |
| ------ | ------ | ---------------- | ------ | ------- | -------- | ----- | ----- | -------------- |
| 227    | 0.651  | 0.618            | 0.701  | 0.592   | 0.638    | 0.701 | 0.641 | 0.679          |
| 527    | 0.806  | 0.806            | 0.863  | 0.783   | 0.805    | 0.849 | 0.816 | 0.831          |
| 1,027  | 0.865  | 0.892            | 0.905  | 0.880   | 0.877    | 0.898 | 0.886 | 0.897          |

What the numbers say:

- **Margin sampling is the strongest strategy here.** It reaches 90% of the ceiling after 429 labels where random needs 690, and 95% after 682 where random needs 1,277. BADGE and cluster-margin are next, with end accuracy about 0.916.
- **Not every uncertainty measure helps, and not at every stage.** Least-confidence and entropy start *below* random (0.618 and 0.592 vs 0.651 at 227 labels) and overtake it later, so entropy has a slightly negative ALC while ending 2 points above random. Margin and BADGE are ahead at every checkpoint above.
- **Pure diversity (k-center) is indistinguishable from random on ALC** and ends about 1 point higher. Banking77's pool is only mildly imbalanced (28 to 149 messages per intent, median 102), so random sampling already covers the intents well: with 177 labels it has seen 68.7 of 77 intents on average against 70.6 for the best strategy, and by 477 labels every strategy has seen 76-77. Diversity has little imbalance to fix on this dataset; a pool with rare intents would be a harder test of it.
- **Suggestions concentrate on what the model gets wrong.** The current model's error rate on the items a strategy suggests, averaged over rounds, is 0.58 for margin, 0.60 for least-confidence and 0.54 for entropy, against 0.19 for random picks. Over the same rounds the whole unlabeled pool has error 0.14 (margin), 0.19 (entropy) and 0.20 (random), so random picks look like the pool and the uncertainty scores pick out the hard items. That shows the *score* is informative; the reasons printed with each suggestion (top intents, nearest labeled neighbours, cluster) are a separate claim and are checked against brute force in the tests. The neighbours shown with each suggestion are recomputed from the embeddings and checked against a brute-force search in the tests.
- **Selection cost:** margin, entropy, least-confidence and cluster-margin take about 0.01 s per round; BADGE 1.4 s, k-center 2.6 s, query-by-committee 3.1 s (measured with 12 jobs running in parallel, one BLAS thread each; cluster-margin's one-off clustering of the pool, a few seconds, is not included).

### Batch size

Margin, BADGE and cluster-margin at three batch sizes, 5 seeds, ALC minus random (the batch-50 row uses the first 5 seeds, so it differs slightly from the table above):

| Batch | margin               | badge                | cluster-margin       |
| ----- | -------------------- | -------------------- | -------------------- |
| 10    | 0.034 [0.027, 0.042] | 0.027 [0.019, 0.036] | 0.020 [0.016, 0.023] |
| 50    | 0.041 [0.031, 0.048] | 0.031 [0.017, 0.044] | 0.023 [0.011, 0.035] |
| 200   | 0.023 [0.015, 0.032] | 0.028 [0.021, 0.035] | 0.029 [0.024, 0.034] |

All three still beat random at every batch size. Margin's advantage is lower at 200 than at 50, and the batch-aware strategies (BADGE, cluster-margin) hold up as the batch grows, so at batch 200 the three are within each other's intervals. That is the direction the theory predicts (a top-b of near-duplicate uncertain items wastes labels), but with 5 seeds the margin drop is suggestive rather than established: its intervals at 50 and 200 touch.

### Knowing when to stop

The stopping signal is the share of pool predictions that change between rounds (Bloodgood & Vijay-Shanker 2009). The rule (threshold, consecutive rounds) was chosen on 3 separate tuning seeds running margin sampling, scored on the *validation* split: stop when the change stays below **0.01 for 2 rounds**, the earliest rule whose accuracy gap to the budget end is at most 0.01. Applied to the 10 benchmark seeds (in a labeling project, `status` evaluates the same signal between rounds at least 50 labels apart, because one-label rounds change almost no predictions by construction):

| Strategy         | Seeds where it fires | Mean labels at stop | Accuracy left on the table |
| ---------------- | -------------------- | ------------------- | -------------------------- |
| margin           | 10/10                | 1,157               | 0.011                      |
| cluster-margin   | 10/10                | 1,272               | 0.007                      |
| least-confidence | 10/10                | 1,337               | 0.008                      |
| qbc              | 10/10                | 1,447               | 0.003                      |
| badge            | 10/10                | 1,457               | 0.003                      |
| entropy          | 10/10                | 1,482               | 0.004                      |
| random, k-center | 0/10                 | never (1,577)       | -                          |

The saving is modest (at best about 420 of 1,577 labels, for margin) and the rule does not transfer to every strategy: for random and k-center the pool predictions keep changing by more than 1% per round, so it never fires. Treat it as a hint to look at the learning curve, not a switch.

## A real session

On the Banking77 pool (150 labels filled in from gold, 64 of 77 intents seen), the CLI explains each suggestion from the model's own state:

```
#2517  Will my card work at all merchant locations?
    model: card_acceptance 15%, country_support 15%, declined_card_payment 7% (margin 0.00)
    near labeled #7946 'Will my new card work outside of the EU?' -> country_support (cosine 0.82)
    near labeled #2516 'Can I use my card no matter where I go?' -> card_acceptance (cosine 0.82)
    near labeled #4716 'My card was declined' -> declined_card_payment (cosine 0.75)
```

With nothing labeled yet the tool says so ("nothing labeled yet: the model has no guess") instead of showing an arbitrary class; the UI offers "Accept the model's guess" only when there is a guess to accept. Cluster-margin additionally shows the item's cluster, its size and how many items in it are already labeled.

## How it is tested

164 tests, no network and no model download, on a synthetic Gaussian-blob pool that stands in for embeddings:

- every strategy returns unique, unlabeled, in-range indices of the right size, including at cold start, with one class labeled, with fewer than `b` items left and with nothing left;
- k-center equals a brute-force farthest-first reference and scikit-activeml's `CoreSet`; the three uncertainty utilities match scikit-activeml's `UncertaintySampling` (margin up to a constant); BADGE's Kronecker distances equal distances between explicitly built gradient embeddings and its selection equals k-means++ run on them; query-by-committee and cluster-margin are checked against hand-computed votes and a hand-built round-robin;
- the head returns valid 77-class distributions with unseen intents at exactly 0;
- the loop is reproducible per seed, paired across strategies, and a run with shuffled evaluation labels picks exactly the same items;
- the benchmark cache is invalidated by changes to the budget, the data or the config and reused otherwise;
- the SQLite store keeps label history (last label wins), rejects labels outside the class list and validates a whole batch before writing any of it;
- `CliRunner` covers every command and error path (exit code 2, one line, no traceback); Streamlit's `AppTest` covers the UI flow.

## Limits

- **Banking77 is friendly to diversity-based and uncertainty-based methods alike:** its intents are only mildly imbalanced and its messages come in tight paraphrase clusters, which is why the diversity strategies add little here. On a skewed or messier pool the ranking could differ.
- **A frozen embedding plus a linear head is a weak model to do active learning on.** Strategies that rely on a representation learned during training (BADGE's gradient embeddings, core-set on learned features) get less to work with than they would with a fine-tuned network.
- **The simulated annotator is perfect.** Real annotators disagree and slip; the UI keeps label history and lets you relabel, but the benchmark does not model noise.
- **One dataset, one head, one embedding model, one cold-start size.** Differences of a few hundredths of ALC between the middle strategies are within what other settings could change.
- The stopping rule was tuned on one strategy and is shown to transfer imperfectly; its tolerance (0.01) is a choice, and margin ends up at 0.011.
- The batch-size study uses 5 seeds, so its intervals are wider than the main table's.
- The UI is verified with Streamlit's `AppTest` and by starting the server and fetching the page; it has not been exercised through a browser.
- The benchmark took about 13 minutes of wall time on 12 worker processes (16 logical cores).

## Layout

`data.py` Banking77 and CSV pools · `embed.py` embeddings, device self-check, cache · `model.py` head · `strategies.py` the eight strategies · `explain.py` reasons · `loop.py` simulated annotator · `stats.py` ALC, labels-to-target, bootstrap, stopping rule · `evaluate.py` / `pipeline.py` job-cached benchmark · `store.py` / `project.py` SQLite label store and project logic · `cli.py` · `app.py` · `results/report.json` the full numbers behind this README.
