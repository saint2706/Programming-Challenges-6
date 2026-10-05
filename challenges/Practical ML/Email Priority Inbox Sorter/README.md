# Email Priority Inbox Sorter

**Category:** Practical ML
**Difficulty:** B (brief: "Lightweight learned classifier, not hand-written rules.")

**Status:** Implemented (Python)

Learns which received emails the mailbox owner will act on, ranks an inbox by
that, and says why for every message. The label is **behavior** -- did the owner
reply to or forward the mail -- not a hand-made notion of "importance", and the
benchmark is a time-split, leakage-controlled run on six real Enron mailboxes. A
Typer CLI runs it; a Streamlit page shows a held-out day's inbox sorted, with a
"why" panel per message.

## Data

[Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)
(Kaggle, about 517,000 raw messages). `cli.py fetch` downloads `emails.csv` into a
gitignored `data/` folder with your Kaggle token in `~/.kaggle/kaggle.json`. The
corpus is real people's mail (a public release), so **nothing from it is
committed**: tests use synthetic fixtures and `results/report.json` holds
aggregates only (a test asserts it contains no message text and no address).

Six mailboxes, picked for having the most sent mail: mann-k, kaminski-v,
dasovich-j, germany-c, shackleton-s, jones-t. After the steps below: **23,841
received messages**, split by time into 16,690 train / 3,575 validation / 3,576
test.

## The label, and what it is not

A received message is **acted** when the owner later sent a message whose
normalized subject (every `Re:`/`Fw:`/`Fwd:` chain stripped) is the same, within
**14 days after receipt**, and it is either

- a reply (`Re:`) addressed to the original sender, or
- a forward (`Fw:`/`Fwd:`).

The corpus has no `In-Reply-To`/`References` headers, so this is a heuristic. A
reply sent before the message arrived is never a match, a reply by someone other
than the owner (an assistant) never is, and mail received in the last 14 days
before the owner's final sent message is dropped (**right-censoring**: its reply
may simply not be in the dump; 273 messages across the six mailboxes). The
owner's address is the most frequent `From` in their sent folders; "received"
means the owner is in To/Cc, is not the sender, counted once (see below), dated
1998-2002 (the corpus has bad clocks stamping 1979 and 2044). Folders are never
a feature: filing is the owner's own behavior.

**Copies.** The same mail sits in several folders of a mailbox (inbox,
all_documents, a project folder...), and **the corpus gives each folder's copy a
different Message-ID**. Deduplicating by ID alone, which my first version did,
left 47,020 folder copies standing for 24,114 distinct messages: about half of
the first dataset was duplicates, which inflated every count and history
feature and put identical messages twice in a day's ranking. I found it when
`rank-inbox` returned the same message three times. Copies are now matched by
Message-ID *and* by content (sender, date, subject, To, Cc, body), for received
and for the owner's sent mail alike; a test covers both a re-stamped copy and
two different mails sent in the same minute.

| mailbox      | folder copies | distinct | kept  | acted | reply | forward |
| ------------ | ------------- | -------- | ----- | ----- | ----- | ------- |
| mann-k       | 6,361         | 2,593    | 2,534 | 27.7% | 24.5% | 3.2%    |
| kaminski-v   | 6,038         | 2,279    | 2,279 | 11.1% | 10.8% | 0.2%    |
| dasovich-j   | 11,663        | 6,498    | 6,498 | 19.7% | 18.7% | 1.0%    |
| germany-c    | 3,207         | 1,685    | 1,586 | 18.9% | 13.7% | 5.1%    |
| shackleton-s | 8,924         | 5,207    | 5,114 | 12.7% | 9.4%  | 3.3%    |
| jones-t      | 10,827        | 5,852    | 5,830 | 16.5% | 14.6% | 1.9%    |

**Precision of the matcher.** I read 50 randomly sampled (received, reply)
pairs (me, not an independent annotator) with the received and sent subjects and
the start of the reply: 49 were plainly genuine replies or forwards and 1 was
borderline (a `Re: FW:` follow-up whose body does not obviously answer the
message it was matched to). That audit was run before the copy fix; the matcher
itself was not changed by it. Recall was not audited: a reply that changes the
subject line, or an answer by phone, is a false negative the matcher cannot see.

**"Acted" is not "important".** It misses read-only FYI mail and mail answered
by phone or in person, and it counts a one-word "ok" the same as a long answer.
The model predicts what this owner *does*, which is what an inbox sorter can
learn from; it cannot tell you what they *should* have read.

## Leakage controls

- **Time splits per mailbox**: oldest 70% train, next 15% validation, newest 15%
  test; a cut never lands between two messages with the same timestamp, so every
  train message is strictly older than every validation message and so on. One
  pooled model across mailboxes (every feature is mailbox-relative).
- **History features see only what was knowable at receipt time** (`features.py`):
  earlier mail from the sender; replies to that sender that had *already been
  sent* before this message arrived (not merely before the dump ended); sent mail
  and thread membership strictly before receipt. They use sorted event arrays and
  `np.searchsorted`, and the tests check them against a brute-force O(n^2)
  reference and check that perturbing every later event leaves earlier messages'
  features byte-identical.
- **Fit on the right split**: TF-IDF, the SVD, early stopping (validation
  PR-AUC) and the isotonic calibrator never see the test period; a test checks a
  word present only in test is outside the vocabulary.
- The `Re:`/`Fw:` prefix on the *received* mail is a legitimate signal (thread
  depth) and is its own ablation group, `thread`.

## Models

| model            | what it sees                                                                                        |
| ---------------- | --------------------------------------------------------------------------------------------------- |
| `random`         | nothing (seeded)                                                                                    |
| `to_me`          | a one-rule baseline: owner is a direct recipient (not just Cc)                                      |
| `tfidf_lr`       | text only: TF-IDF (1-2 grams) on subject + first 2,000 body chars, logistic regression              |
| `lgbm_meta`      | 20 metadata features in 6 groups (recipients, content, time, thread, sender history, owner history) |
| `lgbm_meta_text` | the same plus a 64-component SVD of the TF-IDF                                                      |

LightGBM uses `scale_pos_weight` and early-stops on validation PR-AUC. Time of
day is read in Houston time (a fixed UTC-6; daylight saving ignored).

## Results

Held-out test period (2001-04-06 to 2002-06-11), 3,576 messages of which 23.2%
were acted on. PR-AUC is judged against that base rate: random scores 0.233.

| model            | PR-AUC | ROC-AUC | macro PR-AUC (6 mailboxes) |
| ---------------- | ------ | ------- | -------------------------- |
| `random`         | 0.233  | 0.502   | 0.247                      |
| `to_me`          | 0.248  | 0.540   | 0.256                      |
| `tfidf_lr`       | 0.368  | 0.681   | 0.347                      |
| `lgbm_meta`      | 0.436  | 0.744   | 0.418                      |
| `lgbm_meta_text` | 0.421  | 0.741   | 0.399                      |

**What a user feels is the daily inbox.** For each of the 192 (mailbox, local
day) pairs in the test period with at least 5 received messages and a real mix
of acted and ignored mail (days with none or all acted say nothing about
ranking), how well is the acted mail put first? Mean over days with a 95%
bootstrap CI; ties are broken by a seeded shuffle, so a constant scorer is not
handed the arrival order. Random is higher than the 23% base rate here because
eligible days all contain at least one acted message.

| model            | precision@3          | nDCG@5               | recall in top 20%    |
| ---------------- | -------------------- | -------------------- | -------------------- |
| `random`         | 0.278 [0.241, 0.314] | 0.388 [0.349, 0.428] | 0.230 [0.194, 0.268] |
| `to_me`          | 0.281 [0.247, 0.316] | 0.396 [0.353, 0.440] | 0.271 [0.229, 0.315] |
| `tfidf_lr`       | 0.384 [0.340, 0.429] | 0.515 [0.473, 0.560] | 0.352 [0.311, 0.399] |
| `lgbm_meta`      | 0.438 [0.399, 0.476] | 0.591 [0.554, 0.630] | 0.454 [0.406, 0.499] |
| `lgbm_meta_text` | 0.431 [0.392, 0.470] | 0.582 [0.545, 0.619] | 0.438 [0.390, 0.487] |

What the numbers say, and what they do not:

- Learned models clearly beat random and the one-rule baseline; the intervals
  do not overlap. The metadata model roughly doubles random's recall in the top
  fifth of a day's inbox (0.454 vs 0.230) and nearly doubles its PR-AUC.
- **Text adds nothing once the metadata is in, and may cost a little.**
  `lgbm_meta_text` is no better than `lgbm_meta` on any metric (PR-AUC 0.421 vs
  0.436; the daily intervals overlap almost entirely), and dropping the SVD
  columns changes PR-AUC by -0.014 in the ablation below. Text alone (`tfidf_lr`)
  is clearly weaker than metadata. Who wrote to you, whether you already talk to
  them and how the message is addressed carry the signal; the wording mostly
  restates it, and with 16,690 training messages 64 extra noisy columns have
  room to overfit. I did not tune the SVD size or try an alternative encoder,
  so this says "this text representation did not help here", not "text cannot".
- Absolute quality is modest: ROC-AUC 0.74. Whether someone replies depends on
  things an inbox does not contain. Per mailbox (`lgbm_meta`) kaminski-v, the
  mailbox with the fewest acted messages (9.4% of test), is hardest (PR-AUC
  0.26 against a 0.12 random score) and mann-k easiest (0.55, random 0.38).
- These are one split and one seed; per-mailbox figures rest on 238 to 975 test
  messages each and are indicative at best.

### Calibration

Isotonic regression fit on validation turns scores into probabilities (shown in
the Streamlit page and the CLI). Expected calibration error on the test set,
before -> after: `lgbm_meta_text` 0.071 -> 0.037, `lgbm_meta` 0.081 -> 0.045,
`tfidf_lr` 0.183 -> 0.036. The raw LightGBM output is inflated by
`scale_pos_weight` (its lowest-scoring bin predicts 17% but is acted on 6% of the
time). Ranking always uses the raw score, since isotonic plateaus would create
ties; the calibrated probability is only what is displayed.

### Ablation

PR-AUC lost on test when one metadata feature group is removed and LightGBM is
retrained (positive = the group mattered; one seed, no interval, so differences
under about 0.01 are noise):

| group          | drop   |
| -------------- | ------ |
| sender_history | +0.041 |
| recipients     | +0.029 |
| content        | +0.027 |
| time           | +0.005 |
| thread         | +0.001 |
| owner_history  | -0.006 |
| text (SVD)     | -0.014 |

Only `sender_history`, `recipients` and `content` clear the noise floor.
Negative values mean the model did as well or better without the group; with
overlapping signals (`owner_history` and `thread` partly restate
`sender_history`) removing one group is rarely a clean subtraction.

## Explanations

`explain.py` uses SHAP `TreeExplainer`, which is exact for tree models: a
message's per-feature contributions plus the base value equal the model's raw
log-odds margin (a test checks this to 1e-5). The 64 SVD components are summed
into one "wording of the message" reason because a single component has no
readable meaning. Mean absolute contribution on 2,000 test messages, top of the
list: how often you answer this sender (0.135), wording of the message (0.075),
questions in the text (0.067), number of direct recipients (0.057), you already
wrote in this thread (0.041). (This is `lgbm_meta_text`, the model the CLI and
page serve by default; `lgbm_meta` is selectable and scores a little higher on
this test set.)

## Usage

```bash
uv run python cli.py fetch                       # Kaggle token in ~/.kaggle/kaggle.json
uv run python cli.py prepare                     # parse, label, featurize -> data/dataset.parquet
uv run python cli.py train                       # prepare + fit + evaluate -> results/report.json
uv run python cli.py report                      # print the last benchmark
uv run python cli.py rank-inbox mann-k --day 2001-11-26 --top 10   # held-out days only
uv run python cli.py rank-inbox mann-k --day 2001-11-26 --reveal   # also show what the owner did
uv run streamlit run app.py
```

`rank-inbox` only serves days in the test period: ranking mail the model was
trained on would flatter it. The outcome (replied or ignored) is hidden until
`--reveal` / the checkbox, so you can judge the ordering first.

## Tests

```bash
uv run pytest
```

99 tests on synthetic mailboxes (no real mail): subject normalization, reply vs
forward vs unrelated matching including the sender-must-be-a-recipient rule, the
14-day boundary and pre-receipt replies, right-censoring, dedupe, history
features against a brute-force reference and unchanged by later events, disjoint
and strictly ordered time splits, models beating random on a planted signal,
determinism per seed, metrics against hand-computed values, daily-ranking
exclusions and tie handling, SHAP additivity, an end-to-end pipeline run on a
synthetic CSV (report holds no text or addresses), the CLI via Typer's runner and
the Streamlit page via `AppTest`.

## Limitations

- The label is a subject-matching heuristic over a corpus without threading
  headers; precision was spot-checked (49/50 plain, 1 borderline), recall was not.
- Behavior, not importance (above). A model of one person's habits, pooled over
  six people who are mostly lawyers, traders and analysts at one company in
  2000-2002: do not expect it to transfer to another inbox without retraining.
- Single time split and seed; per-mailbox and ablation numbers carry no
  intervals.
- Houston-time features ignore daylight saving.
- The model never reads attachments or quoted history beyond the first 2,000
  characters, and an exact-subject match misses replies that retitle a thread.
