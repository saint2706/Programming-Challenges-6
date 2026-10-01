# Email Priority Inbox Sorter — design

Practical ML #7 (B). Brief: "Lightweight learned classifier, not hand-written rules."

## Understanding (confirmed with the user)

Rank an inbox so the mail the owner will act on comes first, using a learned
model whose label is **behavior** (did the owner reply to / forward it), not
hand-labelled importance. Success = a leakage-controlled, time-split benchmark on
real mail, honest ranking/calibration metrics, an explanation per message, a
Typer CLI and a Streamlit inbox view. Decisions already made: Enron corpus
(Kaggle `wcukierski/enron-email-dataset`), label = replied/forwarded, LightGBM on
metadata + text features, TF-IDF+LR and metadata-only as comparisons, CLI +
Streamlit, same layout as Practical ML #5/#6, solo build.

## Data

`emails.csv` (517k raw RFC-822 messages; column `file` = `user/folder/N.`).
Headers present: Message-ID, Date, From, To, Cc (inside `X-cc` / `Cc`), Subject,
X-Folder. **No In-Reply-To / References**, so replies are matched heuristically.
Six mailboxes with the most sent mail: mann-k, kaminski-v, dasovich-j,
germany-c, shackleton-s, jones-t. Gitignored `data/`. The corpus is real
people's mail (public release); committed artifacts contain aggregates and
synthetic test fixtures only, never message text or addresses.

## Received mail and the label

- A mailbox owner's address = the most frequent `From` in their sent folders.
- **Received** = owner in To/Cc, `From` != owner, deduplicated by Message-ID per
  mailbox (the same mail sits in several folders), dates restricted to
  1998-2002 (the corpus has bogus years). Folders are *not* used as a feature:
  filing is the owner's own behavior.
- **Acted** (positive) = the owner later sent a message in a sent folder whose
  normalized subject (strip `Re:`/`Fw:`/`Fwd:` chains, whitespace, case) equals
  the received message's, within 14 days after receipt, and either it is a reply
  (`Re:`) whose recipients include the original sender, or it is a forward
  (`Fw:`/`Fwd:`). Kind (`reply`/`forward`) is kept; the primary label is
  `acted`, reply and forward rates are reported separately.
- **Right-censoring.** Messages received within 14 days of the mailbox's last
  sent message are dropped; their replies may simply not be in the dump.
- The matcher is a heuristic. Its precision is judged by hand on 50 sampled
  (received, reply) pairs and reported with the caveat that it is the author's
  judgment; subjects are not committed.

## Splits and leakage controls

Per mailbox, strictly by time: first 70% train, next 15% validation, last 15%
test. One pooled model across mailboxes (all features are mailbox-relative).

- Sender-history features use only information available **at receipt time**:
  messages from the sender received earlier; replies to that sender that were
  **already sent** before this message arrived (not merely sent before the
  dataset ended); whether the owner had already written in this subject thread.
  Computed with sorted event arrays + `searchsorted`, tested against a
  brute-force reference on random data.
- A test makes the label leak loud: shuffling reply timestamps must not change
  the features of earlier messages.
- `Re:`/`Fw:` prefix on the *received* mail is a legitimate feature (thread
  depth) and is reported as its own ablation group.

## Features

- **Metadata:** n_to, n_cc, owner in To (vs Cc only), mass-mail flag (>= 10
  recipients), sender domain is Enron, subject Re/Fw depth, subject length,
  subject all-caps, body length, `?` count, attachment mention, hour/weekday/
  business-hours, sender prior count, sender prior acted rate (known-by-now,
  smoothed), owner->sender prior send count, owner already in thread.
- **Text:** TF-IDF on subject + first 2,000 body chars (fit on train only);
  TruncatedSVD(64) fit on train only; LR on the full TF-IDF.

## Models

1. `popularity`-style trivial baselines for context: random; "owner in To".
2. TF-IDF + LogisticRegression (text only).
3. LightGBM on metadata only.
4. LightGBM on metadata + SVD text.

Class weights handled by `scale_pos_weight` / `class_weight`, early stopping on
validation PR-AUC. Probability calibration (isotonic) fit on validation, reported
before/after.

## Metrics (test, per mailbox and macro-averaged)

PR-AUC (with positive rate shown), ROC-AUC, expected calibration error +
reliability bins, and the inbox metrics that matter: for every (mailbox, day) in
test with >= 5 received messages and >= 1 acted, precision@3, nDCG@5 and
"recall of acted mail in the top 20% of that day's inbox" versus random order.
Bootstrap 95% CIs over days. Feature-group ablation (drop each group, retrain,
PR-AUC change) and SHAP attributions for the LightGBM.

## Components

| File | Role |
|---|---|
| `data.py` | fetch, parse raw messages, mailbox owner address, received-set + dedupe |
| `thread.py` | subject normalization, reply/forward matching, censoring, labels |
| `features.py` | leakage-safe metadata features (searchsorted history), text features |
| `models.py` | the four models, calibration, early stopping |
| `evaluate.py` | metrics, daily inbox ranking, CIs, ablation |
| `explain.py` | SHAP per message, top reasons |
| `pipeline.py` | prepare -> train -> evaluate -> `report.json` and model artifacts |
| `cli.py`, `app.py` | Typer (`fetch`, `prepare`, `train`, `report`, `rank-inbox`), Streamlit inbox |

## Testing

Synthetic mailboxes only: subject normalization, reply vs forward vs unrelated
matching incl. the sender-must-be-recipient rule and the 14-day window,
right-censoring, dedupe, time split disjoint and ordered, history features equal
to a brute-force reference and unaffected by later events, metrics vs
hand-computed values, calibration/ECE, daily ranking metrics, SHAP reasons
returned for a fitted model, CLI via Typer runner, Streamlit via `AppTest`. One
end-to-end test on the real CSV that skips if `data/` is absent.

## Non-goals

No per-user fine-tuning of a language model, no spam filtering, no folder
prediction, no claim that "replied" equals "important" -- the README states the
label is behavior and says what it misses (read-only FYI mail, mail answered by
phone).
