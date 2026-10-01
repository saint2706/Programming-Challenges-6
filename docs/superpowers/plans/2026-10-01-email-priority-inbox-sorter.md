# Email Priority Inbox Sorter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learn which received emails the owner acts on (reply/forward), rank an inbox by it, explain each score, and benchmark honestly on time-split Enron mailboxes.

**Architecture:** Flat modules like Practical ML #5/#6. Pure logic (`thread`, `features`, `evaluate`) is separated from IO (`data`, `pipeline`). All history features are computed with sorted event arrays so they only use what was knowable at receipt time.

**Tech Stack:** Python >=3.12, uv, polars, numpy, scikit-learn, lightgbm, shap, typer, streamlit, kaggle, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-email-priority-inbox-sorter-design.md`

## Global Constraints

- Folder `challenges/Practical ML/Email Priority Inbox Sorter/`; own `pyproject.toml` + `uv.lock`; `requires-python = ">=3.12"`; `>=` bounds; pytest in dev group; `data/` gitignored; Kaggle token only in `~/.kaggle/`.
- Label window 14 days; received-mail dates 1998-2002; censoring cutoff 14 days before the mailbox's last sent message.
- Splits per mailbox by time: 70% train / 15% val / 15% test; fit everything (TF-IDF, SVD, scalers, calibration) on the proper split only.
- Committed artifacts and tests contain no real message text or addresses (synthetic fixtures; aggregates in `results/report.json`).
- ruff (repo `ruff.toml`) must exit 0 and be run **without a pipe that hides the exit code**; dprint via `npx --yes dprint@0.57.4 fmt --config dprint.json <files>`.

## Review Focus

- Message with no/garbled Date, no Subject, empty body, non-ASCII subject -> parsed or dropped with a count, never a crash.
- Same Message-ID in several folders -> one received message.
- Reply by someone other than the owner, or to a different sender with the same subject -> not a positive.
- Reply exactly at the 14-day boundary / before the receipt time -> boundary inclusive after, never before.
- Message received in the censoring tail -> dropped, counted.
- A day with all-acted or none-acted messages -> excluded from daily ranking metrics, not NaN.

---

### Task 1: Scaffold, parsing, mailbox owner, received set

**Files:** Create `pyproject.toml`, `pytest.ini`, `data.py`, `test_data.py`; modify root `.gitignore`.
**Interfaces:** Produces `parse_message(raw: str) -> Message | None` (dataclass: `message_id, date (datetime|None), sender, to, cc, subject, body`), `owner_address(sent: Sequence[Message]) -> str`, `received(msgs, owner) -> list[Message]` (owner in To/Cc, sender != owner, dedupe by message_id, date within 1998-2002), `load_mailboxes(csv_path, users) -> dict[str, MailboxData(sent, others)]`, `fetch(data_dir)`.
- [ ] Failing tests: parse a synthetic RFC-822 string (To/Cc lists, folded headers, `Re:` subject); garbled/missing Date -> `date is None` and message dropped by `received`; owner = most frequent sender of sent mail; dedupe; self-sent excluded; year 2044 excluded.
- [ ] Implement; run; commit `Scaffold Email Priority Inbox Sorter and message parsing`.

### Task 2: Threading and labels

**Files:** Create `thread.py`, `test_thread.py`.
**Interfaces:** `normalize_subject(s) -> str`; `kind_of(subject) -> Literal["reply","forward","other"]`; `label_received(received: list[Message], sent: list[Message], owner, window_days=14) -> pl.DataFrame` with columns `message_id, acted(bool), kind(str|None), acted_at(datetime|None)`; `censor_cutoff(sent) -> datetime`.
- [ ] Failing tests: `Re: Re: FW: Q3 numbers ` normalizes to `q3 numbers`; reply requires original sender in reply recipients; forward needs only the subject; reply before receipt is ignored; at +14d inclusive, +14d+1s not; earliest qualifying reply is `acted_at`; unrelated subject not positive; empty subject never matches anything.
- [ ] Implement with a dict subject -> sorted sent times (bisect); commit `Add reply/forward matching and labels`.

### Task 3: Leakage-safe features

**Files:** Create `features.py`, `test_features.py`.
**Interfaces:** `metadata_features(df: DataFrame, sent_index) -> DataFrame` where `df` has the received messages + label columns; `history_counts(times, keys, event_times) -> ...`; `FEATURE_GROUPS: dict[str, list[str]]` (`recipients`, `content`, `time`, `thread`, `sender_history`, `owner_history`).
- [ ] Failing tests: sender prior count equals a brute-force O(n^2) reference on random data; prior acted rate counts a past message only if its `acted_at` < current receipt time; shuffling later events does not change an earlier message's features; owner-already-in-thread uses only sent mail before receipt; mass-mail, `?` count, hour/weekday, Re/Fw depth on fixtures; every column belongs to exactly one group.
- [ ] Implement with `np.searchsorted`; commit `Add leakage-safe email features`.

### Task 4: Splits and models

**Files:** Create `models.py`, `test_models.py`.
**Interfaces:** `time_split(df, fracs=(0.7,0.15,0.15)) -> (train,val,test)` per mailbox; `TextFeatures.fit(train_texts)` (TF-IDF + SVD64); `fit_models(train,val,seed) -> Models` with `.predict(name, df) -> np.ndarray`, names `random|to_me|tfidf_lr|lgbm_meta|lgbm_meta_text`; `Calibrator.fit(val_scores, val_y)`.
- [ ] Failing tests: splits are ordered and disjoint per mailbox; test strictly after val strictly after train; TF-IDF/SVD fit only sees train text (assert vocabulary excludes a word present only in test); each model beats random on a synthetic task with a planted signal; deterministic for a seed; calibrator output in [0,1] and monotone.
- [ ] Implement (LightGBM early stopping on val PR-AUC); commit `Add time splits, text features and models`.

### Task 5: Evaluation

**Files:** Create `evaluate.py`, `test_evaluate.py`.
**Interfaces:** `pr_auc(y, s)`, `ece(y, p, bins=10)`, `reliability(y, p, bins)`, `daily_inbox_metrics(df, scores, k=3) -> DataFrame` (precision@3, ndcg@5, recall_top20 per eligible mailbox-day), `bootstrap_ci`, `ablation(train,val,test, groups) -> dict`.
- [ ] Failing tests: PR-AUC and ECE against hand values; perfect scores -> daily ndcg 1; days with 0 or all positives excluded; random scores give recall_top20 near 0.2 on a large synthetic set; bootstrap deterministic; ablation drops the planted-signal group the most.
- [ ] Implement; commit `Add evaluation metrics and ablation`.

### Task 6: Explanations

**Files:** Create `explain.py`, `test_explain.py`.
**Interfaces:** `reasons(model, X_row_or_frame, k=4) -> list[list[tuple[str,float]]]` using SHAP TreeExplainer (feature names mapped to readable text).
- [ ] Failing tests: for a fitted LightGBM on planted data, the planted feature is the top reason for a positive; contributions plus base value reproduce the raw margin (additivity) within tolerance.
- [ ] Implement; commit `Add SHAP reasons`.

### Task 7: Pipeline

**Files:** Create `pipeline.py`, `test_pipeline.py`.
**Interfaces:** `prepare(csv, users, out_dir) -> pl.DataFrame` (labelled, featured, split-tagged; also stats: dropped counts, label rates by kind); `run_all(data_dir, out_dir, users, seed) -> dict` writing `results/report.json` + model files.
- [ ] Failing tests on a synthetic CSV in the Kaggle format (two mailboxes): end-to-end produces report with all models/metrics, drop counts, censoring applied, no real text in the report.
- [ ] Implement; commit `Add pipeline`.

### Task 8: CLI and Streamlit

**Files:** Create `cli.py`, `app.py`, `test_cli.py`, `test_app.py`.
**Interfaces:** `fetch`, `prepare`, `train`, `report`, `rank-inbox MAILBOX --day YYYY-MM-DD --top 10`; Streamlit: mailbox + day pickers over the test period, inbox sorted by calibrated probability, "why" panel, acted flag shown only after a toggle.
- [ ] Failing tests: Typer runner on a tiny prepared fixture (ranking order, unknown mailbox/day -> exit 2 with message); `AppTest` renders, sorts and shows reasons.
- [ ] Implement; commit `Add CLI and Streamlit inbox`.

### Task 9: Real run, docs, bookkeeping

- [ ] `prepare` + `train` on the six real mailboxes; audit 50 reply pairs (print subjects only, do not commit); fix matcher defects with a failing test each.
- [ ] README from measured numbers only (label definition + audit, leakage controls, model table with CIs, calibration before/after, daily inbox metrics vs random, ablation, SHAP summary, limitations incl. "replied != important"); root README row 7 -> Implemented, Practical ML 7/30 (23%), Total 62/150 (41%); dprint; ruff exit 0; memory file update; commit. Do not push.
