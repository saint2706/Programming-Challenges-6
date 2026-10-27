# Meeting Notes to Action-Items Extractor

**Category:** Practical Software
**Difficulty:** A
**Status:** Complete

Turns meeting transcripts and notes into a list of action items with an owner, a due date and the
source line each item came from. It reads speaker transcripts, Zoom, Teams and Meet exports, WebVTT and
SRT captions, Markdown and plain notes, action sections, `@mentions` and checkboxes, and writes a
Markdown checklist, JSON, an ICS calendar of VTODO entries, or a table.

```bash
uv run --locked actionitems extract notes.txt --format md --meeting-date 2026-11-02
uv run --locked actionitems extract corpus/test --format ics --out actions.ics
uv run --locked actionitems eval --split all --write
```

## Data

The evaluation corpus is hand-written and clearly synthetic: 48 documents, 190 gold action items, in
two splits. DEV (24 documents, 90 items) is used for rule iteration. TEST (24 documents, 100 items) was
run once after the rules were frozen. Each split has three documents for each of eight formats: speaker
turns, Zoom, Teams, WebVTT, SRT, checklist, ASR-style text without punctuation, and notes. Gold labels
live in `corpus/gold.json`, which records the quote, task, owner, due date and action for each item. A
gold quote must occur exactly once in the document.

No real meeting data is used. The AMI corpus was not used because its licence was not verified.

## Method

The pipeline is rule-based and deterministic. It has four stages.

1. **Ingest.** Read the text (BOM tolerated, cp1252 fallback, NUL bytes rejected), with size and line
   limits and symlinks skipped.
2. **Parse.** Detect the format from its timing lines or headers, then split the text into utterances
   with a speaker, timestamp, section and kind.
3. **Classify.** Rules find commitments ("I'll send it"), requests ("can you", "please", "Eli, follow
   up"), unassigned tasks ("someone should"), conditional items and hedged items. spaCy
   `en_core_web_sm` 3.8.0 runs NER and the dependency parse to refine owners and task spans. Names are
   resolved against the attendee list (exact 1.0, first name 0.9, fuzzy match at ratio 88 or higher
   scaled by 0.85, not on the roster 0.6).
4. **Resolve dates.** `dateparser` 1.4.3 handles absolute dates, with "next week", "this week",
   "by Friday", "EOD" and "in N days" resolved by fixed rules against the meeting day.

Each item gets a confidence score (rule base times owner factor, times 0.85 when conditional). The
default threshold is 0.4, which drops tentative items.

The optional LLM refinement pass (`--llm-url`) sends the transcript to a local llama-server with a JSON
schema. It never deletes rule items. Its output is checked before use: owners must be on the roster or
in the text, due dates must be ISO dates at most two years after the meeting, and a task must share
words with the line it cites. Items the model adds get confidence 0.35, below the default threshold, so
they appear only with `--min-confidence 0.3` or lower. Tests use a mocked server; one real run is
recorded below.

## Results

Commands: `uv run --locked actionitems eval --split all --write` writes `results/metrics.json` and
`results/report.md`.

| split | P     | R     | F1    | owner acc | due acc |
| ----- | ----- | ----- | ----- | --------- | ------- |
| DEV   | 0.989 | 0.967 | 0.978 | 0.943     | 1.000   |
| TEST  | 0.980 | 0.960 | 0.970 | 0.958     | 1.000   |

The DEV score is optimistic. Before the DEV fixes it was F1 0.950. The rule changes were made by
reading DEV errors only, and TEST was run once afterwards. TEST F1 (0.970) is close to DEV, which
suggests the rules are not heavily overfitted, but a sample of 24 documents is small.

Per format on TEST, ASR-style text has the lowest owner accuracy (0.714), while its F1 is 1.000.

### LLM pass on TEST

One real run, with Qwen3-1.7B-Q8_0 (Apache-2.0) served by llama-server on loopback. Full table in
`results/llm-test.md`.

| run                            | P     | R     | F1    | owner acc | due acc |
| ------------------------------ | ----- | ----- | ----- | --------- | ------- |
| rules only (committed)         | 0.980 | 0.960 | 0.970 | 0.958     | 1.000   |
| LLM, default threshold 0.4     | 0.980 | 0.960 | 0.970 | 0.760     | 0.854   |
| LLM, opt-in threshold 0.3      | 0.700 | 0.980 | 0.817 | 0.745     | 0.847   |
| LLM, before the security fixes | 0.717 | 0.990 | 0.832 | 0.747     | 0.818   |

The LLM pass does not improve any measured metric here. At the default threshold it adds nothing and
only fills owners and dates, which lowers owner accuracy from 0.958 to 0.760. With the opt-in threshold
it adds false positives and F1 falls to 0.817. Rules-only stays the default.

## Error analysis

TEST errors, from `results/report.md`:

- **Missed:** "Look at the terms" (speaker turns), "Look at the release checklist" (Zoom), "Check if
  the reserved instances expire soon" (Teams), "Rewrite the headline" (WebVTT). Three of the four use
  soft verbs ("look at", "check if").
- **Spurious:** "Get going with the quarterly numbers" (speaker turns), "Rewrite it" (WebVTT, a
  pronoun-only restatement of an earlier item).

DEV errors left after tuning: "Own the vendor shortlist" (phrased as "I can own", missed), "Test the
export flow" (ASR, a sentence right after a name), and a Teams sentence about reopening an incident
(kept as a task, spurious).

## Limits

- Rules are English-only and tuned on synthetic text. Real transcripts with more disfluency, overlap or
  unusual phrasing will score lower.
- ASR owner resolution depends on names being capitalised and present in the attendee list.
- Soft verbs ("look at", "check if") account for several missed items, which costs recall.
- The LLM pass was run once on TEST, with one small model (Qwen3-1.7B). It lowered owner and due accuracy
  and did not raise F1. Larger models or a different prompt may behave differently; this was not tried.
- The AMI corpus was not used because its licence was not verified.

## Dependency and platform status

- Python 3.14 on Windows: `spacy` 3.8.18 with `en_core_web_sm` 3.8.0 loads and runs. The tests and
  evaluation were run with spaCy enabled.
- Linux runtime: not verified. Wheel resolution for spaCy 3.8.18 on Linux was checked with
  `uv pip compile --only-binary`, but the model was not loaded on Linux. CI runs on Linux, so the
  first CI run will show whether it loads there.
- `--no-nlp` runs the rules without spaCy, so the tests and CLI still work without the model.

## Where this is actually used

This is a practice project and is not deployed for anyone. Commercial meeting tools do the same job
from recordings, and their own pages describe it:

- Otter: "Otter turns every meeting into a clear summary with decisions, action items, and insights."
  (otter.ai)
- Fireflies: "Get detailed notes, action items, and customized summaries instantly after every
  meeting." (fireflies.ai)
- Google Meet: "Gemini in Meet captures real-time notes and action items, so you can stay focused on
  the discussion." (workspace.google.com)
- Microsoft Teams: Copilot "suggests action items and answers any questions you have, in real time
  during or after a meeting", and "Transcription is required to use Copilot." (support.microsoft.com)

This project is a local, rules-first alternative for the same input. It reads a transcript export, a
set of captions or typed notes, and each item keeps the line it came from, so a reviewer can check it
quickly. It runs offline with no hosted service. The ICS output is for calendar and task apps that
read VTODO entries and is tested by a round trip through `icalendar`. The JSON output is for scripts
that post items elsewhere. The Markdown checklist is for a notes file that people edit. Someone who
wants to judge the rules can read `results/report.md` for the error list on the labelled text.

The LLM pass is off by default, and a non-loopback endpoint is refused unless the user passes
`--allow-remote`.

## Security review

Findings from a review of the extractor, with the fix and the test that covers it. Tests are in
`tests/test_security.py` unless noted.

- **Quadratic dedupe (MED).** Items are bucketed by owner and due date before fuzzy matching, and a
  document with more than 2000 candidate items is refused. Tests: `test_dedupe_only_merges_within_the_same_owner_and_due`,
  `test_document_over_the_item_cap_is_a_clean_input_error`.
- **Markdown and HTML in output (MED).** Task, owner and heading text pass through an escaper that
  neutralises Markdown and HTML. JSON uses `ensure_ascii`. Tests: `test_html_and_markdown_in_tasks_are_neutralised`,
  `test_owner_and_heading_are_escaped_too`, `test_json_output_is_ascii_and_escapes_control_characters`.
- **Dates (LOW).** Years outside 1900 to 2200 are a clean input error for `--meeting-date` and date
  headers, and relative dates that overflow are left unresolved. Tests: `test_meeting_date_outside_the_sane_range_is_rejected`,
  `test_date_header_outside_the_range_is_rejected_cleanly`, `test_relative_due_overflow_is_unresolved_not_a_crash`,
  `test_meeting_date_overflow_guard_in_extract_options`.
- **LLM responses (LOW).** Deep JSON nesting, non-string content and lone surrogates are rejected or
  dropped. The response size is capped, and server error text is sanitised and truncated. Tests:
  `test_deeply_nested_json_reply_is_an_llm_error`, `test_non_string_content_is_an_llm_error`,
  `test_lone_surrogate_in_a_task_is_dropped_before_any_writer_sees_it`,
  `test_server_error_text_is_sanitised_in_the_backend`, and `test_llm.py` for the size cap.
- **Output files (LOW).** Writes are atomic. Text output inside an input directory is refused, an
  existing file needs `--force`, and an output that is an input file is always refused. Tests:
  `test_text_output_inside_an_input_directory_is_refused`, `test_existing_output_needs_force`,
  `test_output_that_is_an_input_file_is_refused_even_with_force`,
  `test_failed_atomic_write_leaves_the_target_untouched`, `test_cli_refuses_to_overwrite_without_force`.
- **Prompt injection through the LLM (LOW to MED).** A task must share at least half its words with
  the cited line, owners are capped at 80 characters and must be on the roster or in the text, and
  due dates may be at most two years after the meeting. Added items get confidence 0.35 and are
  tagged "LLM-added" in Markdown and ICS. Tests: `test_ungrounded_tasks_and_overlong_owners_are_dropped_or_cleared`,
  `test_llm_added_items_need_an_explicit_opt_in`, `test_llm_items_are_tagged_in_markdown_and_ics`.
- **Endpoint handling (hardening).** The endpoint is parsed once; userinfo and backslashes are refused,
  and the URL is rebuilt without its fragment. Proxies are ignored and redirects are not followed. The
  response has a wall-clock deadline, but no test exercises it. Only the scheme and host:port appear
  in messages. Tests: `test_endpoint_with_userinfo_or_backslash_is_refused`,
  `test_loopback_endpoint_is_rebuilt_without_fragment`, `test_environment_proxies_are_not_used`,
  `test_redirects_are_not_followed`.
- **Input caps and display (hardening).** At most 500 files and 20 MB in total are read. Bidi and
  zero-width characters are stripped, and error messages are sanitised before printing. Gold entries
  must stay inside the corpus folder. Tests: `test_file_count_and_aggregate_size_are_capped`,
  `test_bidi_and_zero_width_characters_are_stripped`, `test_error_messages_are_printed_safely`,
  `test_gold_entries_must_stay_inside_the_corpus`.

Known gap: when stdin (`-`) is mixed with file paths, stdin documents now come before file documents,
so the output order can differ from the argument order. The extracted items are unchanged.
