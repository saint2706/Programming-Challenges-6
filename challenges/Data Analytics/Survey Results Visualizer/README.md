# Survey Results Visualizer

**Category:** Data Analytics
**Difficulty:** B

**Status:** Implemented (Python)

Likert-scale charts and word clouds for open-ended answers, from a survey CSV
(one row per respondent) to one self-contained HTML report.

## What it does

- **Classifies every column automatically**, then lets you override it. Each
  column's non-blank cells go through these rules in order:
  1. no non-blank cells: `blank`
  2. at least 90% are agreement labels (`Strongly agree` ... `Strongly disagree`), or at least
     90% are integers 1-7 with at least 3 distinct values: **Likert item**
  3. average of 4 or more words per answer: **open-ended text**
  4. 12 or fewer distinct values: **categorical** (demographics such as department or tenure)
  5. anything else (ids, timestamps, ages): ignored
- **Scale detection.** Label columns are 7-point if any `Somewhat`/`Slightly` label appears, else
  5-point. Numeric columns are 5-point if the largest value is 5 or less, else 7-point. `--scale`
  forces either. Labels are case/whitespace-insensitive and accept `Neither agree nor disagree`
  as a neutral alias.
- **Reverse-coded items.** `--reverse` flips the scale (`"I often think about leaving"`) so
  "agree" always means the good direction across every chart and statistic.
- **Diverging stacked bars** (Plotly): neutral is centered on zero and split half/half, disagree
  extends left, agree extends right; items sorted by **net agreement** (top-2-box minus
  bottom-2-box). 5- and 7-point items get separate charts because their bands differ.
- **Item summary table:** n, skipped, mean, top-2-box %, bottom-2-box %, net.
- **Optional breakdown** (`--breakdown department`): grouped top-2-box bars plus a mean-score
  table per group with per-cell n. The 12 largest groups are shown; blank group values become
  `(missing)`.
- **Open-ended answers:** a stopword-filtered word cloud (hand-built inline SVG), a top-terms
  table, and the first few verbatim responses.
- **Nothing is dropped silently.** Unrecognized answers (`Maybe`, `9` on a 5-point scale, `2.5`)
  are excluded from the charts but listed under "Data notes" with their counts. Blank cells are
  reported as "skipped". A text column with no answers still renders, with a note.

## Design notes

**Word cloud layout (no image libraries, no randomness).** Font size is
`14 + 42 * sqrt((count - min) / (max - min))` px; the square root stops one dominant word from
shrinking everything else into illegibility. Words are placed in descending count order (ties
alphabetical). Each starts at the canvas center and walks an elliptical Archimedean spiral; the
first spot whose padded bounding box sits inside the canvas and overlaps no placed word wins.
Boxes are estimated (0.58em per glyph wide, 0.9em tall) rather than measured, which is close
enough for sans-serif text and keeps the layout a pure function of the input, so the same CSV
gives a byte-identical report. Words that never fit are counted and reported.

**Diverging bars with plotly.** `barmode="relative"` stacks positive and negative values
separately, so the neutral band is emitted as two traces (`-n/2` and `+n/2`) sharing one legend
entry, and the tick labels are rewritten to absolute percentages.

**XSS.** Survey text is attacker-controlled. Every respondent-derived string (title, column
names, group names, words, sample responses) goes through `html.escape` before reaching the
page. Column and group names are escaped before Plotly too, because Plotly renders a small HTML
subset in labels; Plotly's own JSON embedding escapes `<` and `&`. The tests use a
`<script>`/`onerror` payload in every one of those places.

**Plotly bundle inlined once** in `<head>` (about 4.8 MB), so the report needs no network. Chart
div ids are fixed rather than random uuids, so reports are reproducible.

**Known limits.** Small-integer numeric columns that are not really Likert (household size 1-5)
look identical to a Likert item; use `--ignore`. The neutral/agreement vocabulary is English
agreement wording only; other scales (satisfaction, frequency) work when the column is coded
numerically.

## Run it

```bash
cd "challenges/Data Analytics/Survey Results Visualizer"

# Everything auto-detected:
uv run python survey_visualizer.py sample_data/employee_engagement.csv -o report.html

# With reverse-coding, a breakdown and a title, via flags or a JSON config:
uv run python survey_visualizer.py sample_data/employee_engagement.csv \
    --reverse "I often think about leaving the company" --breakdown department -o report.html
uv run python survey_visualizer.py sample_data/employee_engagement.csv \
    --config sample_data/config.json -o report.html

uv run pytest -q # 36 tests
```

Flags: `--likert`, `--reverse`, `--text`, `--ignore` (one or more column names), `--breakdown`,
`--scale {5,7}`, `--title`, `--config FILE`. Config keys are the same names (`likert`, `reverse`,
`text`, `ignore`, `breakdown`, `scale`, `title`); flags win over the config file.

Open the generated `.html` file in any browser; it is fully self-contained.

## Sample data

`sample_data/employee_engagement.csv` is 120 synthetic employees generated from a fixed seed:
five 5-point label items (one negatively worded), two 7-point items, one numeric 1-5 item,
department and tenure, two open-ended questions, scattered blanks and two stray `Maybe` answers.
`sample_data/config.json` shows the config form.

## Where this is actually used

This is the core of what SurveyMonkey, Qualtrics and Google Forms summary pages do: a
diverging Likert chart is the standard way to show agreement (as popularized by HR engagement
surveys and NPS-style pulse checks), reverse-coding is how validated instruments such as the
System Usability Scale are scored, and a term-frequency cloud is the quick first read of free-text
comments before anyone does proper coding.
