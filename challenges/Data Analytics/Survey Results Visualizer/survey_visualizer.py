"""Survey Results Visualizer: Likert charts + word clouds in one self-contained HTML report.

Run with:  uv run python survey_visualizer.py survey.csv -o report.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl

# --------------------------------------------------------------------------
# Likert vocabulary
# --------------------------------------------------------------------------

# Every agreement label maps to a rank on the 7-point scale. A 5-point scale
# is the subset {1, 2, 4, 6, 7} re-numbered 1..5 (FIVE_POINT_RANKS).
LABEL_RANKS: dict[str, int] = {
    "strongly disagree": 1,
    "disagree": 2,
    "somewhat disagree": 3,
    "slightly disagree": 3,
    "neutral": 4,
    "neither": 4,
    "neither agree nor disagree": 4,
    "somewhat agree": 5,
    "slightly agree": 5,
    "agree": 6,
    "strongly agree": 7,
}
SEVEN_ONLY_RANKS = {3, 5}
FIVE_POINT_RANKS = {1: 1, 2: 2, 4: 3, 6: 4, 7: 5}

SCALE_LABELS: dict[int, list[str]] = {
    5: ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
    7: [
        "Strongly disagree",
        "Disagree",
        "Somewhat disagree",
        "Neutral",
        "Somewhat agree",
        "Agree",
        "Strongly agree",
    ],
}
SCALE_COLORS: dict[int, list[str]] = {
    5: ["#b2182b", "#ef8a62", "#d9d9d9", "#67a9cf", "#2166ac"],
    7: ["#b2182b", "#d6604d", "#f4a582", "#d9d9d9", "#92c5de", "#4393c3", "#2166ac"],
}

# Only applied to Likert columns: a blank-ish cell there means "skipped".
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--"}

AUTO_LIKERT_MIN_VALID = 0.9  # share of non-missing cells that must parse
AUTO_NUMERIC_MIN_DISTINCT = 3
OPEN_ENDED_MIN_AVG_WORDS = 4.0
CATEGORICAL_MAX_DISTINCT = 12
MAX_BREAKDOWN_GROUPS = 12
TOP_TERMS = 40
SAMPLE_RESPONSES = 5

_STOPWORD_TEXT = """
    a about above after again all also am an and any are aren't as at be because been
    before being below between both but by can can't cannot could couldn't did didn't do
    does doesn't doing don't down during each few for from further get got had hadn't has
    hasn't have haven't having he her here hers herself him himself his how i i'd i'll i'm
    i've if in into is isn't it it's its itself just let's me more most mustn't my myself
    no nor not of off on once only or other ought our ours ourselves out over own really
    same shan't she should shouldn't so some such than that that's the their theirs them
    themselves then there there's these they they'd they'll they're they've this those
    through to too under until up very was wasn't we we'd we'll we're we've were weren't
    what what's when where which while who whom why will with won't would wouldn't you
    you'd you'll you're you've your yours yourself yourselves much many lot lots one
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())

TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class LikertItem:
    name: str
    scale: int
    mode: str  # "labels" or "numbers"
    reverse: bool
    counts: list[int]  # index 0 == scale point 1
    missing: int
    unknown: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return sum(self.counts)

    def pct(self, index: int) -> float:
        return 100.0 * self.counts[index] / self.n if self.n else 0.0

    @property
    def mean(self) -> float | None:
        if not self.n:
            return None
        return sum((i + 1) * c for i, c in enumerate(self.counts)) / self.n

    @property
    def top2_pct(self) -> float:
        return self.pct(self.scale - 1) + self.pct(self.scale - 2)

    @property
    def bottom2_pct(self) -> float:
        return self.pct(0) + self.pct(1)

    @property
    def neutral_pct(self) -> float:
        return self.pct((self.scale - 1) // 2)

    @property
    def net(self) -> float:
        """Net agreement: top-2-box % minus bottom-2-box %."""
        return self.top2_pct - self.bottom2_pct


@dataclass
class TextSummary:
    name: str
    responses: int
    blank: int
    total_tokens: int
    terms: list[tuple[str, int]]
    samples: list[str]


@dataclass
class ColumnRoles:
    likert: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # name -> {"reverse": bool}
    text: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    blank: list[str] = field(default_factory=list)


@dataclass
class SurveyAnalysis:
    title: str
    respondents: int
    roles: ColumnRoles
    items: list[LikertItem]
    texts: list[TextSummary]
    breakdown_column: str | None
    breakdown: dict[
        str, dict[str, dict[str, float | int | None]]
    ]  # group -> item -> stats
    breakdown_groups_omitted: int = 0
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Parsing and detection
# --------------------------------------------------------------------------


def read_survey(path: Path) -> pl.DataFrame:
    """Read every cell as a string so labels and numeric codes share one path."""
    return pl.read_csv(path, infer_schema=False, encoding="utf8-lossy")


def normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold()).rstrip(".!")


def is_missing(value: str | None) -> bool:
    return value is None or normalize_label(value) in MISSING_TOKENS


def _numeric_code(value: str) -> int | None:
    try:
        number = float(value.strip())
    except ValueError:
        return None
    if math.isfinite(number) and number == int(number):
        return int(number)
    return None


def score_column(
    raw: list[str | None], scale_override: int | None = None
) -> tuple[str, int, list[int | None], Counter[str], int]:
    """Convert raw cells to scale points.

    Returns (mode, scale, codes, unknown, missing). ``codes`` is aligned with
    ``raw``: 1..scale for valid cells, None for missing/unknown cells.
    ``unknown`` counts each unrecognized non-blank cell verbatim.
    """
    present = [normalize_label(v) for v in raw if not is_missing(v)]
    label_hits = sum(1 for v in present if v in LABEL_RANKS)
    numeric_hits = sum(1 for v in present if _numeric_code(v) is not None)
    mode = "labels" if label_hits >= numeric_hits else "numbers"

    if mode == "labels":
        ranks = [LABEL_RANKS[v] for v in present if v in LABEL_RANKS]
        scale = scale_override or (
            7 if any(r in SEVEN_ONLY_RANKS for r in ranks) else 5
        )
    else:
        maxima = [c for v in present if (c := _numeric_code(v)) is not None]
        scale = scale_override or (5 if max(maxima, default=5) <= 5 else 7)

    codes: list[int | None] = []
    unknown: Counter[str] = Counter()
    missing = 0
    for value in raw:
        if is_missing(value):
            missing += 1
            codes.append(None)
            continue
        assert value is not None
        code: int | None
        if mode == "labels":
            rank = LABEL_RANKS.get(normalize_label(value))
            if rank is None:
                code = None
            elif scale == 7:
                code = rank
            else:
                code = FIVE_POINT_RANKS.get(
                    rank
                )  # 'somewhat' labels don't exist on 5-pt
        else:
            number = _numeric_code(value)
            code = number if number is not None and 1 <= number <= scale else None
        if code is None:
            unknown[value.strip()] += 1
        codes.append(code)
    return mode, scale, codes, unknown, missing


def _looks_likert(raw: list[str | None]) -> bool:
    present = [normalize_label(v) for v in raw if not is_missing(v)]
    if not present:
        return False
    label_share = sum(1 for v in present if v in LABEL_RANKS) / len(present)
    if label_share >= AUTO_LIKERT_MIN_VALID:
        return True
    codes = [_numeric_code(v) for v in present]
    valid = [c for c in codes if c is not None and 1 <= c <= 7]
    return (
        len(valid) / len(present) >= AUTO_LIKERT_MIN_VALID
        and len(set(valid)) >= AUTO_NUMERIC_MIN_DISTINCT
    )


def _avg_words(values: list[str]) -> float:
    return sum(len(v.split()) for v in values) / len(values)


def detect_roles(
    df: pl.DataFrame,
    *,
    likert: list[str] | None = None,
    reverse: list[str] | None = None,
    text: list[str] | None = None,
    ignore: list[str] | None = None,
) -> ColumnRoles:
    """Assign every column a role. Explicit lists always beat auto-detection.

    Auto rules, applied per column to its non-blank cells:
      1. none non-blank                       -> blank
      2. >=90% are agreement labels, or >=90% are integers in 1..7 with at
         least 3 distinct values                -> likert
      3. average >= 4 words per answer        -> text (open-ended)
      4. <= 12 distinct values                -> categorical (demographic)
      5. anything else (ids, timestamps, ...) -> ignored
    """
    likert, reverse, text, ignore = (
        likert or [],
        reverse or [],
        text or [],
        ignore or [],
    )
    for group in (likert, reverse, text, ignore):
        for name in group:
            if name not in df.columns:
                raise ValueError(
                    f"column {name!r} not found in CSV (columns: {df.columns})"
                )
    overlap = set(likert) & set(text)
    if overlap:
        raise ValueError(f"columns listed as both likert and text: {sorted(overlap)}")

    roles = ColumnRoles()
    for name in df.columns:
        raw = df[name].to_list()
        nonblank = [v.strip() for v in raw if v is not None and v.strip()]
        reverse_flag = name in reverse
        if name in ignore:
            roles.ignored.append(name)
        elif name in likert or name in reverse:
            roles.likert[name] = {"reverse": reverse_flag}
        elif name in text:
            roles.text.append(name)
        elif not nonblank:
            roles.blank.append(name)
        elif _looks_likert(raw):
            roles.likert[name] = {"reverse": False}
        elif _avg_words(nonblank) >= OPEN_ENDED_MIN_AVG_WORDS:
            roles.text.append(name)
        elif len(set(nonblank)) <= CATEGORICAL_MAX_DISTINCT:
            roles.categorical.append(name)
        else:
            roles.ignored.append(name)
    return roles


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, minus stopwords and anything under 3 letters."""
    tokens = TOKEN_RE.findall(text.casefold().replace("’", "'"))
    return [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]


def summarize_text(
    name: str, values: list[str | None], top: int = TOP_TERMS
) -> TextSummary:
    answers = [v.strip() for v in values if v is not None and v.strip()]
    counts: Counter[str] = Counter()
    for answer in answers:
        counts.update(tokenize(answer))
    # Ties break alphabetically so output is deterministic.
    terms = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return TextSummary(
        name=name,
        responses=len(answers),
        blank=len(values) - len(answers),
        total_tokens=sum(counts.values()),
        terms=terms,
        samples=answers[:SAMPLE_RESPONSES],
    )


def analyze_item(
    name: str, raw: list[str | None], reverse: bool = False, scale: int | None = None
) -> tuple[LikertItem, list[int | None]]:
    mode, scale, codes, unknown, missing = score_column(raw, scale)
    if reverse:
        codes = [None if c is None else scale + 1 - c for c in codes]
    counts = [0] * scale
    for c in codes:
        if c is not None:
            counts[c - 1] += 1
    item = LikertItem(name, scale, mode, reverse, counts, missing, dict(unknown))
    return item, codes


def analyze_survey(
    df: pl.DataFrame,
    *,
    likert: list[str] | None = None,
    reverse: list[str] | None = None,
    text: list[str] | None = None,
    ignore: list[str] | None = None,
    breakdown: str | None = None,
    scale: int | None = None,
    title: str = "Survey Results",
) -> SurveyAnalysis:
    if scale is not None and scale not in SCALE_LABELS:
        raise ValueError("scale must be 5 or 7")
    if breakdown is not None and breakdown not in df.columns:
        raise ValueError(f"breakdown column {breakdown!r} not found in CSV")

    roles = detect_roles(df, likert=likert, reverse=reverse, text=text, ignore=ignore)
    if breakdown in roles.likert:
        raise ValueError(
            f"breakdown column {breakdown!r} is a Likert item, not a demographic"
        )

    items: list[LikertItem] = []
    codes_by_item: dict[str, list[int | None]] = {}
    for name, opts in roles.likert.items():
        item, codes = analyze_item(name, df[name].to_list(), opts["reverse"], scale)
        items.append(item)
        codes_by_item[name] = codes

    texts = [summarize_text(name, df[name].to_list()) for name in roles.text]

    groups: dict[str, dict[str, dict[str, float | int | None]]] = {}
    omitted = 0
    if breakdown is not None and items:
        labels = [
            "(missing)" if v is None or not v.strip() else v.strip()
            for v in df[breakdown].to_list()
        ]
        sizes = Counter(labels)
        ordered = sorted(sizes, key=lambda g: (-sizes[g], g))
        kept = ordered[:MAX_BREAKDOWN_GROUPS]
        omitted = len(ordered) - len(kept)
        for group in kept:
            rows = [i for i, label in enumerate(labels) if label == group]
            groups[group] = {}
            for item in items:
                sub = [codes_by_item[item.name][i] for i in rows]
                valid = [c for c in sub if c is not None]
                if valid:
                    top2 = 100.0 * sum(c >= item.scale - 1 for c in valid) / len(valid)
                    mean: float | None = sum(valid) / len(valid)
                else:
                    top2, mean = 0.0, None
                groups[group][item.name] = {"n": len(valid), "mean": mean, "top2": top2}

    analysis = SurveyAnalysis(
        title=title,
        respondents=df.height,
        roles=roles,
        items=items,
        texts=texts,
        breakdown_column=breakdown if groups else None,
        breakdown=groups,
        breakdown_groups_omitted=omitted,
    )
    analysis.notes = _collect_notes(analysis, breakdown)
    return analysis


def _collect_notes(analysis: SurveyAnalysis, breakdown: str | None) -> list[str]:
    notes: list[str] = []
    if breakdown and not analysis.breakdown:
        notes.append(
            f"Breakdown by {breakdown!r} skipped: no Likert items to break down."
        )
    for item in analysis.items:
        bad = sum(item.unknown.values())
        if bad:
            shown = ", ".join(
                f"{v!r} x{c}" for v, c in sorted(item.unknown.items())[:5]
            )
            notes.append(
                f"{item.name!r}: {bad} response(s) not recognized on the "
                f"{item.scale}-point scale and excluded from the charts ({shown})."
            )
    for text in analysis.texts:
        if text.responses == 0:
            notes.append(f"{text.name!r}: no open-ended responses.")
    if analysis.breakdown_groups_omitted:
        notes.append(
            f"Breakdown shows the {MAX_BREAKDOWN_GROUPS} largest groups; "
            f"{analysis.breakdown_groups_omitted} smaller group(s) omitted."
        )
    return notes


# --------------------------------------------------------------------------
# Word cloud (hand-built SVG, deterministic)
# --------------------------------------------------------------------------

CLOUD_WIDTH, CLOUD_HEIGHT = 800, 380
CLOUD_MIN_FONT, CLOUD_MAX_FONT = 14, 56
CLOUD_PALETTE = ["#2166ac", "#b2182b", "#4d9221", "#7b3294", "#e08214", "#01665e"]


def build_word_cloud(
    terms: list[tuple[str, int]], width: int = CLOUD_WIDTH, height: int = CLOUD_HEIGHT
) -> tuple[str, int]:
    """Lay out ``terms`` as SVG text. Returns (svg, number_of_terms_that_did_not_fit).

    Algorithm (no randomness, so identical input gives identical output):
      1. Font size = MIN + (MAX-MIN) * sqrt((count-cmin)/(cmax-cmin)); sqrt keeps
         one dominant word from shrinking everything else to illegibility.
      2. Words are placed in descending count order (ties alphabetical).
      3. Each word starts at the canvas centre and walks an elliptical
         Archimedean spiral; the first position whose padded bounding box lies
         inside the canvas and overlaps no already-placed box wins.
      4. Boxes are estimated (0.58em per glyph wide, 0.9em tall), which is
         close enough for sans-serif text without measuring real glyphs.
    """
    if not terms:
        return "", 0
    counts = [c for _, c in terms]
    cmin, cmax = min(counts), max(counts)
    placed: list[tuple[float, float, float, float]] = []  # (x0, y0, x1, y1)
    parts: list[str] = []
    skipped = 0
    pad = 3.0
    for rank, (word, count) in enumerate(terms):
        frac = 0.5 if cmax == cmin else math.sqrt((count - cmin) / (cmax - cmin))
        size = CLOUD_MIN_FONT + (CLOUD_MAX_FONT - CLOUD_MIN_FONT) * frac
        w, h = len(word) * size * 0.58 + 2 * pad, size * 0.9 + 2 * pad
        position = None
        for step in range(6000):
            angle = 0.15 * step
            radius = 0.9 * angle
            cx = width / 2 + radius * math.cos(angle) * 1.7
            cy = height / 2 + radius * math.sin(angle)
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                continue
            if any(
                box[0] < p[2] and box[2] > p[0] and box[1] < p[3] and box[3] > p[1]
                for p in placed
            ):
                continue
            position = (cx, cy, box)
            break
        if position is None:
            skipped += 1
            continue
        cx, cy, box = position
        placed.append(box)
        color = CLOUD_PALETTE[rank % len(CLOUD_PALETTE)]
        safe = html.escape(word)
        parts.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="{size:.1f}" fill="{color}" '
            f'text-anchor="middle" dominant-baseline="central" font-weight="600">'
            f"<title>{safe}: {count}</title>{safe}</text>"
        )
    svg = (
        f'<svg class="cloud" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Word cloud of frequent terms" xmlns="http://www.w3.org/2000/svg">'
        + "".join(parts)
        + "</svg>"
    )
    return svg, skipped


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _plot_html(fig: go.Figure, div_id: str) -> str:
    # Explicit div ids (plotly defaults to a random uuid) keep reports byte-reproducible.
    chart = pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config={"displaylogo": False},
    )
    return f'<div class="chart">{chart}</div>'


def _label(name: str) -> str:
    # Plotly renders a small HTML subset in labels; escaping first neutralizes
    # any tags in respondent-controlled column names (plotly decodes entities).
    return html.escape(name)


def build_likert_figure(items: list[LikertItem]) -> go.Figure:
    """Diverging stacked bars for items that share one scale, best net agreement on top."""
    scale = items[0].scale
    labels, colors = SCALE_LABELS[scale], SCALE_COLORS[scale]
    mid = (scale - 1) // 2
    ordered = sorted(items, key=lambda i: (-i.net, i.name))
    names = [_label(i.name) for i in ordered]

    def trace(level: int, sign: float, half: bool) -> go.Bar:
        share = [i.pct(level) for i in ordered]
        x = [sign * (s / 2 if half else s) for s in share]
        return go.Bar(
            y=names,
            x=x,
            orientation="h",
            name=labels[level],
            legendgroup=labels[level],
            showlegend=not (half and sign > 0),
            marker_color=colors[level],
            customdata=share,
            hovertemplate="%{y}<br>"
            + labels[level]
            + ": %{customdata:.1f}%<extra></extra>",
        )

    fig = go.Figure()
    fig.add_trace(trace(mid, -1, True))
    for level in range(mid - 1, -1, -1):
        fig.add_trace(trace(level, -1, False))
    fig.add_trace(trace(mid, +1, True))
    for level in range(mid + 1, scale):
        fig.add_trace(trace(level, +1, False))

    fig.update_layout(
        barmode="relative",
        height=max(260, 60 * len(ordered) + 140),
        margin={"l": 20, "r": 20, "t": 30, "b": 60},
        xaxis={
            "title": "% of respondents",
            "ticksuffix": "%",
            "zeroline": True,
            "zerolinewidth": 2,
        },
        yaxis={"autorange": "reversed", "automargin": True},
        legend={"orientation": "h", "y": -0.2},
        template="plotly_white",
    )
    # Show absolute values on the axis even though the negative side is drawn leftwards.
    ticks = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
    fig.update_xaxes(
        tickvals=ticks,
        ticktext=[f"{abs(t)}%" for t in ticks],
        range=[-100, 100],
        title=None,
    )
    return fig


def build_breakdown_figure(analysis: SurveyAnalysis) -> go.Figure:
    fig = go.Figure()
    item_names = [i.name for i in analysis.items]
    for group, stats in analysis.breakdown.items():
        fig.add_trace(
            go.Bar(
                y=[_label(n) for n in item_names],
                x=[stats[n]["top2"] for n in item_names],
                name=f"{_label(group)} (n={max(int(stats[n]['n'] or 0) for n in item_names)})",
                orientation="h",
                hovertemplate="%{y}<br>top-2-box: %{x:.1f}%<extra>"
                + _label(group)
                + "</extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        height=max(
            300, 34 * len(item_names) * max(1, len(analysis.breakdown)) // 2 + 160
        ),
        margin={"l": 20, "r": 20, "t": 30, "b": 60},
        xaxis={"title": "Top-2-box % (agree + strongly agree)", "range": [0, 100]},
        yaxis={"autorange": "reversed", "automargin": True},
        legend={"orientation": "h", "y": -0.2},
        template="plotly_white",
    )
    return fig


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------

CSS = """
:root{--fg:#1d2733;--muted:#5b6673;--bg:#f6f7f9;--card:#fff;--line:#dde1e6;--warn:#8a5a00;--warnbg:#fff4d6}
@media (prefers-color-scheme:dark){:root{--fg:#e6e9ee;--muted:#9aa4b1;--bg:#12161c;--card:#1b2129;--line:#2d3641;--warn:#f0c674;--warnbg:#3a3120}}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:var(--bg);color:var(--fg);line-height:1.5}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}
h1{margin:0 0 4px}h2{margin:0 0 12px;font-size:1.2rem}
.sub{color:var(--muted);margin:0 0 20px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:18px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
.note{background:var(--warnbg);color:var(--warn);padding:8px 12px;border-radius:6px;margin:6px 0}
.cloud{width:100%;height:auto;max-height:420px}
blockquote{margin:8px 0;padding:6px 12px;border-left:3px solid var(--line);color:var(--muted)}
.muted{color:var(--muted)}
.chart{background:#fff;border-radius:8px;padding:8px;margin:8px 0}
"""


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _plotly_bundle() -> str:
    """Plotly's JS as inline <script> tags, emitted once and shared by every chart."""
    page = pio.to_html(go.Figure(), include_plotlyjs="inline", full_html=False)
    # Scripts 1-2 are the PlotlyConfig stub and the bundle; the 3rd draws the empty figure.
    scripts = re.findall(r"<script\b.*?</script\s*>", page, re.DOTALL | re.IGNORECASE)
    return "".join(scripts[:2])


def _summary_table(items: list[LikertItem]) -> str:
    rows = []
    for i in sorted(items, key=lambda i: (-i.net, i.name)):
        tag = " <span class='muted'>(reverse-coded)</span>" if i.reverse else ""
        rows.append(
            f"<tr><td>{html.escape(i.name)}{tag}</td><td>{i.scale}</td><td>{i.n}</td>"
            f"<td>{i.missing}</td><td>{_fmt(i.mean)}</td><td>{i.top2_pct:.1f}%</td>"
            f"<td>{i.bottom2_pct:.1f}%</td><td>{i.net:+.1f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Item</th><th>Scale</th><th>n</th><th>Skipped</th><th>Mean</th>"
        "<th>Top-2-box</th><th>Bottom-2-box</th><th>Net</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _breakdown_table(analysis: SurveyAnalysis) -> str:
    groups = list(analysis.breakdown)
    head = "".join(f"<th>{html.escape(g)}</th>" for g in groups)
    rows = []
    for item in analysis.items:
        cells = []
        for g in groups:
            s = analysis.breakdown[g][item.name]
            cells.append(
                f"<td>{_fmt(s['mean'])} <span class='muted'>(n={s['n']})</span></td>"
            )
        rows.append(f"<tr><td>{html.escape(item.name)}</td>{''.join(cells)}</tr>")
    return (
        f"<table><thead><tr><th>Mean score</th>{head}</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _text_section(summary: TextSummary) -> str:
    out = [f"<section><h2>Open-ended: {html.escape(summary.name)}</h2>"]
    out.append(
        f"<p class='sub'>{summary.responses} response(s), {summary.blank} blank, "
        f"{summary.total_tokens} meaningful words.</p>"
    )
    if not summary.terms:
        out.append(
            "<p class='muted'>Nothing to show: no meaningful words found.</p></section>"
        )
        return "".join(out)
    svg, skipped = build_word_cloud(summary.terms)
    out.append(svg)
    if skipped:
        out.append(
            f"<p class='muted'>{skipped} lower-ranked term(s) did not fit the cloud.</p>"
        )
    rows = "".join(
        f"<tr><td>{html.escape(w)}</td><td>{c}</td></tr>" for w, c in summary.terms[:15]
    )
    out.append(
        "<h3>Top terms</h3><table><thead><tr><th>Term</th><th>Count</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
    quotes = "".join(
        f"<blockquote>{html.escape(s)}</blockquote>" for s in summary.samples
    )
    out.append(f"<h3>Sample responses</h3>{quotes}</section>")
    return "".join(out)


def render_report(analysis: SurveyAnalysis) -> str:
    title = html.escape(analysis.title)
    body: list[str] = [
        f"<h1>{title}</h1>",
        (
            f"<p class='sub'>{analysis.respondents} respondents &middot; "
            f"{len(analysis.items)} Likert item(s) &middot; {len(analysis.texts)} "
            f"open-ended question(s)</p>"
        ),
    ]
    if analysis.notes:
        notes = "".join(
            f"<div class='note'>{html.escape(n)}</div>" for n in analysis.notes
        )
        body.append(f"<section><h2>Data notes</h2>{notes}</section>")

    if analysis.items:
        body.append("<section><h2>Likert responses</h2>")
        for scale in sorted({i.scale for i in analysis.items}):
            group = [i for i in analysis.items if i.scale == scale]
            body.append(
                f"<h3>{scale}-point items</h3>{_plot_html(build_likert_figure(group), f'likert-{scale}')}"
            )
        body.append("</section>")
        body.append(
            f"<section><h2>Item summary</h2>{_summary_table(analysis.items)}</section>"
        )
    if analysis.breakdown:
        column = html.escape(analysis.breakdown_column or "")
        body.append(
            f"<section><h2>Breakdown by {column}</h2>"
            f"{_plot_html(build_breakdown_figure(analysis), 'breakdown')}"
            f"{_breakdown_table(analysis)}</section>"
        )
    body.extend(_text_section(t) for t in analysis.texts)

    r = analysis.roles
    role_rows = [
        ("Likert items", ", ".join(r.likert)),
        ("Open-ended text", ", ".join(r.text)),
        ("Categorical", ", ".join(r.categorical)),
        ("Ignored", ", ".join(r.ignored)),
        ("Entirely blank", ", ".join(r.blank)),
    ]
    rows = "".join(
        f"<tr><td>{label}</td><td style='text-align:left'>{html.escape(cols) or '-'}</td></tr>"
        for label, cols in role_rows
    )
    body.append(
        f"<section><h2>How columns were interpreted</h2><table><tbody>{rows}</tbody></table></section>"
    )

    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{CSS}</style>{_plotly_bundle()}</head><body><main>"
        f"{''.join(body)}</main></body></html>"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

CONFIG_KEYS = {"likert", "reverse", "text", "ignore", "breakdown", "scale", "title"}


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("config must be a JSON object")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"unknown config key(s): {sorted(unknown)} (allowed: {sorted(CONFIG_KEYS)})"
        )
    return config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Turn a survey CSV into an HTML results report."
    )
    p.add_argument("csv", type=Path, help="survey CSV, one row per respondent")
    p.add_argument("-o", "--output", type=Path, default=Path("survey_report.html"))
    p.add_argument(
        "--config", type=Path, help="JSON file with the same options as the flags"
    )
    p.add_argument("--likert", nargs="+", help="force these columns to be Likert items")
    p.add_argument("--reverse", nargs="+", help="Likert columns to reverse-code")
    p.add_argument(
        "--text", nargs="+", help="force these columns to be open-ended text"
    )
    p.add_argument("--ignore", nargs="+", help="columns to skip entirely")
    p.add_argument("--breakdown", help="demographic column to split Likert results by")
    p.add_argument("--scale", type=int, choices=[5, 7], help="force the scale size")
    p.add_argument("--title", help="report title")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config) if args.config else {}
        options = {
            key: getattr(args, key)
            if getattr(args, key) is not None
            else config.get(key)
            for key in (
                "likert",
                "reverse",
                "text",
                "ignore",
                "breakdown",
                "scale",
                "title",
            )
        }
        if options["title"] is None:
            options["title"] = "Survey Results"
        analysis = analyze_survey(read_survey(args.csv), **options)
    except (ValueError, TypeError, OSError, pl.exceptions.PolarsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.write_text(render_report(analysis), encoding="utf-8")
    print(
        f"wrote {args.output} ({analysis.respondents} respondents, {len(analysis.items)} items)"
    )
    for note in analysis.notes:
        print(f"note: {note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
