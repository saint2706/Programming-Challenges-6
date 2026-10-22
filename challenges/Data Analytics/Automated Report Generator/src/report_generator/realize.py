"""Findings to sentences, and the audit that every printed number is a declared fact.

Templates are plain strings with ``{fact}`` placeholders and contain no digits. A sentence can only
show a number by pasting a ``Fact`` text, so ``audit`` can check three things: the templates are
digit-free, each numeric fact's text still matches its raw value, and the finished sentence shows
exactly the numbers its facts declared (nothing doubled, nothing smuggled in).
"""

from __future__ import annotations

import re
from collections import Counter

from report_generator.findings import NUMERIC_KINDS, Finding, check_text

TEMPLATES: dict[str, str] = {
    "overview": "The table has {rows} {row_label} and {cols} columns.",
    "overview.time": (
        "The table has {rows} {row_label} and {cols} columns, covering {start} to {end} (by {time_col})."
    ),
    "trend": "The {what} {verb} {change} between {start} and {end}, from about {start_level} to {end_level}.",
    "trend.sa": (
        "After setting the seasonal pattern aside, the {what} {verb} {change} between {start} and {end},"
        " from about {start_level} to {end_level}."
    ),
    "season.moderate": (
        "A {cycle} cycle shapes the {what}: the high point comes in {peak} and the low in {trough}, a swing of"
        " about {swing} of the average level (cycles correlate at {rho}, across {cycles} cycles)."
    ),
    "season.strong": (
        "A clear {cycle} cycle shapes the {what}: the high point comes in {peak} and the low in {trough}, a swing"
        " of about {swing} of the average level (cycles correlate at {rho}, across {cycles} cycles)."
    ),
    "season.very_strong": (
        "A very regular {cycle} cycle shapes the {what}: the high point comes in {peak} and the low in {trough},"
        " a swing of about {swing} of the average level (cycles correlate at {rho}, across {cycles} cycles)."
    ),
    "shift.up": (
        "The {what} stepped up around {date}: the average over the {window} {unit} after was {after},"
        " against {before} before, {change} higher."
    ),
    "shift.down": (
        "The {what} stepped down around {date}: the average over the {window} {unit} after was {after},"
        " against {before} before, {change} lower."
    ),
    "anomaly.one": (
        "One {what} reading stands out: {date}, at {value} where about {expected} was expected,"
        " {deviation} {side} the norm."
    ),
    "anomaly.many": (
        "{n} {what} readings stand out as unusual; the largest is {date}, at {value} where about {expected}"
        " was expected, {deviation} {side} the norm."
    ),
    "latest": (
        "In the latest {unit} ({period}) the {what} came to {value}, {dir_prev} {chg_prev} on the {unit} before and"
        " {dir_year} {chg_year} on the same {unit} a year earlier."
    ),
    "latest.noyoy": (
        "In the latest {unit} ({period}) the {what} came to {value}, {dir_prev} {chg_prev} on the {unit} before."
    ),
    "concentration.dim": (
        "The {what} is concentrated in a few {dims}: {top} alone accounts for {share}, and the top {k} for"
        " {share_k} (of {n} {dims})."
    ),
    "concentration.entity": (
        "A small group of {entity} drives the {what}: the top {top_n} of {n} account for {share} of the total."
    ),
    "group.ratio": (
        "{measure} differs by {dim}: the median is highest for {hi} ({hi_med}) and lowest for {lo} ({lo_med}),"
        " about {gap} as much. The grouping explains roughly {share} of the variation in ranks."
    ),
    "group.diff": (
        "{measure} differs by {dim}: the median is highest for {hi} ({hi_med}) and lowest for {lo} ({lo_med})."
        " The grouping explains roughly {share} of the variation in ranks."
    ),
    "group2.ratio": (
        "{measure} is higher for {hi} than for {lo} (median {hi_med} against {lo_med}, about {gap} as much);"
        " the split explains roughly {share} of the variation in ranks."
    ),
    "group2.diff": (
        "{measure} is higher for {hi} than for {lo} (median {hi_med} against {lo_med});"
        " the split explains roughly {share} of the variation in ranks."
    ),
    "correlation.pos": "{a} and {b} rise and fall together ({strength}, rank correlation {rho}).",
    "correlation.neg": "{a} and {b} move in opposite directions ({strength}, rank correlation {rho}).",
    "distribution.right.share": (
        "{measure} is heavily right-skewed: the median is {median} but the mean is {mean}, and the largest one percent of"
        " values hold {top_share} of the total."
    ),
    "distribution.right": "{measure} is heavily right-skewed: the median is {median} but the mean is {mean}.",
    "distribution.left": "{measure} is heavily left-skewed: the median is {median} but the mean is {mean}.",
    "quality.missing": "{col} is missing in {share} of rows.",
    "quality.missing.many": (
        "{n_cols} columns are missing more than {limit} of their values; the worst is {col} ({share})."
    ),
    "quality.duplicates": "{n} {row_label} ({share}) are exact duplicates of another row.",
    "quality.gaps": "{n} of the {total} days in the period have no row.",
    "quality.negative": "{share} of {col} values are negative ({n} rows).",
    "quality.identity": "{total} equals {a} plus {b} in all {n} rows, so those columns carry the same information.",
}

DIGIT = re.compile(r"\d")
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
PLACEHOLDER = re.compile(r"\{(\w+)\}")


class AuditError(ValueError):
    """A sentence shows a number its facts do not declare."""


def sentence(finding: Finding) -> str:
    template = TEMPLATES[finding.template]
    missing = [k for k in PLACEHOLDER.findall(template) if k not in finding.facts]
    if missing:
        raise KeyError(f"{finding.template} needs facts {missing}")
    text = PLACEHOLDER.sub(lambda m: finding.facts[m.group(1)].text, template)
    return text


def realize(findings: list[Finding]) -> None:
    for f in findings:
        f.text = sentence(f)


def audit(finding: Finding) -> list[str]:
    """Problems with one finding's sentence; an empty list means every number is accounted for."""
    problems: list[str] = []
    template = TEMPLATES[finding.template]
    if DIGIT.search(template):
        problems.append(f"template {finding.template} contains a digit")
    used = PLACEHOLDER.findall(template)
    for key in used:
        fact = finding.facts[key]
        if not check_text(fact):
            problems.append(
                f"fact {key}: text {fact.text!r} does not match value {fact.value!r}"
            )
    if finding.text != sentence(finding):
        problems.append("the sentence is not the one its template and facts produce")
    expected: Counter[str] = Counter()
    rest = finding.text
    for key in used:
        fact = finding.facts[key]
        if fact.kind in NUMERIC_KINDS:
            expected.update(NUMBER.findall(fact.text))
        else:  # a name from the data (it may hold digits, like 'Q1 2012'): not a claim
            rest = re.sub(re.escape(fact.text), "", rest, count=1, flags=re.IGNORECASE)
    shown: Counter[str] = Counter(NUMBER.findall(rest))
    if shown != expected:
        problems.append(
            f"numbers shown {dict(shown)} differ from numbers declared {dict(expected)}"
        )
    return problems
