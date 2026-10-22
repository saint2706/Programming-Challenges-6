"""The typed findings a report is built from, and the facts (numbers) each one carries.

A finding never holds prose. It holds ``Fact`` objects: a raw value, and the exact text that will be
printed for it. The sentence realiser may only paste those texts in, so any number a reader sees can
be traced back to a raw value, and re-derived from the data by ``verify``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Fact kinds whose text is a number to be audited; "label" and "date" are names, not claims.
NUMERIC_KINDS = ("count", "num", "pct", "ratio", "p")


def fmt_count(v: float) -> str:
    return f"{round(v):,}"


def fmt_num(v: float) -> str:
    """Three-or-so significant figures, without scientific notation: 1,234,567 / 12.3 / 0.0421."""
    if float(v).is_integer():
        return f"{int(v):,}"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    if a == 0:
        return "0"
    return f"{v:.3g}"


def fmt_pct(v: float) -> str:
    """A fraction as a percentage: 0.034 -> '3.4%', 0.253 -> '25%'. Unsigned; prose gives direction."""
    a = abs(v) * 100
    return f"{a:.1f}%" if a < 10 else f"{a:,.0f}%"


def fmt_ratio(v: float) -> str:
    return f"{v:.1f}x" if v < 10 else f"{v:.0f}x"


def fmt_p(v: float) -> str:
    return "p < 0.001" if v < 0.001 else f"p = {v:.3f}"


FORMATTERS = {
    "count": fmt_count,
    "num": fmt_num,
    "pct": fmt_pct,
    "ratio": fmt_ratio,
    "p": fmt_p,
}


@dataclass(frozen=True)
class Fact:
    value: Any
    text: str
    kind: str

    def to_json(self) -> dict:
        v = self.value
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        return {"value": v, "text": self.text, "kind": self.kind}


def count(v: float) -> Fact:
    return Fact(round(v), fmt_count(v), "count")


def num(v: float) -> Fact:
    return Fact(float(v), fmt_num(v), "num")


def pct(v: float) -> Fact:
    return Fact(float(v), fmt_pct(v), "pct")


def ratio(v: float) -> Fact:
    return Fact(float(v), fmt_ratio(v), "ratio")


def pvalue(v: float) -> Fact:
    return Fact(float(v), fmt_p(v), "p")


def label(s: str) -> Fact:
    """A name taken from the data (a column, a category, a date): printed as is, never audited."""
    return Fact(str(s).strip(), str(s).strip(), "label")


def check_text(fact: Fact) -> bool:
    """Does the fact's printed text still match its raw value? (The audit calls this.)"""
    if fact.kind not in FORMATTERS:
        return True
    return FORMATTERS[fact.kind](fact.value) == fact.text


@dataclass
class Finding:
    id: str
    type: str
    section: str
    columns: tuple[str, ...]
    template: str  # which sentence pattern the realiser uses, e.g. "trend.up"
    facts: dict[str, Fact]
    score: (
        float  # 0..1 interestingness: effect size times evidence, used only for ranking
    )
    stats: dict[str, Any] = field(default_factory=dict)  # test statistics, not printed
    chart: dict | None = None
    text: str = ""
    detail: str = ""

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "section": self.section,
            "columns": list(self.columns),
            "template": self.template,
            "score": round(self.score, 4),
            "text": self.text,
            "detail": self.detail,
            "facts": {k: f.to_json() for k, f in self.facts.items()},
            "stats": {
                k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in self.stats.items()
            },
        }
