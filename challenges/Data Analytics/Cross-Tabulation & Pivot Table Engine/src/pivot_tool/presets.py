"""The two bundled datasets and the pivots that show off what the engine does with them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pivot_tool.paths import project_root


def count(label: str = "people", **show: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"agg": "count", "label": label}
    if show:
        out["show_as"] = show
    return out


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    file: str
    presets: list[dict[str, Any]] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return project_root() / "sample_data" / self.file


ADULT = Dataset(
    "adult",
    "UCI Adult census (48,842 people)",
    "adult.csv.gz",
    [
        {
            "name": "Who earns over 50K? By education",
            "description": "Rows are ordered by education level (education_num). 'share' is each row's percentage across the two income columns.",
            "spec": {
                "rows": [{"field": "education_num"}, {"field": "education"}],
                "cols": [{"field": "income"}],
                "values": [count("people"), count("share", kind="pct_row")],
            },
        },
        {
            "name": "Hours worked by age band and sex",
            "description": "Age grouped into bins of 10 years; the mean is of hours_per_week.",
            "spec": {
                "rows": [{"field": "age", "group": {"kind": "bin", "width": 10}}],
                "cols": [{"field": "sex"}],
                "values": [
                    {"agg": "mean", "field": "hours_per_week", "label": "mean hours"},
                    count("people"),
                ],
            },
        },
        {
            "name": "Workclass, then occupation, with subtotals",
            "description": "Only the top 4 occupations per workclass are shown, but each subtotal still covers every row of its workclass.",
            "spec": {
                "rows": [
                    {"field": "workclass"},
                    {
                        "field": "occupation",
                        "top": {"n": 4, "measure": "people"},
                        "sort": {"by": "value", "measure": "people", "desc": True},
                    },
                ],
                "cols": [{"field": "sex"}],
                "values": [count("people")],
                "options": {"subtotals": True, "subtotal_position": "bottom"},
            },
        },
        {
            "name": "Calculated field: capital gain per person",
            "description": "A formula over two aggregated measures, evaluated in every cell including the totals.",
            "spec": {
                "rows": [
                    {
                        "field": "marital_status",
                        "sort": {
                            "by": "value",
                            "measure": "gain per person",
                            "desc": True,
                        },
                    },
                ],
                "values": [
                    count("people"),
                    {"agg": "sum", "field": "capital_gain", "label": "gain"},
                    {"calc": "gain / people", "label": "gain per person"},
                ],
            },
        },
        {
            "name": "Is income tied to sex? (cross-tab statistics)",
            "description": "Chi-square, Cramer's V, odds ratio. The layer shows how much of the gap is explained by marital status.",
            "tab": "crosstab",
            "layer": "marital_status",
            "spec": {
                "rows": [{"field": "sex"}],
                "cols": [{"field": "income"}],
                "values": [count("people")],
            },
        },
    ],
)

UCB = Dataset(
    "berkeley",
    "UC Berkeley graduate admissions, fall 1973 (4,526 applicants)",
    "ucb_admissions.csv",
    [
        {
            "name": "Admission by sex, all departments",
            "description": "Pooled: men were admitted far more often than women.",
            "spec": {
                "rows": [{"field": "sex"}],
                "cols": [{"field": "outcome"}],
                "values": [count("applicants"), count("share", kind="pct_row")],
            },
        },
        {
            "name": "Admission by department and sex",
            "description": "Look at the share column inside each department: in most of them women were admitted at a higher rate.",
            "spec": {
                "rows": [{"field": "dept"}, {"field": "sex"}],
                "cols": [{"field": "outcome"}],
                "values": [count("applicants"), count("share", kind="pct_row")],
                "options": {"subtotals": True, "subtotal_position": "bottom"},
            },
        },
        {
            "name": "Simpson's paradox check",
            "description": "Women applied mostly to the departments that reject most applicants. Layered by department, the association reverses.",
            "tab": "crosstab",
            "layer": "dept",
            "spec": {
                "rows": [{"field": "sex"}],
                "cols": [{"field": "outcome"}],
                "values": [count("applicants")],
            },
        },
    ],
)

DATASETS = {d.key: d for d in (ADULT, UCB)}
