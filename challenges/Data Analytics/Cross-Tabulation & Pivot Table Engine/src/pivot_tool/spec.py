"""The command line's little language for describing a pivot, and the spec dict it produces.

--rows education                    a field
--rows age:bin=10                   grouped: year, quarter, month, year_month, year_quarter,
--rows when:month                   weekday, day, hour, bin=WIDTH[@START], edges=1,2,3, quantile=N
--rows '{"field": "x", "top": {"n": 5, "measure": "count"}}'      the full form, as JSON

--values 'mean(hours_per_week)'
--values 'count() as people @pct_row'          show values as: pct_grand pct_row pct_col pct_parent_row
--values 'weighted_mean(price;units)'            pct_parent_col index diff:rows pct_diff:cols
--values 'calc: units / orders as per_order'     running:rows rank:cols rank_asc:rows

--filter 'sex=Male'  'age>=30'  'income in <=50K|>50K'  'age between 20..30'
--filter 'name contains smith'  'occupation is null'  'occupation not null'
"""

from __future__ import annotations

import json
import re
from typing import Any

AGGS = {
    "sum",
    "count",
    "distinct",
    "mean",
    "median",
    "min",
    "max",
    "std",
    "var",
    "first",
    "last",
    "mode",
    "weighted_mean",
}
SIMPLE_GROUPS = {
    "year",
    "quarter",
    "month",
    "year_month",
    "year_quarter",
    "weekday",
    "day",
    "hour",
}
SHOW_AS = {
    "normal",
    "pct_grand",
    "pct_row",
    "pct_col",
    "pct_parent_row",
    "pct_parent_col",
    "index",
}
ALONG = {"diff", "pct_diff", "running", "rank", "rank_asc"}


class SpecError(ValueError):
    """The command line described a pivot the engine cannot be asked for."""


def _number(text: str, what: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise SpecError(f"{what}: '{text}' is not a number") from exc


def parse_group(text: str) -> dict[str, Any]:
    name, _, arg = text.partition("=")
    if name in SIMPLE_GROUPS and not arg:
        return {"kind": name}
    if name == "bin" and arg:
        width, _, start = arg.partition("@")
        out: dict[str, Any] = {"kind": "bin", "width": _number(width, "bin width")}
        if start:
            out["start"] = _number(start, "bin start")
        return out
    if name == "edges" and arg:
        return {
            "kind": "edges",
            "edges": [_number(p, "edge") for p in arg.split(",") if p],
        }
    if name == "quantile" and arg:
        return {"kind": "quantile", "n": int(_number(arg, "quantile count"))}
    raise SpecError(
        f"unknown grouping '{text}': use one of {', '.join(sorted(SIMPLE_GROUPS))}, bin=WIDTH[@START], edges=A,B,C or quantile=N"
    )


def parse_dim(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{"):
        try:
            dim = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SpecError(f"bad JSON dimension: {exc}") from exc
        if "field" not in dim:
            raise SpecError('a JSON dimension needs a "field"')
        return dim
    field, _, group = text.partition(":")
    if not field:
        raise SpecError("a dimension needs a field name")
    dim: dict[str, Any] = {"field": field}
    if group:
        dim["group"] = parse_group(group)
    return dim


def parse_show_as(text: str) -> dict[str, Any]:
    name, _, axis = text.partition(":")
    if name in SHOW_AS and not axis:
        return {"kind": name}
    if name in ALONG:
        if axis not in ("rows", "cols"):
            raise SpecError(f"'{name}' needs an axis: {name}:rows or {name}:cols")
        if name == "rank_asc":
            return {"kind": "rank", "along": axis, "desc": False}
        return {"kind": name, "along": axis}
    raise SpecError(f"unknown 'show values as' '{text}'")


_AGG_CALL = re.compile(r"^([a-z_]+)\((.*)\)$")


def parse_value(text: str) -> dict[str, Any]:
    body = text.strip()
    show: dict[str, Any] = {"kind": "normal"}
    m = re.search(r"\s@([a-z_]+(?::(?:rows|cols))?)\s*$", body)
    if m:
        show = parse_show_as(m.group(1))
        body = body[: m.start()].rstrip()
    label = None
    m = re.search(r"\s+as\s+([^()]+?)\s*$", body)
    if m:
        label = m.group(1)
        body = body[: m.start()].rstrip()
    value: dict[str, Any] = {"show_as": show}
    if label:
        value["label"] = label
    if body.startswith("calc:"):
        value["calc"] = body[5:].strip()
        return value
    call = _AGG_CALL.match(body)
    if not call:
        raise SpecError(
            f"cannot read the value '{text}': write it as agg(field), e.g. sum(price) or count()"
        )
    agg, args = call.group(1), call.group(2).strip()
    if agg not in AGGS:
        raise SpecError(
            f"unknown aggregation '{agg}': use one of {', '.join(sorted(AGGS))}"
        )
    value["agg"] = agg
    field, _, weight = args.partition(";")
    if field.strip():
        value["field"] = field.strip()
    if weight.strip():
        value["weight"] = weight.strip()
    return value


def _literal(text: str) -> Any:
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


_COMPARE = re.compile(r"^\s*([^\s<>=!]+)\s*(>=|<=|!=|=|>|<)\s*(.+?)\s*$")
_WORDS = re.compile(
    r"^\s*(\S+)\s+(in|!in|between|contains|is null|not null)\s*(.*?)\s*$"
)
_OPS = {"=": "eq", "!=": "ne", ">": "gt", ">=": "ge", "<": "lt", "<=": "le"}


def parse_filter(text: str) -> dict[str, Any]:
    m = _WORDS.match(text)
    if m:
        field, word, rest = m.groups()
        if word == "in" or word == "!in":
            values = [_literal(p) for p in rest.split("|") if p.strip()]
            if not values:
                raise SpecError(
                    f"filter '{text}': give values after 'in', separated by |"
                )
            return {
                "field": field,
                "op": "in" if word == "in" else "not_in",
                "values": values,
            }
        if word == "between":
            lo, sep, hi = rest.partition("..")
            if not sep or not lo.strip() or not hi.strip():
                raise SpecError(f"filter '{text}': write between LOW..HIGH")
            return {
                "field": field,
                "op": "between",
                "values": [_literal(lo), _literal(hi)],
            }
        if word == "contains":
            return {"field": field, "op": "contains", "value": rest}
        return {"field": field, "op": "is_null" if word == "is null" else "not_null"}
    m = _COMPARE.match(text)
    if m:
        field, op, value = m.groups()
        return {"field": field, "op": _OPS[op], "value": _literal(value)}
    raise SpecError(
        f"cannot read the filter '{text}': try 'field=value', 'age>=30', 'x in a|b', 'x between 1..5', 'x contains t', 'x is null'"
    )


def build_spec(
    rows: list[str] | None = None,
    cols: list[str] | None = None,
    values: list[str] | None = None,
    filters: list[str] | None = None,
    *,
    subtotals: bool = False,
    totals_row: bool = True,
    totals_col: bool = True,
    sort: str | None = None,
    top: str | None = None,
) -> dict[str, Any]:
    """A pivot spec from the command line's pieces. ``sort`` and ``top`` apply to the first row field."""
    spec: dict[str, Any] = {
        "rows": [parse_dim(r) for r in rows or []],
        "cols": [parse_dim(c) for c in cols or []],
        "values": [parse_value(v) for v in values or []],
        "filters": [parse_filter(f) for f in filters or []],
        "options": {
            "subtotals": subtotals,
            "totals_row": totals_row,
            "totals_col": totals_col,
        },
    }
    if sort or top:
        if not spec["rows"]:
            raise SpecError(
                "--sort and --top apply to the first row field, but there are no --rows"
            )
        first = spec["rows"][0]
        if sort:
            how, _, rest = sort.partition(":")
            if how == "label":
                first["sort"] = {"by": "label", "desc": rest == "desc"}
            elif how == "value":
                measure, _, direction = rest.rpartition(":")
                if direction not in ("asc", "desc"):
                    measure, direction = rest, "desc"
                if not measure:
                    raise SpecError(
                        "--sort value:MEASURE[:asc|desc] needs a measure label"
                    )
                first["sort"] = {
                    "by": "value",
                    "measure": measure,
                    "desc": direction == "desc",
                }
            else:
                raise SpecError("--sort is label[:desc] or value:MEASURE[:asc|desc]")
        if top:
            n, _, measure = top.partition(":")
            if not n.isdigit() or not measure:
                raise SpecError("--top is N:MEASURE, e.g. 5:count")
            first["top"] = {"n": int(n), "measure": measure, "desc": True}
    return spec
