import pytest
from pivot_tool.spec import (
    SpecError,
    build_spec,
    parse_dim,
    parse_filter,
    parse_group,
    parse_show_as,
    parse_value,
)


def test_dimensions():
    assert parse_dim("education") == {"field": "education"}
    assert parse_dim("when:month") == {"field": "when", "group": {"kind": "month"}}
    assert parse_dim("age:bin=10") == {
        "field": "age",
        "group": {"kind": "bin", "width": 10.0},
    }
    assert parse_dim("age:bin=5@2.5")["group"] == {
        "kind": "bin",
        "width": 5.0,
        "start": 2.5,
    }
    assert parse_dim("x:edges=18,30,65")["group"] == {
        "kind": "edges",
        "edges": [18.0, 30.0, 65.0],
    }
    assert parse_dim("x:quantile=4")["group"] == {"kind": "quantile", "n": 4}
    assert parse_dim('{"field": "a", "hide_blank": true}') == {
        "field": "a",
        "hide_blank": True,
    }


@pytest.mark.parametrize(
    "bad",
    [
        "",
        ":month",
        "x:decade",
        "x:bin=",
        "x:bin=ten",
        "x:edges=",
        "{",
        '{"nofield": 1}',
        "x:month=3",
    ],
)
def test_bad_dimensions(bad):
    with pytest.raises(SpecError):
        parse_dim(bad)


@pytest.mark.parametrize(
    "group",
    [
        "year",
        "quarter",
        "month",
        "year_month",
        "year_quarter",
        "weekday",
        "day",
        "hour",
    ],
)
def test_every_simple_group(group):
    assert parse_group(group) == {"kind": group}


def test_values():
    assert parse_value("mean(hours)") == {
        "show_as": {"kind": "normal"},
        "agg": "mean",
        "field": "hours",
    }
    v = parse_value("count() as people @pct_row")
    assert v == {"show_as": {"kind": "pct_row"}, "label": "people", "agg": "count"}
    assert parse_value("weighted_mean(price;units)")["weight"] == "units"
    assert parse_value("count(product) as with product")["label"] == "with product"
    assert parse_value("calc: gain / people as per person") == {
        "show_as": {"kind": "normal"},
        "label": "per person",
        "calc": "gain / people",
    }
    assert parse_value("sum(x) @running:rows")["show_as"] == {
        "kind": "running",
        "along": "rows",
    }
    assert parse_value("sum(x) @rank_asc:cols")["show_as"] == {
        "kind": "rank",
        "along": "cols",
        "desc": False,
    }


@pytest.mark.parametrize(
    "bad",
    [
        "sum",
        "sum(x",
        "median of x",
        "bogus(x)",
        "sum(x) @nonsense",
        "sum(x) @diff",
        "sum(x) @diff:sideways",
    ],
)
def test_bad_values(bad):
    with pytest.raises(SpecError):
        parse_value(bad)


def test_show_as_names():
    for name in (
        "normal",
        "pct_grand",
        "pct_row",
        "pct_col",
        "pct_parent_row",
        "pct_parent_col",
        "index",
    ):
        assert parse_show_as(name) == {"kind": name}
    assert parse_show_as("pct_diff:cols") == {"kind": "pct_diff", "along": "cols"}


def test_filters():
    assert parse_filter("sex=Male") == {"field": "sex", "op": "eq", "value": "Male"}
    assert parse_filter("age>=30") == {"field": "age", "op": "ge", "value": 30}
    assert parse_filter("price < 2.5") == {"field": "price", "op": "lt", "value": 2.5}
    assert parse_filter("x != y") == {"field": "x", "op": "ne", "value": "y"}
    assert parse_filter("income in <=50K|>50K") == {
        "field": "income",
        "op": "in",
        "values": ["<=50K", ">50K"],
    }
    assert parse_filter("income !in <=50K") == {
        "field": "income",
        "op": "not_in",
        "values": ["<=50K"],
    }
    assert parse_filter("age between 20..30") == {
        "field": "age",
        "op": "between",
        "values": [20, 30],
    }
    assert parse_filter("when between 2024-01-01..2024-12-31")["values"] == [
        "2024-01-01",
        "2024-12-31",
    ]
    assert parse_filter("name contains smith") == {
        "field": "name",
        "op": "contains",
        "value": "smith",
    }
    assert parse_filter("occupation is null")["op"] == "is_null"
    assert parse_filter("occupation not null")["op"] == "not_null"


@pytest.mark.parametrize(
    "bad", ["nonsense", "x in", "x between 5", "x between ..5", "=5"]
)
def test_bad_filters(bad):
    with pytest.raises(SpecError):
        parse_filter(bad)


def test_build_spec_assembles_everything_and_applies_sort_and_top_to_the_first_row_field():
    spec = build_spec(
        ["a", "b:month"],
        ["c"],
        ["sum(x)"],
        ["x>1"],
        subtotals=True,
        totals_row=False,
        sort="value:sum(x):asc",
        top="3:sum(x)",
    )
    assert spec["options"] == {
        "subtotals": True,
        "totals_row": False,
        "totals_col": True,
    }
    assert spec["rows"][0]["sort"] == {
        "by": "value",
        "measure": "sum(x)",
        "desc": False,
    }
    assert spec["rows"][0]["top"] == {"n": 3, "measure": "sum(x)", "desc": True}
    assert "sort" not in spec["rows"][1]
    assert build_spec(["a"], sort="label:desc")["rows"][0]["sort"] == {
        "by": "label",
        "desc": True,
    }
    assert build_spec(["a"], sort="value:count")["rows"][0]["sort"] == {
        "by": "value",
        "measure": "count",
        "desc": True,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sort": "label"},
        {"top": "3:x"},
        {"rows": ["a"], "sort": "sideways"},
        {"rows": ["a"], "sort": "value:"},
        {"rows": ["a"], "top": "x:y"},
        {"rows": ["a"], "top": "3"},
    ],
)
def test_build_spec_rejects_bad_sort_and_top(kwargs):
    with pytest.raises(SpecError):
        build_spec(**kwargs)
