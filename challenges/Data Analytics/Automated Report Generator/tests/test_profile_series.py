"""Column roles, calendar grain and the series builder."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from helpers import bundled, events, planted
from report_generator.profile import profile_table
from report_generator.series import ROWS, build_series, choose_grain


def test_roles_on_the_planted_table():
    p = profile_table(planted())
    roles = {c.name: c.role for c in p.columns.values()}
    assert roles["date"] == "time" and p.time == "date" and p.native_grain == "day"
    assert roles["plan"] == "dimension"
    assert {n for n, r in roles.items() if r == "measure"} == {
        "visits",
        "temp",
        "score",
        "spend",
        "x",
        "y",
        "noise",
    }
    assert p.columns["visits"].agg == "sum" and p.columns["temp"].agg == "mean"
    assert not p.event_table


def test_roles_on_an_event_table():
    p = profile_table(events())
    assert p.event_table
    assert (
        p.columns["customer_id"].role == "entity"
        and p.columns["country"].role == "dimension"
    )
    assert p.columns["quantity"].agg == "sum" and p.columns["price"].agg == "mean"


def test_text_dates_are_parsed_and_overrides_work():
    frame = pl.DataFrame(
        {
            "when": [f"2021-{m:02d}-15" for m in range(1, 13)] * 3,
            "n": [(i * 7) % 11 + 20 for i in range(36)],
            "flag": [True, False] * 18,
        }
    )
    p = profile_table(frame, sum_columns=("n",))
    assert p.time == "when" and p.columns["n"].agg == "sum"
    assert p.columns["flag"].role == "dimension"
    assert profile_table(frame, time="when", ignore=("n",)).columns.keys() == {
        "when",
        "flag",
    }
    with pytest.raises(ValueError, match="no column"):
        profile_table(frame, time="nope")
    with pytest.raises(ValueError, match="does not hold dates"):
        profile_table(frame, time="n")


def test_ids_years_and_constants_are_left_alone():
    frame = pl.DataFrame(
        {
            "row_id": range(200),
            "year_built": [1950 + i % 60 for i in range(200)],
            "const": ["a"] * 200,
            "value": np.random.default_rng(0).normal(size=200),
            "name": [f"user{i}" for i in range(200)],
        }
    )
    roles = {c.name: c.role for c in profile_table(frame).columns.values()}
    assert roles == {
        "row_id": "id",
        "year_built": "year",
        "const": "constant",
        "value": "measure",
        "name": "id",
    }


def test_two_columns_that_carve_the_rows_the_same_way_count_once():
    n = 900
    code = [f"SKU{i % 300}" for i in range(n)]
    frame = pl.DataFrame(
        {"stock_code": code, "description": [c.lower() for c in code], "q": range(n)}
    )
    p = profile_table(frame)
    assert sorted(
        c.role for c in p.columns.values() if c.name in ("stock_code", "description")
    ) == ["entity", "text"]
    assert p.notes


def test_calendar_dimensions_are_recognised():
    bike = profile_table(bundled("bike"))
    assert bike.columns["weekday"].calendar == "weekday"
    assert bike.columns["weather"].calendar is None


def test_choose_grain():
    d = np.datetime64
    assert choose_grain(d("2020-01-01"), d("2020-04-30"), "day") == "day"
    assert choose_grain(d("2019-01-01"), d("2021-12-31"), "day") == "week"
    assert choose_grain(d("2006-01-01"), d("2010-07-31"), "month") == "month"
    assert choose_grain(d("2000-01-01"), d("2023-12-31"), "month") == "quarter"
    assert choose_grain(d("2020-01-01"), d("2020-01-08"), "day") is None


def test_weekly_buckets_are_anchored_and_the_ragged_end_is_dropped():
    p = profile_table(planted())
    s = build_series(p, "visits")
    assert s.grain == "week" and str(s.dates[0]) == "2019-01-01" and s.period == 52
    assert len(s) == 156 and any(
        "incomplete" in d for d in s.dropped
    )  # 1095 days = 156 weeks + 3 days
    days = planted()["visits"].to_numpy()
    assert s.values[0] == pytest.approx(days[:7].sum())
    mean_series = build_series(p, "temp")
    assert mean_series.values[1] == pytest.approx(
        planted()["temp"].to_numpy()[7:14].mean()
    )
    assert (
        "weekly total visits" == s.phrase
        and "weekly average temp" == mean_series.phrase
    )


def test_event_tables_get_a_row_count_series_and_quiet_buckets_are_zero():
    frame = events()
    p = profile_table(frame)
    s = build_series(p, ROWS)
    assert s.agg == "count" and s.values.sum() <= frame.height and (s.values >= 0).all()
    sparse = frame.filter(
        ~pl.col("order_date").is_between(date(2020, 3, 1), date(2020, 3, 20))
    )
    q = build_series(profile_table(sparse), ROWS)
    assert (q.values == 0).any()


def test_a_month_that_looks_cut_short_by_row_count_is_dropped():
    rng = np.random.default_rng(0)
    rows = []
    for m in range(36):
        first = date(2018 + m // 12, m % 12 + 1, 1)
        rows += [first] * (8 if m == 35 else int(rng.integers(70, 90)))
    frame = pl.DataFrame({"month": rows, "price": np.arange(len(rows), dtype=float)})
    s = build_series(profile_table(frame, time="month"), ROWS)
    assert len(s) == 35 and any("rows against a typical" in d for d in s.dropped)


def test_too_short_or_undated_tables_have_no_series():
    short = pl.DataFrame(
        {"d": [date(2020, 1, 1) + timedelta(days=i) for i in range(5)], "v": range(5)}
    )
    assert build_series(profile_table(short), "v") is None
    assert build_series(profile_table(pl.DataFrame({"v": range(30)})), "v") is None
