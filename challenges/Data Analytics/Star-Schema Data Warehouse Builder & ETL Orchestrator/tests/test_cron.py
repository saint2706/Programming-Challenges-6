"""Cron semantics the scheduler relies on: field syntax, the day-of-month/day-of-week rule, Sunday as 0 or 7,
fire sequences, the interval a fire covers, and the once-a-day restriction."""

from __future__ import annotations

from datetime import datetime

import pytest
from star_warehouse import cron


def test_next_fires_for_a_daily_schedule():
    spec = cron.parse("0 2 * * *")
    fire = datetime(2010, 1, 1, 0, 0)
    seen = []
    for _ in range(3):
        fire = cron.next_after(spec, fire)
        seen.append(fire)
    assert seen == [
        datetime(2010, 1, 1, 2, 0),
        datetime(2010, 1, 2, 2, 0),
        datetime(2010, 1, 3, 2, 0),
    ]


def test_steps_lists_and_ranges_expand():
    spec = cron.parse("*/20 9-10 * * 1-5")
    assert spec.minutes == frozenset({0, 20, 40})
    assert spec.hours == frozenset({9, 10})
    assert spec.weekdays == frozenset({1, 2, 3, 4, 5})


def test_sunday_is_zero_or_seven():
    assert (
        cron.parse("0 0 * * 0").weekdays
        == cron.parse("0 0 * * 7").weekdays
        == frozenset({0})
    )
    # 2010-01-03 was a Sunday.
    assert cron.next_after(cron.parse("0 0 * * 7"), datetime(2010, 1, 1)) == datetime(
        2010, 1, 3
    )


def test_day_of_month_and_day_of_week_are_ored_when_both_restricted():
    # Fridays and the 13th. January 2010: Fridays are the 1st, 8th, 15th; the 13th is a Wednesday.
    spec = cron.parse("0 0 13 * 5")
    assert cron.next_after(spec, datetime(2010, 1, 1)) == datetime(2010, 1, 8)
    assert cron.next_after(spec, datetime(2010, 1, 8)) == datetime(2010, 1, 13)


def test_previous_before_mirrors_next_after():
    spec = cron.parse("30 6 * * *")
    assert cron.previous_before(spec, datetime(2010, 3, 5, 6, 30)) == datetime(
        2010, 3, 4, 6, 30
    )
    assert cron.previous_before(spec, datetime(2010, 3, 5, 6, 31)) == datetime(
        2010, 3, 5, 6, 30
    )


def test_due_fires_is_half_open_and_capped():
    spec = cron.parse("0 2 * * *")
    fires = cron.due_fires(spec, datetime(2010, 3, 1, 2, 0), datetime(2010, 3, 4, 2, 0))
    assert fires == [
        datetime(2010, 3, 2, 2, 0),
        datetime(2010, 3, 3, 2, 0),
        datetime(2010, 3, 4, 2, 0),
    ]
    assert (
        len(cron.due_fires(spec, datetime(2010, 1, 1), datetime(2011, 1, 1), limit=5))
        == 5
    )


def test_interval_for_a_fire_covers_the_days_since_the_previous_fire():
    daily = cron.parse("0 2 * * *")
    from datetime import date

    assert cron.interval_for_fire(daily, datetime(2010, 3, 5, 2, 0)) == (
        date(2010, 3, 4),
        date(2010, 3, 5),
    )
    weekly = cron.parse("0 2 * * 1")  # Mondays; 2010-01-04 was a Monday
    assert cron.interval_for_fire(weekly, datetime(2010, 1, 4, 2, 0)) == (
        date(2009, 12, 28),
        date(2010, 1, 4),
    )


@pytest.mark.parametrize("expr", ["*/30 * * * *", "0 2,14 * * *", "0 0-5 * * *"])
def test_schedules_that_fire_more_than_once_a_day_are_rejected(expr):
    with pytest.raises(ValueError, match="at most once a day"):
        cron.check_daily(cron.parse(expr))


@pytest.mark.parametrize("expr", ["* * * *", "61 * * * *", "0 0 0 * *", "a b c d e"])
def test_malformed_expressions_are_rejected(expr):
    with pytest.raises(ValueError):
        cron.parse(expr)
