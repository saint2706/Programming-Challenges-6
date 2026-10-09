from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from cronwatch.cronexpr import CronSyntaxError, parse

NY = ZoneInfo("America/New_York")


def at(*args, tz=UTC):
    return datetime(*args, tzinfo=tz)


def runs(text, after, n=3, tz=UTC):
    return parse(text).next_runs(after, n, tz)


def test_every_minute_and_steps():
    assert runs("* * * * *", at(2026, 1, 1, 0, 0, 30), 2) == [
        at(2026, 1, 1, 0, 1),
        at(2026, 1, 1, 0, 2),
    ]
    assert runs("*/15 * * * *", at(2026, 1, 1, 0, 14), 3) == [
        at(2026, 1, 1, 0, 15),
        at(2026, 1, 1, 0, 30),
        at(2026, 1, 1, 0, 45),
    ]


def test_next_run_is_strictly_after_the_reference_time():
    assert runs("0 3 * * *", at(2026, 1, 1, 3, 0), 1) == [at(2026, 1, 2, 3, 0)]
    assert runs("0 3 * * *", at(2026, 1, 1, 2, 59, 59), 1) == [at(2026, 1, 1, 3, 0)]


def test_lists_ranges_and_range_steps():
    expr = parse("0,30 9-11 * * *")
    assert sorted(expr.minutes) == [0, 30] and sorted(expr.hours) == [9, 10, 11]
    assert sorted(parse("1-10/3 * * * *").minutes) == [1, 4, 7, 10]
    assert sorted(parse("5/20 * * * *").minutes) == [5, 25, 45]


def test_names_are_case_insensitive_and_seven_is_sunday():
    assert parse("0 0 * JAN,Jul MON-FRI").months == frozenset({1, 7})
    assert (
        parse("0 0 * * 1-5").weekdays
        == parse("0 0 * * mon-fri").weekdays
        == frozenset({1, 2, 3, 4, 5})
    )
    assert parse("0 0 * * 7").weekdays == parse("0 0 * * 0").weekdays == frozenset({0})
    assert parse("0 0 * * 0-7").weekdays == frozenset(range(7))


def test_macros_expand():
    assert (
        parse("@hourly").minutes == frozenset({0}) and len(parse("@hourly").hours) == 24
    )
    assert runs("@daily", at(2026, 5, 5, 12), 1) == [at(2026, 5, 6)]
    assert runs("@weekly", at(2026, 10, 9), 1) == [at(2026, 10, 11)]  # Sunday
    assert runs("@monthly", at(2026, 12, 31, 23), 1) == [at(2027, 1, 1)]
    assert runs("@yearly", at(2026, 6, 1), 1) == [at(2027, 1, 1)]


def test_reboot_never_has_a_next_run_but_is_valid():
    expr = parse("@reboot")
    assert (
        expr.reboot
        and expr.next_after(at(2026, 1, 1)) is None
        and expr.describe() == "At system startup"
    )


def test_dom_and_dow_both_restricted_means_either():
    # 13th of the month OR Friday, per Vixie cron
    got = runs("0 0 13 * 5", at(2026, 10, 9, 12), 3)
    assert got == [at(2026, 10, 13), at(2026, 10, 16), at(2026, 10, 23)]


def test_star_prefixed_dom_or_dow_does_not_trigger_the_either_rule():
    # */2 starts with `*`, so day-of-month is "star" and only day-of-week restricts
    got = runs("0 0 */2 * 1", at(2026, 10, 1), 2)
    assert [d.isoweekday() for d in got] == [1, 1]


def test_month_end_and_leap_day():
    assert runs("0 0 31 * *", at(2026, 1, 31, 1), 3) == [
        at(2026, 3, 31),
        at(2026, 5, 31),
        at(2026, 7, 31),
    ]
    assert runs("0 0 29 2 *", at(2026, 1, 1), 2) == [at(2028, 2, 29), at(2032, 2, 29)]


def test_year_rollover():
    assert runs("30 23 31 12 *", at(2026, 12, 31, 23, 30), 1) == [
        at(2027, 12, 31, 23, 30)
    ]


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("* * * *", "expected 5 fields"),
        ("* * * * * *", "expected 5 fields"),
        ("60 * * * *", "minute: 60 is outside 0-59"),
        ("0 24 * * *", "hour: 24 is outside 0-23"),
        ("0 0 0 * *", "day-of-month: 0 is outside 1-31"),
        ("0 0 * 13 *", "month: 13 is outside 1-12"),
        ("0 0 * * 8", "day-of-week: 8 is outside 0-7"),
        ("*/0 * * * *", "step '0'"),
        ("*/x * * * *", "step 'x'"),
        ("a * * * *", "not a number"),
        ("0 0 * foo *", "not a number or name"),
        ("10-5 * * * *", "runs backwards"),
        ("@fortnightly", "unknown macro"),
        ("", "expected 5 fields"),
        (", * * * *", "empty"),
        ("0 0 31 2 *", "never fires"),
        ("0 0 30 2 *", "never fires"),
    ],
)
def test_invalid_expressions_say_what_is_wrong(text, fragment):
    with pytest.raises(CronSyntaxError, match=fragment):
        parse(text)


def test_matches_checks_one_wall_clock_minute():
    expr = parse("30 4 * * 1-5")
    assert expr.matches(datetime(2026, 10, 9, 4, 30))  # noqa: DTZ001 - cron is wall-clock; a Friday
    assert not expr.matches(datetime(2026, 10, 10, 4, 30))  # noqa: DTZ001 - Saturday
    assert not expr.matches(datetime(2026, 10, 9, 4, 31))  # noqa: DTZ001


def test_runs_between_is_half_open_and_limited():
    expr = parse("0 * * * *")
    got = expr.runs_between(at(2026, 1, 1, 0, 0), at(2026, 1, 1, 3, 0))
    assert got == [at(2026, 1, 1, 1), at(2026, 1, 1, 2), at(2026, 1, 1, 3)]
    assert (
        len(parse("* * * * *").runs_between(at(2026, 1, 1), at(2026, 1, 2), limit=10))
        == 10
    )


# --- DST: the cases that make hand-rolled cron tools wrong twice a year ---


def test_a_time_in_the_spring_forward_gap_is_skipped_that_day():
    # US DST began 2026-03-08: 02:00-02:59 local never happened.
    got = runs("30 2 * * *", at(2026, 3, 7, 12, tz=NY), 3, NY)
    assert [d.date().isoformat() for d in got] == [
        "2026-03-09",
        "2026-03-10",
        "2026-03-11",
    ]
    assert all(d.hour == 2 and d.minute == 30 for d in got)


def test_wildcard_jobs_resume_right_after_the_gap_without_inventing_times():
    got = runs("*/30 * * * *", at(2026, 3, 8, 1, 29, tz=NY), 4, NY)
    assert [(d.hour, d.minute) for d in got] == [(1, 30), (3, 0), (3, 30), (4, 0)]
    # 01:30 EST -> 03:00 EDT is 30 real minutes (same-zone subtraction would say 90: compare in UTC)
    assert got[1].astimezone(UTC) - got[0].astimezone(UTC) == timedelta(minutes=30)


def test_fall_back_repeated_hour_fires_once_in_its_first_pass():
    # US DST ended 2026-11-01: 01:00-01:59 happened twice.
    got = runs("30 1 * * *", at(2026, 10, 31, 12, tz=NY), 3, NY)
    assert [d.date().isoformat() for d in got] == [
        "2026-11-01",
        "2026-11-02",
        "2026-11-03",
    ]
    assert got[0].utcoffset() == timedelta(hours=-4)  # first pass, still EDT


def test_results_are_in_the_requested_zone_whatever_the_input_zone():
    after = datetime(2026, 6, 1, 3, 0, tzinfo=UTC)  # 23:00 on May 31 in New York
    got = parse("0 0 * * *").next_after(after, NY)
    assert got == datetime(2026, 6, 1, 0, 0, tzinfo=NY)
    assert got.tzinfo is NY


def test_naive_input_is_read_as_wall_clock_in_the_given_zone():
    got = parse("0 12 * * *").next_after(datetime(2026, 6, 1, 13, 0), NY)  # noqa: DTZ001 - naive on purpose
    assert got == datetime(2026, 6, 2, 12, 0, tzinfo=NY)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("* * * * *", "Every minute"),
        ("*/5 * * * *", "Every 5 minutes"),
        ("0 * * * *", "Every hour on the hour"),
        ("17 * * * *", "At minute 17 of every hour"),
        ("30 2 * * *", "At 02:30"),
        ("0 9,17 * * *", "At 09:00 and 17:00"),
        ("0 3 * * 1-5", "At 03:00 on Monday to Friday"),
        ("0 3 * * 0,6", "At 03:00 on Saturday and Sunday"),
        ("0 0 1 * *", "At 00:00 on day 1 of the month"),
        ("0 0 13 * 5", "At 00:00 on day 13 of the month or on Friday"),
        ("5 4 * jan,jul sun", "At 04:05 on Sunday in January and July"),
        ("0 0 29 2 *", "At 00:00 on day 29 of the month in February"),
        ("15 */6 * * *", "At 00:15, 06:15, 12:15 and 18:15"),
        ("@daily", "At 00:00"),
    ],
)
def test_describe(text, expected):
    assert parse(text).describe() == expected


def test_describe_falls_back_to_explicit_fields_for_irregular_schedules():
    text = parse("0,20,40 8-18 * * *").describe()
    assert text.startswith("At minute 0, 20 and 40 past hour 8, 9")
