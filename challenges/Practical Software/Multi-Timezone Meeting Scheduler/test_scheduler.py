from __future__ import annotations

from datetime import date, time

import pytest
from scheduler import (
    Participant,
    find_overlaps,
    format_window,
    parse_participant_spec,
    search_timezones,
)


def test_full_overlap_same_timezone_same_hours() -> None:
    a = Participant("A", "America/New_York", time(9, 0), time(17, 0))
    b = Participant("B", "America/New_York", time(9, 0), time(17, 0))
    windows = find_overlaps([a, b], date(2026, 6, 15), "America/New_York")
    assert len(windows) == 1
    w = windows[0]
    assert w.available == ("A", "B")
    assert w.duration_minutes() == pytest.approx(8 * 60)


def test_partial_overlap_across_timezones() -> None:
    # New York 9-17 EDT (UTC-4 in June) is 13:00-21:00 UTC.
    # London 9-17 BST (UTC+1 in June) is 08:00-16:00 UTC.
    # Overlap in UTC: 13:00-16:00 -> 3 hours.
    ny = Participant("NY", "America/New_York", time(9, 0), time(17, 0))
    ldn = Participant("LDN", "Europe/London", time(9, 0), time(17, 0))
    windows = find_overlaps([ny, ldn], date(2026, 6, 15), "America/New_York")
    assert len(windows) == 1
    assert windows[0].duration_minutes() == pytest.approx(3 * 60)
    assert windows[0].available == ("LDN", "NY")


def test_no_overlap_returns_empty() -> None:
    # Tokyo 9-17 JST (UTC+9) is 00:00-08:00 UTC. New York 9-17 EDT is 13:00-21:00 UTC. No overlap.
    tokyo = Participant("Tokyo", "Asia/Tokyo", time(9, 0), time(17, 0))
    ny = Participant("NY", "America/New_York", time(9, 0), time(17, 0))
    windows = find_overlaps([tokyo, ny], date(2026, 6, 15), "UTC")
    assert windows == []


def test_min_participants_partial_overlap() -> None:
    a = Participant("A", "UTC", time(9, 0), time(12, 0))
    b = Participant("B", "UTC", time(10, 0), time(13, 0))
    c = Participant("C", "UTC", time(11, 0), time(14, 0))
    # A: 9-12, B: 10-13, C: 11-14.
    # All three overlap only 11-12. Any two overlap 10-13 in various pairs.
    all_three = find_overlaps(
        [a, b, c], date(2026, 1, 5), "UTC"
    )  # min=3 (default = everyone)
    assert len(all_three) == 1
    assert all_three[0].available == ("A", "B", "C")
    assert all_three[0].duration_minutes() == pytest.approx(60)

    at_least_two = find_overlaps([a, b, c], date(2026, 1, 5), "UTC", min_participants=2)
    assert sum(w.duration_minutes() for w in at_least_two) == pytest.approx(
        180
    )  # 10-13 total
    assert all(len(w.available) >= 2 for w in at_least_two)


def test_days_of_week_exclusion() -> None:
    weekday_only = Participant("A", "UTC", time(9, 0), time(17, 0))  # default Mon-Fri
    # 2026-01-10 is a Saturday.
    windows = find_overlaps(
        [weekday_only, weekday_only.__class__("B", "UTC", time(9, 0), time(17, 0))],
        date(2026, 1, 10),
        "UTC",
    )
    assert windows == []


_ALL_DAYS = frozenset(range(7))


def test_dst_spring_forward_shifts_utc_offset() -> None:
    # US DST begins 2026-03-08 (second Sunday of March). Before that date,
    # America/New_York is UTC-5; on/after, it's UTC-4 -- so a fixed local
    # 09:00 start should shift by exactly one hour in UTC across the boundary.
    # Both DST transition dates are Sundays, so these participants must opt
    # into all 7 days rather than the default Mon-Fri.
    p = Participant("A", "America/New_York", time(9, 0), time(17, 0), days=_ALL_DAYS)
    ref = Participant("Ref", "UTC", time(0, 0), time(23, 59), days=_ALL_DAYS)

    before = find_overlaps(
        [p, ref], date(2026, 3, 1), "America/New_York", min_participants=1
    )
    after = find_overlaps(
        [p, ref], date(2026, 3, 8), "America/New_York", min_participants=1
    )

    p_window_before = next(w for w in before if "A" in w.available)
    p_window_after = next(w for w in after if "A" in w.available)

    assert p_window_before.start_utc.hour == 14  # 09:00 EST = 14:00 UTC
    assert p_window_after.start_utc.hour == 13  # 09:00 EDT = 13:00 UTC


def test_dst_fall_back_shifts_utc_offset_back() -> None:
    # US DST ends 2026-11-01 (also a Sunday). Before: UTC-4 (EDT). On/after: UTC-5 (EST).
    p = Participant("A", "America/New_York", time(9, 0), time(17, 0), days=_ALL_DAYS)
    before = find_overlaps([p], date(2026, 10, 25), "America/New_York")
    after = find_overlaps([p], date(2026, 11, 1), "America/New_York")
    assert before[0].start_utc.hour == 13  # EDT
    assert after[0].start_utc.hour == 14  # EST


def test_overnight_shift_rejected() -> None:
    with pytest.raises(ValueError, match="overnight"):
        Participant("A", "UTC", time(22, 0), time(6, 0))


def test_unknown_timezone_rejected() -> None:
    # zoneinfo.ZoneInfoNotFoundError (raised for an unknown IANA name) is a KeyError subclass.
    with pytest.raises(KeyError):
        Participant("A", "Not/AZone", time(9, 0), time(17, 0))


def test_min_participants_out_of_range() -> None:
    a = Participant("A", "UTC", time(9, 0), time(17, 0))
    with pytest.raises(ValueError, match="min_participants"):
        find_overlaps([a], date(2026, 1, 5), "UTC", min_participants=0)
    with pytest.raises(ValueError, match="min_participants"):
        find_overlaps([a], date(2026, 1, 5), "UTC", min_participants=2)


def test_duplicate_names_rejected() -> None:
    a = Participant("A", "UTC", time(9, 0), time(17, 0))
    a2 = Participant("A", "UTC", time(10, 0), time(18, 0))
    with pytest.raises(ValueError, match="unique"):
        find_overlaps([a, a2], date(2026, 1, 5), "UTC")


def test_segments_split_on_membership_change_not_just_threshold() -> None:
    # A: 09-17, B: 09-13, C: 13-17 (all UTC). With min_participants=1 the
    # union is one continuous 09-17 stretch, but the *available* set changes
    # at 13:00, so it must be reported as two windows, not incorrectly merged.
    a = Participant("A", "UTC", time(9, 0), time(17, 0))
    b = Participant("B", "UTC", time(9, 0), time(13, 0))
    c = Participant("C", "UTC", time(13, 0), time(17, 0))
    windows = find_overlaps([a, b, c], date(2026, 1, 5), "UTC", min_participants=1)
    assert len(windows) == 2
    assert windows[0].available == ("A", "B")
    assert windows[1].available == ("A", "C")
    assert windows[0].end_utc == windows[1].start_utc


def test_format_window_localizes_correctly() -> None:
    a = Participant("A", "UTC", time(9, 0), time(17, 0))
    windows = find_overlaps([a], date(2026, 1, 5), "UTC")
    start_str, _end_str = format_window(windows[0], "Asia/Kolkata")
    # 09:00 UTC = 14:30 IST (UTC+5:30)
    assert "14:30" in start_str


def test_parse_participant_spec_basic() -> None:
    p = parse_participant_spec("Asha|Asia/Kolkata|09:00-18:00")
    assert p.name == "Asha"
    assert p.tz == "Asia/Kolkata"
    assert p.work_start == time(9, 0)
    assert p.work_end == time(18, 0)
    assert p.days == frozenset(range(5))


def test_parse_participant_spec_with_days() -> None:
    p = parse_participant_spec("Ben|Europe/London|09:00-17:00|0,1,2,3,4,5")
    assert p.days == frozenset({0, 1, 2, 3, 4, 5})


def test_parse_participant_spec_invalid_shape() -> None:
    with pytest.raises(ValueError, match="Invalid participant spec"):
        parse_participant_spec("just-a-name")


def test_parse_participant_spec_invalid_hours() -> None:
    with pytest.raises(ValueError, match="Invalid hours"):
        parse_participant_spec("A|UTC|not-hours")


def test_search_timezones_finds_known_zone() -> None:
    results = search_timezones("kolkata")
    assert "Asia/Kolkata" in results


def test_search_timezones_case_insensitive_and_limited() -> None:
    results = search_timezones("AMERICA", limit=5)
    assert len(results) <= 5
    assert all("america" in r.lower() for r in results)
