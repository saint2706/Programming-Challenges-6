from datetime import UTC, datetime, timedelta

import pytest
from cronwatch.crontab import Crontab
from cronwatch.display import duration, humanize, last_run_text, local, next_run_text
from cronwatch.store import FAILED, OK, TIMEOUT, Run

NOW = datetime(2026, 10, 9, 12, 20, tzinfo=UTC)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (10, "now"),
        (-10, "just now"),
        (60, "in 1m"),
        (-60, "1m ago"),
        (59 * 60, "in 59m"),
        (3600, "in 1h"),
        (3600 + 5 * 60, "in 1h 5m"),
        (9 * 3600 + 59 * 60, "in 9h 59m"),
        (13 * 3600 + 40 * 60, "in 14h"),
        (-(30 * 3600), "1d 6h ago"),
        (5 * 86400, "in 5d"),
        (5 * 86400 + 3 * 3600, "in 5d 3h"),
        (-(20 * 86400), "20d ago"),
    ],
)
def test_humanize_uses_at_most_two_units_and_rounds(seconds, expected):
    assert humanize(timedelta(seconds=seconds)) == expected


def test_duration():
    assert [duration(x) for x in (None, 0.04, 9.96, 42.3, 75, 3725)] == [
        "—",
        "0.0s",
        "10.0s",
        "42s",
        "1m 15s",
        "1h 02m",
    ]


def test_local_formats_in_the_given_zone():
    from zoneinfo import ZoneInfo

    assert (
        local(NOW, ZoneInfo("Asia/Tokyo")) == "Fri 09 Oct 21:20"
        and local(NOW, UTC, with_date=False) == "12:20"
    )


def job(line):
    return Crontab.from_text(line + "\n").jobs()[0]


def test_next_run_text_states():
    assert (
        next_run_text(job("0 15 * * * x"), NOW, UTC) == "in 2h 40m (Fri 09 Oct 15:00)"
    )
    assert next_run_text(job("#cronwatch:off# 0 15 * * * x"), NOW, UTC) == "disabled"
    assert next_run_text(job("61 * * * * x"), NOW, UTC) == "invalid schedule"
    assert next_run_text(job("@reboot x"), NOW, UTC) == "at startup"


def run(status, exit_code, minutes_ago=5):
    started = NOW - timedelta(minutes=minutes_ago)
    return Run(1, "j", "c", "h", started, started, 1.0, exit_code, status, "")


def test_last_run_text_includes_exit_codes_and_timeouts():
    assert last_run_text(None, NOW, UTC) == ("no runs yet", "dim")
    assert last_run_text(run(OK, 0), NOW, UTC) == ("● 5m ago", "green")
    assert last_run_text(run(FAILED, 3), NOW, UTC) == ("✗ 5m ago exit 3", "bold red")
    assert last_run_text(run(TIMEOUT, 124), NOW, UTC)[0].endswith("timed out")
