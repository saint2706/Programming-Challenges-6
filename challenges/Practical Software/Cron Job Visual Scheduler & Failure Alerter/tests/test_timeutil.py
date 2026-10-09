from datetime import UTC
from zoneinfo import ZoneInfo

import pytest
from cronwatch.timeutil import schedule_zone, system_zone


def test_tz_environment_variable_wins(tmp_path):
    assert system_zone({"TZ": "Europe/Berlin"}, tmp_path) == ZoneInfo("Europe/Berlin")
    assert system_zone({"TZ": ":America/New_York"}, tmp_path) == ZoneInfo(
        "America/New_York"
    )


def test_etc_timezone_then_localtime_symlink_then_utc(tmp_path):
    assert system_zone({}, tmp_path) is UTC
    link = tmp_path / "localtime"
    link.symlink_to("/usr/share/zoneinfo/Asia/Tokyo")
    assert system_zone({}, tmp_path) == ZoneInfo("Asia/Tokyo")
    (tmp_path / "timezone").write_text("Australia/Sydney\n")
    assert system_zone({}, tmp_path) == ZoneInfo("Australia/Sydney")


def test_garbage_values_fall_through_instead_of_crashing(tmp_path):
    (tmp_path / "timezone").write_text("Not/AZone\n")
    assert system_zone({"TZ": "???"}, tmp_path) is UTC


def test_configured_zone_must_be_valid():
    assert schedule_zone("Europe/Paris") == ZoneInfo("Europe/Paris")
    with pytest.raises(ValueError, match="unknown timezone"):
        schedule_zone("Mars/Olympus")
