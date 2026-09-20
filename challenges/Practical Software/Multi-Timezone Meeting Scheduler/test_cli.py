from __future__ import annotations

from cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_find_reports_overlap() -> None:
    result = runner.invoke(
        app,
        [
            "find",
            "--date",
            "2026-06-15",
            "--ref-tz",
            "America/New_York",
            "-p",
            "NY|America/New_York|09:00-17:00",
            "-p",
            "LDN|Europe/London|09:00-17:00",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "NY" in result.stdout
    assert "LDN" in result.stdout


def test_find_reports_no_overlap() -> None:
    result = runner.invoke(
        app,
        [
            "find",
            "--date",
            "2026-06-15",
            "--ref-tz",
            "UTC",
            "-p",
            "Tokyo|Asia/Tokyo|09:00-17:00",
            "-p",
            "NY|America/New_York|09:00-17:00",
        ],
    )
    assert result.exit_code == 0
    assert "No overlapping window" in result.stdout


def test_find_rejects_bad_date() -> None:
    result = runner.invoke(
        app,
        ["find", "--date", "not-a-date", "--ref-tz", "UTC", "-p", "A|UTC|09:00-17:00"],
    )
    assert result.exit_code == 1


def test_find_rejects_bad_participant_spec() -> None:
    result = runner.invoke(
        app,
        ["find", "--date", "2026-06-15", "--ref-tz", "UTC", "-p", "garbage"],
    )
    assert result.exit_code == 1


def test_zones_command_finds_known_zone() -> None:
    result = runner.invoke(app, ["zones", "kolkata"])
    assert result.exit_code == 0
    assert "Asia/Kolkata" in result.stdout


def test_zones_command_no_match() -> None:
    result = runner.invoke(app, ["zones", "not-a-real-place-xyz"])
    assert result.exit_code == 0
    assert "No timezones matching" in result.stdout


def test_min_participants_flag_narrows_to_partial_overlap() -> None:
    result = runner.invoke(
        app,
        [
            "find",
            "--date",
            "2026-01-05",
            "--ref-tz",
            "UTC",
            "-p",
            "A|UTC|09:00-12:00",
            "-p",
            "B|UTC|10:00-13:00",
            "-p",
            "C|UTC|11:00-14:00",
            "--min-participants",
            "2",
        ],
    )
    assert result.exit_code == 0
    # With min-participants=2, the window(s) should mention at least two of the three names.
    assert sum(name in result.stdout for name in ("A", "B", "C")) >= 2
