from __future__ import annotations

from fastapi.testclient import TestClient
from web import app

client = TestClient(app)


def test_index_shows_form() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<form" in resp.text


def test_find_returns_overlap_table() -> None:
    resp = client.post(
        "/find",
        data={
            "date_str": "2026-06-15",
            "ref_tz": "America/New_York",
            "participants": "NY|America/New_York|09:00-17:00\nLDN|Europe/London|09:00-17:00",
        },
    )
    assert resp.status_code == 200
    assert "<table>" in resp.text
    assert "NY" in resp.text and "LDN" in resp.text


def test_find_no_overlap_message() -> None:
    resp = client.post(
        "/find",
        data={
            "date_str": "2026-06-15",
            "ref_tz": "UTC",
            "participants": "Tokyo|Asia/Tokyo|09:00-17:00\nNY|America/New_York|09:00-17:00",
        },
    )
    assert resp.status_code == 200
    assert "No overlapping window" in resp.text


def test_find_rejects_bad_date() -> None:
    resp = client.post(
        "/find",
        data={"date_str": "nope", "ref_tz": "UTC", "participants": "A|UTC|09:00-17:00"},
    )
    assert resp.status_code == 400


def test_find_rejects_bad_participant_line() -> None:
    resp = client.post(
        "/find",
        data={
            "date_str": "2026-06-15",
            "ref_tz": "UTC",
            "participants": "not-a-valid-spec",
        },
    )
    assert resp.status_code == 400


def test_find_respects_display_tz() -> None:
    resp = client.post(
        "/find",
        data={
            "date_str": "2026-01-05",
            "ref_tz": "UTC",
            "display_tz": "Asia/Kolkata",
            "participants": "A|UTC|09:00-17:00",
        },
    )
    assert resp.status_code == 200
    assert "14:30" in resp.text  # 09:00 UTC = 14:30 IST


def test_find_respects_min_participants() -> None:
    resp = client.post(
        "/find",
        data={
            "date_str": "2026-01-05",
            "ref_tz": "UTC",
            "min_participants": "2",
            "participants": "A|UTC|09:00-12:00\nB|UTC|10:00-13:00\nC|UTC|11:00-14:00",
        },
    )
    assert resp.status_code == 200
    assert "<table>" in resp.text
