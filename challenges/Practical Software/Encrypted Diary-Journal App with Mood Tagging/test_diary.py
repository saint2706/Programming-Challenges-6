"""Tests for the encryption, persistence, and entry/mood logic in diary.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from diary import (
    MAGIC,
    Entry,
    JournalStore,
    JournalUnlockError,
    Mood,
    _decrypt,
    _encrypt,
)

# --- low-level encrypt/decrypt -----------------------------------------------


def test_encrypt_decrypt_round_trip():
    plaintext = b'{"entries": []}'
    blob = _encrypt(plaintext, "correct horse battery staple")
    assert _decrypt(blob, "correct horse battery staple") == plaintext


def test_decrypt_wrong_passphrase_raises():
    blob = _encrypt(b"secret diary contents", "right-passphrase")
    with pytest.raises(JournalUnlockError):
        _decrypt(blob, "wrong-passphrase")


def test_decrypt_tampered_ciphertext_raises():
    blob = bytearray(_encrypt(b"secret diary contents", "a-passphrase"))
    blob[-1] ^= 0xFF  # flip the last byte of the GCM tag/ciphertext
    with pytest.raises(JournalUnlockError):
        _decrypt(bytes(blob), "a-passphrase")


def test_decrypt_garbage_data_raises():
    with pytest.raises(JournalUnlockError):
        _decrypt(b"not a real journal file at all", "whatever")


def test_encrypted_blob_does_not_contain_plaintext():
    secret = b"a very identifiable secret string XYZZY"
    blob = _encrypt(secret, "a-passphrase")
    assert secret not in blob
    assert b"XYZZY" not in blob


def test_two_saves_use_different_nonces():
    plaintext = b'{"entries": []}'
    blob1 = _encrypt(plaintext, "same-passphrase")
    blob2 = _encrypt(plaintext, "same-passphrase")
    nonce1 = blob1[len(MAGIC) + 1 + 16 : len(MAGIC) + 1 + 16 + 12]
    nonce2 = blob2[len(MAGIC) + 1 + 16 : len(MAGIC) + 1 + 16 + 12]
    assert nonce1 != nonce2
    assert blob1 != blob2


# --- JournalStore: create / unlock / persistence -----------------------------


def test_new_journal_is_created_on_first_unlock(tmp_path):
    path = tmp_path / "journal.enc"
    assert not path.exists()
    store = JournalStore(path, "a-passphrase")
    assert store.entries == []
    store.add_entry(Mood.HAPPY, "first entry")
    assert path.exists()


def test_reopen_with_correct_passphrase_sees_entries(tmp_path):
    path = tmp_path / "journal.enc"
    store = JournalStore(path, "a-passphrase")
    store.add_entry(Mood.CALM, "feeling okay today")

    reopened = JournalStore(path, "a-passphrase")
    assert len(reopened.entries) == 1
    assert reopened.entries[0].body == "feeling okay today"
    assert reopened.entries[0].mood is Mood.CALM


def test_reopen_with_wrong_passphrase_fails(tmp_path):
    path = tmp_path / "journal.enc"
    store = JournalStore(path, "right-passphrase")
    store.add_entry(Mood.SAD, "a private thought")

    with pytest.raises(JournalUnlockError):
        JournalStore(path, "wrong-passphrase")


def test_file_on_disk_is_opaque_ciphertext(tmp_path):
    path = tmp_path / "journal.enc"
    store = JournalStore(path, "a-passphrase")
    store.add_entry(Mood.ANGRY, "something I would never want leaked in plaintext")

    raw = path.read_bytes()
    assert b"something I would never want leaked" not in raw
    assert raw.startswith(MAGIC)


def test_atomic_write_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "journal.enc"
    store = JournalStore(path, "a-passphrase")
    store.add_entry(Mood.NEUTRAL, "entry one")
    store.add_entry(Mood.HAPPY, "entry two")

    leftovers = list(tmp_path.glob(".*.tmp"))
    assert leftovers == []


def test_atomic_write_survives_failed_write(tmp_path, monkeypatch):
    """If the write step blows up partway through, the original file must be untouched."""
    path = tmp_path / "journal.enc"
    store = JournalStore(path, "a-passphrase")
    store.add_entry(Mood.HAPPY, "safe entry")
    original_bytes = path.read_bytes()

    import diary as diary_module

    def boom(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(diary_module.os, "fsync", boom)
    with pytest.raises(OSError):
        store.add_entry(Mood.SAD, "this save should fail")

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".*.tmp")) == []
    # And the journal is still openable afterwards.
    reopened = JournalStore(path, "a-passphrase")
    assert len(reopened.entries) == 1
    assert reopened.entries[0].body == "safe entry"


def test_delete_entry(tmp_path):
    store = JournalStore(tmp_path / "journal.enc", "pw")
    entry = store.add_entry(Mood.HAPPY, "to be deleted")
    store.add_entry(Mood.SAD, "kept")
    store.delete_entry(entry.id)
    assert [e.body for e in store.entries] == ["kept"]


# --- search / filter -----------------------------------------------------------


def test_search_by_text(tmp_path):
    store = JournalStore(tmp_path / "journal.enc", "pw")
    store.add_entry(Mood.HAPPY, "went hiking today")
    store.add_entry(Mood.SAD, "stayed inside all day")
    results = store.search(text="hiking")
    assert len(results) == 1
    assert "hiking" in results[0].body


def test_search_by_mood(tmp_path):
    store = JournalStore(tmp_path / "journal.enc", "pw")
    store.add_entry(Mood.HAPPY, "great day")
    store.add_entry(Mood.SAD, "rough day")
    results = store.search(mood=Mood.SAD)
    assert len(results) == 1
    assert results[0].mood is Mood.SAD


def test_search_by_date_range(tmp_path):
    store = JournalStore(tmp_path / "journal.enc", "pw")
    now = datetime.now(timezone.utc)
    store.add_entry(Mood.HAPPY, "old entry", created_at=now - timedelta(days=10))
    store.add_entry(Mood.HAPPY, "recent entry", created_at=now)
    results = store.search(since=now - timedelta(days=1))
    assert [e.body for e in results] == ["recent entry"]


def test_entries_sorted_newest_first(tmp_path):
    store = JournalStore(tmp_path / "journal.enc", "pw")
    now = datetime.now(timezone.utc)
    store.add_entry(Mood.HAPPY, "first", created_at=now - timedelta(days=2))
    store.add_entry(Mood.HAPPY, "second", created_at=now - timedelta(days=1))
    store.add_entry(Mood.HAPPY, "third", created_at=now)
    assert [e.body for e in store.entries] == ["third", "second", "first"]


# --- mood trend ------------------------------------------------------------


def test_mood_trend_averages_same_day_entries(tmp_path):
    store = JournalStore(tmp_path / "journal.enc", "pw")
    now = datetime(2026, 1, 15, 10, tzinfo=timezone.utc)
    store.add_entry(Mood.HAPPY, "a", created_at=now)  # score 7
    store.add_entry(Mood.SAD, "b", created_at=now.replace(hour=20))  # score 2
    trend = store.mood_trend(days=5, now=now)
    today_point = next(p for p in trend.points if p.date == "2026-01-15")
    assert today_point.average_score == pytest.approx(4.5)


def test_mood_trend_fills_gaps_with_none(tmp_path):
    store = JournalStore(tmp_path / "journal.enc", "pw")
    now = datetime(2026, 1, 15, tzinfo=timezone.utc)
    store.add_entry(Mood.HAPPY, "only entry", created_at=now)
    trend = store.mood_trend(days=3, now=now)
    assert len(trend.points) == 3
    scores = {p.date: p.average_score for p in trend.points}
    assert scores["2026-01-13"] is None
    assert scores["2026-01-14"] is None
    assert scores["2026-01-15"] == 7.0


def test_mood_trend_counts_moods(tmp_path):
    store = JournalStore(tmp_path / "journal.enc", "pw")
    store.add_entry(Mood.HAPPY, "a")
    store.add_entry(Mood.HAPPY, "b")
    store.add_entry(Mood.SAD, "c")
    trend = store.mood_trend(days=30)
    assert trend.mood_counts[Mood.HAPPY] == 2
    assert trend.mood_counts[Mood.SAD] == 1


def test_sparkline_values_fills_none_with_zero(tmp_path):
    store = JournalStore(tmp_path / "journal.enc", "pw")
    trend = store.mood_trend(days=3)
    assert trend.sparkline_values == [0.0, 0.0, 0.0]


# --- Entry (de)serialization -------------------------------------------------


def test_entry_round_trips_through_dict():
    entry = Entry(
        id="abc123",
        created_at=datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc),
        mood=Mood.EXCITED,
        body="line one\nline two",
    )
    restored = Entry.from_dict(entry.to_dict())
    assert restored == entry


def test_all_moods_have_icon_color_score():
    for mood in Mood:
        assert mood.icon
        assert mood.color
        assert isinstance(mood.score, int)
