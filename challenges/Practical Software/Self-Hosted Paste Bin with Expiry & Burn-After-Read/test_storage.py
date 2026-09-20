from __future__ import annotations

import threading
import time
from pathlib import Path

from storage import PasteStore


def test_create_and_get(tmp_path: Path) -> None:
    store = PasteStore(tmp_path / "db.sqlite")
    paste = store.create("hello world", language="text")
    fetched = store.get_and_consume(paste.id)
    assert fetched is not None
    assert fetched.content == "hello world"


def test_nonexistent_paste_returns_none(tmp_path: Path) -> None:
    store = PasteStore(tmp_path / "db.sqlite")
    assert store.get_and_consume("does-not-exist") is None
    assert store.peek("does-not-exist") is None


def test_burn_after_read_only_readable_once(tmp_path: Path) -> None:
    store = PasteStore(tmp_path / "db.sqlite")
    paste = store.create("secret", burn_after_read=True)

    first = store.get_and_consume(paste.id)
    assert first is not None
    assert first.content == "secret"

    second = store.get_and_consume(paste.id)
    assert second is None


def test_non_burn_paste_readable_multiple_times(tmp_path: Path) -> None:
    store = PasteStore(tmp_path / "db.sqlite")
    paste = store.create("reusable", burn_after_read=False)

    assert store.get_and_consume(paste.id) is not None
    assert store.get_and_consume(paste.id) is not None
    assert store.get_and_consume(paste.id) is not None


def test_ttl_expiry_on_read(tmp_path: Path) -> None:
    store = PasteStore(tmp_path / "db.sqlite")
    paste = store.create("expires soon", ttl_seconds=-1)  # already expired
    assert store.get_and_consume(paste.id) is None
    assert store.peek(paste.id) is None


def test_never_expires_when_ttl_none(tmp_path: Path) -> None:
    store = PasteStore(tmp_path / "db.sqlite")
    paste = store.create("forever", ttl_seconds=None)
    assert paste.expires_at is None
    assert store.get_and_consume(paste.id) is not None


def test_delete_expired_sweeps_only_expired(tmp_path: Path) -> None:
    store = PasteStore(tmp_path / "db.sqlite")
    expired = store.create("old", ttl_seconds=-5)
    fresh = store.create("new", ttl_seconds=3600)
    forever = store.create("permanent", ttl_seconds=None)

    removed = store.delete_expired()

    assert removed == 1
    assert store.count() == 2
    assert store.peek(fresh.id) is not None
    assert store.peek(forever.id) is not None
    assert store.peek(expired.id) is None


def test_peek_does_not_consume_burn_after_read(tmp_path: Path) -> None:
    store = PasteStore(tmp_path / "db.sqlite")
    paste = store.create("still here", burn_after_read=True)

    assert store.peek(paste.id) is not None
    assert store.peek(paste.id) is not None  # peeking never burns it

    consumed = store.get_and_consume(paste.id)
    assert consumed is not None
    assert store.get_and_consume(paste.id) is None  # now it's really gone


def test_concurrent_burn_after_read_only_one_winner(tmp_path: Path) -> None:
    """Two threads racing on the same one-time-view link must not both see it.

    ``get_and_consume`` opens a manual ``BEGIN IMMEDIATE`` transaction per
    connection, so the loser blocks until the winner commits and then finds
    the row already gone. Run many times in one test to make a race
    condition in the implementation very likely to surface as a flake.
    """
    store = PasteStore(tmp_path / "db.sqlite")
    paste = store.create("racy secret", burn_after_read=True)

    results: list[bool] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        result = store.get_and_consume(paste.id)
        results.append(result is not None)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert results.count(True) == 1
    assert results.count(False) == 1


def test_language_and_content_round_trip(tmp_path: Path) -> None:
    store = PasteStore(tmp_path / "db.sqlite")
    paste = store.create("print('hi')", language="python", ttl_seconds=60)
    fetched = store.peek(paste.id)
    assert fetched is not None
    assert fetched.language == "python"
    assert fetched.content == "print('hi')"
    assert fetched.expires_at is not None and fetched.expires_at > time.time()


def test_ids_are_unique_and_url_safe(tmp_path: Path) -> None:
    store = PasteStore(tmp_path / "db.sqlite")
    ids = {store.create("x").id for _ in range(50)}
    assert len(ids) == 50
    for pid in ids:
        assert all(c.isalnum() or c in "-_" for c in pid)
