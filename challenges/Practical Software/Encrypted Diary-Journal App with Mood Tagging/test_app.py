"""Smoke tests for the Textual UI, driven headlessly via Textual's Pilot.

Business logic (encryption, search, mood trend) is covered exhaustively in
test_diary.py against the plain diary.py module; these tests only confirm
the UI actually wires that logic up correctly end to end.
"""

from __future__ import annotations

import pytest
from app import DiaryApp, EntryListItem, MainScreen, UnlockScreen
from diary import JournalStore, Mood
from textual.widgets import Input, ListView, Sparkline

pytestmark = pytest.mark.asyncio


async def test_app_starts_on_unlock_screen(tmp_path):
    app = DiaryApp(journal_path=tmp_path / "journal.enc")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, UnlockScreen)


async def test_unlocking_new_journal_creates_it_and_shows_main_screen(tmp_path):
    journal_path = tmp_path / "journal.enc"
    app = DiaryApp(journal_path=journal_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#passphrase-input")
        await pilot.press(*"test-passphrase")
        await pilot.click("#unlock-button")
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)


async def test_wrong_passphrase_shows_error_and_stays_on_unlock_screen(tmp_path):
    journal_path = tmp_path / "journal.enc"
    JournalStore(journal_path, "correct-passphrase").add_entry(Mood.HAPPY, "seed entry")

    app = DiaryApp(journal_path=journal_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#passphrase-input")
        await pilot.press(*"wrong-passphrase")
        await pilot.click("#unlock-button")
        await pilot.pause()
        assert isinstance(app.screen, UnlockScreen)
        error_widget = app.screen.query_one("#unlock-error")
        error_text = str(error_widget.render())
        assert "wrong passphrase" in error_text or "corrupted" in error_text


async def test_new_entry_appears_in_list_and_updates_trend(tmp_path):
    journal_path = tmp_path / "journal.enc"
    app = DiaryApp(journal_path=journal_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#passphrase-input")
        await pilot.press(*"a-passphrase")
        await pilot.click("#unlock-button")
        await pilot.pause()

        # Focus something other than the search box first -- Input widgets
        # consume printable-character keys (including "n") themselves rather
        # than letting them bubble up to the screen's key bindings.
        await pilot.click("#entry-list")
        await pilot.press("n")
        await pilot.pause()
        await pilot.click("#entry-body")
        for ch in "Today was a good day.":
            await pilot.press(ch if ch != " " else "space")
        await pilot.click("#save-entry")
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, MainScreen)
        list_view = screen.query_one("#entry-list", ListView)
        assert len(list_view.children) == 1
        assert isinstance(list_view.children[0], EntryListItem)

        sparkline = screen.query_one("#trend-sparkline", Sparkline)
        assert any(v != 0.0 for v in sparkline.data)


async def test_search_filters_entry_list(tmp_path):
    journal_path = tmp_path / "journal.enc"
    store = JournalStore(journal_path, "a-passphrase")
    store.add_entry(Mood.HAPPY, "hiking in the mountains")
    store.add_entry(Mood.SAD, "stuck at a desk all day")

    app = DiaryApp(journal_path=journal_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#passphrase-input")
        await pilot.press(*"a-passphrase")
        await pilot.click("#unlock-button")
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, MainScreen)
        search_input = screen.query_one("#search-input", Input)
        search_input.focus()
        await pilot.pause()
        for ch in "hiking":
            await pilot.press(ch)
        await pilot.pause()

        list_view = screen.query_one("#entry-list", ListView)
        assert len(list_view.children) == 1
