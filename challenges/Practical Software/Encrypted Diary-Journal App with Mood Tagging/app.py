"""Textual UI for the encrypted diary. All encryption/persistence/search
logic lives in `diary.py`; this module is purely presentation and input
handling so it stays thin and the security-critical code stays testable
without a terminal.

Run directly:
    uv run --with textual --with cryptography python app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from diary import Entry, JournalStore, JournalUnlockError, Mood
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Sparkline,
    Static,
    TextArea,
)

DEFAULT_JOURNAL_PATH = Path(__file__).parent / "journal.enc"


class UnlockScreen(ModalScreen[JournalStore]):
    """Passphrase prompt shown on startup. Dismisses with an unlocked JournalStore."""

    DEFAULT_CSS = """
    UnlockScreen {
        align: center middle;
    }
    #unlock-box {
        width: 60;
        height: auto;
        border: round $primary;
        padding: 1 2;
    }
    #unlock-error {
        color: $error;
        height: auto;
    }
    """

    def __init__(self, journal_path: Path) -> None:
        super().__init__()
        self.journal_path = journal_path

    def compose(self) -> ComposeResult:
        is_new = not self.journal_path.exists()
        title = "Create a new journal" if is_new else "Unlock your journal"
        subtitle = (
            "No journal found here yet -- the passphrase you enter now will\n"
            "encrypt it. There is no recovery if you forget it."
            if is_new
            else "Enter your passphrase to decrypt this journal."
        )
        with Vertical(id="unlock-box"):
            yield Label(title, id="unlock-title")
            yield Static(subtitle, id="unlock-subtitle")
            yield Input(placeholder="Passphrase", password=True, id="passphrase-input")
            yield Static("", id="unlock-error")
            yield Button("Unlock", variant="primary", id="unlock-button")

    def on_mount(self) -> None:
        self.query_one("#passphrase-input", Input).focus()

    @on(Input.Submitted, "#passphrase-input")
    @on(Button.Pressed, "#unlock-button")
    def attempt_unlock(self) -> None:
        passphrase = self.query_one("#passphrase-input", Input).value
        error = self.query_one("#unlock-error", Static)
        if not passphrase:
            error.update("Passphrase can't be empty.")
            return
        try:
            store = JournalStore(self.journal_path, passphrase)
        except JournalUnlockError as exc:
            error.update(str(exc))
            self.query_one("#passphrase-input", Input).value = ""
            return
        self.dismiss(store)


class EntryEditorScreen(ModalScreen[tuple[Mood, str] | None]):
    """Modal for composing a new entry: mood picker + multi-line body."""

    DEFAULT_CSS = """
    EntryEditorScreen {
        align: center middle;
    }
    #editor-box {
        width: 80;
        height: 24;
        border: round $primary;
        padding: 1 2;
    }
    #entry-body {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="editor-box"):
            yield Label("New entry")
            yield Select(
                [(f"{m.icon}  {m.value.title()}", m) for m in Mood],
                id="mood-select",
                value=Mood.NEUTRAL,
                allow_blank=False,
            )
            yield TextArea(id="entry-body")
            with Horizontal():
                yield Button("Save", variant="primary", id="save-entry")
                yield Button("Cancel", id="cancel-entry")

    def on_mount(self) -> None:
        self.query_one("#entry-body", TextArea).focus()

    @on(Button.Pressed, "#save-entry")
    def save(self) -> None:
        mood = self.query_one("#mood-select", Select).value
        body = self.query_one("#entry-body", TextArea).text.strip()
        if not body or mood is Select.BLANK:
            self.dismiss(None)
            return
        self.dismiss((mood, body))

    @on(Button.Pressed, "#cancel-entry")
    def cancel(self) -> None:
        self.dismiss(None)


class EntryListItem(ListItem):
    def __init__(self, entry: Entry) -> None:
        super().__init__(
            Label(
                f"{entry.mood.icon} [{entry.mood.color}]{entry.mood.value:<8}[/] "
                f"{entry.created_at:%Y-%m-%d %H:%M}  "
                f"{entry.body.splitlines()[0][:60]}"
            )
        )
        self.entry = entry


class MainScreen(Screen):
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("n", "new_entry", "New entry"),
        ("d", "delete_entry", "Delete entry"),
        ("/", "focus_search", "Search"),
        ("escape", "clear_search", "Clear search"),
    ]

    def __init__(self, store: JournalStore) -> None:
        super().__init__()
        self.store = store

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="left-pane"):
                yield Input(placeholder="Search entries...", id="search-input")
                yield ListView(id="entry-list")
            with VerticalScroll(id="right-pane"):
                yield Label("Mood trend (last 30 days)", id="trend-title")
                yield Sparkline([], id="trend-sparkline")
                yield Static("", id="trend-summary")
                yield Label("Entry detail", id="detail-title")
                yield Static("Select an entry to read it.", id="detail-body")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_entries()
        self.refresh_trend()

    def refresh_entries(self, filter_text: str | None = None) -> None:
        entries = (
            self.store.search(text=filter_text) if filter_text else self.store.entries
        )
        list_view = self.query_one("#entry-list", ListView)
        list_view.clear()
        for entry in entries:
            list_view.append(EntryListItem(entry))

    def refresh_trend(self) -> None:
        trend = self.store.mood_trend(days=30)
        self.query_one("#trend-sparkline", Sparkline).data = trend.sparkline_values
        counts = ", ".join(
            f"{mood.icon}{count}" for mood, count in trend.mood_counts.most_common()
        )
        self.query_one("#trend-summary", Static).update(counts or "No entries yet.")

    @on(Input.Changed, "#search-input")
    def on_search_changed(self, event: Input.Changed) -> None:
        self.refresh_entries(event.value or None)

    @on(ListView.Highlighted, "#entry-list")
    def on_entry_highlighted(self, event: ListView.Highlighted) -> None:
        item = event.item
        if isinstance(item, EntryListItem):
            self.query_one("#detail-body", Static).update(
                f"[{item.entry.mood.color}]{item.entry.mood.icon} {item.entry.mood.value}[/] "
                f"-- {item.entry.created_at:%Y-%m-%d %H:%M}\n\n{item.entry.body}"
            )

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()

    def action_clear_search(self) -> None:
        search = self.query_one("#search-input", Input)
        search.value = ""
        self.refresh_entries()

    def action_new_entry(self) -> None:
        # push_screen_wait must run inside a worker task (see DiaryApp.on_mount).
        self.run_worker(self._new_entry_flow())

    async def _new_entry_flow(self) -> None:
        result = await self.app.push_screen_wait(EntryEditorScreen())
        if result is not None:
            mood, body = result
            self.store.add_entry(mood, body)
            self.refresh_entries()
            self.refresh_trend()

    def action_delete_entry(self) -> None:
        list_view = self.query_one("#entry-list", ListView)
        item = list_view.highlighted_child
        if isinstance(item, EntryListItem):
            self.store.delete_entry(item.entry.id)
            self.refresh_entries()
            self.refresh_trend()


class DiaryApp(App):
    TITLE = "Encrypted Diary"
    CSS = """
    #left-pane { width: 45%; }
    #right-pane { width: 55%; padding: 1 2; }
    #entry-list { height: 1fr; }
    """

    def __init__(self, journal_path: Path = DEFAULT_JOURNAL_PATH) -> None:
        super().__init__()
        self.journal_path = journal_path

    def on_mount(self) -> None:
        # push_screen_wait must run inside a worker task, so the unlock ->
        # main-screen handoff runs as a background worker rather than
        # directly in the (non-worker) on_mount coroutine.
        self.run_worker(self._unlock_and_show_main_screen())

    async def _unlock_and_show_main_screen(self) -> None:
        store = await self.push_screen_wait(UnlockScreen(self.journal_path))
        await self.push_screen(MainScreen(store))


if __name__ == "__main__":
    DiaryApp().run()
