"""Textual UI for the Kanban board.

Columns are rendered side-by-side, each with a ListView of cards.
Keybindings: hjkl/arrows to navigate, Shift+Left/Right to move cards
between columns, a to add, e to edit, d to delete, n for new column, q to quit.

All persistence happens through board.py and storage.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from board import Board, Card, Column
from storage import load_board, save_board
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, Static, TextArea

DEFAULT_BOARD_PATH = Path(__file__).parent / "board.json"


class CardEditorScreen(ModalScreen[tuple[str, str, list[str]] | None]):
    """Modal for creating/editing a card: title, description, tags."""

    DEFAULT_CSS = """
    CardEditorScreen {
        align: center middle;
    }
    #editor-box {
        width: 80;
        height: 20;
        border: round $primary;
        padding: 1 2;
    }
    #description {
        height: 10;
    }
    """

    def __init__(self, title: str = "", description: str = "", tags_str: str = "") -> None:
        super().__init__()
        self.initial_title = title
        self.initial_description = description
        self.initial_tags_str = tags_str

    def compose(self) -> ComposeResult:
        with Vertical(id="editor-box"):
            yield Label("Edit Card")
            yield Input(placeholder="Title", id="card-title", value=self.initial_title)
            yield Static("Description:", id="desc-label")
            yield TextArea(id="description", text=self.initial_description)
            yield Static("Tags (comma-separated):", id="tags-label")
            yield Input(placeholder="Tags", id="card-tags", value=self.initial_tags_str)
            with Horizontal():
                yield Button("Save", variant="primary", id="save-card")
                yield Button("Cancel", id="cancel-card")

    def on_mount(self) -> None:
        self.query_one("#card-title", Input).focus()

    @on(Button.Pressed, "#save-card")
    def save(self) -> None:
        title = self.query_one("#card-title", Input).value.strip()
        description = self.query_one("#description", TextArea).text.strip()
        tags_str = self.query_one("#card-tags", Input).value.strip()
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        if not title:
            return
        self.dismiss((title, description, tags))

    @on(Button.Pressed, "#cancel-card")
    def cancel(self) -> None:
        self.dismiss(None)


class CardListItem(ListItem):
    def __init__(self, card: Card) -> None:
        tags_str = f" [{', '.join(card.tags)}]" if card.tags else ""
        label = Label(f"{card.title}{tags_str}")
        super().__init__(label)
        self.card = card


class ColumnWidget(Static):
    """A single column in the board, containing a ListView of cards."""

    DEFAULT_CSS = """
    ColumnWidget {
        width: 1fr;
        height: 100%;
        border: solid $primary;
        padding: 0 1;
    }
    ColumnWidget > #col-header {
        dock: top;
        height: 1;
    }
    ColumnWidget > #col-list {
        height: 1fr;
    }
    """

    def __init__(self, column: Column) -> None:
        super().__init__()
        self.column = column

    def compose(self) -> ComposeResult:
        yield Label(self.column.name, id="col-header")
        with ListView(id="col-list"):
            for card in self.column.cards:
                yield CardListItem(card)

    def get_list_view(self) -> ListView:
        return self.query_one("#col-list", ListView)

    def refresh_cards(self) -> None:
        list_view = self.get_list_view()
        list_view.clear()
        for card in self.column.cards:
            list_view.append(CardListItem(card))


class MainScreen(Screen):
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("h", "focus_prev_column", "Prev column"),
        ("l", "focus_next_column", "Next column"),
        ("left", "focus_prev_column", "Prev column"),
        ("right", "focus_next_column", "Next column"),
        ("j", "focus_next_card", "Next card"),
        ("k", "focus_prev_card", "Prev card"),
        ("down", "focus_next_card", "Next card"),
        ("up", "focus_prev_card", "Prev card"),
        ("a", "add_card", "Add card"),
        ("e", "edit_card", "Edit card"),
        ("d", "delete_card", "Delete card"),
        ("shift+h", "move_card_left", "Move left"),
        ("shift+l", "move_card_right", "Move right"),
        ("n", "new_column", "New column"),
        ("q", "quit_app", "Quit"),
    ]

    def __init__(self, board: Board, board_path: Path) -> None:
        super().__init__()
        self.board = board
        self.board_path = board_path
        self.current_col_index = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="board-container"):
            for col in self.board.columns:
                yield ColumnWidget(col)
        yield Footer()

    def on_mount(self) -> None:
        if self.board.columns:
            col_widget = self.query("ColumnWidget")[0]
            col_widget.get_list_view().focus()

    def get_current_column_widget(self) -> ColumnWidget | None:
        col_widgets = self.query("ColumnWidget")
        if 0 <= self.current_col_index < len(col_widgets):
            return col_widgets[self.current_col_index]
        return None

    def get_selected_card(self) -> Card | None:
        col_widget = self.get_current_column_widget()
        if not col_widget:
            return None
        list_view = col_widget.get_list_view()
        item = list_view.highlighted_child
        if isinstance(item, CardListItem):
            return item.card
        return None

    def action_focus_prev_column(self) -> None:
        if self.board.columns:
            self.current_col_index = (self.current_col_index - 1) % len(self.board.columns)
            col_widget = self.get_current_column_widget()
            if col_widget:
                col_widget.get_list_view().focus()

    def action_focus_next_column(self) -> None:
        if self.board.columns:
            self.current_col_index = (self.current_col_index + 1) % len(self.board.columns)
            col_widget = self.get_current_column_widget()
            if col_widget:
                col_widget.get_list_view().focus()

    def action_focus_prev_card(self) -> None:
        col_widget = self.get_current_column_widget()
        if col_widget:
            list_view = col_widget.get_list_view()
            if list_view.children:
                idx = list_view._selected
                if idx is not None and idx > 0:
                    list_view.highlighted_index = idx - 1

    def action_focus_next_card(self) -> None:
        col_widget = self.get_current_column_widget()
        if col_widget:
            list_view = col_widget.get_list_view()
            if list_view.children:
                idx = list_view._selected
                if idx is not None and idx < len(list_view.children) - 1:
                    list_view.highlighted_index = idx + 1
                elif idx is None:
                    list_view.highlighted_index = 0

    def action_add_card(self) -> None:
        self.run_worker(self._add_card_flow())

    async def _add_card_flow(self) -> None:
        result = await self.app.push_screen_wait(CardEditorScreen())
        if result is not None:
            title, description, tags = result
            self.board.add_card(self.current_col_index, title, description, tags)
            save_board(self.board, self.board_path)
            self.refresh_current_column()

    def action_edit_card(self) -> None:
        card = self.get_selected_card()
        if not card:
            return
        tags_str = ", ".join(card.tags)
        self.run_worker(self._edit_card_flow(card, tags_str))

    async def _edit_card_flow(self, card: Card, tags_str: str) -> None:
        result = await self.app.push_screen_wait(
            CardEditorScreen(card.title, card.description, tags_str)
        )
        if result is not None:
            title, description, tags = result
            self.board.edit_card(self.current_col_index, card.id, title, description, tags)
            save_board(self.board, self.board_path)
            self.refresh_current_column()

    def action_delete_card(self) -> None:
        card = self.get_selected_card()
        if not card:
            return
        self.board.delete_card(self.current_col_index, card.id)
        save_board(self.board, self.board_path)
        self.refresh_current_column()

    def action_move_card_left(self) -> None:
        card = self.get_selected_card()
        if card and self.current_col_index > 0:
            self.board.move_card(card.id, self.current_col_index, self.current_col_index - 1)
            save_board(self.board, self.board_path)
            self.refresh_columns()

    def action_move_card_right(self) -> None:
        card = self.get_selected_card()
        if card and self.current_col_index < len(self.board.columns) - 1:
            self.board.move_card(card.id, self.current_col_index, self.current_col_index + 1)
            save_board(self.board, self.board_path)
            self.refresh_columns()

    def action_new_column(self) -> None:
        self.run_worker(self._new_column_flow())

    async def _new_column_flow(self) -> None:
        app = self.app
        result = await app.push_screen_wait(
            ColumnNameScreen()
        )
        if result:
            self.board.add_column(result)
            save_board(self.board, self.board_path)
            self.refresh_columns()

    def action_quit_app(self) -> None:
        self.app.exit()

    def refresh_current_column(self) -> None:
        col_widget = self.get_current_column_widget()
        if col_widget:
            col_widget.refresh_cards()

    def refresh_columns(self) -> None:
        container = self.query_one("#board-container", Horizontal)
        container.remove_children()
        for col in self.board.columns:
            container.mount(ColumnWidget(col))
        self.current_col_index = min(self.current_col_index, len(self.board.columns) - 1)
        if self.board.columns and self.current_col_index >= 0:
            col_widget = self.query("ColumnWidget")[self.current_col_index]
            col_widget.get_list_view().focus()


class ColumnNameScreen(ModalScreen[str | None]):
    """Modal for entering a new column name."""

    DEFAULT_CSS = """
    ColumnNameScreen {
        align: center middle;
    }
    #name-box {
        width: 50;
        height: auto;
        border: round $primary;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="name-box"):
            yield Label("New Column")
            yield Input(placeholder="Column name", id="col-name-input")
            with Horizontal():
                yield Button("Create", variant="primary", id="create-col")
                yield Button("Cancel", id="cancel-col")

    def on_mount(self) -> None:
        self.query_one("#col-name-input", Input).focus()

    @on(Input.Submitted, "#col-name-input")
    @on(Button.Pressed, "#create-col")
    def create(self) -> None:
        name = self.query_one("#col-name-input", Input).value.strip()
        if name:
            self.dismiss(name)

    @on(Button.Pressed, "#cancel-col")
    def cancel(self) -> None:
        self.dismiss(None)


class KanbanApp(App):
    TITLE = "Kanban Board"

    def __init__(self, board_path: Path = DEFAULT_BOARD_PATH) -> None:
        super().__init__()
        self.board_path = board_path
        self.board = load_board(board_path)
        if not self.board.columns:
            self.board.add_column("To Do")
            self.board.add_column("In Progress")
            self.board.add_column("Done")
            save_board(self.board, board_path)

    def on_mount(self) -> None:
        self.push_screen(MainScreen(self.board, self.board_path))


if __name__ == "__main__":
    KanbanApp().run()
