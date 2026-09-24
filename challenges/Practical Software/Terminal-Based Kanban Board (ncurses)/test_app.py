"""Smoke tests for the Textual UI, driven headlessly via Textual's Pilot.

Business logic (board operations) is covered exhaustively in test_board.py;
these tests only confirm the UI actually wires that logic up correctly end to end.
"""

from __future__ import annotations

import pytest
from app import KanbanApp, MainScreen
from board import Board
from storage import load_board, save_board

pytestmark = pytest.mark.asyncio


async def test_app_starts_on_main_screen(tmp_path):
    board_path = tmp_path / "board.json"
    app = KanbanApp(board_path=board_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)


async def test_app_creates_default_columns_on_first_run(tmp_path):
    board_path = tmp_path / "board.json"
    app = KanbanApp(board_path=board_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.board.columns) == 3
        assert app.board.columns[0].name == "To Do"
        assert app.board.columns[1].name == "In Progress"
        assert app.board.columns[2].name == "Done"


async def test_app_loads_existing_board(tmp_path):
    board_path = tmp_path / "board.json"
    board = Board()
    board.add_column("Tasks")
    board.add_card(0, "Existing Task")
    save_board(board, board_path)

    app = KanbanApp(board_path=board_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.board.columns) == 1
        assert app.board.columns[0].name == "Tasks"
        assert len(app.board.columns[0].cards) == 1


async def test_navigation_between_columns(tmp_path):
    board_path = tmp_path / "board.json"
    app = KanbanApp(board_path=board_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MainScreen)
        assert screen.current_col_index == 0

        await pilot.press("l")
        await pilot.pause()
        assert screen.current_col_index == 1

        await pilot.press("h")
        await pilot.pause()
        assert screen.current_col_index == 0


async def test_delete_card_via_action(tmp_path):
    board_path = tmp_path / "board.json"
    board = Board()
    board.add_column("To Do")
    card = board.add_card(0, "Task to delete")
    save_board(board, board_path)

    app = KanbanApp(board_path=board_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert len(app.board.columns[0].cards) == 1

        list_views = screen.query("ListView")
        if list_views:
            list_views[0].focus()
        await pilot.pause()

        await pilot.press("d")
        await pilot.pause()

        assert len(app.board.columns[0].cards) == 0
        loaded = load_board(board_path)
        assert len(loaded.columns[0].cards) == 0


async def test_add_column_creates_column(tmp_path):
    board_path = tmp_path / "board.json"
    app = KanbanApp(board_path=board_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        initial_col_count = len(app.board.columns)

        await pilot.press("n")
        await pilot.pause()

        from textual.widgets import Input

        col_input = app.screen_stack[-1].query_one("#col-name-input", Input)
        col_input.value = "Review"
        await pilot.press("enter")
        await pilot.pause()

        assert len(app.board.columns) == initial_col_count + 1
        loaded = load_board(board_path)
        assert any(col.name == "Review" for col in loaded.columns)


async def test_board_persistence_on_card_delete(tmp_path):
    board_path = tmp_path / "board.json"
    board = Board()
    board.add_column("To Do")
    card = board.add_card(0, "Task 1")
    save_board(board, board_path)

    app = KanbanApp(board_path=board_path)
    async with app.run_test() as pilot:
        await pilot.pause()

        list_views = app.screen.query("ListView")
        if list_views:
            list_views[0].focus()

        await pilot.press("d")
        await pilot.pause()

        loaded = load_board(board_path)
        assert len(loaded.columns[0].cards) == 0


async def test_navigation_with_hjkl_keys(tmp_path):
    board_path = tmp_path / "board.json"
    board = Board()
    board.add_column("To Do")
    board.add_column("In Progress")
    board.add_column("Done")
    board.add_card(0, "Task 1")
    board.add_card(1, "Task 2")
    save_board(board, board_path)

    app = KanbanApp(board_path=board_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen

        assert screen.current_col_index == 0
        await pilot.press("l")
        await pilot.pause()
        assert screen.current_col_index == 1
        await pilot.press("l")
        await pilot.pause()
        assert screen.current_col_index == 2
