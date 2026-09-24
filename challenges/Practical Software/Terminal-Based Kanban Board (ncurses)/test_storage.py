"""Unit tests for JSON round-trip persistence and atomic writes."""

import json
import os
from pathlib import Path

from board import Board
from storage import load_board, save_board


def test_save_and_load_empty_board(tmp_path):
    board_path = tmp_path / "board.json"
    board = Board()
    board.add_column("To Do")

    save_board(board, board_path)
    loaded = load_board(board_path)

    assert len(loaded.columns) == 1
    assert loaded.columns[0].name == "To Do"


def test_save_and_load_board_with_cards(tmp_path):
    board_path = tmp_path / "board.json"
    board = Board()
    board.add_column("To Do")
    board.add_column("Done")
    board.add_card(0, "Task 1", "Description 1", ["urgent"])
    board.add_card(1, "Task 2", "Description 2", ["review"])

    save_board(board, board_path)
    loaded = load_board(board_path)

    assert len(loaded.columns) == 2
    assert loaded.columns[0].name == "To Do"
    assert loaded.columns[1].name == "Done"
    assert len(loaded.columns[0].cards) == 1
    assert loaded.columns[0].cards[0].title == "Task 1"
    assert loaded.columns[0].cards[0].description == "Description 1"
    assert loaded.columns[0].cards[0].tags == ["urgent"]
    assert len(loaded.columns[1].cards) == 1
    assert loaded.columns[1].cards[0].title == "Task 2"


def test_load_nonexistent_file_returns_empty_board(tmp_path):
    board_path = tmp_path / "nonexistent.json"
    board = load_board(board_path)
    assert len(board.columns) == 0


def test_atomic_write_creates_file(tmp_path):
    board_path = tmp_path / "board.json"
    board = Board()
    board.add_column("To Do")
    board.add_card(0, "Task 1")

    save_board(board, board_path)
    assert board_path.exists()

    with open(board_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["format_version"] == 1
    assert data["columns"][0]["name"] == "To Do"


def test_atomic_write_no_temp_file_after_success(tmp_path):
    board_path = tmp_path / "board.json"
    board = Board()
    board.add_column("To Do")

    save_board(board, board_path)

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "board.json"


def test_save_overwrites_existing_file(tmp_path):
    board_path = tmp_path / "board.json"

    board1 = Board()
    board1.add_column("To Do")
    board1.add_card(0, "Task 1")
    save_board(board1, board_path)

    board2 = Board()
    board2.add_column("To Do")
    board2.add_column("Done")
    board2.add_card(0, "Task 2")
    board2.add_card(1, "Task 3")
    save_board(board2, board_path)

    loaded = load_board(board_path)
    assert len(loaded.columns) == 2
    assert len(loaded.columns[0].cards) == 1
    assert loaded.columns[0].cards[0].title == "Task 2"
    assert len(loaded.columns[1].cards) == 1
    assert loaded.columns[1].cards[0].title == "Task 3"


def test_save_creates_parent_directories(tmp_path):
    board_path = tmp_path / "nested" / "deep" / "board.json"
    board = Board()
    board.add_column("To Do")

    save_board(board, board_path)
    assert board_path.exists()

    loaded = load_board(board_path)
    assert len(loaded.columns) == 1


def test_load_malformed_json_raises_error(tmp_path):
    board_path = tmp_path / "board.json"
    board_path.write_text("not valid json {{{", encoding="utf-8")

    try:
        load_board(board_path)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid board file format" in str(e)


def test_load_missing_format_version_raises_error(tmp_path):
    board_path = tmp_path / "board.json"
    board_path.write_text(json.dumps({"columns": []}), encoding="utf-8")

    try:
        load_board(board_path)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid board file format" in str(e)


def test_load_unsupported_format_version_raises_error(tmp_path):
    board_path = tmp_path / "board.json"
    board_path.write_text(json.dumps({"format_version": 99, "columns": []}), encoding="utf-8")

    try:
        load_board(board_path)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported format version" in str(e)


def test_multiple_saves_and_loads(tmp_path):
    board_path = tmp_path / "board.json"
    board = Board()
    board.add_column("To Do")

    for i in range(5):
        board.add_card(0, f"Task {i}")
        save_board(board, board_path)
        loaded = load_board(board_path)
        assert len(loaded.columns[0].cards) == i + 1


def test_card_ids_preserved_across_save_load(tmp_path):
    board_path = tmp_path / "board.json"
    board = Board()
    board.add_column("To Do")
    card = board.add_card(0, "Task 1")
    original_id = card.id

    save_board(board, board_path)
    loaded = load_board(board_path)

    assert loaded.columns[0].cards[0].id == original_id


def test_card_timestamps_preserved(tmp_path):
    board_path = tmp_path / "board.json"
    board = Board()
    board.add_column("To Do")
    card = board.add_card(0, "Task 1")
    original_timestamp = card.created_at

    save_board(board, board_path)
    loaded = load_board(board_path)

    assert loaded.columns[0].cards[0].created_at == original_timestamp
