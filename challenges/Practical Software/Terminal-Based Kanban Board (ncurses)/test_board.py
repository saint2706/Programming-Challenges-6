"""Unit tests for the Kanban board data model."""

import pytest
from board import Board, Card, Column


def test_create_column():
    col = Column("To Do")
    assert col.name == "To Do"
    assert col.cards == []


def test_add_card_to_column():
    col = Column("To Do")
    card = Card(id="1", title="Task 1")
    col.add_card(card)
    assert len(col.cards) == 1
    assert col.cards[0].title == "Task 1"


def test_remove_card_from_column():
    col = Column("To Do")
    card = Card(id="1", title="Task 1")
    col.add_card(card)
    assert col.remove_card("1")
    assert len(col.cards) == 0


def test_remove_nonexistent_card():
    col = Column("To Do")
    assert not col.remove_card("nonexistent")


def test_get_card():
    col = Column("To Do")
    card = Card(id="1", title="Task 1")
    col.add_card(card)
    found = col.get_card("1")
    assert found is not None
    assert found.title == "Task 1"


def test_get_nonexistent_card():
    col = Column("To Do")
    assert col.get_card("nonexistent") is None


def test_find_card_index():
    col = Column("To Do")
    col.add_card(Card(id="1", title="Task 1"))
    col.add_card(Card(id="2", title="Task 2"))
    assert col.find_card_index("1") == 0
    assert col.find_card_index("2") == 1
    assert col.find_card_index("3") == -1


def test_create_board():
    board = Board()
    assert board.columns == []


def test_add_column_to_board():
    board = Board()
    col = board.add_column("To Do")
    assert len(board.columns) == 1
    assert board.columns[0].name == "To Do"
    assert col is board.columns[0]


def test_rename_column():
    board = Board()
    board.add_column("To Do")
    assert board.rename_column(0, "Tasks")
    assert board.columns[0].name == "Tasks"


def test_rename_nonexistent_column():
    board = Board()
    assert not board.rename_column(0, "Tasks")


def test_delete_column():
    board = Board()
    board.add_column("To Do")
    board.add_column("Done")
    assert board.delete_column(0)
    assert len(board.columns) == 1
    assert board.columns[0].name == "Done"


def test_delete_nonexistent_column():
    board = Board()
    assert not board.delete_column(0)


def test_get_column():
    board = Board()
    board.add_column("To Do")
    col = board.get_column(0)
    assert col is not None
    assert col.name == "To Do"


def test_get_nonexistent_column():
    board = Board()
    assert board.get_column(0) is None


def test_add_card_to_board():
    board = Board()
    board.add_column("To Do")
    card = board.add_card(0, "Task 1", "Description", ["urgent"])
    assert card is not None
    assert card.title == "Task 1"
    assert card.description == "Description"
    assert card.tags == ["urgent"]
    assert len(board.columns[0].cards) == 1


def test_add_card_to_nonexistent_column():
    board = Board()
    card = board.add_card(0, "Task 1")
    assert card is None


def test_edit_card():
    board = Board()
    board.add_column("To Do")
    card = board.add_card(0, "Task 1", "Old description")
    assert board.edit_card(0, card.id, title="Task 1 Updated", description="New description", tags=["updated"])
    assert card.title == "Task 1 Updated"
    assert card.description == "New description"
    assert card.tags == ["updated"]


def test_edit_card_partial():
    board = Board()
    board.add_column("To Do")
    card = board.add_card(0, "Task 1", "Description")
    assert board.edit_card(0, card.id, title="Updated")
    assert card.title == "Updated"
    assert card.description == "Description"


def test_edit_nonexistent_card():
    board = Board()
    board.add_column("To Do")
    assert not board.edit_card(0, "nonexistent", title="Updated")


def test_delete_card():
    board = Board()
    board.add_column("To Do")
    card = board.add_card(0, "Task 1")
    assert board.delete_card(0, card.id)
    assert len(board.columns[0].cards) == 0


def test_delete_nonexistent_card():
    board = Board()
    board.add_column("To Do")
    assert not board.delete_card(0, "nonexistent")


def test_find_card():
    board = Board()
    board.add_column("To Do")
    board.add_column("In Progress")
    card1 = board.add_card(0, "Task 1")
    card2 = board.add_card(1, "Task 2")

    idx, found = board.find_card(card1.id)
    assert idx == 0
    assert found is card1

    idx, found = board.find_card(card2.id)
    assert idx == 1
    assert found is card2

    idx, found = board.find_card("nonexistent")
    assert idx == -1
    assert found is None


def test_move_card_between_columns():
    board = Board()
    board.add_column("To Do")
    board.add_column("Done")
    card = board.add_card(0, "Task 1")

    assert board.move_card(card.id, 0, 1)
    assert len(board.columns[0].cards) == 0
    assert len(board.columns[1].cards) == 1
    assert board.columns[1].cards[0].id == card.id


def test_move_card_same_column():
    board = Board()
    board.add_column("To Do")
    card = board.add_card(0, "Task 1")
    assert not board.move_card(card.id, 0, 0)
    assert len(board.columns[0].cards) == 1


def test_move_nonexistent_card():
    board = Board()
    board.add_column("To Do")
    board.add_column("Done")
    assert not board.move_card("nonexistent", 0, 1)


def test_move_card_invalid_columns():
    board = Board()
    board.add_column("To Do")
    card = board.add_card(0, "Task 1")
    assert not board.move_card(card.id, 0, 5)
    assert not board.move_card(card.id, 5, 0)


def test_move_card_off_first_column():
    board = Board()
    board.add_column("To Do")
    board.add_column("In Progress")
    board.add_column("Done")
    card = board.add_card(0, "Task 1")
    assert board.move_card(card.id, 0, 1)
    assert len(board.columns[0].cards) == 0


def test_move_card_off_last_column():
    board = Board()
    board.add_column("To Do")
    board.add_column("In Progress")
    board.add_column("Done")
    card = board.add_card(2, "Task 1")
    assert board.move_card(card.id, 2, 1)
    assert len(board.columns[2].cards) == 0


def test_delete_column_with_cards():
    board = Board()
    board.add_column("To Do")
    board.add_column("Done")
    board.add_card(0, "Task 1")
    assert board.delete_column(0)
    assert len(board.columns) == 1


def test_card_to_dict():
    card = Card(id="1", title="Task 1", description="Desc", tags=["urgent"])
    data = card.to_dict()
    assert data["id"] == "1"
    assert data["title"] == "Task 1"
    assert data["description"] == "Desc"
    assert data["tags"] == ["urgent"]
    assert "created_at" in data


def test_card_from_dict():
    data = {
        "id": "1",
        "title": "Task 1",
        "description": "Desc",
        "tags": ["urgent"],
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    card = Card.from_dict(data)
    assert card.id == "1"
    assert card.title == "Task 1"
    assert card.description == "Desc"
    assert card.tags == ["urgent"]


def test_column_to_dict():
    col = Column("To Do")
    col.add_card(Card(id="1", title="Task 1"))
    data = col.to_dict()
    assert data["name"] == "To Do"
    assert len(data["cards"]) == 1
    assert data["cards"][0]["title"] == "Task 1"


def test_column_from_dict():
    data = {
        "name": "To Do",
        "cards": [
            {
                "id": "1",
                "title": "Task 1",
                "description": "",
                "tags": [],
                "created_at": "2024-01-01T00:00:00+00:00",
            }
        ],
    }
    col = Column.from_dict(data)
    assert col.name == "To Do"
    assert len(col.cards) == 1
    assert col.cards[0].title == "Task 1"


def test_board_to_dict():
    board = Board()
    board.add_column("To Do")
    board.add_card(0, "Task 1")
    data = board.to_dict()
    assert len(data["columns"]) == 1
    assert data["columns"][0]["name"] == "To Do"


def test_board_from_dict():
    data = {
        "columns": [
            {
                "name": "To Do",
                "cards": [
                    {
                        "id": "1",
                        "title": "Task 1",
                        "description": "",
                        "tags": [],
                        "created_at": "2024-01-01T00:00:00+00:00",
                    }
                ],
            }
        ]
    }
    board = Board.from_dict(data)
    assert len(board.columns) == 1
    assert board.columns[0].name == "To Do"
    assert len(board.columns[0].cards) == 1


def test_empty_board():
    board = Board()
    assert len(board.columns) == 0
    data = board.to_dict()
    board2 = Board.from_dict(data)
    assert len(board2.columns) == 0
