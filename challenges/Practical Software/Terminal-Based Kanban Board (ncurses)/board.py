"""Kanban board data model: cards, columns, board.

Zero dependency on Textual or any UI layer, so all business logic is fully
unit-testable without driving a terminal app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class Card:
    id: str
    title: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
        }

    @staticmethod
    def from_dict(data: dict) -> Card:
        return Card(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class Column:
    name: str
    cards: list[Card] = field(default_factory=list)

    def add_card(self, card: Card) -> None:
        self.cards.append(card)

    def remove_card(self, card_id: str) -> bool:
        for i, card in enumerate(self.cards):
            if card.id == card_id:
                self.cards.pop(i)
                return True
        return False

    def get_card(self, card_id: str) -> Card | None:
        for card in self.cards:
            if card.id == card_id:
                return card
        return None

    def find_card_index(self, card_id: str) -> int:
        for i, card in enumerate(self.cards):
            if card.id == card_id:
                return i
        return -1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cards": [card.to_dict() for card in self.cards],
        }

    @staticmethod
    def from_dict(data: dict) -> Column:
        col = Column(name=data["name"])
        col.cards = [Card.from_dict(card_data) for card_data in data.get("cards", [])]
        return col


@dataclass
class Board:
    columns: list[Column] = field(default_factory=list)

    def add_column(self, name: str) -> Column:
        col = Column(name=name)
        self.columns.append(col)
        return col

    def rename_column(self, col_index: int, new_name: str) -> bool:
        if 0 <= col_index < len(self.columns):
            self.columns[col_index].name = new_name
            return True
        return False

    def delete_column(self, col_index: int) -> bool:
        if 0 <= col_index < len(self.columns):
            self.columns.pop(col_index)
            return True
        return False

    def get_column(self, col_index: int) -> Column | None:
        if 0 <= col_index < len(self.columns):
            return self.columns[col_index]
        return None

    def add_card(self, col_index: int, title: str, description: str = "", tags: list[str] | None = None) -> Card | None:
        if 0 <= col_index < len(self.columns):
            card = Card(
                id=str(uuid4()),
                title=title,
                description=description,
                tags=tags or [],
            )
            self.columns[col_index].add_card(card)
            return card
        return None

    def edit_card(self, col_index: int, card_id: str, title: str | None = None, description: str | None = None, tags: list[str] | None = None) -> bool:
        if 0 <= col_index < len(self.columns):
            card = self.columns[col_index].get_card(card_id)
            if card:
                if title is not None:
                    card.title = title
                if description is not None:
                    card.description = description
                if tags is not None:
                    card.tags = tags
                return True
        return False

    def delete_card(self, col_index: int, card_id: str) -> bool:
        if 0 <= col_index < len(self.columns):
            return self.columns[col_index].remove_card(card_id)
        return False

    def find_card(self, card_id: str) -> tuple[int, Card | None]:
        """Returns (col_index, card) or (-1, None) if not found."""
        for col_idx, col in enumerate(self.columns):
            card = col.get_card(card_id)
            if card:
                return (col_idx, card)
        return (-1, None)

    def move_card(self, card_id: str, from_col_idx: int, to_col_idx: int) -> bool:
        """Move a card from one column to another."""
        if not (0 <= from_col_idx < len(self.columns) and 0 <= to_col_idx < len(self.columns)):
            return False
        if from_col_idx == to_col_idx:
            return False

        from_col = self.columns[from_col_idx]
        to_col = self.columns[to_col_idx]

        card = from_col.get_card(card_id)
        if not card:
            return False

        from_col.remove_card(card_id)
        to_col.add_card(card)
        return True

    def to_dict(self) -> dict:
        return {
            "columns": [col.to_dict() for col in self.columns],
        }

    @staticmethod
    def from_dict(data: dict) -> Board:
        board = Board()
        board.columns = [Column.from_dict(col_data) for col_data in data.get("columns", [])]
        return board
