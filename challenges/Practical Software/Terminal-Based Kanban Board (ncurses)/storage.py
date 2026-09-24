"""JSON persistence for the Kanban board with atomic writes.

Atomic writes prevent a half-written JSON file on crash/power-loss: writes go
to a temp file in the same directory, then are atomically swapped in via
os.replace(), which is atomic on both POSIX and Windows. The on-disk file is
plain JSON (as the challenge asks for) with a `format_version` field, not a
binary-prefixed format — a plain JSON file can be inspected, diffed, or
hand-repaired with any text editor if it's ever corrupted.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from board import Board

FORMAT_VERSION = 1


def load_board(path: Path) -> Board:
    """Load a board from disk. Returns an empty board if the file doesn't exist."""
    if not path.exists():
        return Board()

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid board file format") from e

    if not isinstance(data, dict) or "format_version" not in data:
        raise ValueError("Invalid board file format")
    version = data["format_version"]
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported format version: {version}")

    return Board.from_dict(data)


def save_board(board: Board, path: Path) -> None:
    """Save a board to disk atomically.

    Writes to a temp file in the same directory, then os.replace()s it over
    the real file. A crash mid-write leaves the original file untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data = board.to_dict()
    data["format_version"] = FORMAT_VERSION
    json_bytes = json.dumps(data, indent=2).encode("utf-8")

    fd, temp_path = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(json_bytes)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise
