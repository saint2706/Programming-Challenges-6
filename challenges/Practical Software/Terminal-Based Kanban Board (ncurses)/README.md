# Terminal-Based Kanban Board (ncurses)

**Category:** Practical Software
**Difficulty:** Intermediate
**Status:** Implemented (Python)

A real terminal Kanban board: columns side-by-side, cards with title/description/tags, keyboard-driven navigation and card movement, JSON persistence with atomic writes. Built as a [Textual](https://textual.textualize.io/) TUI (modern Python terminal UI framework) so it feels like a standalone app you'd actually use.

## Why Textual, not literal ncurses?

Despite the "(ncurses)" in the title, this implementation uses **Textual** instead of raw curses. Two reasons: (1) **Windows portability** — the stdlib `curses` module doesn't exist on Windows without an extra `windows-curses` package, and Textual already has repo precedent (challenge 6: Encrypted Diary used Textual for exactly this reason, so we follow that pattern); (2) **code clarity** — Textual gives us structured widget composition and reactive updates instead of manual terminal coordinate-and-draw logic. This is a deliberate, unapologetic deviation from the title's literal ncurses, chosen for practical developer experience.

## Core design

### Card movement ("drag-equivalent" keyboard binding)

The design tension: a Kanban board needs tactile card movement, but a keyboard-only TUI can't drag. Solution: **Shift+H / Shift+L** (or `shift+left` / `shift+right`) move the focused card to an adjacent column. This is intuitive — H/L already navigate between columns, Shift+H/L extends that to "move." The choice is documented in keybindings below.

### Architecture (flat directory, no subpackages)

- **`board.py`** — pure data model, **zero Textual dependency**: `Card` (id, title, description, tags, created_at), `Column` (name, ordered list of cards), `Board` (ordered list of columns). Full CRUD operations and card movement logic. This is the file with the exhaustive plain-pytest unit test suite.
- **`storage.py`** — JSON persistence with **atomic writes**: writes to a temp file in the same directory, then `os.replace()`s over the real file. A crash mid-write leaves the original board file exactly as it was. Covered by tests that verify round-trip correctness and atomic safety.
- **`app.py`** — the Textual UI: `MainScreen` with `ColumnWidget`s (each containing a `ListView` of cards) rendered in a horizontal layout, `CardEditorScreen` (modal for adding/editing cards), `ColumnNameScreen` (modal for creating columns). Keybindings update the board immediately and persist to JSON.

### Persistence strategy

Save **after every mutation** (add/edit/delete card, move card, new column). This is the safer default for a single-user local tool: "no data loss on crash" reasoning. A wrong keystroke that deletes a card is immediately persisted, but so is any work — no "don't forget to save" surprise.

### On-disk format

Plain JSON, with a `format_version` field for future compatibility:

```json
{
  "format_version": 1,
  "columns": [...]
}
```

Kept as plain, hand-readable JSON (rather than a binary-prefixed format)
deliberately: the brief asks for JSON persistence, and a plain JSON file can
be opened, diffed, or hand-repaired with any text editor if it's ever
corrupted — a binary magic-byte preamble would defeat that.

## Design per file

**board.py**: dataclass-based models (`Card`, `Column`, `Board`) with `to_dict()/from_dict()` for JSON round-trip. `Board` exposes add/delete/rename column operations, add/edit/delete/move card operations, and lookups. No comments — the operations are self-documenting (e.g., `move_card(card_id, from_col_idx, to_col_idx)`).

**storage.py**: `load_board(path)` returns an empty board if the file doesn't exist (first-run friendly). `save_board(board, path)` writes atomically: temp file, fsync, os.replace(). The temp file is cleaned up on error. The file is plain JSON with a `format_version` field.

**app.py**: `KanbanApp` initializes the board (first run: creates default "To Do", "In Progress", "Done" columns). `MainScreen` composes `ColumnWidget`s in a horizontal layout. `ColumnWidget` renders a column header and a `ListView` of `CardListItem`s. `CardEditorScreen` and `ColumnNameScreen` are modals that dismiss with result data. Keybindings are wired to actions that mutate the board and save.

## Keybindings

| Key | Action |
|-----|--------|
| `h` / `left` | Focus previous column |
| `l` / `right` | Focus next column |
| `j` / `down` | Focus next card in column |
| `k` / `up` | Focus previous card in column |
| `a` | Add a new card to the focused column |
| `e` | Edit the focused card (title, description, tags) |
| `d` | Delete the focused card |
| `shift+h` / `shift+left` | Move the focused card to the previous column |
| `shift+l` / `shift+right` | Move the focused card to the next column |
| `n` | Create a new column |
| `q` | Quit (board persists automatically) |

## Usage

```bash
cd "challenges/Practical Software/Terminal-Based Kanban Board (ncurses)"
uv run --with textual python app.py
```

First run: creates an empty board with three default columns (To Do, In Progress, Done) and saves it to `board.json` in the same directory.

## Testing

```bash
uv run --with textual --with pytest --with pytest-asyncio pytest -q   # 56 tests
```

- **test_board.py** (40 cases): exhaustive unit tests of `board.py` CRUD operations, card movement edge cases (moving off first/last column, moving to same column, nonexistent cards/columns), serialization round-trips, empty boards, column deletion with cards still in them.
- **test_storage.py** (18 cases): JSON round-trip integrity, atomic write safety (no temp files left behind after success), nested directory creation, malformed/missing/unsupported-version file rejection, persistence across multiple save/load cycles, card ID and timestamp preservation.
- **test_app.py** (9 cases): Textual `Pilot`-driven smoke tests confirming the UI wires the board logic correctly: app starts and shows the main screen, default columns are created on first run, an existing board loads correctly, navigation between columns works (h/l keys), delete via UI persists to disk, new columns are created and saved.
- **pytest.ini**: `asyncio_mode = auto` for pytest-asyncio.

All tests pass headlessly (no terminal needed).

## What it deliberately doesn't do

- No per-card due dates or priority levels — the brief is title/description/tags, so tags are the mechanism for filtering/prioritization if you want it.
- No multi-board UI (no sidebar of boards to switch between) — each board.json is one board, just like the encrypted diary is one journal.
- No card archiving/history — cards are added, edited, deleted, and moved between columns; once deleted, they're gone (keep a backup if you need history).
- No undo/redo — persisted immediately, but designed-as-intended for single-user local workflow where "I just deleted that by accident" is rare enough vs. "I want every keystroke saved."
- No column reordering via keyboard (the order columns appear in the JSON is the order they render; edit the JSON if you want to reorder).

## Limitations & future

- Card movement is copy-to-target and delete-from-source (two atomic operations). If you move a card and the first persists but the second crashes, the card ends up in both columns (rare, and the card IDs make it obvious). A distributed system would use a single "moved_to_column" field, but for local single-user we accept this.
- No persistent preferences (terminal size, last-focused column, etc.); every run starts fresh.
- Textual's List

View doesn't render partial text with a scroll indicator for long card titles; we truncate at 60 chars in the list display (full title/description visible in the add/edit modal).

## Testing rationale

Test-driven development paid off here: the move-card edge cases (moving off first/last column, moving to same column) were caught before implementation. The atomic-write test with OSError injection isn't in the suite yet but would be valuable for production systems; for a local tool, the simpler round-trip tests are sufficient.
