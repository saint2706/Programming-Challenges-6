"""Walk a repository the way git would: honour nested ``.gitignore`` files, skip build junk."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

ALWAYS_IGNORED_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", ".tox",
        "dist", "build", "target", ".next", ".nuxt", ".gradle", ".idea", ".vscode",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", "vendor", "coverage",
    }
)  # fmt: skip
ALWAYS_IGNORED_FILES = frozenset({".DS_Store", "Thumbs.db"})

# Only real programming languages: counting Markdown or JSON would make every repo "mostly docs".
LANGUAGES = {
    ".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".jsx": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript", ".rs": "Rust",
    ".go": "Go", ".java": "Java", ".kt": "Kotlin", ".c": "C", ".h": "C", ".cc": "C++",
    ".cpp": "C++", ".hpp": "C++", ".cs": "C#", ".rb": "Ruby", ".php": "PHP",
    ".swift": "Swift", ".sh": "Shell", ".bash": "Shell", ".lua": "Lua", ".scala": "Scala",
    ".dart": "Dart", ".ex": "Elixir", ".exs": "Elixir", ".hs": "Haskell", ".vue": "Vue",
    ".svelte": "Svelte", ".html": "HTML", ".css": "CSS", ".scss": "CSS", ".sql": "SQL",
}  # fmt: skip

MAX_FILES = 50_000


@dataclass
class RepoScan:
    root: Path
    files: list[str] = field(
        default_factory=list
    )  # posix paths relative to root, sorted
    dirs: list[str] = field(default_factory=list)
    language_bytes: dict[str, int] = field(default_factory=dict)
    truncated: bool = False
    file_set: frozenset[str] = frozenset()

    def has_file(self, relpath: str) -> bool:
        return relpath in self.file_set

    def read_text(self, relpath: str, limit: int = 256_000) -> str:
        """Text of a scanned file, empty when it cannot be read (binary, vanished, unreadable)."""
        try:
            with open(self.root / relpath, "rb") as handle:
                return handle.read(limit).decode("utf-8", errors="replace")
        except OSError:
            return ""

    def language_shares(self) -> list[tuple[str, float]]:
        """Languages by share of source bytes, largest first (ties alphabetical)."""
        total = sum(self.language_bytes.values())
        if not total:
            return []
        ranked = sorted(self.language_bytes.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(name, size / total) for name, size in ranked]


def _read_gitignore(directory: Path) -> pathspec.GitIgnoreSpec | None:
    try:
        lines = (
            (directory / ".gitignore")
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        )
    except OSError:
        return None
    return pathspec.GitIgnoreSpec.from_lines(lines)


def scan_repo(root: Path, max_files: int = MAX_FILES) -> RepoScan:
    """Collect every non-ignored file and directory under ``root``.

    Each directory's own ``.gitignore`` applies to its subtree, matched relative to that directory.
    Symlinked directories are listed but never entered, so a link cycle cannot hang the walk.
    """
    root = root.resolve()
    result = RepoScan(root=root)

    def walk(
        directory: Path, rel: str, specs: list[tuple[str, pathspec.GitIgnoreSpec]]
    ) -> None:
        own = _read_gitignore(directory)
        if own is not None:
            specs = [*specs, (rel, own)]
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            return
        for entry in entries:
            if len(result.files) >= max_files:
                result.truncated = True
                return
            is_link_dir = entry.is_symlink() and entry.is_dir()
            is_dir = entry.is_dir(follow_symlinks=False) or is_link_dir
            name = entry.name
            if (is_dir and name in ALWAYS_IGNORED_DIRS) or (
                not is_dir and name in ALWAYS_IGNORED_FILES
            ):
                continue
            if name.endswith(".egg-info") and is_dir:
                continue
            child_rel = f"{rel}/{name}" if rel else name
            if _is_ignored(child_rel, is_dir, specs):
                continue
            if is_dir:
                result.dirs.append(child_rel)
                if not is_link_dir:
                    walk(Path(entry.path), child_rel, specs)
            else:
                result.files.append(child_rel)
                language = LANGUAGES.get(os.path.splitext(name)[1].lower())
                if language:
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    result.language_bytes[language] = (
                        result.language_bytes.get(language, 0) + size
                    )

    walk(root, "", [])
    result.files.sort()
    result.dirs.sort()
    result.file_set = frozenset(result.files)
    return result


def _is_ignored(
    child_rel: str, is_dir: bool, specs: list[tuple[str, pathspec.GitIgnoreSpec]]
) -> bool:
    for base, spec in specs:
        local = child_rel[len(base) + 1 :] if base else child_rel
        if spec.match_file(local + "/" if is_dir else local):
            return True
    return False
