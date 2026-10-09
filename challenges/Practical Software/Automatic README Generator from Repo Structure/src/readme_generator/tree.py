"""Render the directory tree with one-line annotations for well-known paths."""

from posixpath import basename

from readme_generator.scan import RepoScan

ANNOTATIONS = {
    "src": "source code", "lib": "library code", "app": "application code", "cmd": "command entry points",
    "pkg": "reusable packages", "internal": "private packages", "tests": "test suite", "test": "test suite",
    "__tests__": "test suite", "spec": "test suite", "docs": "documentation", "doc": "documentation",
    "scripts": "helper scripts", "bin": "executables", "examples": "usage examples",
    "migrations": "database migrations", "public": "static assets", "static": "static assets",
    "templates": "HTML templates", "assets": "static assets", "data": "datasets and fixtures",
    "config": "configuration", ".github": "GitHub configuration", "workflows": "CI workflows",
    "pyproject.toml": "Python project metadata", "package.json": "Node.js package manifest",
    "Cargo.toml": "Rust crate manifest", "go.mod": "Go module definition", "Gemfile": "Ruby dependencies",
    "composer.json": "PHP dependencies", "pom.xml": "Maven build", "Makefile": "task shortcuts",
    "Dockerfile": "container image", "requirements.txt": "pinned Python dependencies",
    "tsconfig.json": "TypeScript configuration", "LICENSE": "license text",
}  # fmt: skip

LOCKFILES = frozenset(
    {"uv.lock", "poetry.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock", "go.sum", "Gemfile.lock", "composer.lock", "bun.lockb"}
)  # fmt: skip
MAX_ENTRIES_PER_DIR = 12


def render_tree(
    scan: RepoScan, name: str, max_depth: int = 2, exclude: frozenset[str] = frozenset()
) -> str:
    """A ``tree``-style listing, directories first.

    ``exclude`` lists paths to omit; the README being generated is passed so that its own presence
    cannot change the tree and make regeneration non-idempotent. Lockfiles are always hidden.
    """
    children: dict[str, list[str]] = {}
    for path in [*scan.dirs, *scan.files]:
        if path in exclude or basename(path) in LOCKFILES:
            continue
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        children.setdefault(parent, []).append(path)
    dirs = set(scan.dirs)
    lines = [f"{name}/"]

    def emit(parent: str, prefix: str, depth: int) -> None:
        entries = sorted(
            children.get(parent, []), key=lambda p: (p not in dirs, basename(p).lower())
        )
        hidden = max(0, len(entries) - MAX_ENTRIES_PER_DIR)
        entries = entries[:MAX_ENTRIES_PER_DIR]
        for index, path in enumerate(entries):
            last = index == len(entries) - 1 and not hidden
            is_dir = path in dirs
            label = basename(path) + ("/" if is_dir else "")
            note = ANNOTATIONS.get(basename(path))
            lines.append(
                f"{prefix}{'└── ' if last else '├── '}{label}"
                + (f"  # {note}" if note else "")
            )
            if is_dir and depth < max_depth:
                emit(path, prefix + ("    " if last else "│   "), depth + 1)
        if hidden:
            lines.append(f"{prefix}└── … and {hidden} more")

    emit("", "", 1)
    return "\n".join(lines)
