"""A pipeline on disk: SQL models in layer folders plus a ``sources.toml`` naming the raw tables.

Layout::

    pipeline/
      sources.toml                # [sources.<table>] path = "relative/or/absolute/file"
      staging/stg_lines.sql       # one SELECT per file; the file name is the table name
      intermediate/...
      marts/...

A model refers to other models and to sources by their bare table name; there is no templating, so
the files are plain DuckDB SQL you can paste into any DuckDB shell. The folder a model sits in is its
layer, used only to lay out the graph.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ProjectError(ValueError):
    """The pipeline folder is not usable; the message says how."""


@dataclass(frozen=True, slots=True)
class Source:
    name: str
    path: Path

    @property
    def reader(self) -> str:
        """The DuckDB table function that reads this file."""
        suffix = self.path.suffix.lower()
        if suffix == ".parquet":
            return "read_parquet"
        if suffix in {".csv", ".tsv", ".gz"}:
            return "read_csv"
        raise ProjectError(
            f"source {self.name}: unsupported file type {self.path.suffix!r}"
        )

    def relation(self) -> str:
        """A FROM-clause expression for this source; the path is escaped as a SQL string."""
        quoted = str(self.path).replace("\\", "/").replace("'", "''")
        return f"{self.reader}('{quoted}')"


@dataclass(frozen=True, slots=True)
class Model:
    name: str
    layer: str
    sql: str
    path: Path


@dataclass(frozen=True, slots=True)
class Project:
    root: Path
    sources: tuple[Source, ...]
    models: tuple[Model, ...]

    def model(self, name: str) -> Model:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(name)

    @property
    def layers(self) -> list[str]:
        """Layer names in first-seen order: sources first, then folders by name."""
        seen: list[str] = []
        for m in self.models:
            if m.layer not in seen:
                seen.append(m.layer)
        return seen


def load_project(root: Path) -> Project:
    root = Path(root)
    if not root.is_dir():
        raise ProjectError(f"pipeline folder not found: {root}")
    cfg = root / "sources.toml"
    if not cfg.is_file():
        raise ProjectError(
            f"{cfg} not found: list the raw tables under [sources.<name>]"
        )
    try:
        raw = tomllib.loads(cfg.read_text(encoding="utf-8")).get("sources", {})
    except tomllib.TOMLDecodeError as exc:
        raise ProjectError(f"{cfg}: {exc}") from exc
    sources = []
    for name, spec in raw.items():
        if not IDENT.match(name):
            raise ProjectError(f"source name {name!r} must be a plain identifier")
        if not isinstance(spec, dict) or "path" not in spec:
            raise ProjectError(f'source {name}: needs a path = "..." entry')
        path = Path(spec["path"])
        path = path if path.is_absolute() else (root / path).resolve()
        if not path.is_file():
            raise ProjectError(f"source {name}: file not found: {path}")
        sources.append(Source(name, path))
    if not sources:
        raise ProjectError(f"{cfg}: no sources declared")

    models: list[Model] = []
    seen: dict[str, Path] = {s.name: cfg for s in sources}
    for path in sorted(root.rglob("*.sql"), key=lambda p: (p.parent.name, p.name)):
        name = path.stem
        if not IDENT.match(name):
            raise ProjectError(
                f"model file {path.name}: the name must be a plain identifier"
            )
        if name in seen:
            raise ProjectError(f"{name} is defined twice: {seen[name]} and {path}")
        seen[name] = path
        sql = path.read_text(encoding="utf-8").strip().rstrip(";").strip()
        if not sql:
            raise ProjectError(f"{path} is empty")
        layer = path.parent.name if path.parent != root else "models"
        models.append(Model(name, layer, sql, path))
    if not models:
        raise ProjectError(f"no .sql models under {root}")
    return Project(root, tuple(sources), tuple(models))
