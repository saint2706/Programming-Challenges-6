"""A loaded table and the operations on it."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Self

from pivot_tool.ffi import Engine, engine


def read_bytes(path: Path) -> bytes:
    """The file's bytes; ``.gz`` files are decompressed."""
    raw = Path(path).read_bytes()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


class Table:
    """A CSV held by the engine. Close it (or use ``with``) to release the memory."""

    def __init__(
        self,
        data: bytes,
        *,
        delimiter: str | None = None,
        trim: bool = True,
        eng: Engine | None = None,
    ):
        self._eng = eng or engine()
        payload: dict[str, Any] = {"trim": trim}
        if delimiter:
            payload["delimiter"] = delimiter
        info = self._eng.call("load_csv", payload, data)
        self.id: int = info["table"]
        self.rows: int = info["rows"]
        self.fields: list[dict] = info["fields"]

    @classmethod
    def from_path(cls, path: str | Path, **kwargs) -> Table:
        return cls(read_bytes(Path(path)), **kwargs)

    @property
    def names(self) -> list[str]:
        return [f["name"] for f in self.fields]

    def pivot(self, spec: dict) -> dict:
        return self._eng.call("pivot", {"table": self.id, "spec": spec})

    def crosstab(self, spec: dict) -> dict:
        return self._eng.call("crosstab", {"table": self.id, "spec": spec})

    def drill(
        self,
        spec: dict,
        row_path: list[str] | None = None,
        col_path: list[str] | None = None,
        limit: int = 100,
    ) -> dict:
        payload = {
            "table": self.id,
            "spec": spec,
            "row_path": row_path or [],
            "col_path": col_path or [],
            "limit": limit,
        }
        return self._eng.call("drill", payload)

    def values(self, field: str, limit: int = 500) -> dict:
        return self._eng.call(
            "values", {"table": self.id, "field": field, "limit": limit}
        )

    def close(self) -> None:
        if getattr(self, "id", None) is not None:
            try:
                self._eng.call("drop", {"table": self.id})
            finally:
                self.id = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - the interpreter may be shutting down
            pass
