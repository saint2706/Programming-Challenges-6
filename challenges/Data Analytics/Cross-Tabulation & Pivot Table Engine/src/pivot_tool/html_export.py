"""One self-contained HTML file: the engine as WebAssembly, the data, and the pivot page.

The file makes no network request. The WebAssembly module and the CSV travel gzip-compressed and
base64-encoded in two inert ``<script type="application/octet-stream">`` blocks (base64 has no
characters that can end a script); the page configuration travels as JSON with ``<``, ``>``, ``&``
and the two JavaScript line separators escaped, so a title or preset name cannot break out.
"""

from __future__ import annotations

import base64
import gzip
import html
import json
from importlib import resources
from typing import Any

WASM_NAME = "pivot_ffi.wasm"


def web_asset(name: str) -> bytes:
    return resources.files("pivot_tool").joinpath("web", name).read_bytes()


def _u(code: int) -> str:
    """The six characters backslash, u and four hex digits: JSON decodes them back to ``code``."""
    return chr(92) + "u" + format(code, "04x")


_ESCAPES = {chr(c): _u(c) for c in (ord("<"), ord(">"), ord("&"), 0x2028, 0x2029)}


def embed_json(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return "".join(_ESCAPES.get(ch, ch) for ch in text)


def b64_gzip(data: bytes) -> str:
    return base64.b64encode(gzip.compress(data, compresslevel=9, mtime=0)).decode(
        "ascii"
    )


def build_html(
    csv: bytes,
    title: str,
    presets: list[dict[str, Any]] | None = None,
    initial: dict[str, Any] | None = None,
    wasm: bytes | None = None,
) -> str:
    page = web_asset("app.html").decode("utf-8")
    page = page.replace(
        "<title>Pivot table</title>", f"<title>{html.escape(title)}</title>", 1
    )
    page = page.replace("/*CSS*/", web_asset("app.css").decode("utf-8"))
    page = page.replace("/*ENGINE*/", web_asset("engine.js").decode("utf-8"))
    page = page.replace("/*APP*/", web_asset("app.js").decode("utf-8"))
    config = {"title": title, "presets": presets or [], "initial": initial or {}}
    # last, so nothing in the data can be mistaken for a placeholder
    page = page.replace("/*CONFIG*/", embed_json(config))
    page = page.replace(
        "/*WASM*/", b64_gzip(wasm if wasm is not None else web_asset(WASM_NAME))
    )
    return page.replace("/*DATA*/", b64_gzip(csv))
