"""One self-contained HTML file: the graph, the viewer and, optionally, the empirical verdicts.

Everything is inlined (no fonts, scripts or styles are fetched) and the data travels as JSON inside
a ``<script type="application/json">`` block with ``<``, ``>``, ``&`` and the two JavaScript line
separators escaped, so a column called ``</script><img onerror=...>`` is inert. The viewer builds
its DOM with ``textContent`` only.
"""

from __future__ import annotations

import html
import json
from importlib import resources

from data_lineage.graph import Graph
from data_lineage.verify import Verification


def _u(code: int) -> str:
    """The six characters backslash, u and four hex digits: JSON decodes them back to ``code``."""
    return chr(92) + "u" + format(code, "04x")


_ESCAPES = {chr(c): _u(c) for c in (ord("<"), ord(">"), ord("&"), 0x2028, 0x2029)}


def _asset(name: str) -> str:
    return (
        resources.files("data_lineage")
        .joinpath("assets", name)
        .read_text(encoding="utf-8")
    )


def embed_json(payload: dict) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return "".join(_ESCAPES.get(ch, ch) for ch in text)


def build_html(
    graph: Graph, verification: Verification | None = None, title: str = "Data lineage"
) -> str:
    payload = graph.to_json()
    payload["title"] = title
    payload["verification"] = verification.to_json() if verification else None
    page = _asset("report.html")
    page = page.replace(
        "<title>Data lineage</title>", f"<title>{html.escape(title)}</title>", 1
    )
    page = page.replace("/*CSS*/", _asset("report.css"))
    page = page.replace("/*JS*/", _asset("report.js"))
    # last, so a "/*JS*/" inside the data can never be expanded
    return page.replace("/*DATA*/", embed_json(payload))
