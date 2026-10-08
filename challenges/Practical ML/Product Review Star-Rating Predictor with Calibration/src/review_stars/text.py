"""What the model reads: title/text assembly with the auto-title rule, and the de-duplication key.

Amazon pre-fills a review's title with "Five Stars" / "One Star" / ... from the rating the
reviewer already picked. In the 2023 data 10-15% of titles are exactly that and match the true
rating 99.8% of the time, so a model that sees them is partly reading the answer. The default
``title_text`` mode blanks them and keeps real titles ("Perfect fit"), which carry honest signal.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from review_stars.config import TEXT_MODES

AUTO_TITLE = re.compile(
    r"^\s*(one|two|three|four|five|[1-5])[\s-]*stars?[.!]*\s*$", re.IGNORECASE
)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)


def is_auto_title(title: str | None) -> bool:
    return bool(title) and AUTO_TITLE.match(title) is not None


def _clean(s: str | None) -> str:
    return " ".join(_BR.sub(" ", s or "").split())


def build_text(title: str | None, text: str | None, mode: str = "title_text") -> str:
    """The string that gets embedded. Never raises on missing parts: empty in, empty out."""
    if mode not in TEXT_MODES:
        raise ValueError(f"text_mode must be one of {TEXT_MODES}, not {mode!r}")
    body = _clean(text)
    if mode == "text_only":
        return body
    head = _clean(title)
    if mode == "title_text" and is_auto_title(head):
        head = ""
    if not head:
        return body
    return f"{head}. {body}" if body else head


def dedup_key(title: str | None, text: str | None) -> int:
    """A stable 63-bit key of the normalized (title, text) pair (NFKC, casefold, whitespace)."""

    def norm(s):
        return " ".join(unicodedata.normalize("NFKC", s or "").casefold().split())

    digest = hashlib.blake2b(
        f"{norm(title)}\x00{norm(text)}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") >> 1
