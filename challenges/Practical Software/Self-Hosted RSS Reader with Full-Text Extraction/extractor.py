import trafilatura
import httpx
from typing import NamedTuple
from lxml import etree


class ExtractedArticle(NamedTuple):
    title: str | None
    text: str | None


def extract_from_html(html_content: str) -> ExtractedArticle:
    """
    Extract article title and text from raw HTML.
    Takes HTML string (not URL) for offline testability.
    """
    if not html_content:
        return ExtractedArticle(None, None)

    doc = trafilatura.extract(html_content, include_comments=False, output_format="txt")

    try:
        tree = trafilatura.parse(html_content)
        title = tree.xpath("//title/text()")
        title = title[0] if title else None
    except Exception:
        title = None

    return ExtractedArticle(title=title, text=doc)


def fetch_and_extract(url: str, timeout: int = 10) -> ExtractedArticle:
    """
    Fetch a URL and extract its article content.
    For real CLI usage; tests mock this or the HTTP layer.
    """
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        return extract_from_html(response.text)
    except Exception:
        return ExtractedArticle(None, None)
