import feedparser
from typing import NamedTuple
from datetime import datetime


class FeedArticle(NamedTuple):
    title: str
    link: str
    guid: str | None
    published_at: str | None
    summary: str | None


def parse_feed(feed_content: str) -> tuple[str | None, list[FeedArticle]]:
    """
    Parse RSS/Atom feed content and return (feed_title, articles).
    Feed content is raw XML string for offline testability.
    """
    feed = feedparser.parse(feed_content)

    if not feed.get("entries"):
        return feed.get("feed", {}).get("title"), []

    feed_title = feed.get("feed", {}).get("title")
    articles = []

    for entry in feed.entries:
        title = entry.get("title", "")
        link = entry.get("link", "")
        guid = entry.get("id") or entry.get("guid")
        published_at = _extract_published_date(entry)
        summary = entry.get("summary") or entry.get("description")

        if link and link.startswith(("http://", "https://")):
            articles.append(
                FeedArticle(
                    title=title,
                    link=link,
                    guid=guid,
                    published_at=published_at,
                    summary=summary,
                )
            )

    return feed_title, articles


def _extract_published_date(entry: dict) -> str | None:
    """Extract published date from feed entry in ISO format."""
    for key in ["published", "updated", "created"]:
        if key in entry:
            try:
                time_tuple = entry[f"{key}_parsed"]
                dt = datetime(*time_tuple[:6])
                return dt.isoformat()
            except (KeyError, TypeError, ValueError):
                pass
    return None
