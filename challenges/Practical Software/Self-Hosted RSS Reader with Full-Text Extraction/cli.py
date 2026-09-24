import typer
import httpx
from pathlib import Path
from storage import ArticleStore
from feeds import parse_feed
from extractor import fetch_and_extract
from datetime import datetime

app = typer.Typer()

DB_PATH = Path.home() / ".rss_reader" / "feeds.db"
DB_PATH.parent.mkdir(exist_ok=True)
store = ArticleStore(DB_PATH)


@app.command()
def add(url: str):
    """Register a new feed by URL."""
    try:
        response = httpx.get(url, timeout=10, follow_redirects=True)
        response.raise_for_status()
        feed_content = response.text
    except Exception as e:
        typer.echo(f"Error: Failed to fetch feed: {e}", err=True)
        raise typer.Exit(1)

    feed_title, articles = parse_feed(feed_content)

    feed_id = store.add_feed(url, feed_title)
    typer.echo(f"Added feed: {feed_title or url} (ID: {feed_id})")
    typer.echo(f"Found {len(articles)} articles")


@app.command()
def remove(feed_id: int):
    """Remove a feed by ID."""
    store.remove_feed(feed_id)
    typer.echo(f"Removed feed {feed_id}")


@app.command()
def list_feeds():
    """List all registered feeds."""
    feeds = store.list_feeds()
    if not feeds:
        typer.echo("No feeds registered")
        return

    for feed in feeds:
        typer.echo(f"[{feed['id']}] {feed['title'] or feed['url']}")
        typer.echo(f"     {feed['url']}")


@app.command()
def refresh():
    """Poll all feeds for new articles and extract full text."""
    feeds = store.list_feeds()
    if not feeds:
        typer.echo("No feeds to refresh")
        return

    total_new = 0
    for feed in feeds:
        new_count = 0
        try:
            response = httpx.get(feed["url"], timeout=10, follow_redirects=True)
            response.raise_for_status()
            feed_content = response.text
        except Exception as e:
            typer.echo(f"Error fetching {feed['url']}: {e}", err=True)
            continue

        feed_title, articles = parse_feed(feed_content)

        for article in articles:
            existing = store.get_article_by_link(feed["id"], article.link)
            if existing:
                continue

            article_id = store.add_article(
                feed["id"],
                article.title,
                article.link,
                article.published_at,
                article.summary,
            )

            if article_id and article.link:
                extracted = fetch_and_extract(article.link)
                if extracted.text:
                    store.update_article_full_text(article_id, extracted.text, extracted.title)

            new_count += 1

        total_new += new_count
        typer.echo(f"{feed['title'] or feed['url']}: +{new_count} articles")

    typer.echo(f"Total new articles: {total_new}")


@app.command()
def list(
    feed_id: int | None = typer.Option(None, "--feed", help="Filter by feed ID"),
    unread: bool = typer.Option(False, "--unread", help="Show only unread articles"),
):
    """List articles."""
    articles = store.list_articles(feed_id=feed_id, unread_only=unread)
    if not articles:
        typer.echo("No articles found")
        return

    for article in articles:
        read_marker = " " if article["read_flag"] else "*"
        typer.echo(f"[{read_marker}] [{article['id']}] {article['title']}")
        if article["published_at"]:
            typer.echo(f"    {article['published_at']}")


@app.command()
def read(article_id: int):
    """Print full extracted text of an article and mark as read."""
    article = store.get_article(article_id)
    if not article:
        typer.echo(f"Article {article_id} not found", err=True)
        raise typer.Exit(1)

    typer.echo(f"Title: {article['title']}")
    typer.echo(f"Link: {article['link']}")
    typer.echo()

    if article["full_text"]:
        typer.echo(article["full_text"])
    elif article["summary"]:
        typer.echo(article["summary"])
    else:
        typer.echo("(No full text available)")

    store.mark_as_read(article_id)


@app.command()
def search(query: str):
    """Full-text search articles."""
    results = store.search_articles(query)
    if not results:
        typer.echo(f"No articles found matching '{query}'")
        return

    typer.echo(f"Found {len(results)} articles:")
    for article in results:
        read_marker = " " if article["read_flag"] else "*"
        typer.echo(f"[{read_marker}] [{article['id']}] {article['title']}")


if __name__ == "__main__":
    app()
