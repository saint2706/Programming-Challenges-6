import sqlite3
import tempfile
from pathlib import Path
from storage import ArticleStore


def test_init_db_creates_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        conn = sqlite3.connect(store.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "feeds" in tables
        assert "articles" in tables
        assert "articles_fts" in tables


def test_add_feed():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        feed_id = store.add_feed("https://example.com/feed", "Example Feed")
        assert feed_id > 0
        feed = store.get_feed(feed_id)
        assert feed["url"] == "https://example.com/feed"
        assert feed["title"] == "Example Feed"


def test_add_feed_duplicate_url_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        url = "https://example.com/feed"
        store.add_feed(url, "First")
        try:
            store.add_feed(url, "Second")
            assert False, "Should have raised IntegrityError"
        except sqlite3.IntegrityError:
            pass


def test_list_feeds():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        id1 = store.add_feed("https://example.com/feed1", "Feed 1")
        id2 = store.add_feed("https://example.com/feed2", "Feed 2")
        feeds = store.list_feeds()
        assert len(feeds) == 2
        assert feeds[0]["id"] == id2  # reverse chronological
        assert feeds[1]["id"] == id1


def test_remove_feed():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        feed_id = store.add_feed("https://example.com/feed", "Feed")
        store.remove_feed(feed_id)
        feed = store.get_feed(feed_id)
        assert feed is None


def test_add_article():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        feed_id = store.add_feed("https://example.com/feed", "Feed")
        article_id = store.add_article(
            feed_id,
            "Article Title",
            "https://example.com/article1",
            "2024-01-15T10:00:00",
            "Summary text",
            "Full article text here",
        )
        assert article_id > 0
        article = store.get_article(article_id)
        assert article["title"] == "Article Title"
        assert article["link"] == "https://example.com/article1"
        assert article["full_text"] == "Full article text here"


def test_add_article_duplicate_link_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        feed_id = store.add_feed("https://example.com/feed", "Feed")
        link = "https://example.com/article"
        id1 = store.add_article(feed_id, "Title 1", link)
        assert id1 is not None
        id2 = store.add_article(feed_id, "Title 2", link)
        assert id2 is None


def test_list_articles_ordered_by_published():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        feed_id = store.add_feed("https://example.com/feed", "Feed")
        store.add_article(feed_id, "Old", "https://example.com/1", "2024-01-10")
        store.add_article(feed_id, "New", "https://example.com/2", "2024-01-20")
        articles = store.list_articles()
        assert len(articles) == 2
        assert articles[0]["title"] == "New"
        assert articles[1]["title"] == "Old"


def test_list_articles_by_feed():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        feed1_id = store.add_feed("https://example.com/feed1", "Feed 1")
        feed2_id = store.add_feed("https://example.com/feed2", "Feed 2")
        store.add_article(feed1_id, "Article 1", "https://example.com/a1")
        store.add_article(feed2_id, "Article 2", "https://example.com/a2")
        articles = store.list_articles(feed_id=feed1_id)
        assert len(articles) == 1
        assert articles[0]["title"] == "Article 1"


def test_mark_as_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        feed_id = store.add_feed("https://example.com/feed", "Feed")
        article_id = store.add_article(feed_id, "Title", "https://example.com/article")
        article = store.get_article(article_id)
        assert article["read_flag"] == 0
        store.mark_as_read(article_id)
        article = store.get_article(article_id)
        assert article["read_flag"] == 1


def test_mark_as_unread():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        feed_id = store.add_feed("https://example.com/feed", "Feed")
        article_id = store.add_article(feed_id, "Title", "https://example.com/article")
        store.mark_as_read(article_id)
        store.mark_as_unread(article_id)
        article = store.get_article(article_id)
        assert article["read_flag"] == 0


def test_list_articles_unread_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        feed_id = store.add_feed("https://example.com/feed", "Feed")
        id1 = store.add_article(feed_id, "Unread", "https://example.com/1")
        id2 = store.add_article(feed_id, "Read", "https://example.com/2")
        store.mark_as_read(id2)
        articles = store.list_articles(unread_only=True)
        assert len(articles) == 1
        assert articles[0]["id"] == id1


def test_search_articles():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        feed_id = store.add_feed("https://example.com/feed", "Feed")
        store.add_article(
            feed_id, "Python Tutorial", "https://example.com/1", full_text="Learn Python programming"
        )
        store.add_article(
            feed_id, "JavaScript Guide", "https://example.com/2", full_text="Learn JavaScript"
        )
        store.add_article(feed_id, "Other Article", "https://example.com/3", full_text="Something else")
        results = store.search_articles("python")
        assert len(results) == 1
        assert results[0]["title"] == "Python Tutorial"


def test_search_articles_multiple_matches():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        feed_id = store.add_feed("https://example.com/feed", "Feed")
        store.add_article(
            feed_id, "Python Basics", "https://example.com/1", full_text="Python is great"
        )
        store.add_article(feed_id, "Python Advanced", "https://example.com/2", full_text="Advanced Python")
        results = store.search_articles("python")
        assert len(results) == 2


def test_update_article_full_text():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        feed_id = store.add_feed("https://example.com/feed", "Feed")
        article_id = store.add_article(
            feed_id, "Title", "https://example.com/article", full_text="Original text"
        )
        store.update_article_full_text(article_id, "Updated text", "New Title")
        article = store.get_article(article_id)
        assert article["full_text"] == "Updated text"
        assert article["title"] == "New Title"
        results = store.search_articles("updated")
        assert len(results) == 1


def test_update_article_full_text_removes_stale_fts_entries():
    # Regression test: a prior implementation used `INSERT OR REPLACE` against
    # the external-content FTS5 index, which doesn't remove the old row's
    # postings — searches for content that no longer exists kept matching.
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArticleStore(Path(tmpdir) / "test.db")
        feed_id = store.add_feed("https://example.com/feed", "Feed")
        article_id = store.add_article(
            feed_id, "Original Title", "https://example.com/article", full_text="Foxtrot content"
        )
        store.update_article_full_text(article_id, "Zulu content", "Replacement Title")

        assert store.search_articles("foxtrot") == []
        assert store.search_articles("original") == []
        results = store.search_articles("zulu")
        assert len(results) == 1
        assert results[0]["id"] == article_id
