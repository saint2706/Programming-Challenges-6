import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
import cli
from storage import ArticleStore

runner = CliRunner()

SAMPLE_RSS = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <link>https://example.com</link>
    <item>
      <title>Article One</title>
      <link>https://example.com/article1</link>
      <guid>article-1</guid>
      <pubDate>Mon, 15 Jan 2024 10:00:00 GMT</pubDate>
      <description>Summary one</description>
    </item>
  </channel>
</rss>
"""

SAMPLE_ARTICLE_HTML = """
<html>
<head><title>Article One Full Text</title></head>
<body>
  <article>
    <h1>Article One</h1>
    <p>This is the full text of article one with substantial content.</p>
  </article>
</body>
</html>
"""


@patch("cli.httpx.get")
def test_add_feed(mock_get):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.object(cli, "DB_PATH", db_path):
            cli.store = ArticleStore(db_path)

            mock_response = MagicMock()
            mock_response.text = SAMPLE_RSS
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = runner.invoke(cli.app, ["add", "https://example.com/feed"])
            assert result.exit_code == 0
            assert "Added feed" in result.stdout
            assert "Test Feed" in result.stdout


@patch("cli.httpx.get")
def test_add_feed_fetch_error(mock_get):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.object(cli, "DB_PATH", db_path):
            cli.store = ArticleStore(db_path)
            mock_get.side_effect = Exception("Network error")
            result = runner.invoke(cli.app, ["add", "https://example.com/feed"])
            assert result.exit_code == 1
            assert "Error" in result.stdout or "Error" in result.stderr


def test_list_feeds_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.object(cli, "DB_PATH", db_path):
            cli.store = ArticleStore(db_path)
            result = runner.invoke(cli.app, ["list-feeds"])
            assert result.exit_code == 0
            assert "No feeds" in result.stdout


@patch("cli.httpx.get")
def test_list_feeds(mock_get):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.object(cli, "DB_PATH", db_path):
            cli.store = ArticleStore(db_path)

            mock_response = MagicMock()
            mock_response.text = SAMPLE_RSS
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            runner.invoke(cli.app, ["add", "https://example.com/feed"])
            result = runner.invoke(cli.app, ["list-feeds"])
            assert result.exit_code == 0
            assert "Test Feed" in result.stdout


def test_remove_feed():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.object(cli, "DB_PATH", db_path):
            cli.store = ArticleStore(db_path)
            feed_id = cli.store.add_feed("https://example.com/feed", "Test")
            result = runner.invoke(cli.app, ["remove", str(feed_id)])
            assert result.exit_code == 0
            assert "Removed feed" in result.stdout


@patch("cli.fetch_and_extract")
@patch("cli.httpx.get")
def test_refresh_feed(mock_get, mock_extract):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.object(cli, "DB_PATH", db_path):
            cli.store = ArticleStore(db_path)

            mock_response = MagicMock()
            mock_response.text = SAMPLE_RSS
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            mock_extract.return_value = MagicMock(
                text="Full article content", title="Article One"
            )

            runner.invoke(cli.app, ["add", "https://example.com/feed"])
            result = runner.invoke(cli.app, ["refresh"])
            assert result.exit_code == 0
            assert "articles" in result.stdout.lower()


def test_list_articles_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.object(cli, "DB_PATH", db_path):
            cli.store = ArticleStore(db_path)
            result = runner.invoke(cli.app, ["list"])
            assert result.exit_code == 0
            assert "No articles" in result.stdout


@patch("cli.httpx.get")
def test_list_articles(mock_get):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.object(cli, "DB_PATH", db_path):
            cli.store = ArticleStore(db_path)

            mock_response = MagicMock()
            mock_response.text = SAMPLE_RSS
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            runner.invoke(cli.app, ["add", "https://example.com/feed"])

            feed_id = cli.store.list_feeds()[0]["id"]
            cli.store.add_article(feed_id, "Test Article", "https://example.com/test")

            result = runner.invoke(cli.app, ["list"])
            assert result.exit_code == 0
            assert "Test Article" in result.stdout


def test_list_articles_unread_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.object(cli, "DB_PATH", db_path):
            cli.store = ArticleStore(db_path)

            feed_id = cli.store.add_feed("https://example.com/feed", "Feed")
            id1 = cli.store.add_article(feed_id, "Unread", "https://example.com/1")
            id2 = cli.store.add_article(feed_id, "Read", "https://example.com/2")
            cli.store.mark_as_read(id2)

            result = runner.invoke(cli.app, ["list", "--unread"])
            assert result.exit_code == 0
            assert "Unread" in result.stdout
            assert "Read" not in result.stdout or id2 not in result.stdout


def test_read_article():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.object(cli, "DB_PATH", db_path):
            cli.store = ArticleStore(db_path)

            feed_id = cli.store.add_feed("https://example.com/feed", "Feed")
            article_id = cli.store.add_article(
                feed_id,
                "Article",
                "https://example.com/article",
                full_text="Full text content here",
            )

            result = runner.invoke(cli.app, ["read", str(article_id)])
            assert result.exit_code == 0
            assert "Full text content" in result.stdout
            assert "Article" in result.stdout

            article = cli.store.get_article(article_id)
            assert article["read_flag"] == 1


def test_read_article_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.object(cli, "DB_PATH", db_path):
            cli.store = ArticleStore(db_path)
            result = runner.invoke(cli.app, ["read", "999"])
            assert result.exit_code == 1
            assert "not found" in result.stdout or "not found" in result.stderr


def test_search_articles():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.object(cli, "DB_PATH", db_path):
            cli.store = ArticleStore(db_path)

            feed_id = cli.store.add_feed("https://example.com/feed", "Feed")
            cli.store.add_article(
                feed_id,
                "Python Tutorial",
                "https://example.com/1",
                full_text="Learn Python programming",
            )
            cli.store.add_article(
                feed_id, "JavaScript", "https://example.com/2", full_text="Learn JS"
            )

            result = runner.invoke(cli.app, ["search", "python"])
            assert result.exit_code == 0
            assert "Python Tutorial" in result.stdout


def test_search_no_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch.object(cli, "DB_PATH", db_path):
            cli.store = ArticleStore(db_path)
            result = runner.invoke(cli.app, ["search", "nonexistent"])
            assert result.exit_code == 0
            assert "No articles" in result.stdout
