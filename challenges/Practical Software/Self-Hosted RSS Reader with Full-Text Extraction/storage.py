import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager


class ArticleStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feeds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feed_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    link TEXT NOT NULL,
                    published_at TEXT,
                    summary TEXT,
                    full_text TEXT,
                    read_flag INTEGER DEFAULT 0,
                    FOREIGN KEY(feed_id) REFERENCES feeds(id),
                    UNIQUE(feed_id, link)
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
                    title, full_text, content=articles, content_rowid=id
                )
                """
            )
            conn.commit()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def add_feed(self, url: str, title: str | None = None) -> int:
        with self._connection() as conn:
            cursor = conn.execute(
                "INSERT INTO feeds (url, title, created_at) VALUES (?, ?, ?)",
                (url, title, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            return cursor.lastrowid

    def remove_feed(self, feed_id: int) -> bool:
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM articles WHERE feed_id = ?", (feed_id,))
            conn.execute("DELETE FROM feeds WHERE id = ?", (feed_id,))
            conn.commit()
            return cursor.rowcount > 0 or True

    def get_feed(self, feed_id: int) -> dict | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT id, url, title, created_at FROM feeds WHERE id = ?", (feed_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_feeds(self) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id, url, title, created_at FROM feeds ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def add_article(
        self,
        feed_id: int,
        title: str,
        link: str,
        published_at: str | None = None,
        summary: str | None = None,
        full_text: str | None = None,
    ) -> int | None:
        with self._connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO articles (feed_id, title, link, published_at, summary, full_text)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (feed_id, title, link, published_at, summary, full_text),
                )
                article_id = cursor.lastrowid

                if title or full_text:
                    conn.execute(
                        "INSERT INTO articles_fts (rowid, title, full_text) VALUES (?, ?, ?)",
                        (article_id, title or "", full_text or ""),
                    )
                conn.commit()
                return article_id
            except sqlite3.IntegrityError:
                return None

    def get_article(self, article_id: int) -> dict | None:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT id, feed_id, title, link, published_at, summary, full_text, read_flag
                FROM articles WHERE id = ?
                """,
                (article_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_articles(
        self, feed_id: int | None = None, unread_only: bool = False
    ) -> list[dict]:
        with self._connection() as conn:
            query = "SELECT id, feed_id, title, link, published_at, summary, full_text, read_flag FROM articles WHERE 1=1"
            params = []
            if feed_id is not None:
                query += " AND feed_id = ?"
                params.append(feed_id)
            if unread_only:
                query += " AND read_flag = 0"
            query += " ORDER BY published_at DESC, id DESC"

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def mark_as_read(self, article_id: int) -> bool:
        with self._connection() as conn:
            cursor = conn.execute("UPDATE articles SET read_flag = 1 WHERE id = ?", (article_id,))
            conn.commit()
            return cursor.rowcount > 0

    def mark_as_unread(self, article_id: int) -> bool:
        with self._connection() as conn:
            cursor = conn.execute("UPDATE articles SET read_flag = 0 WHERE id = ?", (article_id,))
            conn.commit()
            return cursor.rowcount > 0

    def search_articles(self, query: str) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT articles.id, articles.feed_id, articles.title, articles.link,
                       articles.published_at, articles.summary, articles.full_text, articles.read_flag
                FROM articles
                JOIN articles_fts ON articles.id = articles_fts.rowid
                WHERE articles_fts MATCH ?
                ORDER BY articles.published_at DESC, articles.id DESC
                """,
                (query,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_article_by_link(self, feed_id: int, link: str) -> dict | None:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT id, feed_id, title, link, published_at, summary, full_text, read_flag
                FROM articles WHERE feed_id = ? AND link = ?
                """,
                (feed_id, link),
            ).fetchone()
            return dict(row) if row else None

    def update_article_full_text(self, article_id: int, full_text: str, title: str | None = None) -> bool:
        with self._connection() as conn:
            old = conn.execute(
                "SELECT title, full_text FROM articles WHERE id = ?", (article_id,)
            ).fetchone()
            if old is None:
                return False

            new_title = title if title else old["title"]
            cursor = conn.execute(
                "UPDATE articles SET full_text = ?, title = ? WHERE id = ?",
                (full_text, new_title, article_id),
            )

            # articles_fts is an external-content table (content=articles): the
            # index isn't kept in sync automatically, and `INSERT OR REPLACE`
            # against it does NOT remove the old entry's postings — it silently
            # corrupts the index so stale tokens keep matching forever. The
            # documented fix is the explicit delete-with-old-values-then-insert
            # pair below.
            conn.execute(
                "INSERT INTO articles_fts (articles_fts, rowid, title, full_text) VALUES ('delete', ?, ?, ?)",
                (article_id, old["title"] or "", old["full_text"] or ""),
            )
            conn.execute(
                "INSERT INTO articles_fts (rowid, title, full_text) VALUES (?, ?, ?)",
                (article_id, new_title or "", full_text or ""),
            )
            conn.commit()
            return cursor.rowcount > 0
