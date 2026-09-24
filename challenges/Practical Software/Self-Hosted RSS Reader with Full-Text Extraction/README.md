# Self-Hosted RSS Reader with Full-Text Extraction

**Category:** Practical Software
**Difficulty:** I (brief: "Parse feeds, fetch full article via readability algorithm.")

**Status:** Implemented (Python)

A local RSS reader that polls feed URLs, stores articles in SQLite, extracts
full article text from HTML using readability heuristics, and makes everything
searchable with full-text search. All processing is local; no external service
accounts needed.

## The actually-hard part: full-text extraction is inherently heuristic

Real-world HTML is messy. Sites use different markup patterns for articles —
some semantic, some not. Paywalls hide content. Scripts and ads inject noise.
`trafilatura` implements industry-standard heuristics (similar to Readability,
Diffbot, etc.), but **heuristics have limits.** No single algorithm works for
all sites.

**What works well:** articles with semantic HTML (`<article>`, proper
`<title>`), clean newspaper-style sites, and blogs. **What's hard:**
content behind login walls, dynamic JS-rendered content (needs a headless
browser, which we don't run here), wildly unusual layouts, mixed
English/non-Latin scripts. Some sites intentionally strip article text from
their feeds and force you to visit the web page for the full story — we have
no way around that.

**Feed deduplication** also requires care. Articles can appear in multiple feeds
or get reshared. We deduplicate by feed + link (preferring link as it's
globally unique), and separately by GUID when present. Re-polling a feed should
never duplicate articles in storage. Covered by `test_storage.py` with
duplicate-insertion tests.

## Design

- **`storage.py`** — SQLite: `feeds` table (url, title, created_at), `articles`
  table (feed_id, title, link, published_at, summary, full_text, read_flag),
  and an FTS5 virtual table for full-text search over title/text. CRUD, read
  state, and search are all handled here. One connection per operation; no
  connection pooling (single-user tool). Dedup is enforced at the database
  level via UNIQUE(feed_id, link).

- **`extractor.py`** — Core extraction function `extract_from_html(html: str)`
  takes raw HTML and returns `ExtractedArticle(title, text)`. Offline-testable
  with fixture HTML. Separate `fetch_and_extract(url)` wraps the HTTP call
  (using `httpx`) for real usage; CLI and tests can mock the fetch layer
  independently of extraction logic.

- **`feeds.py`** — Feed parsing with `feedparser`. Parses RSS 2.0, Atom, and
  variants. `parse_feed(content: str)` takes feed XML (not a URL) for offline
  testing. Returns `(feed_title, [FeedArticle, ...])`. Handles missing or
  malformed dates gracefully (returns None rather than crashing). Articles
  without HTTP(S) links are filtered out (some feeds have GUIDs that look like
  URIs but aren't URLs).

- **`cli.py`** — Typer app with commands: `add <url>` (fetch feed, register),
  `remove <id>`, `list-feeds`, `refresh` (poll all feeds, fetch+extract new
  articles), `list [--feed ID] [--unread]`, `read <id>` (print full text, mark
  read), `search <query>` (FTS5). Database lives in `~/.rss_reader/feeds.db`
  for persistence across runs.

## What it deliberately doesn't do

- **No database abstraction layer.** SQLite only. No migrations or schema
  versioning — breaking schema changes would reset the store.
- **No account/multi-user support.** Single shared database.
- **No OPML import/export.** Add feeds one at a time via CLI.
- **No feed refresh scheduling.** Call `refresh` manually or set up a cron job
  externally.
- **No paywalled-content workaround.** If a site strips content from the feed
  and requires login, we can't extract what isn't there.
- **No proxy/SOCKS support in the HTTP client.** Not needed for a home tool
  behind your own network.

## Usage

```bash
cd "challenges/Practical Software/Self-Hosted RSS Reader with Full-Text Extraction"

uv run --with feedparser --with trafilatura --with httpx --with typer \
    python cli.py add https://xkcd.com/feed.xml
# -> Added feed: xkcd (ID: 1)
# -> Found 50 articles

uv run --with feedparser --with trafilatura --with httpx --with typer \
    python cli.py refresh
# -> Polling all feeds...
# -> xkcd: +3 articles

uv run --with feedparser --with trafilatura --with httpx --with typer \
    python cli.py list --unread
# -> [*] [15] xkcd: Sights of 2024
#       2024-01-20T12:00:00

uv run --with feedparser --with trafilatura --with httpx --with typer \
    python cli.py read 15
# -> Title: xkcd: Sights of 2024
# -> Link: https://xkcd.com/2856/
# -> [extracted full article text...]

uv run --with feedparser --with trafilatura --with httpx --with typer \
    python cli.py search python
# -> Found 3 articles
# -> [*] [42] Python Release Candidate

uv run --with feedparser --with trafilatura --with httpx --with typer \
    --with pytest pytest -q    # 43 tests
```

Feeds are polled from arbitrary URLs — no centralized directory. `refresh`
fetches each registered feed, extracts new articles' full text in real-time,
and stores them. Article full text is fetched on demand during `refresh`, not
re-fetched every time you call `read` — the text is cached in the database.

`search <query>` uses SQLite FTS5, which supports boolean operators (`AND`,
`OR`, prefix queries with `*`, phrase queries with double quotes). Example:
`search "python AND django"` finds articles mentioning both.

## Tests

43 pytest cases across four files.

`test_storage.py` (19 cases) covers CRUD operations, duplicate rejection,
ordering by published date, the read/unread state, and FTS5 search (basic,
phrase, and multiple-match queries).

`test_feeds.py` (7 cases) parse fixture RSS and Atom feeds (with and without
GUIDs, with and without publish dates) to verify feedparser integration and
date extraction. Also tests that items without valid URLs are filtered.

`test_extractor.py` (8 cases) feed the extraction function fixture HTML pages
(articles with boilerplate nav/sidebar, clean articles, articles with embedded
ads/scripts, empty/minimal HTML) and verify sensible text is extracted and
obvious noise (tracking scripts, "Tracking" strings) is not. Tests don't make
HTTP requests — all inputs are raw HTML strings.

`test_cli.py` (9 cases) drive the Typer app through `CliRunner`, mocking all
HTTP calls (feed fetches and article fetches). Tests cover `add`, `list-feeds`,
`remove`, `refresh`, `list` (all, by feed, unread only), `read`, `search`, and
error cases (article not found, no results).

The suite runs warning-free and with zero network access: all fixtures are
inline strings, all HTTP is mocked, and no external services are contacted.

Deprecation warning about `datetime.utcnow()` is silenced in favor of
`datetime.now(timezone.utc)`.
