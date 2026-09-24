from feeds import parse_feed

SIMPLE_RSS = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Example Feed</title>
    <link>https://example.com</link>
    <item>
      <title>Article One</title>
      <link>https://example.com/article1</link>
      <guid>article-1</guid>
      <pubDate>Mon, 15 Jan 2024 10:00:00 GMT</pubDate>
      <description>Summary of article one</description>
    </item>
    <item>
      <title>Article Two</title>
      <link>https://example.com/article2</link>
      <guid>article-2</guid>
      <pubDate>Mon, 16 Jan 2024 10:00:00 GMT</pubDate>
      <description>Summary of article two</description>
    </item>
  </channel>
</rss>
"""

RSS_NO_GUID = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Feed Without GUID</title>
    <link>https://example.com</link>
    <item>
      <title>Article No GUID</title>
      <link>https://example.com/article1</link>
      <pubDate>Mon, 15 Jan 2024 10:00:00 GMT</pubDate>
      <description>No GUID here</description>
    </item>
  </channel>
</rss>
"""

ATOM_FEED = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Feed</title>
  <link href="https://example.com"/>
  <entry>
    <title>Atom Article One</title>
    <link href="https://example.com/atom1"/>
    <id>urn:uuid:1234</id>
    <published>2024-01-15T10:00:00Z</published>
    <summary>Summary of atom article</summary>
  </entry>
  <entry>
    <title>Atom Article Two</title>
    <link href="https://example.com/atom2"/>
    <id>urn:uuid:5678</id>
    <published>2024-01-16T10:00:00Z</published>
    <summary>Another atom article</summary>
  </entry>
</feed>
"""

EMPTY_FEED = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Empty Feed</title>
    <link>https://example.com</link>
  </channel>
</rss>
"""


def test_parse_simple_rss():
    title, articles = parse_feed(SIMPLE_RSS)
    assert title == "Example Feed"
    assert len(articles) == 2
    assert articles[0].title == "Article One"
    assert articles[0].link == "https://example.com/article1"
    assert articles[0].guid == "article-1"
    assert articles[0].summary == "Summary of article one"


def test_parse_rss_without_guid():
    title, articles = parse_feed(RSS_NO_GUID)
    assert title == "Feed Without GUID"
    assert len(articles) == 1
    assert articles[0].guid is None
    assert articles[0].link == "https://example.com/article1"


def test_parse_atom_feed():
    title, articles = parse_feed(ATOM_FEED)
    assert title == "Atom Feed"
    assert len(articles) == 2
    assert articles[0].title == "Atom Article One"
    assert articles[0].link == "https://example.com/atom1"
    assert articles[0].guid == "urn:uuid:1234"


def test_parse_empty_feed():
    title, articles = parse_feed(EMPTY_FEED)
    assert title == "Empty Feed"
    assert len(articles) == 0


def test_published_date_extraction_rss():
    title, articles = parse_feed(SIMPLE_RSS)
    assert articles[0].published_at is not None
    assert "2024-01-15" in articles[0].published_at


def test_published_date_extraction_atom():
    title, articles = parse_feed(ATOM_FEED)
    assert articles[0].published_at is not None
    assert "2024-01-15" in articles[0].published_at


def test_parse_feed_skips_items_without_link():
    feed_no_link = """<?xml version="1.0"?>
    <rss version="2.0">
      <channel>
        <title>Feed</title>
        <item>
          <title>No Link</title>
          <guid>guid-1</guid>
        </item>
        <item>
          <title>Has Link</title>
          <link>https://example.com/article</link>
          <guid>guid-2</guid>
        </item>
      </channel>
    </rss>
    """
    title, articles = parse_feed(feed_no_link)
    assert len(articles) == 1
    assert articles[0].link == "https://example.com/article"


def test_parse_feed_handles_missing_summary():
    feed_no_summary = """<?xml version="1.0"?>
    <rss version="2.0">
      <channel>
        <title>Feed</title>
        <item>
          <title>Article</title>
          <link>https://example.com/article</link>
        </item>
      </channel>
    </rss>
    """
    title, articles = parse_feed(feed_no_summary)
    assert len(articles) == 1
    assert articles[0].summary is None
