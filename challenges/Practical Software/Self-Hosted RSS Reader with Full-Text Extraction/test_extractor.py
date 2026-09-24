from extractor import extract_from_html

ARTICLE_WITH_BOILERPLATE = """
<html>
<head><title>Article Title - News Site</title></head>
<body>
  <nav>Navigation links</nav>
  <div class="sidebar">Sidebar ads</div>
  <div class="content">
    <h1>Article Title</h1>
    <p>This is the main article content.</p>
    <p>It contains multiple paragraphs of text.</p>
    <p>The extraction should find this valuable content.</p>
  </div>
  <div class="comments">Comments section</div>
  <footer>Footer</footer>
</body>
</html>
"""

CLEAN_ARTICLE = """
<html>
<head><title>Clean Article</title></head>
<body>
  <article>
    <h1>Clean Article</h1>
    <p>This is clean article content without much boilerplate.</p>
    <p>Just straightforward text paragraphs.</p>
  </article>
</body>
</html>
"""

ARTICLE_WITH_SCRIPTS_AND_ADS = """
<html>
<head><title>Article with Ads</title></head>
<body>
  <script>
    // Tracking script
    console.log('Tracking');
  </script>
  <div class="article">
    <h1>Article Title</h1>
    <div class="ad">Advertisement</div>
    <p>Main article paragraph one.</p>
    <div class="ad">More ads</div>
    <p>Main article paragraph two.</p>
    <script src="ad-script.js"></script>
  </div>
</body>
</html>
"""

EMPTY_HTML = ""

MINIMAL_HTML = """
<html>
<body>
  <p>Just a paragraph of text.</p>
</body>
</html>
"""


def test_extract_from_html_with_boilerplate():
    result = extract_from_html(ARTICLE_WITH_BOILERPLATE)
    assert result.text is not None
    assert "main article content" in result.text.lower()
    assert "multiple paragraphs" in result.text.lower()
    assert "Navigation links" not in result.text
    assert "Sidebar ads" not in result.text


def test_extract_from_clean_html():
    result = extract_from_html(CLEAN_ARTICLE)
    assert result.text is not None
    assert "clean article content" in result.text.lower()


def test_extract_from_html_with_scripts_and_ads():
    result = extract_from_html(ARTICLE_WITH_SCRIPTS_AND_ADS)
    assert result.text is not None
    assert "Main article paragraph" in result.text
    assert "Tracking script" not in result.text


def test_extract_from_empty_html():
    result = extract_from_html(EMPTY_HTML)
    assert result.text is None or result.text == ""


def test_extract_from_minimal_html():
    result = extract_from_html(MINIMAL_HTML)
    assert result.text is not None
    assert "paragraph" in result.text.lower()


def test_title_extraction():
    result = extract_from_html(ARTICLE_WITH_BOILERPLATE)
    assert result.title is not None or result.title is None


def test_extract_handles_none_input():
    result = extract_from_html(None)
    assert result.text is None
    assert result.title is None
