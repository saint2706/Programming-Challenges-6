"""The report must escape engine-supplied text and reference no external resources."""

import re

from dist_pipeline.report import render_html, summary_markdown

HOSTILE = '<script>alert("x")</script><img src=x onerror=alert(1)>'


def _payload(label: str) -> dict:
    row = {
        "engine": "e",
        "label": label,
        "query": "q0_scan_sum",
        "scale": "1m",
        "fmt": "parquet",
        "status": "error",
        "error": label,
        "time": {"median": None, "iqr": None},
        "peak_rss_mb": 1.0,
        "verified": None,
        "max_rel_error": None,
    }
    ok = dict(
        row,
        label="fine",
        status="ok",
        error=None,
        time={"median": 0.5, "iqr": 0.1},
        verified=True,
    )
    return {"machine": {"platform": "test"}, "matrix": [row, ok], "scaling": None}


def test_html_escapes_hostile_text():
    page = render_html(_payload(HOSTILE))
    assert "<script" not in page
    assert "<img" not in page
    assert "&lt;script&gt;" in page


def test_html_has_no_external_resources():
    page = render_html(_payload("fine"))
    assert not re.search(r"(src|href)\s*=\s*[\"']?https?:", page)
    assert "<script" not in page


def test_markdown_escapes_pipes():
    md = summary_markdown(_payload("a|b"))
    assert "a\\|b" in md
