import re

import pytest
from helpers import tone, write_wav
from litestar.testing import TestClient
from voicememo.asr import FakeTranscriber, Segment
from voicememo.ingest import FixedEngine, ingest_file
from voicememo.web import CSRF_COOKIE, create_app, highlight, host_name, parse_range

BASE = "http://127.0.0.1"
HOST = "127.0.0.1"


@pytest.fixture
def memo_id(library, tmp_path):
    wav = write_wav(tmp_path / "walk.wav", tone(2.0))
    engine = FixedEngine(
        FakeTranscriber({"walk": "Walked the dog. The river was cold."})
    )
    return ingest_file(library, wav, engine).memo_id


@pytest.fixture
def client(library):
    with TestClient(create_app(library), base_url=BASE) as c:
        yield c


def csrf_from(client, path="/"):
    client.get(path)
    token = client.cookies.get(CSRF_COOKIE)
    assert token
    return token


# ----- helpers -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "name"),
    [
        ("localhost:8000", "localhost"),
        ("127.0.0.1", "127.0.0.1"),
        ("[::1]:8000", "::1"),
        ("EVIL.example:80", "evil.example"),
        ("[::1", ""),
    ],
)
def test_host_name_strips_port_and_brackets(header, name):
    assert host_name(header) == name


def test_highlight_escapes_before_marking():
    out = str(highlight("<b>[walk]</b> & [dog]"))
    assert out == "&lt;b&gt;<mark>walk</mark>&lt;/b&gt; &amp; <mark>dog</mark>"
    assert "<b>" not in out


@pytest.mark.parametrize(
    ("header", "size", "expected"),
    [
        (None, 100, None),
        ("", 100, None),
        ("bytes=0-9", 100, (0, 9)),
        ("bytes=10-", 100, (10, 99)),
        ("bytes=-20", 100, (80, 99)),
        ("bytes=90-500", 100, (90, 99)),
        ("bytes=0-0", 1, (0, 0)),
        (" bytes=5-6 ", 100, (5, 6)),
    ],
)
def test_parse_range_valid(header, size, expected):
    assert parse_range(header, size) == expected


@pytest.mark.parametrize(
    "header",
    [
        "items=0-1",
        "bytes=-",
        "bytes=abc",
        "bytes=9-3",
        "bytes=100-",
        "bytes=-0",
        "bytes=0-1,4-5",
    ],
)
def test_parse_range_rejects_bad_headers(header):
    with pytest.raises(ValueError):
        parse_range(header, 100)


def test_parse_range_rejects_empty_file():
    with pytest.raises(ValueError):
        parse_range("bytes=0-0", 0)


# ----- Host and Origin guards ----------------------------------------------------------


@pytest.mark.parametrize("host", ["evil.example", "testserver", "127.0.0.1.nip.io"])
def test_foreign_host_header_is_refused(client, host):
    response = client.get("/", headers={"Host": host})
    assert response.status_code == 400


def test_loopback_host_is_accepted(client):
    assert client.get("/", headers={"Host": "localhost:8000"}).status_code == 200


def test_cross_origin_post_is_refused_before_csrf(client, memo_id):
    token = csrf_from(client, f"/memos/{memo_id}")
    response = client.post(
        f"/memos/{memo_id}/tags",
        data={"csrf_token": token, "tag": "x"},
        headers={"Origin": "http://evil.example"},
    )
    assert response.status_code == 403


def test_same_origin_post_with_token_is_accepted(client, memo_id):
    token = csrf_from(client, f"/memos/{memo_id}")
    response = client.post(
        f"/memos/{memo_id}/tags",
        data={"csrf_token": token, "tag": "x"},
        headers={"Origin": BASE},
        follow_redirects=False,
    )
    assert response.status_code == 303


def test_post_without_origin_header_still_needs_csrf(client, memo_id):
    csrf_from(client, f"/memos/{memo_id}")
    response = client.post(f"/memos/{memo_id}/tags", data={"tag": "x"})
    assert response.status_code == 403


# ----- CSRF ----------------------------------------------------------------------------


def test_csrf_cookie_is_httponly_strict(client):
    response = client.get("/")
    cookie = next(
        c for c in response.headers.get_list("set-cookie") if CSRF_COOKIE in c
    )
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie.lower() or "samesite=strict" in cookie.lower()


def test_missing_csrf_field_is_refused(client, memo_id):
    csrf_from(client, f"/memos/{memo_id}")
    response = client.post(f"/memos/{memo_id}/tags", data={"tag": "x"})
    assert response.status_code == 403
    assert "csrf" in response.text


def test_wrong_csrf_token_is_refused(client, memo_id):
    csrf_from(client, f"/memos/{memo_id}")
    response = client.post(
        f"/memos/{memo_id}/tags", data={"csrf_token": "forged", "tag": "x"}
    )
    assert response.status_code == 403


def test_token_without_cookie_is_refused(library, memo_id):
    with TestClient(create_app(library), base_url=BASE) as fresh:
        response = fresh.post(
            f"/memos/{memo_id}/tags", data={"csrf_token": "anything", "tag": "x"}
        )
    assert response.status_code == 403


def test_remove_tag_also_requires_csrf(client, library, memo_id):
    library.add_tag(memo_id, "keep")
    csrf_from(client, f"/memos/{memo_id}")
    response = client.post(f"/memos/{memo_id}/tags/remove", data={"tag": "keep"})
    assert response.status_code == 403
    assert ("keep", "user") in library.tags(memo_id)


# ----- tags ----------------------------------------------------------------------------


def test_add_tag_redirects_for_plain_form_post(client, library, memo_id):
    token = csrf_from(client, f"/memos/{memo_id}")
    response = client.post(
        f"/memos/{memo_id}/tags",
        data={"csrf_token": token, "tag": "Work Notes"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/memos/{memo_id}"
    assert ("work notes", "user") in library.tags(memo_id)


def test_add_tag_returns_fragment_for_htmx(client, library, memo_id):
    token = csrf_from(client, f"/memos/{memo_id}")
    response = client.post(
        f"/memos/{memo_id}/tags",
        data={"csrf_token": token, "tag": "kitchen"},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    assert "kitchen" in response.text


def test_invalid_tag_gets_400_with_message(client, memo_id):
    token = csrf_from(client, f"/memos/{memo_id}")
    response = client.post(
        f"/memos/{memo_id}/tags", data={"csrf_token": token, "tag": "   "}
    )
    assert response.status_code == 400
    assert "must not be empty" in response.text


def test_remove_tag_round_trip(client, library, memo_id):
    library.add_tag(memo_id, "gone")
    token = csrf_from(client, f"/memos/{memo_id}")
    response = client.post(
        f"/memos/{memo_id}/tags/remove",
        data={"csrf_token": token, "tag": "gone"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert library.tags(memo_id) == []


def test_tag_on_unknown_memo_is_404(client):
    token = csrf_from(client)
    response = client.post("/memos/999/tags", data={"csrf_token": token, "tag": "x"})
    assert response.status_code == 404


# ----- rendering and XSS ---------------------------------------------------------------


def test_hostile_title_and_transcript_are_escaped_on_index_and_detail(
    client, library, tmp_path
):
    memo = library.create_pending("xss1", "<img src=x onerror=alert(1)>.m4a", "x.m4a")
    library.complete(
        memo.id,
        [Segment(0.0, 1.0, "<script>alert(1)</script> ok.")],
        duration=1.0,
        language="en",
        backend="fake",
    )
    library.add_tag(memo.id, "<b>bold</b>")

    for path in ["/", f"/memos/{memo.id}"]:
        body = client.get(path).text
        assert "<script>alert(1)</script>" not in body
        assert "<img src=x onerror" not in body
        assert "<b>bold</b>" not in body
        assert "&lt;img src=x onerror=alert(1)&gt;" in body
    assert (
        "&lt;script&gt;alert(1)&lt;/script&gt;" in client.get(f"/memos/{memo.id}").text
    )


def test_search_snippets_are_escaped(client, library, tmp_path):
    wav = write_wav(tmp_path / "s.wav", tone(1.0))
    engine = FixedEngine(FakeTranscriber(default_text="<i>needle</i> here."))
    ingest_file(library, wav, engine)
    body = client.get("/", params={"q": "needle"}).text
    assert "<i>needle</i>" not in body
    assert "&lt;i&gt;" in body


def test_detail_for_unknown_memo_is_404(client):
    assert client.get("/memos/12345").status_code == 404


def test_non_numeric_memo_id_is_not_routed(client):
    assert client.get("/memos/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_index_filters_by_tag(client, library, memo_id):
    library.add_tag(memo_id, "walk")
    assert f'href="/memos/{memo_id}"' in client.get("/", params={"tag": "walk"}).text
    assert (
        f'href="/memos/{memo_id}"' not in client.get("/", params={"tag": "other"}).text
    )


def test_long_query_is_truncated_not_rejected(client):
    assert client.get("/", params={"q": "a" * 5000}).status_code == 200


# ----- audio ---------------------------------------------------------------------------


@pytest.fixture
def audio_size(library, memo_id):
    return library.audio_path(library.get(memo_id)).stat().st_size


def test_audio_full_response(client, memo_id, audio_size):
    response = client.get(f"/memos/{memo_id}/audio")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["accept-ranges"] == "bytes"
    assert len(response.content) == audio_size


def test_audio_partial_content(client, memo_id, audio_size):
    response = client.get(f"/memos/{memo_id}/audio", headers={"Range": "bytes=0-9"})
    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 0-9/{audio_size}"
    assert len(response.content) == 10


def test_audio_suffix_range(client, memo_id, audio_size):
    response = client.get(f"/memos/{memo_id}/audio", headers={"Range": "bytes=-16"})
    assert response.status_code == 206
    assert len(response.content) == 16


def test_audio_unsatisfiable_range_is_416(client, memo_id, audio_size):
    response = client.get(
        f"/memos/{memo_id}/audio", headers={"Range": f"bytes={audio_size + 10}-"}
    )
    assert response.status_code == 416
    assert response.headers["content-range"] == f"bytes */{audio_size}"


def test_audio_unknown_memo_is_404(client):
    assert client.get("/memos/77/audio").status_code == 404


def test_audio_missing_file_is_404(client, library, memo_id):
    library.audio_path(library.get(memo_id)).unlink()
    assert client.get(f"/memos/{memo_id}/audio").status_code == 404


def test_audio_is_served_by_id_only(client, memo_id):
    # The path comes from the stored hash-derived name, so request text cannot reach the disk.
    for probe in [
        "/memos/..%2Fdata/audio",
        f"/memos/{memo_id}/audio?file=../../secret",
    ]:
        response = client.get(probe)
        assert response.status_code in {200, 404}
        if response.status_code == 200:
            assert b"secret" not in response.content


# ----- headers -------------------------------------------------------------------------


def test_security_headers_on_every_page(client, memo_id):
    for path in ["/", f"/memos/{memo_id}", f"/memos/{memo_id}/audio"]:
        headers = client.get(path).headers
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"
        assert headers["referrer-policy"] == "no-referrer"
        csp = headers["content-security-policy"]
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp


def test_csp_forbids_inline_scripts(client):
    csp = client.get("/").headers["content-security-policy"]
    assert not re.search(r"script-src[^;]*'unsafe-inline'", csp)
