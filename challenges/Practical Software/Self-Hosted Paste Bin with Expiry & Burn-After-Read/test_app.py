from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import quote

import app as app_module
import pytest
from fastapi.testclient import TestClient
from storage import Paste


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Point the module-level store at a fresh temp DB per test.
    from storage import PasteStore

    monkeypatch.setattr(app_module, "store", PasteStore(tmp_path / "test.db"))
    with TestClient(app_module.app) as c:
        yield c


def test_index_shows_form(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<form" in resp.text
    assert "New paste" in resp.text


def test_create_and_view_paste(client: TestClient) -> None:
    resp = client.post(
        "/pastes", data={"content": "hello", "language": "text", "expiry": "1h"}
    )
    assert resp.status_code == 200
    assert "Paste created" in resp.text
    assert "/p/" in resp.text

    link = resp.text.split('value="')[1].split('"')[0]
    path = link.split("://", 1)[1].split("/", 1)[1]  # strip scheme+host

    view = client.get(f"/{path}")
    assert view.status_code == 200
    assert "hello" in view.text


def test_burn_after_read_via_http_only_works_once(client: TestClient) -> None:
    resp = client.post(
        "/pastes",
        data={
            "content": "top secret",
            "language": "text",
            "expiry": "never",
            "burn_after_read": "true",
        },
    )
    link = resp.text.split('value="')[1].split('"')[0]
    path = "/" + link.split("://", 1)[1].split("/", 1)[1]

    first = client.get(path)
    assert first.status_code == 200
    assert "top secret" in first.text
    assert "burned" in first.text.lower()

    second = client.get(path)
    assert second.status_code == 404
    assert "gone" in second.text.lower()


def test_empty_content_rejected(client: TestClient) -> None:
    resp = client.post("/pastes", data={"content": "   ", "expiry": "never"})
    assert resp.status_code == 400


def test_unknown_paste_id_is_404(client: TestClient) -> None:
    resp = client.get("/p/does-not-exist")
    assert resp.status_code == 404


def test_raw_endpoint_returns_plain_text_and_consumes_burn(client: TestClient) -> None:
    resp = client.post(
        "/pastes",
        data={"content": "raw content", "expiry": "never", "burn_after_read": "true"},
    )
    link = resp.text.split('value="')[1].split('"')[0]
    path = "/" + link.split("://", 1)[1].split("/", 1)[1]
    raw_path = path.replace("/p/", "/raw/")

    raw = client.get(raw_path)
    assert raw.status_code == 200
    assert raw.text == "raw content"
    assert raw.headers["content-type"].startswith("text/plain")

    raw_again = client.get(raw_path)
    assert raw_again.status_code == 404


def test_countdown_fragment_reflects_remaining_time(client: TestClient) -> None:
    resp = client.post("/pastes", data={"content": "ttl paste", "expiry": "1h"})
    link = resp.text.split('value="')[1].split('"')[0]
    path = "/" + link.split("://", 1)[1].split("/", 1)[1]
    paste_id = path.rsplit("/", 1)[-1]

    fragment = client.get(f"/p/{paste_id}/countdown")
    assert fragment.status_code == 200
    assert "expires in" in fragment.text


def test_countdown_fragment_for_missing_paste(client: TestClient) -> None:
    fragment = client.get("/p/does-not-exist/countdown")
    assert fragment.status_code == 200
    assert "expired" in fragment.text


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_syntax_highlighting_applied_for_python(client: TestClient) -> None:
    resp = client.post(
        "/pastes",
        data={
            "content": "def f():\n    return 1\n",
            "language": "python",
            "expiry": "never",
        },
    )
    link = resp.text.split('value="')[1].split('"')[0]
    path = "/" + link.split("://", 1)[1].split("/", 1)[1]

    view = client.get(path)
    assert view.status_code == 200
    assert "highlight" in view.text  # pygments wraps output in a .highlight block


# --- paste_id is a URL path segment: it must be escaped wherever it is echoed into HTML ---

HOSTILE_ID = '"><img src=x onerror=alert(1)>'


def _paste_with_hostile_id(expires_at: float | None) -> Paste:
    return Paste(
        id=HOSTILE_ID,
        content="x",
        language="text",
        created_at=time.time(),
        expires_at=expires_at,
        burn_after_read=False,
    )


@pytest.mark.parametrize("expires_at", [None, time.time() + 600])
def test_render_view_escapes_paste_id_in_every_attribute(
    expires_at: float | None,
) -> None:
    body = app_module.render_view(
        _paste_with_hostile_id(expires_at), HOSTILE_ID
    ).body.decode()
    assert "<img" not in body
    assert "onerror=alert(1)>" not in body  # attribute breakout must not survive


@pytest.mark.parametrize("expires_at", [None, time.time() + 600])
def test_countdown_escapes_paste_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, expires_at: float | None
) -> None:
    monkeypatch.setattr(
        app_module.store, "peek", lambda _pid: _paste_with_hostile_id(expires_at)
    )
    resp = client.get("/p/" + quote(HOSTILE_ID, safe="") + "/countdown")
    assert resp.status_code == 200
    assert "<img" not in resp.text


def test_escaped_paste_id_still_routes_to_the_same_paste(client: TestClient) -> None:
    resp = client.post(
        "/pastes", data={"content": "hello", "language": "text", "expiry": "never"}
    )
    paste_id = resp.text.split("/p/")[1].split('"')[0]
    view = client.get(f"/p/{paste_id}")
    assert (
        f'hx-get="/p/{paste_id}/countdown"' in view.text
    )  # ordinary ids are unchanged
