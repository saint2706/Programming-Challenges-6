from __future__ import annotations

import json
from pathlib import Path

import pytest
import server as server_module
from fastapi.testclient import TestClient

TOKEN = "test-shared-token-123"


@pytest.fixture()
def served_root(tmp_path: Path) -> Path:
    root = tmp_path / "share"
    root.mkdir()
    (root / "hello.txt").write_text("hello world")
    (root / "notes").mkdir()
    (root / "notes" / "todo.txt").write_text("buy milk")
    return root


@pytest.fixture()
def client(served_root: Path, tmp_path: Path) -> TestClient:
    server_module.configure(
        root=served_root, token=TOKEN, log_path=tmp_path / "access.log.jsonl"
    )
    return TestClient(server_module.app)


@pytest.fixture()
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "access.log.jsonl"


def _tail(log_file: Path) -> list[dict]:
    if not log_file.exists():
        return []
    return [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------
# Public routes
# --------------------------------------------------------------------------


def test_health_is_public(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_index_is_public(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Local Network File Server" in resp.text


def test_login_form_is_public(client: TestClient) -> None:
    resp = client.get("/login")
    assert resp.status_code == 200
    assert "<form" in resp.text


# --------------------------------------------------------------------------
# Auth gate
# --------------------------------------------------------------------------


def test_browse_without_auth_returns_401_for_api_client(client: TestClient) -> None:
    resp = client.get("/browse/", follow_redirects=False)
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


def test_browse_without_auth_redirects_to_login_for_browser(client: TestClient) -> None:
    resp = client.get(
        "/browse/", headers={"Accept": "text/html"}, follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/login?next=")


def test_download_without_auth_rejected(client: TestClient) -> None:
    resp = client.get("/browse/hello.txt", follow_redirects=False)
    assert resp.status_code == 401


def test_logs_page_without_auth_rejected(client: TestClient) -> None:
    resp = client.get("/logs", follow_redirects=False)
    assert resp.status_code == 401


def test_wrong_bearer_token_rejected(client: TestClient) -> None:
    resp = client.get("/browse/", headers={"Authorization": "Bearer wrong-token"})
    assert resp.status_code == 401


def test_correct_bearer_token_grants_access(client: TestClient) -> None:
    resp = client.get("/browse/", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    assert "hello.txt" in resp.text


# --------------------------------------------------------------------------
# Session login flow
# --------------------------------------------------------------------------


def test_login_wrong_password_returns_401_with_error(client: TestClient) -> None:
    resp = client.post("/login", data={"password": "nope"}, follow_redirects=False)
    assert resp.status_code == 401
    assert "Incorrect token" in resp.text


def test_login_correct_password_sets_cookie_and_redirects(client: TestClient) -> None:
    resp = client.post(
        "/login", data={"password": TOKEN, "next": "/browse/"}, follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/browse/"
    assert "fs_session" in resp.cookies


def test_login_open_redirect_next_is_rejected(client: TestClient) -> None:
    resp = client.post(
        "/login",
        data={"password": TOKEN, "next": "//evil.example.com/steal"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/browse/"


def test_session_cookie_grants_browse_access(client: TestClient) -> None:
    login = client.post("/login", data={"password": TOKEN}, follow_redirects=False)
    assert "fs_session" in login.cookies

    resp = client.get("/browse/")
    assert resp.status_code == 200
    assert "hello.txt" in resp.text


def test_logout_invalidates_session(client: TestClient) -> None:
    client.post("/login", data={"password": TOKEN}, follow_redirects=False)
    assert client.get("/browse/").status_code == 200

    client.post("/logout", follow_redirects=False)
    assert client.get("/browse/", follow_redirects=False).status_code == 401


# --------------------------------------------------------------------------
# Directory listing / download correctness
# --------------------------------------------------------------------------


def test_directory_listing_shows_files_and_subdirs(client: TestClient) -> None:
    resp = client.get("/browse/", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    assert "hello.txt" in resp.text
    assert "notes" in resp.text


def test_subdirectory_listing(client: TestClient) -> None:
    resp = client.get("/browse/notes/", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    assert "todo.txt" in resp.text


def test_download_returns_byte_identical_content(
    client: TestClient, served_root: Path
) -> None:
    resp = client.get("/browse/hello.txt", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    assert resp.content == (served_root / "hello.txt").read_bytes()
    assert resp.headers["content-type"].startswith("text/plain")


def test_download_nested_file(client: TestClient, served_root: Path) -> None:
    resp = client.get(
        "/browse/notes/todo.txt", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert resp.status_code == 200
    assert resp.content == (served_root / "notes" / "todo.txt").read_bytes()


def test_nonexistent_file_is_404(client: TestClient) -> None:
    resp = client.get(
        "/browse/does-not-exist.txt", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert resp.status_code == 404


# --------------------------------------------------------------------------
# Path-traversal battery (HTTP level, authenticated -- these must be
# rejected on their own merits, not merely because auth also failed)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attack_path",
    [
        "/browse/..%2Fsecret.txt",
        "/browse/%2e%2e%2Fsecret.txt",
        "/browse/%252e%252e%252Fsecret.txt",
        "/browse/notes/..%2F..%2Fsecret.txt",
        "/browse/..%5Csecret.txt",  # encoded backslash
    ],
)
def test_traversal_attacks_over_http_are_rejected(
    client: TestClient, tmp_path: Path, attack_path: str
) -> None:
    # A real secret file one level above the served root, so a successful
    # escape would be detectable by content, not just by status code.
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret, must never be served")

    resp = client.get(attack_path, headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code in (400, 404)
    assert b"top secret" not in resp.content


def test_literal_dotdot_in_url_never_reaches_secret(
    client: TestClient, tmp_path: Path
) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret, must never be served")

    resp = client.get(
        "/browse/../secret.txt", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    # Whether the router 404s the literal ".." itself or our own handler
    # rejects it with 400, the content must never leak either way.
    assert resp.status_code in (400, 404)
    assert b"top secret" not in resp.content


# --------------------------------------------------------------------------
# Access log correctness
# --------------------------------------------------------------------------


def test_successful_download_is_logged(client: TestClient, log_path: Path) -> None:
    client.get("/browse/hello.txt", headers={"Authorization": f"Bearer {TOKEN}"})
    entries = _tail(log_path)
    downloads = [e for e in entries if e["action"] == "download"]
    assert len(downloads) == 1
    assert downloads[0]["path"] == "/browse/hello.txt"
    assert downloads[0]["identity"] == "token"
    assert downloads[0]["status"] == 200


def test_denied_request_is_logged(client: TestClient, log_path: Path) -> None:
    client.get("/browse/hello.txt", follow_redirects=False)
    entries = _tail(log_path)
    denials = [e for e in entries if e["action"] == "denied"]
    assert len(denials) == 1
    assert denials[0]["status"] == 401
    assert denials[0]["identity"] == "anonymous"


def test_traversal_attempt_is_logged_as_denied(
    client: TestClient, log_path: Path
) -> None:
    client.get("/browse/..%2Fsecret.txt", headers={"Authorization": f"Bearer {TOKEN}"})
    entries = _tail(log_path)
    assert any(e["action"] == "denied" and e["status"] == 400 for e in entries)


def test_login_success_and_failure_are_logged(
    client: TestClient, log_path: Path
) -> None:
    client.post("/login", data={"password": "wrong"}, follow_redirects=False)
    client.post("/login", data={"password": TOKEN}, follow_redirects=False)
    entries = _tail(log_path)
    assert any(e["action"] == "login_failed" for e in entries)
    assert any(e["action"] == "login_success" for e in entries)


def test_logs_page_renders_recorded_entries(client: TestClient) -> None:
    client.get("/browse/hello.txt", headers={"Authorization": f"Bearer {TOKEN}"})
    resp = client.get("/logs", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    assert "/browse/hello.txt" in resp.text
    assert "download" in resp.text


# --------------------------------------------------------------------------
# XSS hardening -- attacker-controlled values must never render as raw HTML
# --------------------------------------------------------------------------


def test_html_bearing_filename_is_escaped_in_listing(
    client: TestClient, served_root: Path
) -> None:
    # `<` `>` `"` are illegal in Windows filenames, so the real HTML-markup
    # characters that matter most for XSS can't be used directly here --
    # `&` and `'` are legal everywhere and still prove `html.escape` runs
    # on every listed name (default `quote=True` escapes both).
    evil_name = "Q&A's notes.txt"
    (served_root / evil_name).write_text("x")

    resp = client.get("/browse/", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    assert "Q&A's notes.txt" not in resp.text
    assert "Q&amp;A&#x27;s notes.txt" in resp.text


def test_malicious_request_path_is_escaped_in_log_viewer(client: TestClient) -> None:
    # The path itself can't literally contain `<` over a real URL (it'd be
    # percent-encoded), but a bearer token attempt can carry one, and its
    # failure is still logged and rendered -- exercise the log-viewer
    # escaping directly via a crafted, already-logged entry.
    server_module._access_logger.log(
        identity='"><img src=x onerror=alert(1)>',
        ip="1.2.3.4",
        method="GET",
        path="/browse/whatever",
        action="denied",
        status=401,
    )
    resp = client.get("/logs", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    assert "<img src=x onerror=alert(1)>" not in resp.text
    assert "&lt;img" in resp.text


def test_malicious_next_param_is_escaped_on_login_page(client: TestClient) -> None:
    resp = client.get("/login", params={"next": '"><script>alert(1)</script>'})
    assert resp.status_code == 200
    assert "<script>alert(1)</script>" not in resp.text
