import json
from datetime import date

import pytest
from actionitems import llm
from actionitems.errors import LLMError
from actionitems.extract import Extraction
from actionitems.models import ActionItem, Source
from helpers import reply, run

MEETING = date(2026, 11, 2)
TEXT = "Priya Shah: Let's start.\nMarco Rossi: I'll send the deck by Friday.\nDana Lee: I will book the room."


# ---- endpoint policy ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/v1/chat",
        "http://localhost:8080/v1/chat",
        "http://[::1]:8080/v1/chat",
    ],
)
def test_loopback_endpoints_are_allowed(url):
    assert llm.check_endpoint(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://10.0.0.5:8080/v1/chat",
        "https://example.com/v1/chat",
        "http://192.168.1.2/x",
    ],
)
def test_remote_endpoints_need_the_explicit_flag(url):
    with pytest.raises(LLMError, match="non-loopback"):
        llm.check_endpoint(url)
    assert llm.check_endpoint(url, allow_remote=True) == url


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://127.0.0.1/x", "http:///nohost", "not a url"]
)
def test_non_http_urls_are_rejected(url):
    with pytest.raises(LLMError):
        llm.check_endpoint(url, allow_remote=True)


def test_payload_pins_the_json_schema_and_numbers_lines():
    payload = llm.build_payload(TEXT, ["Dana Lee"], MEETING)
    dumped = json.dumps(payload)
    assert "json_schema" in dumped
    assert "[2] Marco Rossi" in dumped or "2" in dumped
    assert "Dana Lee" in dumped


# ---- response validation --------------------------------------------------------------------


def _content(*entries) -> str:
    return json.dumps({"action_items": list(entries)})


def test_valid_entries_are_kept():
    out = llm.parse_items(
        _content(
            {
                "line": 2,
                "task": "Send the deck",
                "owner": "Marco Rossi",
                "due": "2026-11-06",
            }
        ),
        TEXT,
        {"Marco Rossi"},
        MEETING,
    )
    assert len(out) == 1
    assert out[0].owner == "Marco Rossi"
    assert out[0].due == date(2026, 11, 6)


def test_owner_not_on_roster_or_in_text_is_dropped():
    out = llm.parse_items(
        _content({"line": 2, "task": "Send the deck", "owner": "Zoe Ng", "due": None}),
        TEXT,
        {"Marco Rossi"},
        MEETING,
    )
    assert out[0].owner is None


@pytest.mark.parametrize(
    "entry",
    [
        {"line": 0, "task": "x", "owner": None, "due": None},
        {"line": 99, "task": "x", "owner": None, "due": None},
        {"line": "2", "task": "x", "owner": None, "due": None},
        {"line": 2, "task": "   ", "owner": None, "due": None},
        {"line": 2, "task": "x" * 201, "owner": None, "due": None},
        "not an object",
    ],
)
def test_invalid_entries_are_dropped(entry):
    assert llm.parse_items(_content(entry), TEXT, set(), MEETING) == []


def test_bad_due_string_becomes_none():
    out = llm.parse_items(
        _content({"line": 2, "task": "Send", "owner": None, "due": "next blursday"}),
        TEXT,
        set(),
        MEETING,
    )
    assert out[0].due is None


def test_non_json_content_is_an_llm_error():
    with pytest.raises(LLMError, match="not JSON"):
        llm.parse_items("sure! here you go", TEXT, set(), MEETING)


def test_missing_action_items_list_is_an_llm_error():
    with pytest.raises(LLMError, match="action_items"):
        llm.parse_items('{"items": []}', TEXT, set(), MEETING)


# ---- transport ------------------------------------------------------------------------------


def test_call_server_returns_message_content(llm_server):
    state, url = llm_server
    reply(state, _content())
    assert llm.call_server(url, {"x": 1}) == _content()
    assert state["requests"] == [{"x": 1}]


def test_call_server_http_error_is_an_llm_error(llm_server):
    state, url = llm_server
    state["status"] = 500
    reply(state, "{}")
    with pytest.raises(LLMError, match="request to .* failed: HTTP 500"):
        llm.call_server(url, {})


def test_call_server_unexpected_shape_is_an_llm_error(llm_server):
    state, url = llm_server
    state["body"] = b'{"nope": true}'
    with pytest.raises(LLMError, match="unexpected shape"):
        llm.call_server(url, {})


def test_call_server_refuses_oversized_responses(llm_server):
    state, url = llm_server
    state["body"] = b"x" * (llm.MAX_RESPONSE_BYTES + 10)
    with pytest.raises(LLMError, match="too large"):
        llm.call_server(url, {})


def test_call_server_connection_refused_is_an_llm_error():
    with pytest.raises(LLMError):
        llm.call_server("http://127.0.0.1:1/v1/chat", {}, timeout=2)


# ---- refine ---------------------------------------------------------------------------------


def _rules_extraction() -> Extraction:
    item = ActionItem(
        task="Send the deck",
        owner=None,
        due=None,
        due_phrase=None,
        due_confidence=0.0,
        item_type="unassigned",
        confidence=0.55,
        reasons=[],
        sources=[Source(2, None, None, "x")],
    )
    return Extraction("notes.txt", MEETING, "UTC", "rules-only", [item])


def test_refine_fills_missing_owner_and_due_but_keeps_rule_items(llm_server):
    state, url = llm_server
    reply(
        state,
        _content(
            {
                "line": 2,
                "task": "Send the deck",
                "owner": "Marco Rossi",
                "due": "2026-11-06",
            }
        ),
    )
    out = llm.refine(_rules_extraction(), TEXT, {"Marco Rossi"}, url)
    assert len(out.items) == 1
    assert out.items[0].owner == "Marco Rossi"
    assert out.items[0].due == "2026-11-06"
    assert "owner filled by the LLM pass" in out.items[0].reasons


def test_refine_adds_items_the_rules_missed(llm_server):
    state, url = llm_server
    reply(
        state,
        _content(
            {"line": 3, "task": "Book the room", "owner": "Dana Lee", "due": None}
        ),
    )
    out = llm.refine(_rules_extraction(), TEXT, {"Dana Lee"}, url)
    added = [i for i in out.items if i.item_type == "llm"]
    assert len(added) == 1
    assert added[0].owner == "Dana Lee"
    assert (
        added[0].confidence == 0.35
    )  # below the default --min-confidence: opt-in only


def test_refine_never_deletes_rule_items(llm_server):
    state, url = llm_server
    reply(state, _content())
    out = llm.refine(_rules_extraction(), TEXT, set(), url)
    assert len(out.items) == 1


def test_refine_failure_keeps_rules_output_and_records_reason(llm_server):
    state, url = llm_server
    state["status"] = 503
    reply(state, "{}")
    before = _rules_extraction()
    out = llm.refine(before, TEXT, set(), url)
    assert out.items == before.items
    assert "+llm-failed" in out.backend


def test_refine_refuses_remote_hosts_before_sending_anything():
    with pytest.raises(LLMError, match="non-loopback"):
        llm.refine(_rules_extraction(), TEXT, set(), "http://203.0.113.9/v1/chat")


def test_refine_with_a_real_extraction_and_fake_server(llm_server):
    state, url = llm_server
    reply(state, _content())
    ext = run(TEXT)
    out = llm.refine(ext, TEXT, {"Marco Rossi"}, url)
    assert out.items == ext.items
    assert len(state["requests"]) == 1
