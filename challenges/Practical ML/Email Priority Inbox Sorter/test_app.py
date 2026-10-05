import pytest
from streamlit.testing.v1 import AppTest

import inbox


@pytest.fixture
def at(monkeypatch, trained):
    import streamlit as st

    st.cache_resource.clear()
    monkeypatch.setattr(inbox, "load_context", lambda: trained)
    app = AppTest.from_file("app.py", default_timeout=90)
    app.run()
    return app


def test_page_renders_a_sorted_inbox_for_the_first_mailbox_and_day(at):
    assert not at.exception
    assert "Email Priority Inbox" in at.title[0].value
    assert at.selectbox(key="mailbox").value == "aa-b"
    labels = [e.label for e in at.expander]
    assert labels and labels[0].startswith("1.")


def test_each_message_has_a_why_panel_and_outcome_is_hidden_until_toggled(at):
    assert any("Why" in m.value for m in at.markdown)
    assert not any("replied" in e.label or "ignored" in e.label for e in at.expander)
    at.checkbox(key="reveal").check().run()
    assert not at.exception
    assert any("replied" in e.label or "ignored" in e.label for e in at.expander)


def test_switching_mailbox_changes_the_day_choices(at):
    at.selectbox(key="mailbox").set_value("cc-d").run()
    assert not at.exception
    assert at.selectbox(key="mailbox").value == "cc-d"
    assert at.selectbox(key="day").options
