"""Streamlit inbox: pick a mailbox and a held-out day, see mail sorted by priority and why.

Run:  uv run streamlit run src/inbox_sorter/app.py
"""

from __future__ import annotations

import streamlit as st

from inbox_sorter import inbox

st.set_page_config(page_title="Email Priority Inbox Sorter", layout="wide")


@st.cache_resource(show_spinner="Loading the trained models...")
def get_context():
    return inbox.load_context()


st.title("Email Priority Inbox Sorter")
art, df = get_context()

with st.sidebar:
    mailbox = st.selectbox("Mailbox", inbox.mailboxes(df), key="mailbox")
    days = inbox.days_with_mail(df, mailbox)
    day = st.selectbox(
        "Day (held-out test period)",
        list(days),
        format_func=lambda d: f"{d.isoformat()}  ({days[d]} messages)",
        key="day",
    )
    top = st.slider("Messages to show", 3, 30, 10, key="top")
    model = st.radio(
        "Model",
        inbox.EXPLAINED_MODELS,
        key="model",
        help="lgbm_meta_text = metadata + message wording; lgbm_meta = metadata only.",
    )
    reveal = st.checkbox("Reveal what the owner actually did", key="reveal")
    st.caption(
        "Priority = calibrated probability the owner replies to or forwards the "
        "message within 14 days. That is behaviour, not importance: it misses "
        "read-only FYI mail and mail answered by phone."
    )

for item in inbox.rank_inbox(art, df, mailbox, day, top=top, model=model):
    outcome = ""
    if reveal:
        outcome = " - replied/forwarded" if item.acted else " - ignored"
    with st.expander(
        f"{item.rank}. {item.probability:.0%}  {item.sender} | {item.subject}{outcome}",
        expanded=item.rank == 1,
    ):
        st.markdown("**Why this rank**")
        st.markdown(
            "\n".join(
                f"- {'raises' if w > 0 else 'lowers'} the score: {label} (`{w:+.2f}` log-odds)"
                for label, w in item.reasons
            )
        )
