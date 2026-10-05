from datetime import datetime

import polars as pl
import pytest

from data import Message, load_mailboxes, owner_address, parse_message, received

RAW = r"""Message-ID: <1.JavaMail.evans@thyme>
Date: Mon, 14 May 2001 16:39:00 -0700 (PDT)
From: Sam.Sender@enron.com
To: owner@enron.com, Other.Person@enron.com
Cc: third@example.com,
 fourth@example.com
Subject: Re: Q3 numbers
Mime-Version: 1.0
X-Folder: \Owner\Inbox

Here is the forecast?
Thanks.
"""


def _msg(mid, sender, to, subject="hello", date="2001-05-14 10:00", cc=(), body="body"):
    return Message(
        message_id=mid,
        date=datetime.fromisoformat(date) if date else None,
        sender=sender,
        to=list(to),
        cc=list(cc),
        subject=subject,
        body=body,
    )


def test_parse_message_reads_headers_lowercases_addresses_and_unfolds_cc():
    m = parse_message(RAW)
    assert m.message_id == "<1.JavaMail.evans@thyme>"
    assert m.sender == "sam.sender@enron.com"
    assert m.to == ["owner@enron.com", "other.person@enron.com"]
    assert m.cc == ["third@example.com", "fourth@example.com"]
    assert m.subject == "Re: Q3 numbers"
    assert m.date == datetime.fromisoformat(
        "2001-05-14T23:39"
    )  # converted to naive UTC
    assert "forecast?" in m.body


@pytest.mark.parametrize(
    "date_line",
    ["Date: not a date at all", "", "Date: Mon, 99 Foo 2001 16:39:00 -0700"],
)
def test_garbled_or_missing_date_gives_none_not_a_crash(date_line):
    raw = RAW.replace("Date: Mon, 14 May 2001 16:39:00 -0700 (PDT)", date_line)
    assert parse_message(raw).date is None


def test_missing_subject_and_empty_body_and_non_ascii_do_not_crash():
    raw = "Message-ID: <2@x>\nDate: Mon, 14 May 2001 16:39:00 -0700\nFrom: a@b.com\nTo: c@d.com\n\n"
    m = parse_message(raw)
    assert m.subject == "" and m.body == ""
    raw2 = "Message-ID: <3@x>\nFrom: a@b.com\nTo: c@d.com\nSubject: caf\xe9 \u2013 r\xe9sum\xe9\n\nbody"
    assert "caf" in parse_message(raw2).subject


def test_missing_message_id_gets_a_stable_fallback_id():
    raw = RAW.replace("Message-ID: <1.JavaMail.evans@thyme>\n", "")
    a, b = parse_message(raw), parse_message(raw)
    assert a.message_id and a.message_id == b.message_id


def test_owner_is_the_most_frequent_sender_of_sent_mail():
    sent = [
        _msg("1", "me@enron.com", ["a@x"]),
        _msg("2", "me@enron.com", ["b@x"]),
        _msg("3", "assistant@enron.com", ["c@x"]),
    ]
    assert owner_address(sent) == "me@enron.com"


def test_received_keeps_mail_to_or_cc_owner_and_drops_own_and_others():
    me = "me@enron.com"
    msgs = [
        _msg("1", "a@x.com", [me]),
        _msg("2", "b@x.com", ["z@x.com"], cc=[me]),
        _msg("3", me, ["a@x.com"]),  # sent by me
        _msg("4", "c@x.com", ["z@x.com"]),  # not addressed to me
    ]
    assert [m.message_id for m in received(msgs, me)] == ["1", "2"]


def test_received_dedupes_same_message_id_across_folders():
    me = "me@enron.com"
    msgs = [
        _msg("1", "a@x.com", [me]),
        _msg("1", "a@x.com", [me]),
        _msg("2", "a@x.com", [me], date="2001-06-01 09:00"),
    ]
    assert [m.message_id for m in received(msgs, me)] == ["1", "2"]


def test_received_drops_missing_dates_and_bogus_years_and_sorts_by_time():
    me = "me@enron.com"
    msgs = [
        _msg("late", "a@x.com", [me], date="2001-07-01 00:00"),
        _msg("nodate", "a@x.com", [me], date=None),
        _msg("y2044", "a@x.com", [me], date="2044-01-01 00:00"),
        _msg("y1980", "a@x.com", [me], date="1980-01-01 00:00"),
        _msg("early", "a@x.com", [me], date="2000-01-01 00:00"),
    ]
    assert [m.message_id for m in received(msgs, me)] == ["early", "late"]


def _csv(tmp_path):
    def row(
        user,
        folder,
        n,
        mid,
        sender,
        to,
        subject,
        date="Mon, 14 May 2001 16:39:00 -0700",
    ):
        raw = f"Message-ID: <{mid}>\nDate: {date}\nFrom: {sender}\nTo: {to}\nSubject: {subject}\n\nbody {mid}\n"
        return (f"{user}/{folder}/{n}.", raw)

    rows = [
        row("mann-k", "_sent_mail", 1, "s1", "kay.mann@enron.com", "a@x.com", "Re: hi"),
        row("mann-k", "sent", 2, "s2", "kay.mann@enron.com", "b@x.com", "hello"),
        row("mann-k", "inbox", 3, "r1", "a@x.com", "kay.mann@enron.com", "hi"),
        row("mann-k", "all_documents", 4, "r1", "a@x.com", "kay.mann@enron.com", "hi"),
        row("other-u", "inbox", 5, "z1", "q@x.com", "other@enron.com", "ignored user"),
    ]
    p = tmp_path / "emails.csv"
    pl.DataFrame(rows, schema=["file", "message"], orient="row").write_csv(p)
    return p


def test_load_mailboxes_splits_sent_from_other_folders_and_filters_users(tmp_path):
    boxes = load_mailboxes(_csv(tmp_path), ["mann-k"])
    assert set(boxes) == {"mann-k"}
    box = boxes["mann-k"]
    assert box.owner == "kay.mann@enron.com"
    assert sorted(m.message_id for m in box.sent) == ["<s1>", "<s2>"]
    assert len(received(box.others, box.owner)) == 1  # r1 appears in two folders
