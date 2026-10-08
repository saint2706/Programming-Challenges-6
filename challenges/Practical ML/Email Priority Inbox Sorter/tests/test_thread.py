from datetime import datetime, timedelta

import pytest
from inbox_sorter.data import Message
from inbox_sorter.thread import (
    censor_cutoff,
    kind_of,
    label_received,
    normalize_subject,
)

ME = "me@enron.com"
T0 = datetime.fromisoformat("2001-05-01T10:00")


def _msg(mid, sender, to, subject, date=T0, cc=()):
    return Message(
        message_id=mid,
        date=date,
        sender=sender,
        to=list(to),
        cc=list(cc),
        subject=subject,
        body="",
    )


def _label(received, sent, **kw):
    df = label_received(received, sent, ME, **kw)
    return {r["message_id"]: r for r in df.iter_rows(named=True)}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Re: Re: FW: Q3 numbers ", "q3 numbers"),
        ("  RE:   Q3   Numbers", "q3 numbers"),
        ("Fwd: Q3 numbers", "q3 numbers"),
        ("FW:Q3 numbers", "q3 numbers"),
        ("Q3 numbers", "q3 numbers"),
        ("", ""),
        ("Re:", ""),
        ("Reservations for Friday", "reservations for friday"),  # not a Re: prefix
    ],
)
def test_normalize_subject_strips_reply_and_forward_chains(raw, expected):
    assert normalize_subject(raw) == expected


@pytest.mark.parametrize(
    ("subject", "kind"),
    [
        ("Re: hi", "reply"),
        ("RE: Fw: hi", "reply"),
        ("Fw: hi", "forward"),
        ("FWD: Re: hi", "forward"),
        ("hi", "other"),
        ("Reservations", "other"),
    ],
)
def test_kind_of_uses_the_outermost_prefix(subject, kind):
    assert kind_of(subject) == kind


def test_reply_to_the_original_sender_is_a_positive_with_its_time():
    rec = [_msg("r", "a@x.com", [ME], "Q3 numbers")]
    reply_at = T0 + timedelta(hours=3)
    sent = [_msg("s", ME, ["a@x.com"], "Re: Q3 numbers", date=reply_at)]
    row = _label(rec, sent)["r"]
    assert row["acted"] is True
    assert row["kind"] == "reply"
    assert row["acted_at"] == reply_at


def test_reply_must_include_the_original_sender_among_its_recipients():
    rec = [_msg("r", "a@x.com", [ME], "Q3 numbers")]
    sent = [
        _msg("s", ME, ["someone.else@x.com"], "Re: Q3 numbers", T0 + timedelta(hours=1))
    ]
    assert _label(rec, sent)["r"]["acted"] is False


def test_reply_to_original_sender_in_cc_counts():
    rec = [_msg("r", "a@x.com", [ME], "Q3 numbers")]
    sent = [
        _msg(
            "s",
            ME,
            ["b@x.com"],
            "Re: Q3 numbers",
            T0 + timedelta(hours=1),
            cc=["a@x.com"],
        )
    ]
    assert _label(rec, sent)["r"]["acted"] is True


def test_forward_needs_only_the_subject():
    rec = [_msg("r", "a@x.com", [ME], "Q3 numbers")]
    sent = [_msg("s", ME, ["boss@x.com"], "Fw: Q3 numbers", T0 + timedelta(hours=1))]
    row = _label(rec, sent)["r"]
    assert row["acted"] is True and row["kind"] == "forward"


def test_reply_before_receipt_is_ignored():
    rec = [_msg("r", "a@x.com", [ME], "Q3 numbers")]
    sent = [_msg("s", ME, ["a@x.com"], "Re: Q3 numbers", T0 - timedelta(seconds=1))]
    assert _label(rec, sent)["r"]["acted"] is False


def test_reply_at_the_instant_of_receipt_counts():
    rec = [_msg("r", "a@x.com", [ME], "Q3 numbers")]
    sent = [_msg("s", ME, ["a@x.com"], "Re: Q3 numbers", T0)]
    assert _label(rec, sent)["r"]["acted"] is True


def test_window_is_inclusive_at_14_days_and_not_a_second_later():
    rec = [_msg("r", "a@x.com", [ME], "Q3 numbers")]
    on = [_msg("s", ME, ["a@x.com"], "Re: Q3 numbers", T0 + timedelta(days=14))]
    late = [
        _msg("s", ME, ["a@x.com"], "Re: Q3 numbers", T0 + timedelta(days=14, seconds=1))
    ]
    assert _label(rec, on)["r"]["acted"] is True
    assert _label(rec, late)["r"]["acted"] is False


def test_window_days_is_a_parameter():
    rec = [_msg("r", "a@x.com", [ME], "Q3 numbers")]
    sent = [_msg("s", ME, ["a@x.com"], "Re: Q3 numbers", T0 + timedelta(days=3))]
    assert _label(rec, sent, window_days=2)["r"]["acted"] is False
    assert _label(rec, sent, window_days=3)["r"]["acted"] is True


def test_earliest_qualifying_reply_is_acted_at():
    rec = [_msg("r", "a@x.com", [ME], "Q3 numbers")]
    sent = [
        _msg("s2", ME, ["a@x.com"], "Re: Q3 numbers", T0 + timedelta(days=2)),
        _msg(
            "s0", ME, ["z@x.com"], "Re: Q3 numbers", T0 + timedelta(hours=1)
        ),  # wrong recipient
        _msg("s1", ME, ["a@x.com"], "Re: Q3 numbers", T0 + timedelta(days=1)),
    ]
    assert _label(rec, sent)["r"]["acted_at"] == T0 + timedelta(days=1)


def test_a_reply_by_someone_other_than_the_owner_is_not_a_positive():
    rec = [_msg("r", "a@x.com", [ME], "Q3 numbers")]
    sent = [
        _msg(
            "s",
            "assistant@enron.com",
            ["a@x.com"],
            "Re: Q3 numbers",
            T0 + timedelta(hours=1),
        )
    ]
    assert _label(rec, sent)["r"]["acted"] is False


def test_unrelated_subject_is_not_a_positive():
    rec = [_msg("r", "a@x.com", [ME], "Q3 numbers")]
    sent = [_msg("s", ME, ["a@x.com"], "Re: Q4 numbers", T0 + timedelta(hours=1))]
    row = _label(rec, sent)["r"]
    assert row["acted"] is False and row["kind"] is None and row["acted_at"] is None


def test_empty_subject_never_matches_anything():
    rec = [_msg("r", "a@x.com", [ME], "")]
    sent = [_msg("s", ME, ["a@x.com"], "Re:", T0 + timedelta(hours=1))]
    assert _label(rec, sent)["r"]["acted"] is False


def test_a_plain_new_message_with_the_same_subject_is_not_a_reply():
    rec = [_msg("r", "a@x.com", [ME], "Q3 numbers")]
    sent = [_msg("s", ME, ["a@x.com"], "Q3 numbers", T0 + timedelta(hours=1))]
    assert _label(rec, sent)["r"]["acted"] is False


def test_each_received_message_is_labelled_independently():
    rec = [
        _msg("r1", "a@x.com", [ME], "Q3 numbers", T0),
        _msg("r2", "b@x.com", [ME], "Lunch?", T0 + timedelta(hours=1)),
    ]
    sent = [_msg("s", ME, ["a@x.com"], "Re: Q3 numbers", T0 + timedelta(hours=2))]
    out = _label(rec, sent)
    assert out["r1"]["acted"] is True and out["r2"]["acted"] is False


def test_censor_cutoff_is_window_before_the_last_sent_message():
    sent = [
        _msg("a", ME, ["x"], "s", T0),
        _msg("b", ME, ["x"], "s", T0 + timedelta(days=100)),
    ]
    assert censor_cutoff(sent) == T0 + timedelta(days=86)
    assert censor_cutoff(sent, window_days=10) == T0 + timedelta(days=90)
