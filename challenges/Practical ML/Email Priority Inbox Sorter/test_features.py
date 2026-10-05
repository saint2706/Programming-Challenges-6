import random
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from data import Message
from features import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    build_sent_index,
    history_counts,
    metadata_features,
    received_frame,
)
from thread import label_received

ME = "me@enron.com"
T0 = datetime.fromisoformat("2001-05-01T10:00")


def _msg(mid, sender, to, subject="hello", date=T0, cc=(), body="body"):
    return Message(
        message_id=mid,
        date=date,
        sender=sender,
        to=list(to),
        cc=list(cc),
        subject=subject,
        body=body,
    )


def _features(received, sent):
    labels = label_received(received, sent, ME)
    df = received_frame(received, labels, mailbox="mb", owner=ME)
    feats = metadata_features(df, build_sent_index(sent, ME))
    return {r["message_id"]: r for r in feats.iter_rows(named=True)}


# --- history_counts vs a brute-force reference -------------------------------


def _brute(qk, qt, ek, et):
    return [
        sum(1 for k, t in zip(ek, et, strict=True) if k == key and t < when)
        for key, when in zip(qk, qt, strict=True)
    ]


def test_history_counts_equals_brute_force_on_random_data():
    rng = random.Random(7)
    base = datetime.fromisoformat("2001-01-01")
    qk = [rng.choice("abcd") for _ in range(300)]
    qt = [base + timedelta(minutes=rng.randrange(0, 5000)) for _ in range(300)]
    ek = [rng.choice("abcde") for _ in range(400)]
    et = [base + timedelta(minutes=rng.randrange(0, 5000)) for _ in range(400)]
    got = history_counts(qk, qt, ek, et)
    assert got.tolist() == _brute(qk, qt, ek, et)


def test_history_counts_is_strictly_before_and_handles_no_events():
    t = datetime.fromisoformat("2001-01-01")
    got = history_counts(["a", "a", "z"], [t, t + timedelta(seconds=1), t], ["a"], [t])
    assert got.tolist() == [0, 1, 0]
    assert history_counts([], [], [], []).tolist() == []


# --- individual features ------------------------------------------------------


def test_recipient_content_and_time_features():
    body = "Is this ok? Really?\nSee the attached file."
    m = _msg(
        "r",
        "boss@enron.com",
        [ME, "b@x.com", "c@x.com"],
        subject="URGENT BUDGET REVIEW",
        cc=["d@x.com"],
        body=body,
        date=datetime.fromisoformat("2001-05-02T17:30"),  # Wed 11:30 Houston time
    )
    f = _features([m], [])["r"]
    assert (f["n_to"], f["n_cc"], f["owner_in_to"], f["mass_mail"]) == (3, 1, 1, 0)
    assert f["sender_is_enron"] == 1
    assert f["subject_caps"] == 1 and f["subject_len"] == len("URGENT BUDGET REVIEW")
    assert f["question_marks"] == 2 and f["mentions_attachment"] == 1
    assert f["body_len"] == len(body)
    assert (f["hour"], f["weekday"], f["business_hours"]) == (11, 2, 1)


def test_cc_only_external_sender_and_weekend_night_are_flagged():
    m = _msg(
        "r",
        "someone@gmail.com",
        ["x@x.com"],
        cc=[ME],
        date=datetime.fromisoformat(
            "2001-05-06T05:00"
        ),  # Sun 05:00Z = Sat 23:00 Houston
    )
    f = _features([m], [])["r"]
    assert f["owner_in_to"] == 0 and f["sender_is_enron"] == 0
    assert f["business_hours"] == 0 and f["weekday"] == 5


def test_mass_mail_flag_at_ten_recipients():
    nine = _msg(
        "a", "x@enron.com", [ME] + [f"u{i}@x.com" for i in range(7)], cc=["c@x.com"]
    )
    ten = _msg(
        "b", "x@enron.com", [ME] + [f"u{i}@x.com" for i in range(8)], cc=["c@x.com"]
    )
    f = _features([nine, ten], [])
    assert f["a"]["mass_mail"] == 0 and f["b"]["mass_mail"] == 1


def test_subject_prefix_depths_and_all_caps_ignores_the_prefix():
    m = _msg("r", "a@x.com", [ME], subject="RE: Re: FW: STATUS UPDATE")
    f = _features([m], [])["r"]
    assert (f["re_depth"], f["fw_depth"]) == (2, 1)
    assert f["subject_caps"] == 1
    plain = _features([_msg("p", "a@x.com", [ME], subject="Status update")], [])["p"]
    assert plain["subject_caps"] == 0 and plain["re_depth"] == 0


def test_missing_subject_and_empty_body_give_zeros_not_errors():
    f = _features([_msg("r", "a@x.com", [ME], subject="", body="")], [])["r"]
    assert f["subject_len"] == 0 and f["body_len"] == 0 and f["subject_caps"] == 0


# --- sender / owner history ---------------------------------------------------


def test_sender_history_counts_only_earlier_messages_and_acted_only_once_known():
    r1 = _msg("r1", "a@x.com", [ME], subject="one", date=T0)
    r2 = _msg("r2", "a@x.com", [ME], subject="two", date=T0 + timedelta(days=2))
    r3 = _msg("r3", "a@x.com", [ME], subject="three", date=T0 + timedelta(days=5))
    # owner answered r1 at +3 days: unknown when r2 arrives (+2d), known at r3 (+5d)
    sent = [_msg("s1", ME, ["a@x.com"], "Re: one", T0 + timedelta(days=3))]
    f = _features([r1, r2, r3], sent)
    assert [f[k]["sender_prior_count"] for k in ("r1", "r2", "r3")] == [0, 1, 2]
    assert [f[k]["sender_prior_acted"] for k in ("r1", "r2", "r3")] == [0, 0, 1]
    assert f["r1"]["sender_prior_acted_rate"] < 0.5  # smoothed prior, no history
    assert f["r3"]["sender_prior_acted_rate"] > f["r2"]["sender_prior_acted_rate"]


def test_owner_history_counts_prior_sends_to_the_sender_and_thread_membership():
    r = _msg("r", "a@x.com", [ME], subject="Q3 numbers", date=T0 + timedelta(days=2))
    sent = [
        _msg("s1", ME, ["a@x.com"], "hello", T0),  # before: counts
        _msg(
            "s2", ME, ["a@x.com", "b@x.com"], "other", T0 + timedelta(days=1)
        ),  # counts
        _msg("s3", ME, ["a@x.com"], "later", T0 + timedelta(days=3)),  # after: ignored
        _msg(
            "s4", ME, ["b@x.com"], "Q3 numbers", T0 + timedelta(days=1)
        ),  # thread, before
    ]
    f = _features([r], sent)["r"]
    assert f["owner_prior_sends_to_sender"] == 2
    assert f["owner_in_thread"] == 1


def test_owner_in_thread_ignores_sent_mail_after_receipt():
    r = _msg("r", "a@x.com", [ME], subject="Q3 numbers", date=T0)
    sent = [_msg("s", ME, ["a@x.com"], "Re: Q3 numbers", T0 + timedelta(hours=1))]
    assert _features([r], sent)["r"]["owner_in_thread"] == 0


def test_later_events_never_change_earlier_messages_features():
    rng = random.Random(3)
    senders = ["a@x.com", "b@x.com", "c@x.com"]
    received = [
        _msg(
            f"r{i}",
            rng.choice(senders),
            [ME],
            subject=f"topic {i % 5}",
            date=T0 + timedelta(hours=7 * i),
        )
        for i in range(60)
    ]
    cut = T0 + timedelta(hours=7 * 30)
    sent = [
        _msg(
            f"s{i}",
            ME,
            [rng.choice(senders)],
            f"Re: topic {i % 5}",
            T0 + timedelta(hours=7 * i + 2),
        )
        for i in range(60)
    ]
    base = _features(received, sent)

    # change everything the owner does at or after `cut`
    sent2 = [s for s in sent if s.date < cut]
    sent2 += [
        _msg(f"x{i}", ME, ["a@x.com"], f"Re: topic {i % 5}", cut + timedelta(hours=i))
        for i in range(25)
    ]
    changed = _features(received, sent2)
    early = [m.message_id for m in received if m.date < cut]
    assert len(early) >= 25
    for mid in early:
        assert changed[mid] == base[mid], mid


def test_every_feature_column_belongs_to_exactly_one_group_and_is_emitted():
    flat = [c for cols in FEATURE_GROUPS.values() for c in cols]
    assert len(flat) == len(set(flat))
    assert sorted(flat) == sorted(FEATURE_COLUMNS)
    assert set(FEATURE_GROUPS) == {
        "recipients",
        "content",
        "time",
        "thread",
        "sender_history",
        "owner_history",
    }
    feats = metadata_features(
        received_frame(
            [_msg("r", "a@x.com", [ME])],
            pl.DataFrame(
                schema={
                    "message_id": pl.Utf8,
                    "acted": pl.Boolean,
                    "kind": pl.Utf8,
                    "acted_at": pl.Datetime("us"),
                }
            ),
            "mb",
            ME,
        ),
        build_sent_index([], ME),
    )
    assert feats.columns == ["message_id", *FEATURE_COLUMNS]
    assert np.isfinite(feats.select(FEATURE_COLUMNS).to_numpy()).all()
