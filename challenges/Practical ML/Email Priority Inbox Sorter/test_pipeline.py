import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import numpy as np
import polars as pl
import pytest

from features import FEATURE_COLUMNS
from models import MODEL_NAMES
from pipeline import load_artifacts, prepare, run_all

BASE = datetime(2000, 3, 1, tzinfo=UTC)
SECRET = "do-not-leak-this-subject-xyz"


def _raw(mid, when, sender, to, subject, body):
    return (
        f"Message-ID: <{mid}>\nDate: {format_datetime(when)}\nFrom: {sender}\n"
        f"To: {to}\nSubject: {subject}\n\n{body}\n"
    )


def synthetic_csv(tmp_path, n=360):
    """Two mailboxes. The owner nearly always answers `boss`, rarely anyone else."""
    rng = np.random.default_rng(0)
    rows = []
    for user, owner in (("aa-b", "aa.b@enron.com"), ("cc-d", "cc.d@enron.com")):
        k = 0
        for i in range(n):
            when = BASE + timedelta(hours=7 * i)
            sender = ["boss@enron.com", "peer@enron.com", "news@spam.com"][i % 3]
            subject = f"{SECRET} {user} {i}"
            body = (
                "urgent please review? "
                if sender == "boss@enron.com"
                else "newsletter "
            ) * 3
            rows.append(
                (
                    f"{user}/inbox/{k}.",
                    _raw(f"{user}-r{i}", when, sender, owner, subject, body),
                )
            )
            k += 1
            p = 0.9 if sender == "boss@enron.com" else 0.08
            if rng.random() < p:
                rows.append(
                    (
                        f"{user}/_sent_mail/{k}.",
                        _raw(
                            f"{user}-s{i}",
                            when + timedelta(hours=2),
                            owner,
                            sender,
                            f"Re: {subject}",
                            "ok",
                        ),
                    )
                )
                k += 1
        # keep the owner active to the end so nothing but the final window is censored
        for j in range(5):
            when = BASE + timedelta(hours=7 * n + 24 * j)
            rows.append(
                (
                    f"{user}/_sent_mail/{k}.",
                    _raw(
                        f"{user}-x{j}", when, owner, "someone@x.com", f"misc {j}", "hi"
                    ),
                )
            )
            k += 1
    p = tmp_path / "emails.csv"
    pl.DataFrame(rows, schema=["file", "message"], orient="row").write_csv(p)
    return p


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("enron")
    csv = synthetic_csv(tmp)
    df, stats = prepare(csv, ["aa-b", "cc-d"], tmp)
    return tmp, csv, df, stats


def test_prepare_labels_features_and_censoring(prepared):
    _, _, df, stats = prepared
    assert set(df["mailbox"]) == {"aa-b", "cc-d"}
    assert set(FEATURE_COLUMNS) <= set(df.columns)
    assert {
        "acted",
        "kind",
        "acted_at",
        "date",
        "message_id",
        "body",
        "subject",
    } <= set(df.columns)
    boss = df.filter(pl.col("sender") == "boss@enron.com")["acted"].mean()
    other = df.filter(pl.col("sender") != "boss@enron.com")["acted"].mean()
    assert boss > 0.8 and other < 0.2
    # right-censoring: the last 14 days before the last sent message are dropped
    assert stats["aa-b"]["censored"] > 0
    assert stats["aa-b"]["kept"] == df.filter(pl.col("mailbox") == "aa-b").height
    last_sent = BASE + timedelta(hours=7 * 360 + 24 * 4)
    cutoff = (last_sent - timedelta(days=14)).replace(tzinfo=None)
    assert df["date"].max() <= cutoff
    assert stats["aa-b"]["reply_rate"] > 0 and stats["aa-b"]["forward_rate"] == 0


def test_prepare_writes_a_reloadable_dataset(prepared):
    tmp, _, df, _ = prepared
    again = pl.read_parquet(tmp / "dataset.parquet")
    assert again.height == df.height


@pytest.fixture(scope="module")
def report(prepared, tmp_path_factory):
    tmp, csv, _, _ = prepared
    out = tmp_path_factory.mktemp("results")
    rep = run_all(tmp, out, ["aa-b", "cc-d"], seed=0, csv=csv)
    return rep, out, tmp


def test_report_has_every_model_metric_and_section(report):
    rep, out, _ = report
    assert set(rep["models"]) == set(MODEL_NAMES)
    for name in MODEL_NAMES:
        m = rep["models"][name]
        assert {"pr_auc", "roc_auc", "pos_rate", "n"} <= set(m["overall"])
        assert set(m["per_mailbox"]) == {"aa-b", "cc-d"}
        assert {"precision_at_3", "ndcg_at_5", "recall_top20"} <= set(m["daily"])
        assert {"mean", "lo", "hi"} <= set(m["daily"]["ndcg_at_5"])
    # the planted signal makes the learned models clearly better than random
    assert (
        rep["models"]["lgbm_meta"]["overall"]["pr_auc"]
        > rep["models"]["random"]["overall"]["pr_auc"] + 0.1
    )
    assert {"before", "after"} <= set(rep["calibration"]["lgbm_meta_text"])
    assert set(rep["ablation"]) >= {"sender_history", "text"}
    assert rep["shap"]["top"] and rep["data"]["splits"]["test"] > 0
    assert (out / "report.json").exists()
    assert json.loads((out / "report.json").read_text())["seed"] == 0


def test_report_contains_no_message_text_or_addresses(report):
    _, out, _ = report
    text = (out / "report.json").read_text()
    assert SECRET not in text
    assert "@" not in text


def test_artifacts_reload_and_score(report, prepared):
    _, _, df, _ = prepared
    _, _, tmp = report
    art = load_artifacts(tmp)
    sample = df.head(10)
    assert art.models.predict("lgbm_meta_text", sample).shape == (10,)
    cal = art.calibrators["lgbm_meta_text"].predict(
        art.models.predict("lgbm_meta_text", sample)
    )
    assert ((cal >= 0) & (cal <= 1)).all()
