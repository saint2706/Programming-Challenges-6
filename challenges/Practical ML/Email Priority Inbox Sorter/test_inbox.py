from datetime import date

import polars as pl
import pytest

from features import LOCAL_OFFSET
from inbox import days_with_mail, rank_inbox
from models import time_split


def _a_busy_test_day(df, box="aa-b"):
    days = days_with_mail(df, box)
    assert days, "synthetic data should have test days"
    return max(days, key=days.get)


def test_test_days_are_only_in_the_test_period(trained):
    _, df = trained
    test = time_split(df)[2]
    first = (
        test.filter(pl.col("mailbox") == "aa-b")["date"].min() + LOCAL_OFFSET
    ).date()
    days = days_with_mail(df, "aa-b")
    assert min(days) >= first
    assert all(isinstance(d, date) and n > 0 for d, n in days.items())


def test_rank_inbox_orders_by_score_and_calibrates_probabilities(trained):
    art, df = trained
    day = _a_busy_test_day(df)
    items = rank_inbox(art, df, "aa-b", day, top=5)
    assert 1 <= len(items) <= 5
    raws = [i.raw_score for i in items]
    assert raws == sorted(raws, reverse=True)
    assert [i.rank for i in items] == list(range(1, len(items) + 1))
    assert all(0 <= i.probability <= 1 for i in items)
    assert all(len(i.reasons) == 4 for i in items)
    assert all(isinstance(i.acted, bool) for i in items)


def test_unknown_mailbox_and_empty_day_raise_value_errors(trained):
    art, df = trained
    with pytest.raises(ValueError, match="unknown mailbox"):
        rank_inbox(art, df, "nobody", date(2000, 6, 1))
    with pytest.raises(ValueError, match="no mail"):
        rank_inbox(art, df, "aa-b", date(1999, 1, 1))


def test_the_metadata_only_model_can_be_chosen(trained):
    art, df = trained
    items = rank_inbox(art, df, "aa-b", _a_busy_test_day(df), model="lgbm_meta")
    assert items and items[0].reasons
    with pytest.raises(ValueError, match="model"):
        rank_inbox(art, df, "aa-b", _a_busy_test_day(df), model="tfidf_lr")
