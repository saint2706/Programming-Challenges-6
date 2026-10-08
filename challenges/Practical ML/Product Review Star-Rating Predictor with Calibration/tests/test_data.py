import io
import json
import urllib.error

import numpy as np
import polars as pl
import pytest
from helpers import month_ms, read_gz_lines, write_reviews_gz
from review_stars import data
from review_stars.config import Config
from review_stars.text import dedup_key

# ---------------------------------------------------------------- download


class FakeResponse(io.BytesIO):
    def __init__(self, payload, status, headers, fail_after=None):
        super().__init__(payload)
        self.status, self.headers, self.fail_after, self.sent = (
            status,
            headers,
            fail_after,
            0,
        )

    def read(self, n=-1):
        if self.fail_after is not None and self.sent >= self.fail_after:
            raise ConnectionError("connection dropped")
        if self.fail_after is not None:
            n = min(n if n > 0 else self.fail_after, self.fail_after - self.sent)
        block = super().read(n)
        self.sent += len(block)
        return block


class FakeServer:
    """An ``opener`` serving ``payload`` over HTTP semantics: Range, 416, or ignoring Range."""

    def __init__(self, payload, honour_range=True, fail_first_after=None):
        self.payload, self.honour_range = payload, honour_range
        self.fail_first_after, self.calls = fail_first_after, []

    def __call__(self, req, timeout=None):
        rng = req.headers.get("Range")
        self.calls.append(rng)
        total = len(self.payload)
        fail = self.fail_first_after if len(self.calls) == 1 else None
        if rng and self.honour_range:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= total:
                raise urllib.error.HTTPError(
                    req.full_url,
                    416,
                    "range",
                    {"Content-Range": f"bytes */{total}"},
                    None,
                )
            body = self.payload[start:]
            headers = {"Content-Range": f"bytes {start}-{total - 1}/{total}"}
            return FakeResponse(body, 206, headers, fail)
        return FakeResponse(self.payload, 200, {"Content-Length": str(total)}, fail)


PAYLOAD = bytes(range(256)) * 400  # 102,400 bytes


def test_download_writes_the_file_atomically(tmp_path):
    dest = tmp_path / "raw" / "A.jsonl.gz"
    data.download("https://x/A.jsonl.gz", dest, opener=FakeServer(PAYLOAD), chunk=4096)
    assert dest.read_bytes() == PAYLOAD
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_skips_a_file_that_already_exists(tmp_path):
    dest = tmp_path / "A.gz"
    dest.write_bytes(b"done")
    server = FakeServer(PAYLOAD)
    data.download("https://x/A.gz", dest, opener=server)
    assert server.calls == [] and dest.read_bytes() == b"done"


def test_download_resumes_from_the_partial_file_after_a_dropped_connection(tmp_path):
    dest = tmp_path / "A.gz"
    server = FakeServer(PAYLOAD, fail_first_after=30_000)
    with pytest.raises(ConnectionError):
        data.download("https://x/A.gz", dest, opener=server, chunk=4096)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size
    assert 0 < have < len(PAYLOAD) and not dest.exists()
    data.download("https://x/A.gz", dest, opener=server, chunk=4096)
    assert dest.read_bytes() == PAYLOAD
    assert server.calls == [
        None,
        f"bytes={have}-",
    ]  # the second call asked only for the rest


def test_download_restarts_cleanly_when_the_server_ignores_range(tmp_path):
    dest = tmp_path / "A.gz"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(b"stale-garbage")
    data.download(
        "https://x/A.gz", dest, opener=FakeServer(PAYLOAD, honour_range=False)
    )
    assert dest.read_bytes() == PAYLOAD


def test_download_treats_a_complete_partial_file_as_finished(tmp_path):
    dest = tmp_path / "A.gz"
    dest.with_name(dest.name + ".part").write_bytes(PAYLOAD)  # all bytes, never renamed
    data.download("https://x/A.gz", dest, opener=FakeServer(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD


def test_download_discards_an_oversized_stale_partial_file(tmp_path):
    dest = tmp_path / "A.gz"
    dest.with_name(dest.name + ".part").write_bytes(PAYLOAD + b"extra")
    data.download("https://x/A.gz", dest, opener=FakeServer(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD


def test_download_refuses_a_non_https_url(tmp_path):
    with pytest.raises(ValueError, match="https"):
        data.download("http://x/A.gz", tmp_path / "A.gz", opener=FakeServer(PAYLOAD))


def test_download_rejects_a_short_body_but_keeps_the_partial_file_for_resume(tmp_path):
    class Short(FakeServer):
        def __call__(self, req, timeout=None):
            return FakeResponse(
                self.payload[:5000], 200, {"Content-Length": str(len(self.payload))}
            )

    dest = tmp_path / "A.gz"
    with pytest.raises(OSError, match="truncated"):
        data.download("https://x/A.gz", dest, opener=Short(PAYLOAD))
    assert not dest.exists() and dest.with_name(dest.name + ".part").exists()


# ---------------------------------------------------------------- scanning


def row(rating, title, text, ts, asin="P1", **extra):
    return {
        "rating": rating,
        "title": title,
        "text": text,
        "timestamp": ts,
        "parent_asin": asin,
        "user_id": "U",
        "helpful_vote": 0,
        "verified_purchase": True,
        **extra,
    }


def test_scan_index_keeps_valid_rows_and_counts_what_it_drops(tmp_path):
    path = tmp_path / "A.jsonl.gz"
    ts = month_ms(2023, 1)
    write_reviews_gz(
        path,
        [
            row(5.0, "Great", "good", ts),
            row(0.0, "Bad rating", "x", ts),
            row(3.5, "Half star", "x", ts),
            row(4.0, "No text", "   ", ts),
            row(4.0, "", "title blank but text ok", ts),
            row(None, "None rating", "x", ts),
            {"title": "missing fields"},
            row(2.0, "Two", "fine", ts),
        ],
        blank_line_at=3,
    )
    idx = data.scan_index(path)
    # ``row`` is the physical line number (the injected blank line counts), so it matches
    # what ``extract_rows`` reads in its second pass
    assert idx.df["row"].to_list() == [0, 5, 8]
    lines = read_gz_lines(path)
    assert [json.loads(lines[r])["text"] for r in idx.df["row"]] == [
        "good",
        "title blank but text ok",
        "fine",
    ]
    assert idx.stats == {
        "lines": 9,
        "kept": 3,
        "unparseable": 1,  # the blank line
        "bad_rating": 4,  # 0.0, 3.5, None, and a row with no rating at all
        "bad_timestamp": 0,
        "empty_text": 1,
    }


def test_scan_index_drops_a_missing_or_nonsense_timestamp(tmp_path):
    path = tmp_path / "A.jsonl.gz"
    write_reviews_gz(
        path,
        [
            row(5.0, "t", "ok", month_ms(2023, 1)),
            row(5.0, "t", "no ts", None),
            row(5.0, "t", "neg", -5),
        ],
    )
    idx = data.scan_index(path)
    assert idx.df["row"].to_list() == [0] and idx.stats["bad_timestamp"] == 2


def test_a_truncated_archive_is_an_error_not_silently_fewer_rows(tmp_path):
    path = tmp_path / "A.jsonl.gz"
    write_reviews_gz(
        path,
        [row(5.0, "t", f"review number {i}", month_ms(2023, 1)) for i in range(3000)],
    )
    path.write_bytes(path.read_bytes()[:-200])
    with pytest.raises(OSError, match="truncated"):
        data.scan_index(path)


# ---------------------------------------------------------------- dedupe, months, windows


def index_df(rows):
    return pl.DataFrame(
        rows,
        schema={"row": pl.Int64, "ts": pl.Int64, "key": pl.UInt64},
        orient="row",
    )


def test_dedupe_keeps_the_earliest_review_across_categories_and_counts_removals():
    a = index_df(
        [(0, 300, 1), (1, 100, 2), (2, 200, 2)]
    )  # key 2 twice: row 1 is earliest
    b = index_df([(0, 50, 1), (1, 400, 3)])  # key 1 also in A, but earlier here
    out, stats = data.dedupe({"in": a, "ood": b})
    assert out["in"]["row"].to_list() == [1] and out["ood"]["row"].to_list() == [0, 1]
    assert stats == {"in": 2, "ood": 0}


def test_dedupe_breaks_timestamp_ties_by_category_order_then_row():
    a = index_df([(5, 100, 9)])
    b = index_df([(1, 100, 9)])
    out, _ = data.dedupe({"in": a, "ood": b})
    assert out["in"].height == 1 and out["ood"].height == 0


def test_month_counts_label_months_in_calendar_order():
    ts = [month_ms(2022, 12), month_ms(2022, 12, day=20), month_ms(2023, 2)]
    counts = data.month_counts(index_df([(i, t, i) for i, t in enumerate(ts)]))
    assert counts == {"2022-12": 2, "2023-02": 1}


def month_counts(spec):
    """{(year, month): n} -> {"YYYY-MM": n} for the window tests."""
    return {f"{y}-{m:02d}": n for (y, m), n in spec.items()}


def cfg(**kw):
    base = {
        "n_train": 100,
        "n_val": 50,
        "n_cal": 50,
        "n_test": 100,
        "n_ood": 40,
        "n_ood_pool": 20,
        "min_month_rows": 5,
    }
    return Config(**{**base, **kw})


def monthly(start, end, n):
    """n reviews in every month from ``start`` to ``end`` inclusive (both (year, month))."""
    out, (y, m) = {}, start
    while (y, m) <= end:
        out[(y, m)] = n
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return month_counts(out)


def test_windows_count_back_from_the_last_month_both_categories_have_data():
    inn = monthly((2019, 1), (2023, 6), 40)
    ood = monthly((2019, 1), (2023, 3), 40)  # the out-of-domain data ends earlier
    ood["2023-04"] = (
        2  # a trailing sliver below min_month_rows must not become "the end"
    )
    w = data.choose_windows(inn, ood, cfg())
    assert w.end == "2023-03"
    assert (w.test[0], w.test[1]) == ("2022-04", "2023-03")  # 12 months
    assert (w.cal[0], w.cal[1]) == ("2021-10", "2022-03")  # 6 months before
    assert (w.val[0], w.val[1]) == ("2021-04", "2021-09")  # 6 months before that
    assert w.train_before == "2021-04" and w.notes == []


def test_a_thin_calibration_window_is_widened_to_nine_then_twelve_months():
    inn = monthly(
        (2018, 1), (2023, 6), 5
    )  # 6 months = 30 reviews, 9 = 45, 12 = 60; n_cal = 50
    inn.update(
        monthly((2022, 7), (2023, 6), 40)
    )  # the test window itself is well stocked
    w = data.choose_windows(inn, dict(inn), cfg())
    assert w.cal_months == 12 and w.val_months == 12
    assert any("cal window widened from 6 to 12" in n for n in w.notes)
    assert w.cal == ("2021-07", "2022-06") and w.val == ("2020-07", "2021-06")
    assert w.val[1] < w.cal[0]  # still contiguous and ordered
    assert not any("still short" in n for n in w.notes)


def test_a_window_that_stays_short_at_twelve_months_is_recorded_not_hidden():
    inn = monthly((2018, 1), (2023, 6), 1)
    inn.update(monthly((2022, 4), (2023, 6), 40))
    w = data.choose_windows(
        inn, dict(inn), cfg(min_month_rows=1, n_cal=10_000, n_val=10_000)
    )
    assert w.cal_months == 12 and w.val_months == 12
    assert any("still short" in n for n in w.notes)


def test_no_month_with_enough_rows_in_both_categories_is_an_error():
    with pytest.raises(ValueError, match="at least"):
        data.choose_windows({"2023-01": 3}, {"2023-02": 3}, cfg(min_month_rows=50))


def test_window_bounds_are_half_open_epoch_milliseconds():
    w = data.choose_windows(
        monthly((2019, 1), (2023, 3), 40), monthly((2019, 1), (2023, 3), 40), cfg()
    )
    lo, hi = w.bounds("test")
    assert lo == month_ms(2022, 4) and hi == month_ms(2023, 4)
    assert w.split_of(lo) == "test" and w.split_of(hi - 1) == "test"
    assert w.split_of(hi) is None  # after the end: used by nobody
    assert w.split_of(month_ms(2022, 3, day=31)) == "cal"
    assert w.split_of(month_ms(2010, 1)) == "train"


# ---------------------------------------------------------------- sampling and prepare


def two_categories(tmp_path, per_month=30):
    """Tiny in-domain/out-of-domain gz files 2019-01..2023-03 with duplicates and auto titles."""
    rng = np.random.default_rng(0)
    paths = {}
    for name, offset in (("Appliances", 0), ("Software", 1)):
        rows = []
        for y in range(2019, 2024):
            for m in range(1, 13):
                if (y, m) > (2023, 3):
                    break
                for j in range(per_month):
                    rating = float(rng.integers(1, 6))
                    title = {5: "Five Stars", 1: "One Star"}.get(int(rating), "Decent")
                    # like the real data, auto titles stop appearing after the 2018-ish era
                    auto_era = j % 3 and y <= 2020
                    rows.append(
                        row(
                            rating,
                            title if auto_era else "Real title",
                            f"{name} review {y}-{m}-{j}",
                            month_ms(y, m, day=1 + j % 27),
                            asin=f"B{(y * 12 + m + offset) % 7}",
                        )
                    )
        rows.append(rows[10])  # an exact duplicate inside the category
        rows.append(
            {**rows[11], "timestamp": month_ms(2023, 3, day=28)}
        )  # a later duplicate
        paths[name] = tmp_path / f"{name}.jsonl.gz"
        write_reviews_gz(paths[name], rows)
    # the same review text in both categories
    return paths


def small_cfg(**kw):
    base = {
        "n_train": 300,
        "n_val": 100,
        "n_cal": 100,
        "n_test": 150,
        "n_ood": 80,
        "n_ood_pool": 40,
        "min_month_rows": 10,
    }
    return Config(**{**base, **kw})


def test_prepare_builds_disjoint_time_ordered_splits_and_records_everything(tmp_path):
    paths = two_categories(tmp_path)
    out = tmp_path / "prepared"
    summary = data.prepare(paths, out, small_cfg())
    df = pl.read_parquet(out / "reviews.parquet")
    assert set(df["split"].unique()) == set(data.SPLITS)
    assert df["id"].n_unique() == df.height

    # in-domain splits are time ordered and never overlap
    def span(s):
        ts = df.filter(pl.col("split") == s)["ts"]
        return ts.min(), ts.max()

    assert span("train")[1] < span("val")[0] and span("val")[1] < span("cal")[0]
    assert span("cal")[1] < span("test")[0]
    # out-of-domain test lives in the test window, its pool in the calibration window
    test_lo, test_hi = month_ms(2022, 4), month_ms(2023, 4)
    assert test_lo <= span("ood_test")[0] and span("ood_test")[1] < test_hi
    assert month_ms(2021, 10) <= span("ood_pool")[0] and span("ood_pool")[1] < test_lo
    w = summary["windows"]
    assert w["test"] == ["2022-04", "2023-03"] and w["cal"] == ["2021-10", "2022-03"]
    assert summary["config"]["text_mode"] == "title_text"
    assert (out / "windows.json").exists()


def test_prepare_never_lets_a_duplicate_straddle_splits_or_categories(tmp_path):
    paths = two_categories(tmp_path)
    # the identical review in both categories, in different months
    a, b = paths["Appliances"], paths["Software"]
    shared = row(4.0, "Shared", "the very same words", month_ms(2020, 5))
    write_reviews_gz(a, [json.loads(x) for x in read_gz_lines(a)] + [shared])
    later = {**shared, "timestamp": month_ms(2022, 11)}
    write_reviews_gz(b, [json.loads(x) for x in read_gz_lines(b)] + [later])
    out = tmp_path / "prepared"
    summary = data.prepare({"Appliances": a, "Software": b}, out, small_cfg())
    # each category holds 2 internal duplicates; the later copy of the shared review is
    # removed from Software (it is the later of the two), the earlier one stays in Appliances
    assert summary["duplicates_removed"] == {"Appliances": 2, "Software": 3}
    df = pl.read_parquet(out / "reviews.parquet")
    keys = [dedup_key(t, x) for t, x in zip(df["title"], df["text"], strict=True)]
    assert len(keys) == len(set(keys))
    assert df.filter(
        (pl.col("text") == "the very same words") & (pl.col("cat") == "ood")
    ).is_empty()


def test_prepare_records_how_much_of_each_split_it_could_fill(tmp_path):
    paths = two_categories(tmp_path, per_month=30)
    summary = data.prepare(paths, tmp_path / "prepared", small_cfg(n_test=10_000))
    short = summary["splits"]["test"]
    assert short["selected"] == short["available"] < 10_000  # took them all, said so


def test_prepare_is_deterministic_for_a_seed(tmp_path):
    paths = two_categories(tmp_path)
    data.prepare(paths, tmp_path / "p1", small_cfg())
    data.prepare(paths, tmp_path / "p2", small_cfg())
    a = pl.read_parquet(tmp_path / "p1" / "reviews.parquet")
    b = pl.read_parquet(tmp_path / "p2" / "reviews.parquet")
    assert a.equals(b)
    data.prepare(paths, tmp_path / "p3", small_cfg(data_seed=1))
    c = pl.read_parquet(tmp_path / "p3" / "reviews.parquet")
    assert not a["id"].equals(c["id"])


def test_prepare_reports_the_auto_title_share_per_category_and_per_split(tmp_path):
    paths = two_categories(tmp_path)
    summary = data.prepare(paths, tmp_path / "prepared", small_cfg())
    share, by_split = summary["auto_title_share"], summary["auto_title_share_by_split"]
    # Appliances contributes old train rows; Software only the recent out-of-domain windows
    assert 0.0 < share["Appliances"] < 0.5 and share["Software"] == 0.0
    assert set(by_split) == set(data.SPLITS)
    # the fixture, like the real archives, has auto titles only in the older era: the share
    # is large in the train window and exactly zero in the test window
    assert (
        by_split["train"] > 0.1
        and by_split["test"] == 0.0
        and by_split["ood_test"] == 0.0
    )


def test_extract_rows_returns_only_the_requested_physical_lines(tmp_path):
    path = tmp_path / "A.jsonl.gz"
    write_reviews_gz(
        path,
        [row(5.0, f"t{i}", f"text {i}", month_ms(2023, 1)) for i in range(10)],
        blank_line_at=4,
    )
    df = data.extract_rows(path, np.array([0, 3, 6, 10]))
    # line 4 is blank, so physical line 6 is review 5 and line 10 is review 9
    assert df.sort("row")["text"].to_list() == ["text 0", "text 3", "text 5", "text 9"]
    assert df["rating"].dtype == pl.Int8


def test_load_prepared_fails_with_a_pointer_to_the_command_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="review-stars prepare"):
        data.load_prepared(tmp_path / "nowhere")
