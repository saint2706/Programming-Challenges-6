import gzip

import data
import numpy as np
import polars as pl
import pytest
from helpers import make_frame

RAW_COLUMNS = {*data.FEATURES, "treatment", "conversion", "visit", "exposure"}


def write_raw_gz(path, df):
    path.write_bytes(gzip.compress(df.write_csv().encode()))


def test_download_rejects_non_https(tmp_path):
    with pytest.raises(ValueError, match="https"):
        data.download("http://example.com/x.gz", tmp_path / "x.gz")
    with pytest.raises(ValueError, match="https"):
        data.download("file:///etc/passwd", tmp_path / "x.gz")


def test_download_is_a_no_op_when_the_file_exists(tmp_path):
    dest = tmp_path / "x.gz"
    dest.write_bytes(b"already here")
    assert data.download("https://example.invalid/x.gz", dest) == dest
    assert dest.read_bytes() == b"already here"


def test_sample_raw_reads_gz_and_caps_rows(tmp_path):
    raw = tmp_path / "raw.csv.gz"
    write_raw_gz(raw, make_frame(500))
    out = data.sample_raw(raw, n=200, seed=0)
    assert out.height == 200
    assert set(out.columns) == RAW_COLUMNS
    assert out["treatment"].dtype == pl.Int8
    assert data.sample_raw(raw, n=10_000, seed=0).height == 500


def test_sample_raw_is_seeded(tmp_path):
    raw = tmp_path / "raw.csv.gz"
    write_raw_gz(raw, make_frame(500))
    assert data.sample_raw(raw, 100, 1).equals(data.sample_raw(raw, 100, 1))
    assert not data.sample_raw(raw, 100, 1).equals(data.sample_raw(raw, 100, 2))


def test_fetch_uses_existing_raw_file_and_caches(tmp_path):
    write_raw_gz(tmp_path / data.RAW_NAME, make_frame(300))
    target = data.fetch(tmp_path, n=100, seed=0, url="https://example.invalid/none")
    assert target.exists() and data.load(tmp_path).height == 100
    stamp = target.stat().st_mtime_ns
    again = data.fetch(tmp_path, n=100, seed=0, url="https://example.invalid/none")
    assert again == target
    assert target.stat().st_mtime_ns == stamp


def test_load_without_fetch_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch"):
        data.load(tmp_path)


def test_balance_passes_on_a_randomized_frame():
    bal = data.assert_randomized(make_frame(20000))
    assert bal["smd"].abs().max() < 0.1 and bal.height == 12


def test_balance_fails_when_treatment_depends_on_a_feature():
    with pytest.raises(data.RandomizationError, match="f0"):
        data.assert_randomized(make_frame(20000, confounded=True))


def test_split_is_a_disjoint_cover_with_the_requested_fractions():
    df = make_frame(10000).with_row_index("rid")
    tr, va, te = data.split(df, seed=0)
    ids = [set(p["rid"].to_list()) for p in (tr, va, te)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert sum(len(i) for i in ids) == 10000
    assert tr.height == pytest.approx(6000, abs=60)
    assert va.height == pytest.approx(2000, abs=60)


def test_split_is_stratified_on_treatment_and_visit():
    df = make_frame(20000)
    for part in data.split(df, seed=0):
        assert part["treatment"].mean() == pytest.approx(
            df["treatment"].mean(), abs=0.01
        )
        assert part["visit"].mean() == pytest.approx(df["visit"].mean(), abs=0.01)


def test_split_is_seeded():
    df = make_frame(2000).with_row_index("rid")
    a, b = data.split(df, seed=4)[0], data.split(df, seed=4)[0]
    assert a["rid"].to_list() == b["rid"].to_list()


def test_xy_shapes_and_dtypes():
    X, t, y = data.xy(make_frame(500), "conversion")
    assert X.shape == (500, 12) and X.dtype == np.float32
    assert t.dtype == np.int8 and y.dtype == np.int8


@pytest.mark.parametrize("bad", ["exposure", "treatment", "visit", "conversion"])
def test_xy_refuses_post_treatment_and_outcome_columns(bad):
    with pytest.raises(ValueError, match=bad):
        data.xy(make_frame(100), features=["f0", bad])


def test_propensity_is_the_treated_share():
    assert data.propensity(make_frame(20000)) == pytest.approx(0.85, abs=0.01)
