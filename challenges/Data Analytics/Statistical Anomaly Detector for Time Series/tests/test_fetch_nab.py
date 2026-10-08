from __future__ import annotations

import json

import pytest
from ts_anomaly import fetch_nab

CSV = b"timestamp,value\n2014-01-01 00:00:00,1\n2014-01-01 00:05:00,2\n"
LABELS = {
    "realTweets/Twitter_volume_X.csv": [
        ["2014-01-01 00:00:00.000000", "2014-01-01 01:00:00.000000"]
    ],
    "other/y.csv": [],
}


def stub(requested):
    def download(url: str) -> bytes:
        requested.append(url)
        return (
            json.dumps(LABELS).encode()
            if url.endswith("combined_windows.json")
            else CSV
        )

    return download


def test_fetch_writes_csv_and_merges_labels(tmp_path):
    requested: list[str] = []
    csv_path, labels_path, digest = fetch_nab.fetch(
        "realTweets/Twitter_volume_X", tmp_path, stub(requested)
    )
    assert (
        csv_path == tmp_path / "Twitter_volume_X.csv" and csv_path.read_bytes() == CSV
    )
    assert requested[0] == f"{fetch_nab.RAW}/data/realTweets/Twitter_volume_X.csv"
    assert len(digest) == 64
    assert list(json.loads(labels_path.read_text())) == [
        "realTweets/Twitter_volume_X.csv"
    ]

    # a second series merges into the same labels file instead of replacing it
    LABELS["realTweets/Second.csv"] = []
    fetch_nab.fetch("realTweets/Second.csv", tmp_path, stub([]))
    assert set(json.loads(labels_path.read_text())) == {
        "realTweets/Twitter_volume_X.csv",
        "realTweets/Second.csv",
    }
    del LABELS["realTweets/Second.csv"]


@pytest.mark.parametrize(
    "key",
    [
        "../etc/passwd",
        "a/../b",
        "nyc_taxi",
        "a/b/c.csv",
        "cat/.hidden",
        "cat/x y",
        "http://evil/x",
        "",
    ],
)
def test_keys_that_could_escape_the_cache_or_url_are_rejected(key):
    with pytest.raises(ValueError, match="not a NAB series key"):
        fetch_nab.normalise_key(key)


def test_a_non_csv_response_and_an_unlabelled_series_are_refused(tmp_path):
    with pytest.raises(ValueError, match="does not look like a NAB CSV"):
        fetch_nab.fetch(
            "realTweets/Twitter_volume_X", tmp_path, lambda url: b"<html>404</html>"
        )
    with pytest.raises(ValueError, match="no labelled windows"):
        fetch_nab.fetch("realTweets/Unknown", tmp_path, stub([]))
    assert not list(tmp_path.glob("*.csv"))  # nothing half-written


def test_cli_reports_failures_with_exit_code_2(monkeypatch, capsys):
    monkeypatch.setattr(
        fetch_nab, "fetch", lambda key: (_ for _ in ()).throw(ValueError("boom"))
    )
    assert fetch_nab.main(["realTweets/x"]) == 2
    assert "boom" in capsys.readouterr().err
