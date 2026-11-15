import json

import pytest
from fastapi.testclient import TestClient
from spc_stream import nab
from spc_stream.cli import main
from spc_stream.server import app


def test_load_maps_windows_to_indices():
    ts, xs, windows = nab.load("art_daily_jumpsdown")
    assert len(ts) == len(xs) == 4032
    ((a, b),) = windows
    assert str(ts[a]) == "2014-04-10 16:15:00" and str(ts[b]) == "2014-04-12 01:45:00"
    with pytest.raises(ValueError):
        nab.load("../../etc/passwd")


def test_score_counts_hits_and_false_alarms():
    s = nab.score("art_daily_jumpsdown", "cusum", prewhiten=False)
    assert (s.windows_hit, s.windows) == (1, 1)
    assert s.false_alarms <= 3
    assert s.mean_delay is not None


def events(text):
    out = []
    for block in text.strip().split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if ": " in line
        )
        if "event" in fields:
            out.append(fields)
    return out


def test_sse_stream():
    c = TestClient(app)
    assert "nyc_taxi" in c.get("/api/streams").json()
    r = c.get(
        "/api/stream/nyc_taxi", params={"chart": "cusum", "speed": 5000, "limit": 30}
    )
    assert r.headers["content-type"].startswith("text/event-stream")
    ev = events(r.text)
    assert [e["event"] for e in ev[:2]] == ["meta", "point"]
    assert ev[-1]["event"] == "end"
    meta = json.loads(ev[0]["data"])
    assert meta["n"] == 10320 and meta["baseline_n"] == 1548
    points = [json.loads(e["data"]) for e in ev if e["event"] == "point"]
    assert [p["i"] for p in points] == list(range(30))
    assert points[0]["stat"] is None  # still in Phase I


def test_sse_rejects_bad_input():
    c = TestClient(app)
    assert c.get("/api/stream/..%2F..%2Fpyproject").status_code in (404, 422)
    assert c.get("/api/stream/not_a_stream").status_code == 422
    assert c.get("/api/stream/nyc_taxi", params={"speed": 0}).status_code == 422
    assert c.get("/api/stream/nyc_taxi", params={"chart": "xbar"}).status_code == 422


def test_index_page():
    html = TestClient(app).get("/").text
    assert "EventSource" in html and "textContent" in html


def test_cli_monitor_prints_alarm(tmp_path, capsys):
    f = tmp_path / "s.csv"
    vals = [10 + (i % 7) * 0.1 for i in range(250)] + [14.0] * 10
    f.write_text(
        "timestamp,value\n" + "".join(f"t{i},{v}\n" for i, v in enumerate(vals))
    )
    assert main(["monitor", str(f), "--chart", "shewhart", "--baseline", "200"]) == 0
    out = capsys.readouterr().out
    assert "t250\t14\tWE1" in out
    assert "# " in out


def test_cli_monitor_errors(tmp_path):
    f = tmp_path / "s.csv"
    f.write_text("1\n2\nabc\n")
    with pytest.raises(SystemExit, match="not a number"):
        main(["monitor", str(f)])
    f.write_text("1\n2\n")
    with pytest.raises(SystemExit, match="Phase I"):
        main(["monitor", str(f)])


def test_cli_arl_and_nab_write_results(tmp_path, monkeypatch):
    from spc_stream import arl

    monkeypatch.setattr(arl, "SHIFTS", (2.0,))
    assert main(["arl", "--reps", "20", "--out", str(tmp_path)]) == 0
    assert set(json.loads((tmp_path / "arl.json").read_text())) == set(arl.CONFIGS)
    assert main(["nab", "--out", str(tmp_path)]) == 0
    rows = json.loads((tmp_path / "nab.json").read_text())
    assert len(rows) == len(nab.STREAMS) * 6
    md = (tmp_path / "nab.md").read_text()
    assert md.endswith("\n") and "| nyc_taxi" in md
