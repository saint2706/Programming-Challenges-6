from __future__ import annotations

import json

import numpy as np
import pytest
from comp_finder import cli, pipeline
from comp_finder.engine import Settings
from typer.testing import CliRunner

runner = CliRunner()
PROPERTY = {
    "lat": 47.6,
    "long": -122.3,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "bedrooms": 3,
    "bathrooms": 2,
    "floors": 1,
    "grade": 8,
    "condition": 3,
}


@pytest.fixture(autouse=True)
def synthetic_engine(engine, tmp_path, monkeypatch):
    monkeypatch.setenv("COMP_FINDER_HOME", str(tmp_path))
    monkeypatch.setattr(pipeline, "build_engine", lambda *_: (engine, Settings()))


def sale_id(engine, offset=-30):
    return int(engine.ids[engine.sales.height + offset])


def test_find_a_sale_prints_the_estimate_comps_and_reasons(engine):
    out = runner.invoke(
        cli.app, ["find", "--sale-id", str(sale_id(engine)), "--k", "3"]
    )
    assert out.exit_code == 0, out.output
    assert "comps-based estimate" in out.output and "GBM AVM" in out.output
    assert "actual sale price" in out.output
    assert out.output.count("\n#") == 3
    assert "+ Location" in out.output


def test_find_json_is_machine_readable_and_matches_the_engine(engine):
    sid = sale_id(engine)
    out = runner.invoke(
        cli.app,
        ["find", "--sale-id", str(sid), "--method", "hedonic", "--k", "4", "--json"],
    )
    assert out.exit_code == 0, out.output
    got = json.loads(out.output)
    q = int(np.flatnonzero(engine.ids == sid)[-1])
    expected = engine.predict("hedonic", np.array([q]), Settings(k=4))
    assert got["estimate"] == pytest.approx(expected.price[0])
    assert [c["row"] for c in got["comps"]] == expected.comps[0].tolist()
    assert got["actual"] == engine.price[q]


def test_find_a_custom_property(engine):
    out = runner.invoke(
        cli.app, ["find", "--property", json.dumps(PROPERTY), "--k", "3", "--bracket"]
    )
    assert out.exit_code == 0, out.output
    assert "actual sale price" not in out.output
    assert out.output.count("\n#") == 3


@pytest.mark.parametrize(
    "args",
    [
        ["find"],
        ["find", "--sale-id", "1", "--property", "{}"],
        ["find", "--sale-id", "7"],
        ["find", "--sale-id", "1000001", "--method", "magic"],
        ["find", "--property", "not json"],
        ["find", "--property", "[1]"],
        ["find", "--property", '{"lat": 47.6}'],
        ["find", "--property", json.dumps(PROPERTY), "--date", "2014-01-01"],
        ["find", "--property", json.dumps(PROPERTY), "--date", "tomorrow"],
    ],
)
def test_bad_input_exits_2_with_a_message(args):
    out = runner.invoke(cli.app, args)
    assert out.exit_code == 2
    assert "error:" in out.output


def test_saved_settings_follow_the_last_evaluation(tmp_path):
    assert pipeline.saved_settings() == Settings()
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "metrics.json").write_text(
        json.dumps({"settings": {"extrapolate": "flat", "combine": "mean"}}),
        encoding="utf-8",
    )
    saved = pipeline.saved_settings()
    assert (saved.extrapolate, saved.combine) == ("flat", "mean")
    (tmp_path / "results" / "metrics.json").write_text("{broken", encoding="utf-8")
    assert pipeline.saved_settings() == Settings()


def test_data_command_summarises_the_real_table():
    out = runner.invoke(cli.app, ["data"])
    assert out.exit_code == 0, out.output
    assert "21,613 sales" in out.output and "177 resales" in out.output
    assert "fit 14,633 / validation 2,228 / test 4,752" in out.output
