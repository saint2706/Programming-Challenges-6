from __future__ import annotations

import json

import pytest
from credit_scorecard.cli import main


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    assert main(["train", "--out", str(out)]) == 0
    return out


def test_train_writes_the_model_report_and_charts(trained_dir):
    names = {p.name for p in trained_dir.iterdir()}
    assert {"scorecard.json", "metrics.json", "report.md"} <= names
    assert {"woe.png", "roc.png", "calibration.png", "score_distribution.png"} <= names
    report = (trained_dir / "report.md").read_text(encoding="utf-8")
    assert "Discrimination" in report and "Feature selection" in report
    metrics = json.loads((trained_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["scorecard_test"]["auc"] > 0.84


def test_score_prints_points_pd_and_reasons(trained_dir, capsys):
    applicant = {
        "age": 30,
        "RevolvingUtilizationOfUnsecuredLines": 1.2,
        "NumberOfTimes90DaysLate": 2,
    }
    assert (
        main(
            [
                "score",
                "--model",
                str(trained_dir / "scorecard.json"),
                "--applicant",
                json.dumps(applicant),
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["score"] == sum(out["points"].values())
    assert 0 < out["probability_of_default"] < 1
    assert out["reason_codes"][0]["feature"] in out["points"]
    assert "MonthlyIncome" in out["unspecified_fields"]


def test_safer_applicant_scores_higher(trained_dir, capsys):
    model = str(trained_dir / "scorecard.json")
    scores = []
    for utilization in (1.2, 0.05):
        main(
            [
                "score",
                "--model",
                model,
                "--applicant",
                json.dumps({"RevolvingUtilizationOfUnsecuredLines": utilization}),
            ]
        )
        scores.append(json.loads(capsys.readouterr().out)["score"])
    assert scores[1] > scores[0]


@pytest.mark.parametrize("bad", ["not json", "[1, 2]", '{"shoe_size": 9}'])
def test_score_rejects_bad_input_with_exit_code_2(trained_dir, capsys, bad):
    assert (
        main(
            [
                "score",
                "--model",
                str(trained_dir / "scorecard.json"),
                "--applicant",
                bad,
            ]
        )
        == 2
    )
    assert capsys.readouterr().err.startswith("error:")


def test_score_reads_an_applicant_file(trained_dir, tmp_path, capsys):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"age": 70}), encoding="utf-8")
    assert (
        main(
            [
                "score",
                "--model",
                str(trained_dir / "scorecard.json"),
                "--applicant-file",
                str(path),
            ]
        )
        == 0
    )
    assert "score" in json.loads(capsys.readouterr().out)


def test_devices_lists_the_cpu(capsys):
    assert main(["devices"]) == 0
    assert "CPU" in capsys.readouterr().out
