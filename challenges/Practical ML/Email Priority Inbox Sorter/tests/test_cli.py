import pytest
from inbox_sorter import cli
from inbox_sorter.inbox import days_with_mail
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def fake_context(monkeypatch, trained):
    monkeypatch.setattr(cli, "get_context", lambda: trained)


def _day(trained, box="aa-b"):
    days = days_with_mail(trained[1], box)
    return max(days, key=days.get)


def test_rank_inbox_prints_ranked_messages_with_reasons(trained):
    day = _day(trained)
    result = runner.invoke(
        cli.app, ["rank-inbox", "aa-b", "--day", day.isoformat(), "--top", "3"]
    )
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].lstrip().startswith("1.")
    assert "because:" in result.output
    assert "replied" not in result.output  # the outcome stays hidden by default


def test_reveal_shows_what_the_owner_actually_did(trained):
    day = _day(trained)
    result = runner.invoke(
        cli.app, ["rank-inbox", "aa-b", "--day", day.isoformat(), "--reveal"]
    )
    assert result.exit_code == 0
    assert "replied" in result.output or "ignored" in result.output


def test_unknown_mailbox_exits_two_with_a_message(trained):
    result = runner.invoke(cli.app, ["rank-inbox", "nobody", "--day", "2000-06-01"])
    assert result.exit_code == 2
    assert "unknown mailbox" in result.output


def test_a_day_with_no_mail_exits_two_and_names_the_range(trained):
    result = runner.invoke(cli.app, ["rank-inbox", "aa-b", "--day", "1999-01-01"])
    assert result.exit_code == 2
    assert "no mail" in result.output and "test days run" in result.output


def test_a_malformed_day_exits_two(trained):
    result = runner.invoke(cli.app, ["rank-inbox", "aa-b", "--day", "yesterday"])
    assert result.exit_code == 2
    assert "YYYY-MM-DD" in result.output


def test_report_without_a_report_file_says_what_to_run(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "RESULTS", tmp_path / "missing.json")
    result = runner.invoke(cli.app, ["report"])
    assert result.exit_code == 2
    assert "train" in result.output


def test_report_prints_a_table_from_a_report_file(monkeypatch, tmp_path):
    import json

    from inbox_sorter.models import MODEL_NAMES

    def model():
        return {
            "overall": {
                "pr_auc": 0.5,
                "roc_auc": 0.7,
                "pos_rate": 0.2,
                "n": 10,
                "positives": 2,
            },
            "macro_pr_auc": 0.5,
            "daily": {
                "n_days": 3,
                "precision_at_3": {"mean": 0.4, "lo": 0.3, "hi": 0.5},
                "ndcg_at_5": {"mean": 0.6, "lo": 0.5, "hi": 0.7},
                "recall_top20": {"mean": 0.3, "lo": 0.2, "hi": 0.4},
            },
        }

    p = tmp_path / "report.json"
    p.write_text(
        json.dumps(
            {
                "models": {n: model() for n in MODEL_NAMES},
                "data": {"splits": {"test": 10}},
            }
        )
    )
    monkeypatch.setattr(cli, "RESULTS", p)
    result = runner.invoke(cli.app, ["report"])
    assert result.exit_code == 0, result.output
    assert "lgbm_meta_text" in result.output and "0.600" in result.output
