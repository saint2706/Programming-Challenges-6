import cli
import pytest
from ranker import Ranker
from test_ranker import HashEncoder, _frames
from typer.testing import CliRunner

runner = CliRunner()
RESUME = "nurse patient ward triage dosage clinic nurse patient team worked"


@pytest.fixture(autouse=True)
def fake_ranker(monkeypatch, tmp_path):
    jobs, resumes = _frames()
    r = Ranker.build(
        jobs, resumes, HashEncoder(), index_dir=tmp_path / "idx", fusion_weight=0.5
    )
    monkeypatch.setattr(cli, "get_ranker", lambda: r)


def _file(tmp_path, text, name="resume.txt"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_rank_jobs_prints_ranked_matches_with_evidence_and_gaps(tmp_path):
    result = runner.invoke(
        cli.app,
        ["rank-jobs", str(_file(tmp_path, RESUME)), "--top", "3", "--scorer", "tfidf"],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith(" 1.") and "HEALTHCARE" in lines[0]
    assert "evidence:" in result.output and "nurse" in result.output
    assert "missing:" in result.output
    assert result.output.count("HEALTHCARE") >= 3


def test_dense_scorer_says_its_evidence_is_only_lexical_overlap(tmp_path):
    result = runner.invoke(
        cli.app,
        [
            "rank-jobs",
            str(_file(tmp_path, RESUME)),
            "--scorer",
            "embedding",
            "--top",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "lexical overlap" in result.output


def test_explain_dense_prints_post_hoc_sentences(tmp_path):
    text = "Led nurse patient ward triage work. Collects stamps on weekends. Enjoys long walks."
    result = runner.invoke(
        cli.app,
        [
            "rank-jobs",
            str(_file(tmp_path, text)),
            "--scorer",
            "embedding",
            "--top",
            "1",
            "--explain-dense",
        ],
    )
    assert result.exit_code == 0
    assert "post-hoc" in result.output and "Led nurse patient" in result.output


def test_rank_resumes_is_the_reverse_direction(tmp_path):
    job = "kitchen menu sauce saute pastry plating kitchen menu"
    result = runner.invoke(
        cli.app,
        [
            "rank-resumes",
            str(_file(tmp_path, job, "job.txt")),
            "--top",
            "3",
            "--scorer",
            "bm25",
        ],
    )
    assert result.exit_code == 0
    assert result.output.count("CHEF") >= 3


def test_empty_file_is_a_clear_usage_error(tmp_path):
    result = runner.invoke(cli.app, ["rank-jobs", str(_file(tmp_path, "  \n"))])
    assert result.exit_code == 2
    assert "empty" in result.output.lower()


def test_missing_file_is_a_clear_usage_error(tmp_path):
    result = runner.invoke(cli.app, ["rank-jobs", str(tmp_path / "nope.txt")])
    assert result.exit_code == 2
    assert "not found" in result.output.lower()


def test_unknown_scorer_is_a_clear_usage_error(tmp_path):
    result = runner.invoke(
        cli.app, ["rank-jobs", str(_file(tmp_path, RESUME)), "--scorer", "magic"]
    )
    assert result.exit_code == 2
    assert "unknown scorer" in result.output.lower()


def test_format_report_lists_every_scorer_with_interval():
    metric = {"mean": 0.5, "lo": 0.4, "hi": 0.6}
    one = {
        s: {"ndcg@10": metric, "mrr": metric, "p@10": metric, "map@50": metric}
        for s in ("random", "tfidf", "bm25", "embedding", "fusion", "rrf")
    }
    report = {
        "results": {
            v: {"resume->jobs": one, "jobs->resumes": one} for v in ("full", "stripped")
        },
        "fusion_weight": {"full": 0.7, "stripped": 0.6},
        "n": {"test_resumes": 10, "test_jobs": 20, "val_resumes": 10, "val_jobs": 20},
        "skipped_categories": {},
        "backend": {"name": "openvino-gpu", "tried": []},
    }
    text = cli.format_report(report)
    for name in (
        "random",
        "tfidf",
        "bm25",
        "embedding",
        "fusion",
        "rrf",
        "openvino-gpu",
    ):
        assert name in text
    assert "0.500 [0.400, 0.600]" in text
