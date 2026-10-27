import hashlib
import json
from datetime import date

import pytest
from actionitems import evaluate
from actionitems.cli import main
from actionitems.errors import InputError
from actionitems.extract import Extraction
from actionitems.models import ActionItem, Source
from actionitems.paths import corpus_dir, results_dir
from helpers import run

GOLD = evaluate.load_gold()


def _item(task, owner=None, due=None, line=1) -> ActionItem:
    return ActionItem(
        task=task,
        owner=owner,
        due=due,
        due_phrase=None,
        due_confidence=0.9 if due else 0.0,
        item_type="commitment",
        confidence=0.9,
        reasons=[],
        sources=[Source(line, None, None, "x")],
    )


# ---- evaluation arithmetic -------------------------------------------------------------------


def test_locate_quote_returns_its_line():
    assert evaluate.locate_quote("a\nb c\nd", "b c") == 2


@pytest.mark.parametrize("quote", ["missing", "a", ""])
def test_locate_quote_rejects_absent_or_ambiguous_quotes(quote):
    with pytest.raises(InputError):
        evaluate.locate_quote("a\nb\na", quote)


def test_locate_quote_rejects_multi_line_quotes():
    with pytest.raises(InputError):
        evaluate.locate_quote("one\ntwo", "one\ntwo")


@pytest.mark.parametrize(
    ("gold", "predicted", "expected"),
    [
        ("Send the deck", "Send the deck to the client", True),
        ("Send the deck to the client", "Send the deck", True),  # 3 of 5 words is 0.6
        ("Send the deck to the client", "Send it", False),  # 0 of 5 shared words
        ("Send the deck", "Book the room", False),
        ("", "anything", False),
    ],
)
def test_task_matching_uses_half_of_the_gold_words(gold, predicted, expected):
    assert evaluate.task_matches(gold, predicted) is expected


@pytest.mark.parametrize(
    ("gold", "predicted", "expected"),
    [
        (None, None, True),
        ("Marco Rossi", None, False),
        (None, "Marco Rossi", False),
        ("Marco", "Marco Rossi", True),
        ("Marco Rossi", "Marco Rossi", True),
        ("Marco Rossi", "Dana Lee", False),
    ],
)
def test_owner_matching(gold, predicted, expected):
    assert evaluate.owner_matches(gold, predicted) is expected


def test_score_document_counts_tp_fp_fn():
    text = "Marco: send the deck by Friday.\nDana: book the room.\nNoise line."
    entry = {
        "id": "t-1",
        "format": "speaker",
        "items": [
            {
                "quote": "send the deck by Friday.",
                "task": "Send the deck",
                "owner": "Marco",
                "due": "2026-11-06",
                "action": True,
            },
            {
                "quote": "book the room.",
                "task": "Book the room",
                "owner": "Dana",
                "due": None,
                "action": True,
            },
            {
                "quote": "Noise line.",
                "task": None,
                "owner": None,
                "due": None,
                "action": False,
            },
        ],
    }
    extraction = Extraction(
        "t.txt",
        date(2026, 11, 2),
        "UTC",
        "rules-only",
        [
            _item("Send the deck", "Marco Rossi", "2026-11-06", line=1),
            _item("Read the minutes", None, None, line=3),
        ],
    )
    score = evaluate.score_document(entry, text, extraction)
    assert (score.tp, score.fp, score.fn) == (1, 1, 1)
    assert score.missed == ["Book the room"]
    assert score.spurious == ["Read the minutes"]
    assert score.pairs == [(True, True)]


def test_score_document_requires_the_same_line():
    text = "Marco: send the deck by Friday.\nSomething else."
    entry = {
        "id": "t-1",
        "format": "speaker",
        "items": [
            {
                "quote": "send the deck by Friday.",
                "task": "Send the deck",
                "owner": "Marco",
                "due": None,
                "action": True,
            },
        ],
    }
    extraction = Extraction(
        "t.txt",
        date(2026, 11, 2),
        "UTC",
        "rules-only",
        [_item("Send the deck", line=2)],
    )
    score = evaluate.score_document(entry, text, extraction)
    assert (score.tp, score.fp, score.fn) == (0, 1, 1)


def test_aggregate_handles_no_predictions():
    result = evaluate.aggregate([])
    assert result["precision"] == 0.0 and result["recall"] == 0.0
    assert result["owner_accuracy"] is None and result["due_accuracy"] is None


def test_aggregate_computes_f1():
    score = evaluate.DocScore(
        "a", "speaker", tp=2, fp=1, fn=1, pairs=[(True, True), (False, True)]
    )
    result = evaluate.aggregate([score])
    assert result["precision"] == pytest.approx(0.667, abs=1e-3)
    assert result["recall"] == pytest.approx(0.667, abs=1e-3)
    assert result["f1"] == pytest.approx(0.667, abs=1e-3)
    assert result["owner_accuracy"] == 0.5
    assert result["due_accuracy"] == 1.0


# ---- corpus integrity -------------------------------------------------------------------------


def test_corpus_has_48_documents_split_24_and_24():
    docs = GOLD["documents"]
    assert len(docs) == 48
    assert sum(d["split"] == "dev" for d in docs) == 24
    assert sum(d["split"] == "test" for d in docs) == 24


@pytest.mark.parametrize("split", ["dev", "test"])
def test_each_split_has_three_documents_per_format(split):
    counts: dict[str, int] = {}
    for doc in GOLD["documents"]:
        if doc["split"] == split:
            counts[doc["format"]] = counts.get(doc["format"], 0) + 1
    assert set(counts.values()) == {3}
    assert len(counts) == 8


def test_document_ids_and_paths_are_unique_and_match_their_split():
    ids = [d["id"] for d in GOLD["documents"]]
    files = [d["file"] for d in GOLD["documents"]]
    assert len(set(ids)) == len(ids)
    assert len(set(files)) == len(files)
    for doc in GOLD["documents"]:
        assert doc["id"].startswith(doc["split"] + "-")
        assert doc["file"].startswith(doc["split"] + "/")


@pytest.mark.parametrize("doc", GOLD["documents"], ids=lambda d: d["id"])
def test_every_gold_quote_locates_exactly_once(doc):
    text = (corpus_dir() / doc["file"]).read_text(encoding="utf-8")
    for item in doc["items"]:
        assert evaluate.locate_quote(text, item["quote"]) >= 1


@pytest.mark.parametrize("doc", GOLD["documents"], ids=lambda d: d["id"])
def test_gold_labels_are_consistent(doc):
    for item in doc["items"]:
        if item["action"]:
            assert item["task"], item
            assert item["owner"] is None or isinstance(item["owner"], str)
        else:
            assert (
                item["task"] is None and item["owner"] is None and item["due"] is None
            )


def test_dev_split_evaluates_end_to_end():
    metrics, scores = evaluate.evaluate_splits(("dev",), gold=GOLD, base=None)
    dev = metrics["splits"]["dev"]
    assert dev["documents"] == 24
    assert dev["overall"]["gold_actions"] == sum(
        1
        for d in GOLD["documents"]
        if d["split"] == "dev"
        for i in d["items"]
        if i["action"]
    )
    assert len(scores["dev"]) == 24


def test_gold_digest_matches_file_bytes():
    path = corpus_dir() / "gold.json"
    assert evaluate.gold_digest() == hashlib.sha256(path.read_bytes()).hexdigest()


def test_write_results_creates_metrics_and_report(tmp_path):
    metrics, scores = evaluate.evaluate_splits(("dev",), gold=GOLD)
    metrics_path, report_path = evaluate.write_results(
        metrics, {"dev": scores["dev"]}, tmp_path
    )
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert data["splits"]["dev"]["documents"] == 24
    assert data["gold_sha256"] == evaluate.gold_digest()
    report = report_path.read_text(encoding="utf-8")
    assert "# Evaluation report" in report
    assert "## DEV split" in report


def test_summary_lists_each_split():
    metrics, _ = evaluate.evaluate_splits(("dev",), gold=GOLD)
    assert "split: dev" in evaluate.summary(metrics)


def test_committed_results_match_the_current_gold_file():
    path = results_dir() / "metrics.json"
    assert path.exists(), (
        "run `actionitems eval --write` to create results/metrics.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["gold_sha256"] == evaluate.gold_digest(), (
        "gold.json changed; re-run eval --write"
    )
    assert set(data["splits"]) == {"dev", "test"}


# ---- CLI ----------------------------------------------------------------------------------------

SAMPLE = "Marco Rossi: I'll send the deck by Friday.\nDana Lee: Okay, I will book the room tomorrow.\n"


def test_extract_prints_markdown_by_default(tmp_path, capsys):
    path = tmp_path / "notes.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    assert main(["extract", str(path), "--no-nlp", "--meeting-date", "2026-11-02"]) == 0
    out = capsys.readouterr().out
    assert "- [ ] " in out
    assert "Marco Rossi" in out


def test_extract_json_to_stdout(tmp_path, capsys):
    path = tmp_path / "notes.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    assert (
        main(
            [
                "extract",
                str(path),
                "--no-nlp",
                "--format",
                "json",
                "--meeting-date",
                "2026-11-02",
            ]
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert len(data["documents"][0]["items"]) == 2


def test_extract_ics_to_file_uses_lf_free_crlf_output(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE, encoding="utf-8")
    target = tmp_path / "out.ics"
    code = main(
        [
            "extract",
            str(source),
            "--no-nlp",
            "--format",
            "ics",
            "--out",
            str(target),
            "--meeting-date",
            "2026-11-02",
        ]
    )
    assert code == 0
    raw = target.read_bytes()
    assert b"BEGIN:VCALENDAR" in raw
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n")


def test_extract_reads_a_directory(tmp_path, capsys):
    (tmp_path / "a.txt").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "b.md").write_text(
        "- [ ] Read the minutes (Dana Lee)\n", encoding="utf-8"
    )
    assert main(["extract", str(tmp_path), "--no-nlp", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [d["name"] for d in data["documents"]] == ["a.txt", "b.md"]


def test_missing_input_is_a_usage_error(tmp_path, capsys):
    assert main(["extract", str(tmp_path / "nope.txt"), "--no-nlp"]) == 2
    assert "no such file" in capsys.readouterr().err


def test_out_path_that_overwrites_input_is_a_usage_error(tmp_path, capsys):
    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE, encoding="utf-8")
    assert main(["extract", str(source), "--no-nlp", "--out", str(source)]) == 2


@pytest.mark.parametrize("value", ["1.5", "-0.1"])
def test_min_confidence_outside_range_is_a_usage_error(tmp_path, value):
    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE, encoding="utf-8")
    assert main(["extract", str(source), "--no-nlp", "--min-confidence", value]) == 2


def test_too_many_attendees_is_a_usage_error(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE, encoding="utf-8")
    attendees = ",".join(f"Person {n}" for n in range(201))
    assert main(["extract", str(source), "--no-nlp", "--attendees", attendees]) == 2


def test_unknown_format_exits_with_argparse_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["extract", "-", "--format", "xml"])
    assert exc.value.code == 2


def test_eval_write_needs_all_splits(capsys):
    assert main(["eval", "--split", "dev", "--write", "--no-nlp"]) == 2


def test_eval_prints_a_summary_without_writing(capsys):
    assert main(["eval", "--split", "dev", "--no-nlp"]) == 0
    out = capsys.readouterr().out
    assert "split: dev" in out
    assert "overall" in out


def test_extract_with_unreachable_llm_keeps_rules_output(tmp_path, capsys):
    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE, encoding="utf-8")
    code = main(
        [
            "extract",
            str(source),
            "--no-nlp",
            "--format",
            "json",
            "--meeting-date",
            "2026-11-02",
            "--llm-url",
            "http://127.0.0.1:1/v1/chat",
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert "+llm-failed" in data["documents"][0]["backend"]
    assert len(data["documents"][0]["items"]) == 2


def test_extract_refuses_remote_llm_without_flag(tmp_path, capsys):
    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE, encoding="utf-8")
    code = main(
        ["extract", str(source), "--no-nlp", "--llm-url", "http://203.0.113.5/v1/chat"]
    )
    assert code == 1
    assert "non-loopback" in capsys.readouterr().err


def test_run_helper_is_rules_only():
    assert run(SAMPLE).backend == "rules-only"
