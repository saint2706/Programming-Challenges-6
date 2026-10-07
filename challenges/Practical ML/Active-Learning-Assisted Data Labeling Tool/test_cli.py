import cli
import data
import pipeline
import pytest
from helpers import fake_choice, make_problem, tiny_config
from typer.testing import CliRunner

runner = CliRunner()
INTENTS = ["balance", "card", "transfer"]
TEXTS = [f"{INTENTS[i % 3]} request number {i}" for i in range(60)]


def run(*args):
    return runner.invoke(cli.app, [str(a) for a in args])


@pytest.fixture
def csv_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "get_choice", fake_choice)
    monkeypatch.setattr(data, "DATA_DIR", tmp_path / "data")
    pool = tmp_path / "pool.csv"
    rows = ["text,intent"] + [f"{t},{INTENTS[i % 3]}" for i, t in enumerate(TEXTS)]
    pool.write_text("\n".join(rows) + "\n", encoding="utf-8")
    classes = tmp_path / "classes.txt"
    classes.write_text("\n".join(INTENTS) + "\n", encoding="utf-8")
    return pool, classes


@pytest.fixture
def project(tmp_path, csv_files):
    pool, classes = csv_files
    proj = tmp_path / "proj"
    result = run(
        "init", proj, "--csv", pool, "--classes", classes, "--label-col", "intent"
    )
    assert result.exit_code == 0, result.output
    return proj


def test_init_from_csv_then_suggest_label_status_and_export(tmp_path, project):
    assert "0 of 60" in run("status", project).output
    out = run("suggest", project, "--batch", 5).output
    assert sum(1 for ln in out.splitlines() if ln.startswith("#")) == 5
    assert "nothing labeled yet" in out
    for i in range(6):
        r = run("label", project, i, INTENTS[i % 3])
        assert r.exit_code == 0 and f"labeled #{i}" in r.output
    status = run("status", project).output
    assert "6 of 60" in status and "3 of 3 intents" in status and "round 6" in status
    out = run("suggest", project, "--strategy", "k-center", "--batch", 3).output
    assert "near labeled" in out
    export = tmp_path / "out.csv"
    assert run("export", project, export).exit_code == 0
    assert len(export.read_text(encoding="utf-8").splitlines()) == 61


def test_every_strategy_works_from_the_command_line(project):
    for name in (
        "random",
        "least-confidence",
        "margin",
        "entropy",
        "k-center",
        "badge",
        "qbc",
        "cluster-margin",
    ):
        r = run("suggest", project, "--strategy", name, "--batch", 4)
        assert r.exit_code == 0, (name, r.output)


@pytest.mark.parametrize(
    "args,fragment",
    [
        (("label", "{p}", 0, "zzz"), "not one of"),
        (("label", "{p}", 99999, "card"), "no item"),
        (("suggest", "{p}", "--strategy", "nope"), "unknown strategy"),
        (("suggest", "{p}", "--batch", 0), ""),
    ],
)
def test_user_errors_exit_2_with_one_line(project, args, fragment):
    r = run(*[str(project) if a == "{p}" else a for a in args])
    assert r.exit_code == 2
    assert "Traceback" not in r.output and fragment in r.output


def test_commands_on_a_missing_project_say_to_run_init(tmp_path):
    for cmd in (("status",), ("suggest",), ("export", tmp_path / "o.csv")):
        args = [cmd[0], tmp_path / "nope", *cmd[1:]]
        r = run(*args)
        assert r.exit_code == 2 and "init" in r.output and "Traceback" not in r.output
    assert run("label", tmp_path / "nope", 0, "a").exit_code == 2


def test_init_validates_its_inputs(tmp_path, csv_files, project):
    pool, classes = csv_files
    bad = tmp_path / "bad.csv"
    bad.write_text("body\nhello\n", encoding="utf-8")
    one_class = tmp_path / "one.csv"
    one_class.write_text("text,intent\nhello,a\nworld,a\n", encoding="utf-8")
    cases = [
        (("init", tmp_path / "p1"), "exactly one"),
        (("init", tmp_path / "p2", "--demo", "--csv", pool), "exactly one"),
        (
            ("init", tmp_path / "p3", "--csv", tmp_path / "missing.csv"),
            "does not exist",
        ),
        (
            ("init", tmp_path / "p4", "--csv", bad, "--classes", classes),
            "no 'text' column",
        ),
        (("init", tmp_path / "p5", "--csv", pool), "--classes"),
        (
            ("init", tmp_path / "p6", "--csv", one_class, "--label-col", "intent"),
            "fewer than two",
        ),
        (
            (
                "init",
                tmp_path / "p7",
                "--csv",
                pool,
                "--classes",
                classes,
                "--label-col",
                "text",
            ),
            "not in the class list",
        ),
        (("init", project, "--csv", pool, "--classes", classes), "exists"),
    ]
    for args, fragment in cases:
        r = run(*args)
        assert (
            r.exit_code == 2 and fragment in r.output and "Traceback" not in r.output
        ), (args, r.output)


def test_init_demo_builds_a_project_with_gold_labels_and_an_evaluation_set(
    tmp_path, banking_dir, monkeypatch
):
    loaded = pipeline.load_problem(banking_dir, get_choice=fake_choice, val_size=30)
    monkeypatch.setattr(cli, "load_problem", lambda: loaded)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path / "no-results")
    proj = tmp_path / "demo"
    assert run("init", proj, "--demo").exit_code == 0
    assert (proj / "eval.npz").exists()
    assert run("suggest", proj, "--strategy", "k-center", "--batch", 3).exit_code == 0
    assert run("label", proj, 0, "c0").exit_code == 0
    status = run("status", proj).output
    assert (
        "accuracy 0." in status or "accuracy 1." in status
    )  # an evaluation set exists


def test_benchmark_prints_the_report_and_a_later_run_completes_it(
    tmp_path, monkeypatch
):
    prob = make_problem(n=300)
    monkeypatch.setattr(cli, "load_problem", lambda: (prob, None, {"backend": "fake"}))
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path / "results")
    base = ["benchmark", "--seeds", 2, "--budget", 30, "--batch", 10, "--jobs", 1]
    first = run(*base, "--stage", "main")
    assert first.exit_code == 0, first.output
    assert "margin" in first.output and "random" in first.output
    assert (
        "ceiling" in first.output.lower()
        and "not computed: batch, stop" in first.output
    )
    assert (tmp_path / "results" / "report.json").exists()
    second = run(*base, "--stage", "batch", "--stage", "stop")
    assert second.exit_code == 0 and "not computed" not in second.output
    assert "stopping rule" in second.output


def test_benchmark_rejects_unknown_stages_and_reports_missing_data(
    tmp_path, monkeypatch
):
    assert run("benchmark", "--stage", "nope").exit_code == 2
    monkeypatch.setattr(data, "DATA_DIR", tmp_path / "empty")
    r = run("benchmark")
    assert r.exit_code == 2 and "fetch" in r.output and "Traceback" not in r.output


def test_fetch_and_embed_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "fetch", lambda *a, **k: tmp_path)
    assert "data in" in run("fetch").output

    def boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(data, "fetch", boom)
    r = run("fetch")
    assert r.exit_code == 2 and "no network" in r.output

    prob = make_problem(n=100)
    info = {
        "backend": "fake",
        "tried": [["openvino-gpu", "ok: mean cosine 0.9990"]],
        "model": "m",
    }
    monkeypatch.setattr(cli, "load_problem", lambda: (prob, None, info))
    out = run("embed").output
    assert "fake" in out and "openvino-gpu" in out and "C = 10.0" in out


def test_format_report_handles_a_real_small_report(tmp_path):
    report = pipeline.run_all(
        make_problem(n=300), tmp_path, tiny_config(), n_jobs=1, meta={"backend": "fake"}
    )
    text = cli.format_report(report)
    for name in ("random", "margin", "badge", "cluster-margin", "qbc", "k-center"):
        assert name in text
    assert "ceiling" in text and "stopping rule" in text and "batch size 5" in text


def test_suggesting_from_a_fully_labeled_project_says_so_instead_of_crashing(project):
    from project import Project

    p = Project(project)
    p.submit({i: INTENTS[i % 3] for i in range(60)})
    for name in ("margin", "k-center", "cluster-margin"):
        r = run("suggest", project, "--strategy", name)
        assert r.exit_code == 0 and "every item is labeled" in r.output, (
            name,
            r.output,
        )
