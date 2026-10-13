import pytest
from helpers import make_pdf, page_texts
from pdftoolkit import jobs
from pdftoolkit.cli import main
from pdftoolkit.errors import PdfToolError
from pypdf import PdfReader


@pytest.fixture
def workspace(tmp_path):
    for name, n in (("a", 3), ("b", 2), ("c", 4)):
        make_pdf(tmp_path / "in" / f"{name}.pdf", n)
    return tmp_path


def write_job(tmp_path, text, name="jobs.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_batch_runs_every_job_relative_to_the_file(workspace, capsys):
    job = write_job(
        workspace,
        """
[[job]]
name = "book"
op = "merge"
inputs = ["in/a.pdf", "in/b.pdf[1]"]
output = "out/book.pdf"

[[job]]
op = "watermark"
inputs = ["in/*.pdf"]
output_dir = "out/marked"
text = "DRAFT"

[[job]]
op = "split"
inputs = ["in/c.pdf"]
output_dir = "out/c"
every = 2

[[job]]
op = "rotate"
inputs = ["in/a.pdf"]
output = "out/a-turned.pdf"
degrees = 90
pages = "1"
""",
    )
    elsewhere = workspace / "elsewhere"
    elsewhere.mkdir()
    # Paths must resolve against the job file, not the process's working directory.
    import os

    cwd = os.getcwd()
    os.chdir(elsewhere)
    try:
        assert main(["batch", str(job)]) == 0
    finally:
        os.chdir(cwd)
    out = workspace / "out"
    assert page_texts(out / "book.pdf") == ["a 1", "a 2", "a 3", "b 1"]
    assert sorted(p.name for p in (out / "marked").iterdir()) == [
        "a_watermarked.pdf",
        "b_watermarked.pdf",
        "c_watermarked.pdf",
    ]
    assert sorted(p.name for p in (out / "c").iterdir()) == ["c_p1-2.pdf", "c_p3-4.pdf"]
    assert PdfReader(out / "a-turned.pdf").pages[0].rotation == 90
    printed = capsys.readouterr().out
    assert "[1] book" in printed and "[2] watermark" in printed


def test_failed_job_does_not_stop_the_rest_and_sets_status(workspace, capsys):
    job = write_job(
        workspace,
        """
[[job]]
op = "extract"
inputs = ["in/missing.pdf"]
pages = "1"

[[job]]
op = "extract"
inputs = ["in/a.pdf"]
output = "out/a1.pdf"
pages = "1"
""",
    )
    assert main(["batch", str(job)]) == 1
    assert (workspace / "out" / "a1.pdf").exists()
    assert "[1] extract: FAILED - no such file" in capsys.readouterr().err


def test_fail_fast_stops_at_first_failure(workspace):
    job = write_job(
        workspace,
        """
[[job]]
op = "extract"
inputs = ["in/nope.pdf"]
pages = "1"

[[job]]
op = "extract"
inputs = ["in/a.pdf"]
output = "out/a1.pdf"
pages = "1"
""",
    )
    assert main(["batch", str(job), "--fail-fast"]) == 1
    assert not (workspace / "out").exists()


def test_dry_run_writes_nothing(workspace, capsys):
    job = write_job(
        workspace,
        """
[[job]]
op = "merge"
inputs = ["in/a.pdf", "in/b.pdf"]
output = "out/m.pdf"

[[job]]
op = "split"
inputs = ["in/a.pdf"]
output_dir = "out/s"
each = true
""",
    )
    assert main(["batch", str(job), "--dry-run"]) == 0
    assert not (workspace / "out").exists()
    assert "(dry run)" in capsys.readouterr().out


def test_dry_run_still_rejects_bad_plans(workspace, capsys):
    job = write_job(
        workspace,
        """
[[job]]
op = "extract"
inputs = ["in/a.pdf"]
output = "in/a.pdf"
pages = "1"
""",
    )
    assert main(["batch", str(job), "--dry-run"]) == 1
    assert "also an input" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('op = "extract"\ninputs = ["in/a.pdf"]\npagez = "1"', "unknown option"),
        ('op = "explode"\ninputs = ["in/a.pdf"]', "unknown op"),
        ('op = "extract"', "needs op and inputs"),
        ('op = "extract"\ninputs = "in/a.pdf"', "needs op and inputs"),
        ('op = "rotate"\ninputs = ["in/a.pdf"]\ndegrees = "ninety"', "invalid option"),
        ('op = "merge"\ninputs = ["in/a.pdf"]', "needs an output file"),
        (
            'op = "split"\ninputs = ["in/a.pdf"]\noutput = "x.pdf"\nevery = 1',
            "output directory",
        ),
    ],
)
def test_bad_jobs_are_reported_not_crashed(workspace, capsys, body, message):
    job = write_job(workspace, f"[[job]]\n{body}\n")
    assert main(["batch", str(job)]) == 1
    assert message in capsys.readouterr().err


def test_bad_batch_files(workspace, capsys):
    assert main(["batch", str(workspace / "missing.toml")]) == 1
    assert main(["batch", str(write_job(workspace, "this is = not toml ["))]) == 1
    assert (
        main(
            [
                "batch",
                str(write_job(workspace, "[defaults]\nforce = true", "empty.toml")),
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "cannot read batch file" in err and "no [[job]]" in err


def test_secrets_come_from_the_environment_not_the_file(workspace, monkeypatch):
    job = write_job(
        workspace,
        """
[[job]]
op = "encrypt"
inputs = ["in/a.pdf"]
output = "out/a.enc.pdf"
user_password_env = "BATCH_PW"
allow_copy = false
""",
    )
    monkeypatch.delenv("BATCH_PW", raising=False)
    assert main(["batch", str(job)]) == 1  # variable missing -> clear failure, no file
    assert not (workspace / "out").exists()
    monkeypatch.setenv("BATCH_PW", "hunter2")
    assert main(["batch", str(job)]) == 0
    assert PdfReader(workspace / "out" / "a.enc.pdf").decrypt("hunter2")
    assert "hunter2" not in job.read_text()


def test_batch_decrypts_with_password_env(workspace, monkeypatch):
    monkeypatch.setenv("PW", "pw")
    jobs_file = write_job(
        workspace,
        """
[[job]]
op = "encrypt"
inputs = ["in/a.pdf"]
output = "out/e.pdf"
user_password_env = "PW"

[[job]]
op = "decrypt"
inputs = ["out/e.pdf"]
output = "out/d.pdf"
password_env = "PW"
""",
    )
    assert main(["batch", str(jobs_file)]) == 0
    assert page_texts(workspace / "out" / "d.pdf") == ["a 1", "a 2", "a 3"]


def test_run_batch_yields_per_job_outcomes(workspace):
    job = write_job(
        workspace,
        '[[job]]\nop = "extract"\ninputs = ["in/a.pdf"]\npages = "1"\n'
        '[[job]]\nop = "extract"\ninputs = ["in/zzz.pdf"]\npages = "1"\n',
    )
    outcomes = list(jobs.run_batch(job))
    assert [type(o[2]) for o in outcomes] == [list, PdfToolError]


def test_batch_watermark_image_path_resolves_against_job_file(workspace):
    from PIL import Image

    Image.new("RGBA", (40, 20), (255, 0, 0, 255)).save(workspace / "logo.png")
    job = write_job(
        workspace,
        """
[[job]]
op = "watermark"
inputs = ["in/a.pdf"]
output = "out/logo.pdf"
image = "logo.png"
position = "bottom-right"
scale = 0.2
""",
    )
    assert main(["batch", str(job)]) == 0
    assert (workspace / "out" / "logo.pdf").exists()
