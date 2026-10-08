import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from helpers import fake_choice, make_features, month_ms, tiny_config, write_reviews_gz
from review_stars import cli, config, data, predict
from typer.testing import CliRunner

runner = CliRunner()
SMALL = [
    "--set", "n_train=300", "--set", "n_val=60", "--set", "n_cal=60", "--set", "n_test=100",
    "--set", "n_ood=60", "--set", "n_ood_pool=30", "--set", "min_month_rows=10",
]  # fmt: skip


def run(*args, **kw):
    return runner.invoke(cli.app, [str(a) for a in args], **kw)


def raw_files():
    """Tiny Appliances / Software archives where they belong: ``data/raw``."""
    rng = np.random.default_rng(0)
    for name in ("Appliances", "Software"):
        rows = []
        for y in range(2019, 2024):
            for m in range(1, 13):
                if (y, m) > (2023, 3):
                    break
                for j in range(30):
                    rows.append(
                        {
                            "rating": float(rng.integers(1, 6)),
                            "title": "Five Stars"
                            if (y <= 2020 and j % 3)
                            else "A title",
                            "text": f"{name} review {y}-{m}-{j}",
                            "timestamp": month_ms(y, m, 1 + j % 27),
                            "parent_asin": f"B{(y + m + j) % 9}",
                            "verified_purchase": True,
                            "helpful_vote": 0,
                        }
                    )
        path = data.raw_path(config.data_dir(), name)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_reviews_gz(path, rows)


# ---------------------------------------------------------------- the surface


def test_help_lists_every_command():
    out = run("--help").output
    for name in (
        "fetch",
        "prepare",
        "embed",
        "train",
        "calibrate",
        "benchmark",
        "report",
        "predict",
    ):
        assert name in out


def test_the_module_runs_as_python_dash_m_and_the_script_entry_point_exists():
    env = {**os.environ, "PYTHONUTF8": "1"}
    res = subprocess.run(
        [sys.executable, "-m", "review_stars", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=120,
    )
    assert res.returncode == 0 and "benchmark" in res.stdout
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert 'review-stars = "review_stars.cli:app"' in pyproject.read_text(
        encoding="utf-8"
    )


def test_overrides_are_parsed_by_the_field_type_and_validated():
    cfg = cli.apply_overrides(
        tiny_config(),
        [
            "n_boot=7",
            "alpha=0.2",
            "recal_ns=10,20,30",
            "text_mode=text_only",
            "l2_grid=1e-3,1e-2",
        ],
    )
    assert (cfg.n_boot, cfg.alpha, cfg.recal_ns, cfg.text_mode) == (
        7,
        0.2,
        (10, 20, 30),
        "text_only",
    )
    assert cfg.l2_grid == (1e-3, 1e-2)
    for bad in ("nope=1", "n_boot", "n_boot=zero", "text_mode=bogus", "alpha=2"):
        with pytest.raises(Exception, match="."):
            cli.apply_overrides(tiny_config(), [bad])


def test_a_bad_override_is_a_usage_error_not_a_traceback():
    res = run("benchmark", "--set", "nope=1")
    assert res.exit_code == 2 and "nope" in res.output and "Traceback" not in res.output


# ---------------------------------------------------------------- data commands


def test_fetch_reports_what_it_downloaded(monkeypatch, tmp_path):
    files = {"Appliances": tmp_path / "a.gz", "Software": tmp_path / "s.gz"}
    for p in files.values():
        p.write_bytes(b"x" * 10)
    monkeypatch.setattr(data, "fetch", lambda data_dir, cfg: files)
    out = run("fetch").output
    assert "Appliances" in out and "Software" in out


def test_prepare_without_the_raw_files_points_at_fetch():
    res = run("prepare")
    assert res.exit_code == 2 and "review-stars fetch" in res.output


def test_prepare_prints_the_month_table_the_windows_and_what_was_selected():
    raw_files()
    res = run("prepare", *SMALL)
    assert res.exit_code == 0, res.output
    for needle in ("2023-03", "test", "Appliances", "Software", "duplicates", "auto"):
        assert needle in res.output
    assert (config.data_dir() / "prepared" / "reviews.parquet").exists()
    assert json.loads(
        (config.data_dir() / "prepared" / "windows.json").read_text(encoding="utf-8")
    )["windows"]


def test_embed_reports_the_backend_and_resumes(monkeypatch):
    raw_files()
    assert run("prepare", *SMALL).exit_code == 0
    monkeypatch.setattr(cli, "get_choice", lambda: fake_choice())
    first = run("embed")
    assert first.exit_code == 0, first.output
    assert "fake" in first.output and "computed" in first.output
    second = run("embed")
    assert "reused" in second.output and second.exit_code == 0


def test_embed_without_prepared_data_points_at_prepare():
    res = run("embed")
    assert res.exit_code == 2 and "review-stars prepare" in res.output


# ---------------------------------------------------------------- stages, report, predict


@pytest.fixture
def synthetic(monkeypatch):
    features = make_features(seed=0)
    monkeypatch.setattr(cli, "load_features", lambda cfg, progress=None: features)
    monkeypatch.setattr(cli, "default_config", tiny_config)
    return features


def test_train_then_calibrate_then_benchmark_then_report(synthetic):
    assert "train" in run("train").output
    out = run("calibrate").output
    assert "calibrate" in out
    res = run("benchmark", "--stage", "evaluate", "--jobs", "1")
    assert res.exit_code == 0, res.output
    assert (
        "ens-classification" in res.output and "not computed" in res.output
    )  # conformal etc. still missing
    full = run("benchmark")
    assert full.exit_code == 0 and "ablation" in full.output
    rep = run("report")
    assert rep.exit_code == 0, rep.output
    for name in ("tables.md", "reliability.png", "framings.png"):
        assert (config.results_dir() / name).exists()


def test_benchmark_stages_are_cached_between_commands(synthetic):
    run("benchmark", "--stage", "train")
    again = run("benchmark", "--stage", "train")
    assert "computed this run: none" in again.output


def test_an_unknown_stage_is_a_usage_error(synthetic):
    res = run("benchmark", "--stage", "bogus")
    assert res.exit_code == 2 and "bogus" in res.output


def test_report_before_a_benchmark_says_what_to_run():
    res = run("report")
    assert res.exit_code == 2 and "review-stars benchmark" in res.output


def test_predict_prints_each_framing_with_its_set_and_abstain_decision(
    synthetic, monkeypatch
):
    assert run("benchmark").exit_code == 0
    monkeypatch.setattr(predict, "get_encoder", lambda: fake_choice(dim=12).encoder)
    res = run("predict", "great item works perfect", "--threshold", "0.4")
    assert res.exit_code == 0, res.output
    for needle in (
        "classification",
        "regression",
        "ordinal",
        "stars",
        "confidence",
        "set",
    ):
        assert needle in res.output
    titled = run("predict", "", "--title", "Works great")
    assert titled.exit_code == 0


def test_predict_refuses_an_empty_review_and_a_missing_benchmark(
    synthetic, monkeypatch
):
    monkeypatch.setattr(predict, "get_encoder", lambda: fake_choice(dim=12).encoder)
    res = run("predict", "anything")
    assert res.exit_code == 2 and "review-stars benchmark" in res.output
    assert run("benchmark").exit_code == 0
    empty = run("predict", "   ")
    assert empty.exit_code == 2 and "empty" in empty.output


def test_printing_a_review_never_crashes_on_a_stream_that_cannot_encode_it(monkeypatch):
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    cli._safe_output()
    print(
        "\u975e\u5e38\u597d", file=stream, flush=True
    )  # would raise UnicodeEncodeError before
    assert stream.buffer.getvalue().startswith(b"???")


def test_the_cli_entry_point_survives_a_non_utf8_output_stream_end_to_end():
    env = {**os.environ, "PYTHONIOENCODING": "cp1252:strict", "PYTHONUTF8": "0"}
    res = subprocess.run(
        [
            sys.executable,
            "-c",
            "from review_stars import cli; cli._safe_output(); print('\u975e')",
        ],
        capture_output=True,
        env=env,
        check=False,
        timeout=120,
    )
    assert res.returncode == 0 and res.stdout.strip() == b"?"
