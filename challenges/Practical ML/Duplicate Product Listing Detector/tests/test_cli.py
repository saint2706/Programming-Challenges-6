"""Tests for cli.py via Typer's CliRunner -- fake encoders, generated catalog, no network."""

from pathlib import Path

import polars as pl
import pytest
from duplicate_listings import cli, data, pipeline
from helpers import FakeImage, FakeText, make_catalog
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch):
    make_catalog(tmp_path, n_groups=14)
    monkeypatch.setattr(
        data, "fetch_catalog", lambda data_dir=tmp_path: data_dir / "train.csv"
    )
    monkeypatch.setattr(data, "fetch_images", lambda names, data_dir=tmp_path: [])
    monkeypatch.setattr(pipeline, "default_encoders", lambda: (FakeText(), FakeImage()))
    return tmp_path


def invoke(workdir: Path, *args: str):
    return runner.invoke(cli.app, [args[0], "--data-dir", str(workdir), *args[1:]])


def test_full_flow_fetch_embed_evaluate_dedupe(workdir: Path):
    r = invoke(workdir, "fetch", "--groups", "12", "--seed", "0")
    assert r.exit_code == 0, r.output
    assert (workdir / "slice.csv").exists()

    r = invoke(workdir, "embed")
    assert r.exit_code == 0, r.output
    assert (workdir / "embeddings.npz").exists()

    r = invoke(workdir, "evaluate", "--k", "5")
    assert r.exit_code == 0, r.output
    assert "fused" in r.output and "candidate recall" in r.output.lower()
    assert (workdir / "report.json").exists()

    out = workdir / "dupes.csv"
    r = invoke(workdir, "dedupe", "--split", "test", "--out", str(out))
    assert r.exit_code == 0, r.output
    table = pl.read_csv(out)
    assert {"cluster_id", "posting_id", "title", "image"} <= set(table.columns)
    assert table.height > 0


def test_fetch_rejects_non_positive_group_count(workdir: Path):
    r = invoke(workdir, "fetch", "--groups", "0")
    assert r.exit_code != 0


def test_embed_before_fetch_explains_what_to_run(workdir: Path):
    r = invoke(workdir, "embed")
    assert r.exit_code != 0
    assert "fetch" in r.output


def test_dedupe_before_evaluate_explains_what_to_run(workdir: Path):
    invoke(workdir, "fetch", "--groups", "12")
    invoke(workdir, "embed")
    r = invoke(workdir, "dedupe", "--split", "test")
    assert r.exit_code != 0
    assert "evaluate" in r.output


def test_dedupe_rejects_unknown_split(workdir: Path):
    invoke(workdir, "fetch", "--groups", "12")
    invoke(workdir, "embed")
    invoke(workdir, "evaluate", "--k", "5")
    r = invoke(workdir, "dedupe", "--split", "train")
    assert r.exit_code != 0
