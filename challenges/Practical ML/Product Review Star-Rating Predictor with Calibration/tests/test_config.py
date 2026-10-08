from pathlib import Path

import pytest
from review_stars import config
from review_stars.config import Config


def test_project_root_honours_the_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv("REVIEW_STARS_HOME", str(tmp_path))
    assert config.project_root() == tmp_path
    assert config.data_dir() == tmp_path / "data"
    assert config.results_dir() == tmp_path / "results"


def test_project_root_defaults_to_the_challenge_folder_not_the_cwd(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("REVIEW_STARS_HOME")
    monkeypatch.chdir(tmp_path)
    root = config.project_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "review_stars" / "config.py").exists()
    assert root != Path.cwd()


def test_defaults_match_the_spec():
    c = Config()
    assert (c.in_domain, c.out_domain) == ("Appliances", "Software")
    assert (c.test_months, c.cal_months, c.val_months) == (12, 6, 6)
    assert (c.n_train, c.n_val, c.n_cal, c.n_test, c.n_ood) == (
        150_000,
        20_000,
        20_000,
        30_000,
        30_000,
    )
    assert c.n_boot == 1000 and c.alpha == 0.1 and c.ece_bins == 15
    assert c.mlp_seeds == 5 and c.recal_draws == 20
    assert c.recal_ns == (50, 100, 250, 500, 1000, 2500, 5000)
    assert c.text_mode == "title_text"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text_mode": "bogus"},
        {"n_train": 0},
        {"test_months": 0},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"mlp_seeds": 0},
        {"n_boot": 0},
        {"ece_bins": 1},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_to_dict_is_json_ready_and_leaves_out_run_only_settings():
    import json

    d = Config().to_dict()
    json.dumps(d)
    assert "n_jobs" not in d and d["l2_grid"] == [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    assert Config(n_jobs=8).to_dict() == d
