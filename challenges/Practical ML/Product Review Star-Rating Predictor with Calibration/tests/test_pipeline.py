import json
import shutil
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from helpers import fake_choice, make_features, tiny_config
from review_stars import calibrate, data, metrics, pipeline
from review_stars.config import Config
from review_stars.pipeline import STAGES, run_all

CFG = tiny_config()


@pytest.fixture(scope="module")
def features():
    return make_features(seed=0)


@pytest.fixture(scope="module")
def finished(features, tmp_path_factory):
    """One full run, shared by the read-only tests."""
    out = tmp_path_factory.mktemp("results")
    return run_all(features, out, CFG), out


@pytest.fixture
def results_copy(finished, tmp_path):
    """A private copy of the finished run: cache tests start from 'everything is computed'."""
    dest = tmp_path / "results"
    shutil.copytree(finished[1], dest)
    return dest


def strict_load(path):
    def refuse(const):
        raise ValueError(f"non-strict JSON constant {const}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


# ---------------------------------------------------------------- what a run produces


def test_a_full_run_computes_every_stage_for_every_model_and_writes_strict_json(
    finished,
):
    report, out = finished
    assert report["ran"] == list(STAGES) and report["missing"] == []
    assert report["models"] == [
        "tfidf-classification", "tfidf-regression",
        "linear-classification", "linear-regression", "linear-ordinal",
        "mlp-classification", "mlp-regression", "mlp-ordinal",
        "ens-classification", "ens-regression", "ens-ordinal",
    ]  # fmt: skip
    on_disk = strict_load(out / "report.json")
    assert set(on_disk["stages"]) == set(STAGES)
    assert (
        on_disk["data"]["sizes"]["train"] == 700
        and len(on_disk["data"]["fingerprint"]) == 16
    )
    assert (out / "models" / "linear-ordinal.pt").exists()
    assert len(list((out / "models").glob("mlp-classification-s*.pt"))) == CFG.mlp_seeds


def test_evaluation_scores_every_model_and_calibrator_with_intervals(finished):
    report, _ = finished
    test = report["stages"]["evaluate"]["splits"]["test"]
    keys = set(test["rows"])
    assert "linear-classification|none" in keys and "ens-ordinal|isotonic" in keys
    assert (
        "linear-classification|vector" in keys
        and "linear-regression|vector" not in keys
    )
    for row in test["rows"].values():
        for name in metrics.BUNDLE_KEYS:
            ci = row["ci"][name]
            assert row["point"][name] is not None or name == "qwk"
            if ci["lo"] is not None:
                assert ci["lo"] <= ci["hi"]
        assert len(row["coverage"]) == len(CFG.levels)
        assert sum(row["reliability"]["n"]) == test["n"]
    assert test["slices"]["linear-classification|none"]["all"]["n"] == test["n"]
    assert set(test["intervals"]) == {
        "tfidf-regression",
        "linear-regression",
        "mlp-regression",
    }


def test_paired_differences_are_reported_for_calibrators_framings_and_capacities(
    finished,
):
    report, _ = finished
    labels = {
        d["label"] for d in report["stages"]["evaluate"]["splits"]["test"]["diffs"]
    }
    assert "linear-classification: temperature - none" in labels
    assert "linear: classification - regression [temperature]" in labels
    assert "classification: mlp - linear [temperature]" in labels
    assert "classification: ens - mlp [temperature]" in labels


def test_temperature_scaling_improves_the_calibration_of_a_model_that_needed_it(
    finished,
):
    report, _ = finished
    rows = report["stages"]["evaluate"]["splits"]["ood_test"]["rows"]
    improved = [
        m
        for m in report["models"]
        if rows[f"{m}|temperature"]["point"]["nll"]
        <= rows[f"{m}|none"]["point"]["nll"] + 0.02
    ]
    assert (
        len(improved) >= 8
    )  # temperature is fit on the calibration split, not on this one


def test_conformal_shift_and_selective_stages_report_their_payloads(finished):
    report, _ = finished
    conf = report["stages"]["conformal"]["models"]["ens-classification"]["temperature"]
    assert set(conf) == {"aps", "aps_det"}
    assert conf["aps_det"]["test"]["mean_size"] >= conf["aps"]["test"]["mean_size"]
    assert "interval" in report["stages"]["conformal"]["models"]["linear-regression"]
    assert "interval" not in report["stages"]["conformal"]["models"]["linear-ordinal"]
    shift = report["stages"]["shift"]["models"]["mlp-ordinal"]
    assert [r["n"] for r in shift["rows"]] == [30, 80] and shift["skipped"] == []
    sel = report["stages"]["selective"]["models"]["linear-classification"][
        "temperature"
    ]["test"]
    assert set(sel) == {"summary", "transfer", "fixed"} and len(sel["fixed"]) == len(
        pipeline.FIXED_TAUS
    )


# ---------------------------------------------------------------- the stage cache


def test_a_second_identical_run_recomputes_nothing(features, results_copy):
    again = run_all(features, results_copy, CFG)
    assert again["ran"] == [] and again["missing"] == []


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"alpha": 0.2}, ["conformal"]),
        ({"levels": (0.5, 0.9)}, ["evaluate"]),
        ({"recal_draws": 5}, ["shift"]),
        ({"ablation_train": 200}, ["ablation"]),
        ({"n_boot": 20}, ["evaluate", "conformal"]),
        ({"n_jobs": 2}, []),  # how the work is run is never part of a result key
        ({"mlp_hidden": 16}, list(STAGES)),
        ({"l2_grid": (1e-2,)}, list(STAGES)),
    ],
)
def test_changing_a_setting_recomputes_exactly_what_depends_on_it(
    features, results_copy, change, expected
):
    assert run_all(features, results_copy, replace(CFG, **change))["ran"] == expected


def test_changing_the_data_recomputes_everything(features, results_copy):
    changed = make_features(seed=0)
    changed.splits["test"].X[3, 2] += 0.01  # one embedding value in one split
    assert changed.fingerprint() != features.fingerprint()
    assert run_all(changed, results_copy, CFG)["ran"] == list(STAGES)


def test_the_fingerprint_sees_texts_groups_and_labels_too():
    base = make_features(seed=1)
    fp = base.fingerprint()
    assert fp == make_features(seed=1).fingerprint()
    for edit in (
        lambda f: f.splits["train"].texts.__setitem__(0, "different words"),
        lambda f: f.splits["cal"].groups.__setitem__(1, "B-other"),
        lambda f: f.splits["ood_test"].y.__setitem__(
            2, (f.splits["ood_test"].y[2] % 5) + 1
        ),
    ):
        other = make_features(seed=1)
        edit(other)
        assert other.fingerprint() != fp


def test_asking_for_one_stage_runs_its_dependencies_but_never_drops_finished_stages(
    features, tmp_path
):
    first = run_all(features, tmp_path, CFG, stages=("conformal",))
    assert first["ran"] == ["train", "calibrate", "conformal"]
    assert first["missing"] == ["evaluate", "shift", "selective", "ablation"]
    second = run_all(features, tmp_path, CFG, stages=("evaluate",))
    assert second["ran"] == ["evaluate"]
    assert "conformal" in second["stages"] and "evaluate" in second["stages"]
    assert second["missing"] == ["shift", "selective", "ablation"]


def test_fresh_recomputes_only_the_requested_stages(features, results_copy):
    assert run_all(features, results_copy, CFG, stages=("conformal",), fresh=True)[
        "ran"
    ] == ["conformal"]


def test_unknown_stage_is_an_error(features, tmp_path):
    with pytest.raises(ValueError, match="unknown stage"):
        run_all(features, tmp_path, CFG, stages=("train", "nope"))


def test_a_torn_stage_file_is_recomputed_not_trusted(features, tmp_path):
    run_all(features, tmp_path, CFG)
    (tmp_path / "stages" / "evaluate.json").write_text(
        '{"key": "x", "payl', encoding="utf-8"
    )
    (tmp_path / "stages" / "train.npz").write_bytes(b"PK\x03\x04 torn")
    again = run_all(features, tmp_path, CFG)
    assert again["ran"] == ["train", "evaluate"] and again["missing"] == []


def test_a_stage_computed_under_an_older_version_is_not_reused(
    features, results_copy, monkeypatch
):
    monkeypatch.setattr(pipeline, "STAGE_VERSION", pipeline.STAGE_VERSION + 1)
    assert run_all(features, results_copy, CFG)["ran"] == list(STAGES)


def test_progress_messages_are_reported(features, tmp_path):
    seen = []
    run_all(features, tmp_path, CFG, stages=("train",), progress=seen.append)
    assert "== stage train ==" in seen and any(
        "linear classification" in m for m in seen
    )


# ---------------------------------------------------------------- the leakage guarantees


def test_calibrators_are_fit_on_the_calibration_split_and_nothing_else(
    features, tmp_path, monkeypatch
):
    seen = []
    real = calibrate.fit_all

    def spy(preds, y, *a, **k):
        seen.append((len(preds), np.asarray(y).copy()))
        return real(preds, y, *a, **k)

    monkeypatch.setattr(calibrate, "fit_all", spy)
    run_all(features, tmp_path, CFG, stages=("calibrate",))
    cal_y = features.splits["cal"].y
    assert len(seen) == len(pipeline.model_names(CFG))
    assert all(n == len(cal_y) and np.array_equal(y, cal_y) for n, y in seen)


def test_heads_see_only_train_and_validation_never_calibration_or_test(
    features, tmp_path, monkeypatch
):
    from review_stars import heads

    seen = []
    real = heads.fit_linear

    def spy(framing, X, y, Xv, yv, **kw):
        seen.append((len(X), len(Xv)))
        return real(framing, X, y, Xv, yv, **kw)

    monkeypatch.setattr(heads, "fit_linear", spy)
    run_all(features, tmp_path, CFG, stages=("train",))
    assert set(seen) == {
        (len(features.splits["train"].y), len(features.splits["val"].y))
    }


# ---------------------------------------------------------------- awkward splits


def test_a_split_where_every_review_has_the_same_star_still_completes(tmp_path):
    f = make_features(seed=2)
    f.splits["ood_test"].y[:] = 5
    report = run_all(
        f, tmp_path, CFG, stages=("train", "calibrate", "evaluate", "shift")
    )
    row = report["stages"]["evaluate"]["splits"]["ood_test"]["rows"][
        "linear-classification|none"
    ]
    # predictions vary while the truth is constant: kappa is exactly chance level (0), not undefined
    assert row["point"]["qwk"] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 <= row["point"]["acc"] <= 1.0
    strict_load(tmp_path / "report.json")


def test_a_calibration_split_missing_a_star_still_completes(tmp_path):
    f = make_features(seed=3)
    keep = f.splits["cal"].y != 1
    cal = f.splits["cal"]
    f.splits["cal"] = type(cal)(
        X=cal.X[keep], y=cal.y[keep], groups=cal.groups[keep],
        texts=[t for t, k in zip(cal.texts, keep, strict=True) if k],
        n_tok=cal.n_tok[keep], verified=cal.verified[keep],
        ids=[i for i, k in zip(cal.ids, keep, strict=True) if k],
    )  # fmt: skip
    report = run_all(f, tmp_path, CFG, stages=("evaluate", "selective"))
    summary = report["stages"]["calibrate"]["models"]
    # no one-star reviews to learn from: the isotonic map for star 1 is left alone and flagged
    assert all(summary[m]["degenerate_stars"] == [1] for m in summary)
    assert report["missing"] == ["conformal", "shift", "ablation"]
    rows = report["stages"]["evaluate"]["splits"]["test"]["rows"]
    assert all(np.isfinite(r["point"]["nll"]) for r in rows.values())


# ---------------------------------------------------------------- the input-text ablation


def test_the_ablation_shows_a_leak_that_exists_in_the_old_era_but_not_in_the_test_window(
    features, tmp_path
):
    F = features

    def provider(mode, subsets):
        out = {}
        for split, idx in subsets.items():
            X = F.splits[split].X[idx]
            leak = np.zeros((len(idx), 1), dtype=np.float32)
            if mode == "raw_title" and split == "train":
                leak[:, 0] = (
                    F.splits["train"].y[idx] - 3.0
                ) * 2.0  # the auto title: the rating itself
            out[split] = np.concatenate([X, leak], axis=1)
        return out

    report = run_all(F, tmp_path, CFG, stages=("ablation",), ablation_provider=provider)
    modes = report["stages"]["ablation"]["modes"]
    assert set(modes) == set(pipeline.ABLATION_MODES)
    honest, text_only, leaky = (
        modes["title_text"],
        modes["text_only"],
        modes["raw_title"],
    )
    for framing in ("classification", "regression", "ordinal"):
        # same era as the training rows: the leak makes the model look far better
        assert (
            leaky[framing]["holdout"]["acc"] > honest[framing]["holdout"]["acc"] + 0.1
        )
        # the time-split test window has no such titles: the shortcut is gone and it costs accuracy
        assert leaky[framing]["test"]["acc"] < leaky[framing]["holdout"]["acc"] - 0.1
        assert (
            abs(
                text_only[framing]["holdout"]["acc"] - honest[framing]["holdout"]["acc"]
            )
            < 0.08
        )
    assert report["stages"]["ablation"]["n_holdout"] == CFG.ablation_train // 10


def test_without_a_provider_the_other_input_settings_are_skipped_and_say_so(finished):
    modes = finished[0]["stages"]["ablation"]["modes"]
    assert "skipped" in modes["text_only"] and "skipped" in modes["raw_title"]
    assert "classification" in modes["title_text"]


# ---------------------------------------------------------------- helpers and loading


def test_clean_makes_nested_results_strict_json():
    out = pipeline.clean(
        {
            "a": np.float64("nan"),
            "b": [np.inf, np.int64(3), np.bool_(True)],
            "c": (1.5, np.array([1.0, np.nan])),
        }
    )
    assert out == {"a": None, "b": [None, 3, True], "c": [1.5, [1.0, None]]}


def prepared_dir(tmp_path, per_split=6):
    rows = []
    for split in data.SPLITS:
        for i in range(per_split):
            rows.append(
                {
                    "id": f"{split}:{i}", "cat": "in", "split": split, "row": i,
                    "rating": 1 + i % 5,
                    "title": "Five Stars" if i == 0 else ("" if i == 1 else f"Title {i}"),
                    "text": "" if i == 2 else f"review text number {i} about {split}",
                    "ts": 1_600_000_000_000 + i, "parent_asin": f"B{i % 3}",
                    "verified": i % 2 == 0, "helpful_vote": 0,
                }
            )  # fmt: skip
    out = tmp_path / "prepared"
    out.mkdir()
    pl.DataFrame(rows).write_parquet(out / "reviews.parquet")
    return tmp_path


def test_load_features_builds_the_model_text_embeds_once_and_keeps_every_split(
    tmp_path,
):
    root = prepared_dir(tmp_path)
    cfg = Config(text_mode="title_text")
    f = pipeline.load_features(cfg, root, fake_choice)
    assert set(f.splits) == set(data.SPLITS)
    train = f.splits["train"]
    assert train.X.shape == (6, 8) and train.y.tolist() == [1, 2, 3, 4, 5, 1]
    assert (
        train.texts[0] == "review text number 0 about train"
    )  # the auto title is blanked
    assert train.texts[1].startswith(
        "review text number 1"
    )  # an empty title adds nothing
    assert train.texts[2] == "Title 2"  # a title-only review keeps its real title
    assert np.allclose(np.linalg.norm(train.X, axis=1), 1.0, atol=1e-5)
    assert (
        f.meta["embedding"]["backend"] == "fake"
        and f.meta["embedding"]["shards"]["computed"] == 1
    )
    again = pipeline.load_features(cfg, root, fake_choice)
    assert again.meta["embedding"]["shards"] == {"total": 1, "reused": 1, "computed": 0}
    assert np.array_equal(again.splits["test"].X, f.splits["test"].X)


def test_the_text_mode_changes_what_is_embedded(tmp_path):
    root = prepared_dir(tmp_path)
    raw = pipeline.load_features(Config(text_mode="raw_title"), root, fake_choice)
    assert raw.splits["train"].texts[0].startswith("Five Stars. ")
    only = pipeline.load_features(Config(text_mode="text_only"), root, fake_choice)
    assert (
        only.splits["train"].texts[2] == ""
    )  # nothing left to embed, and that is allowed
    assert only.meta["empty_texts"] == len(data.SPLITS)
    assert np.isfinite(only.splits["train"].X).all()


def test_load_features_names_the_command_when_the_data_is_not_prepared(tmp_path):
    with pytest.raises(FileNotFoundError, match="review-stars prepare"):
        pipeline.load_features(Config(), tmp_path, fake_choice)
    root = prepared_dir(tmp_path)
    df = pl.read_parquet(root / "prepared" / "reviews.parquet").filter(
        pl.col("split") != "cal"
    )
    df.write_parquet(root / "prepared" / "reviews.parquet")
    with pytest.raises(ValueError, match="'cal'"):
        pipeline.load_features(Config(), root, fake_choice)


def test_the_ablation_provider_embeds_only_the_requested_rows_under_the_requested_mode(
    tmp_path,
):
    root = prepared_dir(tmp_path)
    provider = pipeline.make_ablation_provider(Config(), root, fake_choice)
    out = provider("raw_title", {"train": np.array([0, 3]), "test": np.array([1])})
    assert out["train"].shape == (2, 8) and out["test"].shape == (1, 8)
    main = pipeline.load_features(Config(text_mode="raw_title"), root, fake_choice)
    assert np.allclose(out["train"], main.splits["train"].X[[0, 3]])


def test_the_reliability_bands_contain_the_diagonal_for_a_calibrated_model_and_not_for_an_overconfident_one():
    from helpers import make_probs
    from review_stars import stats

    groups = np.repeat(np.arange(2000), 10)
    covered = {}
    for name, sharpen in (("calibrated", 1.0), ("overconfident", 2.5)):
        P, y = make_probs(n=20_000, sharpen=sharpen, seed=3)
        reps = stats.boot_metrics(
            {"m": P},
            y,
            groups,
            n_boot=200,
            seed=0,
            bins=10,
            curve_bins=pipeline.RELIABILITY_BINS,
        )["m"]
        conf, correct = metrics.top_label(P, y)
        band = pipeline.reliability_bands(
            conf, correct, reps["reliability_n"], reps["reliability_k"]
        )
        centers = (np.array(band["edges"][:-1]) + np.array(band["edges"][1:])) / 2
        populated = np.array(band["n"]) >= 300
        assert populated.sum() >= 3
        assert (np.array(band["n"]).sum() == len(y)) and (
            band["lo"][populated] <= band["hi"][populated]
        ).all()
        inside = (band["lo"][populated] <= centers[populated]) & (
            centers[populated] <= band["hi"][populated]
        )
        covered[name] = inside.mean()
    assert covered["calibrated"] > 0.6 and covered["overconfident"] < 0.4


def test_the_evaluation_stage_stores_a_banded_reliability_diagram_for_every_model_and_calibrator(
    finished,
):
    rows = finished[0]["stages"]["evaluate"]["splits"]["test"]["rows"]
    for row in rows.values():
        rel = row["reliability_fixed"]
        assert len(rel["edges"]) == pipeline.RELIABILITY_BINS + 1
        assert sum(rel["n"]) == finished[0]["stages"]["evaluate"]["splits"]["test"]["n"]
        for lo, acc, hi in zip(rel["lo"], rel["acc"], rel["hi"], strict=True):
            assert (lo is None) == (hi is None)
            if lo is not None and acc is not None:
                assert lo <= hi
