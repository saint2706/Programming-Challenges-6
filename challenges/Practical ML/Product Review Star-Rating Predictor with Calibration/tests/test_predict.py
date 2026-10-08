import numpy as np
import pytest
from helpers import fake_choice, make_features, tiny_config
from review_stars import config, pipeline, predict


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    report = pipeline.run_all(make_features(seed=0), out, tiny_config())
    return report, out


@pytest.fixture(scope="module")
def bundle(run):
    return predict.load_bundle(run[1])


@pytest.fixture(scope="module")
def encoder():
    return fake_choice(dim=12).encoder  # the synthetic features are 12-d


def test_the_bundle_uses_the_seed_ensembles_their_temperature_and_their_conformal_threshold(
    run, bundle
):
    report, _ = run
    assert set(bundle.members) == {"classification", "regression", "ordinal"}
    assert all(
        len(m) == 2 for m in bundle.members.values()
    )  # tiny_config trains 2 seeds
    T = report["stages"]["calibrate"]["models"]["ens-classification"]["T"]
    assert bundle.calibrators["classification"].T == pytest.approx(T)
    q = report["stages"]["conformal"]["models"]["ens-ordinal"]["temperature"]["aps"][
        "qhat"
    ]
    assert bundle.qhat["ordinal"] == pytest.approx(q)
    assert bundle.alpha == 0.1 and bundle.text_mode == "title_text"


def test_one_review_gives_a_calibrated_distribution_a_set_and_an_abstain_flag_per_framing(
    bundle, encoder
):
    out = predict.predict_review(
        bundle, encoder, "great item works perfect", threshold=0.5
    )
    assert set(out["framings"]) == {"classification", "regression", "ordinal"}
    for r in out["framings"].values():
        assert len(r["probs"]) == 5 and sum(r["probs"]) == pytest.approx(1.0)
        assert 1 <= r["star"] <= 5 and r["confidence"] == pytest.approx(max(r["probs"]))
        assert r["star"] == int(np.argmax(r["probs"])) + 1
        assert r["star"] in r["set"] and r["set"] == sorted(r["set"])
        assert 1.0 <= r["expected"] <= 5.0 and r["abstain"] == (r["confidence"] < 0.5)
    assert (
        out["n_tokens"] >= 3
        and out["truncated"] is False
        and out["text"] == "great item works perfect"
    )


def test_the_same_review_always_gets_the_same_answer_including_its_random_tie_break(
    bundle, encoder
):
    a = predict.predict_review(bundle, encoder, "item broken refund useless")
    b = predict.predict_review(bundle, encoder, "item broken refund useless")
    assert a == b
    assert (
        predict.review_draw("a") == predict.review_draw("a") != predict.review_draw("b")
    )
    assert 0.0 <= predict.review_draw("anything") < 1.0


def test_the_threshold_decides_abstention(bundle, encoder):
    text = "the item"
    low = predict.predict_review(bundle, encoder, text, threshold=0.0)
    high = predict.predict_review(bundle, encoder, text, threshold=1.01)
    assert not any(r["abstain"] for r in low["framings"].values())
    assert all(r["abstain"] for r in high["framings"].values())


def test_the_calibrated_probabilities_differ_from_the_raw_ones_by_the_fitted_temperature(
    bundle, encoder
):
    out = predict.predict_review(bundle, encoder, "perfect happy love it")
    r = out["framings"]["classification"]
    assert r["T"] == pytest.approx(bundle.calibrators["classification"].T)
    assert not np.allclose(r["probs"], r["raw_probs"]) or r["T"] == pytest.approx(
        1.0, abs=1e-3
    )


@pytest.mark.parametrize(
    "title,text",
    [("", ""), ("   ", "  \n "), ("Five Stars", ""), (None, None), ("", "<br />")],
)
def test_an_empty_review_is_refused_with_a_clear_message(bundle, encoder, title, text):
    with pytest.raises(ValueError, match="empty"):
        predict.predict_review(bundle, encoder, text, title=title)


def test_a_title_only_review_is_scored_on_its_title(bundle, encoder):
    out = predict.predict_review(bundle, encoder, "", title="Works great")
    assert out["text"] == "Works great" and set(out["framings"]) == {
        "classification",
        "regression",
        "ordinal",
    }


def test_one_word_emoji_and_non_english_reviews_work(bundle, encoder):
    for text in (
        "Great",
        "\U0001f44d\U0001f44d\U0001f44d",
        "非常好",
        "Très bien, merci",
    ):
        out = predict.predict_review(bundle, encoder, text)
        for r in out["framings"].values():
            assert np.isfinite(r["probs"]).all() and sum(r["probs"]) == pytest.approx(
                1.0
            )


def test_a_review_longer_than_512_tokens_is_truncated_and_says_so(bundle, encoder):
    out = predict.predict_review(bundle, encoder, "word " * 5000)
    assert out["truncated"] is True and out["n_tokens"] == 512
    assert all(np.isfinite(r["probs"]).all() for r in out["framings"].values())


def test_the_auto_generated_title_is_blanked_like_in_training(bundle, encoder):
    a = predict.predict_review(bundle, encoder, "great item", title="Five Stars")
    b = predict.predict_review(bundle, encoder, "great item")
    assert a["text"] == b["text"] == "great item" and a["framings"] == b["framings"]


def test_loading_from_a_folder_without_results_says_how_to_make_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="review-stars benchmark"):
        predict.load_bundle(tmp_path)


def test_a_bundle_missing_a_head_file_is_a_clear_error_not_a_crash(run, tmp_path):
    import shutil

    shutil.copytree(run[1], tmp_path / "r")
    (tmp_path / "r" / "models" / "mlp-ordinal-s1.pt").unlink()
    with pytest.raises(FileNotFoundError, match="mlp-ordinal-s1.pt"):
        predict.load_bundle(tmp_path / "r")


def test_the_default_results_folder_follows_the_environment(run, monkeypatch):
    monkeypatch.setenv("REVIEW_STARS_HOME", str(run[1].parent))
    assert config.results_dir() == run[1].parent / "results"
    with pytest.raises(FileNotFoundError):
        predict.load_bundle()
