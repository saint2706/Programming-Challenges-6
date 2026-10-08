"""Tests for evaluate.py -- synthetic clustered embeddings with a known right answer."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from duplicate_listings import embed, evaluate


def synthetic(
    n_groups: int, seed: int, text_noise: float, image_noise: float, dim: int = 16
):
    """Listings in duplicate groups: each modality = group centre + noise."""
    rng = np.random.default_rng(seed)
    rows, text, image = [], [], []
    n = 0
    for g in range(n_groups):
        tc, ic = rng.normal(size=dim), rng.normal(size=dim)
        for _ in range(int(rng.integers(2, 5))):
            rows.append(
                {
                    "posting_id": f"s{seed}_{n}",
                    "title": f"t{n}",
                    "image": f"{n}.jpg",
                    "label_group": 1000 * seed + g,
                }
            )
            text.append(tc + rng.normal(scale=text_noise, size=dim))
            image.append(ic + rng.normal(scale=image_noise, size=dim))
            n += 1
    df = pl.DataFrame(rows)
    emb = embed.Embeddings(
        df["posting_id"].to_list(),
        embed.l2_normalize(np.array(text, dtype=np.float32)),
        embed.l2_normalize(np.array(image, dtype=np.float32)),
        [],
    )
    return df, emb


def build(tmp_path: Path, name: str, df, emb, k: int = 10):
    return evaluate.prepare_split(df, emb, tmp_path / "db", name, k=k)


def test_prepare_split_labels_pairs_by_group(tmp_path: Path):
    df, emb = synthetic(6, seed=1, text_noise=0.1, image_noise=0.1)
    split = build(tmp_path, "val", df, emb)
    groups = df["label_group"].to_numpy()
    expected = (groups[split.pairs[:, 0]] == groups[split.pairs[:, 1]]).astype(int)
    np.testing.assert_array_equal(split.labels, expected)
    assert split.n_true == sum(
        c * (c - 1) // 2 for c in df.group_by("label_group").len()["len"].to_list()
    )


def test_clean_clusters_give_near_perfect_scores(tmp_path: Path):
    vdf, vemb = synthetic(20, seed=1, text_noise=0.05, image_noise=0.05)
    tdf, temb = synthetic(20, seed=2, text_noise=0.05, image_noise=0.05)
    report = evaluate.evaluate(
        build(tmp_path, "val", vdf, vemb), build(tmp_path, "test", tdf, temb)
    )
    for mode in ("text", "image", "fused"):
        assert report.modes[mode].f1 > 0.95
        assert report.modes[mode].cluster_f1 > 0.95
    assert report.candidate_recall == pytest.approx(1.0)


def test_fused_beats_either_modality_when_each_is_half_noisy(tmp_path: Path):
    # Text and image are each individually unreliable; averaging two independent
    # noisy views is the reason to fuse at all, so fused must win.
    vdf, vemb = synthetic(60, seed=3, text_noise=0.9, image_noise=0.9, dim=32)
    tdf, temb = synthetic(60, seed=4, text_noise=0.9, image_noise=0.9, dim=32)
    report = evaluate.evaluate(
        build(tmp_path, "val", vdf, vemb, k=20),
        build(tmp_path, "test", tdf, temb, k=20),
    )
    f = report.modes
    assert f["fused"].f1 >= max(f["text"].f1, f["image"].f1)
    assert f["fused"].f1 > min(f["text"].f1, f["image"].f1)


def test_fused_weight_follows_the_informative_modality(tmp_path: Path):
    vdf, vemb = synthetic(40, seed=5, text_noise=0.05, image_noise=3.0)
    tdf, temb = synthetic(40, seed=6, text_noise=0.05, image_noise=3.0)
    report = evaluate.evaluate(
        build(tmp_path, "val", vdf, vemb), build(tmp_path, "test", tdf, temb)
    )
    assert report.modes["fused"].weight >= 0.7
    assert report.modes["text"].f1 > report.modes["image"].f1


def test_thresholds_are_tuned_on_validation_only(tmp_path: Path):
    """Test-split labels must not influence the operating point."""
    vdf, vemb = synthetic(30, seed=7, text_noise=0.4, image_noise=0.4)
    tdf, temb = synthetic(30, seed=8, text_noise=0.4, image_noise=0.4)
    val = build(tmp_path, "val", vdf, vemb)
    test = build(tmp_path, "test", tdf, temb)
    a = evaluate.evaluate(val, test)
    # Corrupt the test labels completely; the tuned operating points must not move.
    test.labels = 1 - test.labels
    b = evaluate.evaluate(val, test)
    for mode in ("text", "image", "fused"):
        assert a.modes[mode].threshold == b.modes[mode].threshold
        assert a.modes[mode].weight == b.modes[mode].weight


def test_flag_pairs_respects_weight_and_threshold(tmp_path: Path):
    df, emb = synthetic(10, seed=9, text_noise=0.1, image_noise=0.1)
    split = build(tmp_path, "val", df, emb)
    none = evaluate.flag_pairs(split, weight=0.5, threshold=2.0)
    everything = evaluate.flag_pairs(split, weight=0.5, threshold=-2.0)
    assert len(none) == 0
    assert len(everything) == len(split.pairs)


def test_duplicate_clusters_table_lists_only_multi_member_clusters(tmp_path: Path):
    df, emb = synthetic(8, seed=10, text_noise=0.05, image_noise=0.05)
    split = build(tmp_path, "val", df, emb)
    table = evaluate.duplicate_clusters(split, weight=0.5, threshold=0.5)
    assert set(table.columns) >= {"cluster_id", "posting_id", "title", "image"}
    sizes = table.group_by("cluster_id").len()["len"]
    assert (sizes >= 2).all()
    assert table["posting_id"].n_unique() == table.height


def test_pair_level_metrics_match_evaluate_at_same_operating_point(tmp_path: Path):
    vdf, vemb = synthetic(20, seed=11, text_noise=0.3, image_noise=0.3)
    tdf, temb = synthetic(20, seed=12, text_noise=0.3, image_noise=0.3)
    val, test = build(tmp_path, "val", vdf, vemb), build(tmp_path, "test", tdf, temb)
    report = evaluate.evaluate(val, test)
    m = report.modes["fused"]
    live = evaluate.pair_level(test, m.weight, m.threshold)
    assert live.precision == pytest.approx(m.precision)
    assert live.recall == pytest.approx(m.recall)
    assert live.f1 == pytest.approx(m.f1)


def test_pair_level_counts_flagged_and_moves_with_threshold(tmp_path: Path):
    df, emb = synthetic(15, seed=13, text_noise=0.3, image_noise=0.3)
    split = build(tmp_path, "val", df, emb)
    loose = evaluate.pair_level(split, 0.5, -1.0)
    strict = evaluate.pair_level(split, 0.5, 0.95)
    assert loose.n_flagged == len(split.pairs)
    assert strict.n_flagged < loose.n_flagged
    assert loose.recall >= strict.recall


def test_review_pairs_sorted_limited_and_labeled(tmp_path: Path):
    df, emb = synthetic(10, seed=14, text_noise=0.3, image_noise=0.3)
    split = build(tmp_path, "val", df, emb)
    table = evaluate.review_pairs(split, weight=0.5, threshold=0.6, limit=7)
    assert table.height == 7
    assert table["fused"].to_list() == sorted(table["fused"].to_list(), reverse=True)
    assert set(table.columns) >= {
        "title_a",
        "title_b",
        "image_a",
        "image_b",
        "text",
        "image",
        "fused",
        "flagged",
        "is_duplicate",
    }
    assert all(table["flagged"] == (table["fused"] >= 0.6))


def test_review_pairs_borderline_band_keeps_only_near_threshold(tmp_path: Path):
    df, emb = synthetic(12, seed=15, text_noise=0.4, image_noise=0.4)
    split = build(tmp_path, "val", df, emb)
    table = evaluate.review_pairs(split, weight=0.5, threshold=0.5, band=0.1, limit=500)
    assert table.height > 0
    assert ((table["fused"] - 0.5).abs() <= 0.1 + 1e-9).all()
