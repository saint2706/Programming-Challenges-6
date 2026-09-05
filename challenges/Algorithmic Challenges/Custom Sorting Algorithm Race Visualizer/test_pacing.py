"""Tests for the caption pacing.

Pure arithmetic over the step stream, so none of this needs Manim installed --
which is the point of keeping `pacing.py` separate from `visualize.py`.

Run with:  uv run --with pytest pytest -q
"""

from __future__ import annotations

import pytest

from pacing import (
    CLOSING_HOLD,
    CPS_FIRST,
    CPS_REPEAT,
    MAX_HOLD,
    MIN_HOLD_FIRST,
    MIN_HOLD_REPEAT,
    MOTION_COMPARE,
    MOTION_PLAIN,
    MOTION_SWAP,
    MOTION_WRITE,
    TITLE_MIN_HOLD,
    CaptionPacer,
    caption_template,
    motion_time,
    novel_characters,
    reading_time,
    scene_duration,
)
from sorting_algorithms import ALGORITHMS, BASE_ARRAY, SUBTITLES, Step


# ---------------------------------------------------------------------------
# The two text measures
# ---------------------------------------------------------------------------


def test_caption_template_erases_numbers():
    assert caption_template("Compare adjacent pair 3, 4") == "Compare adjacent pair #, #"
    assert caption_template("Compare adjacent pair 3, 4") == caption_template(
        "Compare adjacent pair 12, 13"
    )
    assert caption_template("Sorted") == "Sorted"


def test_caption_template_distinguishes_different_sentences():
    assert caption_template("Swap 1, 2") != caption_template("Compare 1, 2")


def test_novel_characters_counts_only_the_middle():
    assert novel_characters("Compare adjacent pair 3, 4", "Compare adjacent pair 4, 5") == 4
    assert novel_characters("Swap 1, 2", "Swap 1, 2") == 0
    assert novel_characters(None, "anything") == len("anything")
    assert novel_characters("", "abc") == 3
    assert novel_characters("abc", "") == 0


def test_novel_characters_is_never_negative():
    for a in ("", "a", "abc", "abcabc", "xyz"):
        for b in ("", "a", "abc", "abcabc", "xyz"):
            assert 0 <= novel_characters(a, b) <= len(b)


def test_reading_time_is_clamped_at_both_ends():
    assert reading_time("", 0.5) == 0.5
    assert reading_time("x" * 5000, 0.5) == MAX_HOLD
    mid = "x" * 28
    assert reading_time(mid, 0.5) == pytest.approx(28 / CPS_FIRST)


# ---------------------------------------------------------------------------
# The hold budget
# ---------------------------------------------------------------------------


def test_first_sight_of_a_template_pays_the_full_read():
    pacer = CaptionPacer()
    caption = "Scan for minimum: compare 0 and 1"
    assert pacer.hold_for(caption) == pytest.approx(
        max(MIN_HOLD_FIRST, len(caption) / CPS_FIRST)
    )


def test_a_repeated_template_pays_only_for_what_changed():
    pacer = CaptionPacer()
    first = pacer.hold_for("Compare adjacent pair 3, 4")
    second = pacer.hold_for("Compare adjacent pair 4, 5")
    assert second < first
    assert second == pytest.approx(max(MIN_HOLD_REPEAT, 4 / CPS_REPEAT))


def test_an_identical_caption_costs_nothing():
    pacer = CaptionPacer()
    pacer.hold_for("Rotate cycle")
    assert pacer.hold_for("Rotate cycle") == 0.0
    assert pacer.hold_for("Rotate cycle") == 0.0


def test_returning_to_a_template_after_a_detour_is_still_a_repeat():
    pacer = CaptionPacer()
    pacer.hold_for("Compare adjacent pair 3, 4")
    pacer.hold_for("Index 9 finalized")
    back = pacer.hold_for("Compare adjacent pair 0, 1")
    assert back <= MIN_HOLD_FIRST


def test_every_hold_is_inside_its_bounds():
    pacer = CaptionPacer()
    captions = [
        "Sorted",
        "Compare adjacent pair 0, 1",
        "Compare adjacent pair 1, 2",
        "Compare adjacent pair 1, 2",
        "x" * 500,
        "y",
    ]
    for caption in captions:
        hold = pacer.hold_for(caption)
        assert 0.0 <= hold <= MAX_HOLD


def test_a_very_long_caption_is_capped():
    pacer = CaptionPacer()
    assert pacer.hold_for("z" * 1000) == MAX_HOLD


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------


def test_motion_time_ranks_swap_and_write_above_compare():
    plain = Step([1])
    assert motion_time(Step([1], swap=(0, 1))) == MOTION_SWAP
    assert motion_time(Step([1], write=(0,))) == MOTION_WRITE
    assert motion_time(Step([1], compare=(0, 1))) == MOTION_COMPARE
    assert motion_time(plain) == MOTION_PLAIN
    assert MOTION_COMPARE <= MOTION_PLAIN <= MOTION_SWAP


def test_a_swap_that_is_also_a_write_is_paced_as_a_swap():
    assert motion_time(Step([1], swap=(0, 1), write=(0,))) == MOTION_SWAP


# ---------------------------------------------------------------------------
# Whole scenes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(ALGORITHMS))
def test_every_scene_is_watchable_but_not_endless(key):
    steps = list(ALGORITHMS[key](list(BASE_ARRAY)))
    info = scene_duration(steps, key, SUBTITLES[key])
    assert 15 <= info["total"] <= 90, f"{key} runs {info['total']:.1f}s"
    # More than half of every scene is a still frame; that is the whole change.
    assert info["hold"] > info["motion"]


@pytest.mark.parametrize("key", list(ALGORITHMS))
def test_no_scene_averages_less_than_a_third_of_a_second_per_step(key):
    steps = list(ALGORITHMS[key](list(BASE_ARRAY)))
    info = scene_duration(steps, key, SUBTITLES[key])
    assert info["total"] / info["steps"] >= 1 / 3


def test_scene_duration_accounts_for_the_title_and_the_ending():
    empty = scene_duration([], "Bubble Sort", SUBTITLES["Bubble Sort"])
    assert empty["steps"] == 0
    assert empty["motion"] == 0.0
    assert empty["total"] == pytest.approx(
        reading_time("Bubble Sort " + SUBTITLES["Bubble Sort"], TITLE_MIN_HOLD)
        + CLOSING_HOLD
    )


def test_pacing_is_much_slower_than_the_original_fixed_run_times():
    """The original driver: 0.15-0.25 s per step, caption in motion throughout."""
    for key, fn in ALGORITHMS.items():
        steps = list(fn(list(BASE_ARRAY)))
        original = sum(
            0.25 if (s.swap or s.write) else (0.15 if s.compare else 0.2) for s in steps
        )
        now = scene_duration(steps, key, SUBTITLES[key])["total"]
        assert now > 2 * original, key


def test_the_template_trick_actually_saves_time():
    """Charging every caption a full first read would be far longer."""
    total_now = total_naive = 0.0
    for key, fn in ALGORITHMS.items():
        steps = list(fn(list(BASE_ARRAY)))
        total_now += scene_duration(steps, key, SUBTITLES[key])["total"]
        total_naive += sum(
            max(MIN_HOLD_FIRST, min(len(s.caption) / CPS_FIRST, MAX_HOLD)) for s in steps
        )
    assert total_naive > 1.4 * total_now


def test_every_algorithm_has_a_subtitle():
    assert set(SUBTITLES) == set(ALGORITHMS)
    for key, text in SUBTITLES.items():
        assert text and "|" in text, key


def test_hold_scale_zero_removes_only_the_holds(monkeypatch):
    import pacing

    steps = list(ALGORITHMS["Bubble Sort"](list(BASE_ARRAY)))
    full = scene_duration(steps, "Bubble Sort", SUBTITLES["Bubble Sort"])
    monkeypatch.setattr(pacing, "HOLD_SCALE", 0.0)
    stripped = pacing.scene_duration(steps, "Bubble Sort", SUBTITLES["Bubble Sort"])
    assert stripped["hold"] == 0.0
    assert stripped["motion"] == full["motion"]
    assert stripped["total"] == full["motion"]
