"""Tests for the MST animation's caption pacing.

Pure arithmetic over the Step stream -- no Manim import, so this can be
checked in milliseconds. Same technique as the sorting-visualizer's
pacing.py in this repo: a step is a short motion phase plus a hold phase
sized by subtitle-reading-speed rules, with repeats of the same caption
*shape* (digits normalized) charged only for what changed.
"""

from __future__ import annotations

import pytest
from mst import kruskal_steps
from pacing import (
    CaptionPacer,
    caption_template,
    motion_time,
    novel_characters,
    reading_time,
    scene_duration,
)


def test_caption_template_erases_numbers():
    assert caption_template("Accept 0-1 (w=4)") == "Accept #-# (w=#)"
    assert caption_template("Accept 0-1 (w=4)") == caption_template(
        "Accept 12-13 (w=99)"
    )


def test_novel_characters_counts_only_the_middle():
    assert novel_characters("Accept 0-1 (w=4)", "Accept 0-2 (w=4)") == 1
    assert novel_characters(None, "anything") == len("anything")
    assert novel_characters("same", "same") == 0


def test_reading_time_has_a_floor_and_ceiling():
    assert reading_time("", minimum=0.5) == 0.5
    assert reading_time("x" * 1000, minimum=0.1) <= 4.0  # MAX_HOLD


def test_caption_pacer_charges_nothing_for_an_exact_repeat():
    pacer = CaptionPacer()
    pacer.hold_for("Accept 0-1 (w=4)")
    assert pacer.hold_for("Accept 0-1 (w=4)") == 0.0


def test_caption_pacer_charges_full_read_for_a_new_shape_then_less_for_a_repeat_shape():
    pacer = CaptionPacer()
    first = pacer.hold_for("Accept 0-1 (w=4)")
    second = pacer.hold_for("Accept 1-2 (w=7)")  # same shape, different numbers
    assert first > second > 0.0


def test_motion_time_distinguishes_accept_from_reject():
    steps = list(kruskal_steps(3, [(0, 1, 1), (1, 2, 2), (0, 2, 100)]))
    accepted = [s for s in steps if s.accepted]
    rejected = [s for s in steps if not s.accepted]
    assert accepted and rejected
    for s in accepted:
        assert motion_time(s) > 0
    for s in rejected:
        assert motion_time(s) > 0


def test_scene_duration_is_longer_than_naive_fixed_time_per_step():
    # The original implementation gave every step a flat ~0.4s with no
    # separate hold -- this must produce a longer, readable total for a
    # caption-bearing step stream.
    steps = list(
        kruskal_steps(
            6,
            [
                (0, 1, 4),
                (0, 2, 4),
                (1, 2, 2),
                (1, 3, 5),
                (2, 3, 8),
                (2, 4, 10),
                (3, 4, 2),
                (3, 5, 6),
                (4, 5, 3),
            ],
        )
    )
    info = scene_duration(steps, "Kruskal")
    naive = 0.4 * len(steps)
    assert info["total"] > naive
    assert info["steps"] == len(steps)


@pytest.mark.parametrize("bad_hold", [-1.0])
def test_caption_pacer_never_returns_negative(bad_hold):
    pacer = CaptionPacer()
    assert pacer.hold_for("") >= 0.0
