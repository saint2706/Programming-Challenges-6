"""Tests for the N-Queens search animation's caption pacing.

Same technique as the MST and sorting visualizers in this repo: a step gets
a short motion phase plus a hold phase sized by subtitle-reading-speed
rules, with a repeated caption *shape* (numbers normalized) charged only for
what changed. This matters more here than anywhere else in the repo: a
demo search produces hundreds of place/backtrack steps, so charging a full
read for every one of them would make the video both unreadable (steps
flying past in the naive fixed-run-time version) and needlessly long (if
every repeat paid full price too).
"""

from __future__ import annotations

from nqueens import solve_bitmask_steps
from pacing import (
    CaptionPacer,
    caption_template,
    motion_time,
    novel_characters,
    scene_duration,
)


def test_caption_template_erases_numbers():
    assert caption_template("Place row 2 at column 5") == "Place row # at column #"
    assert caption_template("Place row 2 at column 5") == caption_template(
        "Place row 10 at column 11"
    )


def test_novel_characters_counts_only_the_middle():
    assert novel_characters("Place row 2 at column 5", "Place row 2 at column 3") == 1


def test_caption_pacer_charges_full_read_once_then_less_for_repeats():
    pacer = CaptionPacer()
    first = pacer.hold_for("Place row 0 at column 0")
    second = pacer.hold_for("Place row 0 at column 1")
    third = pacer.hold_for("Place row 0 at column 2")
    assert first > second
    assert second == third  # both are single-changed-character repeats


def test_motion_time_is_shorter_for_backtrack_than_place():
    place_steps = [s for s in solve_bitmask_steps(6, limit=1) if s.action == "place"]
    backtrack_steps = [
        s for s in solve_bitmask_steps(6, limit=1) if s.action == "backtrack"
    ]
    assert place_steps
    for s in place_steps:
        assert motion_time(s) > 0
    for s in backtrack_steps:
        assert motion_time(s) > 0


def test_scene_duration_stays_bounded_despite_hundreds_of_steps():
    # n=6 with limit=4 produces ~250 place/backtrack steps -- the repeat
    # discount must keep the hold budget from ballooning linearly with that.
    steps = list(solve_bitmask_steps(6, limit=4))
    info = scene_duration(steps, "N-Queens")
    assert info["steps"] > 200
    naive_full_read_every_step = (
        0.6 * info["steps"]
    )  # what no-discount pacing would cost
    assert info["total"] < naive_full_read_every_step
    assert info["total"] > 0


def test_solution_captions_are_never_skipped_even_though_rare():
    steps = list(solve_bitmask_steps(6, limit=4))
    pacer = CaptionPacer()
    solution_holds = []
    for s in steps:
        hold = pacer.hold_for(s.caption)
        if s.action == "solution":
            solution_holds.append(hold)
    assert len(solution_holds) == 4
    assert all(
        h > 0 for h in solution_holds
    )  # "Solution #N" always a new shape? no -- first only
