"""How long each place/backtrack/solution step stays on screen, and why.

The first version of `visualize.py` gave every place/backtrack a flat
0.1-0.12s `run_time` with no separate reading pause -- for a demo search
that produces hundreds of steps, that is fast enough that no caption is
ever actually legible, only a blur of queens appearing and vanishing. This
is the same fix as the MST and sorting visualizers in this repo: split a
step into a short **motion** phase and a still **hold** phase sized by
subtitle-reading-speed practice (14 characters/second first read, below
Netflix's 17 cps ceiling since these captions are denser than prose).

This matters more here than in the other two visualizers: a search over
n=6 finding just 4 solutions already produces ~250 place/backtrack steps
(see `test_scene_duration_stays_bounded_despite_hundreds_of_steps`), and
"Place row 2 at column 5" / "Place row 2 at column 3" are the same caption
*shape* differing only in the placed column -- so a repeat is charged only
for the changed digits, at roughly double the read rate, which is what
keeps the total from scaling linearly with the (large) step count.

    uv run python pacing.py
"""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher

CPS_FIRST = 14.0
CPS_REPEAT = 30.0
MIN_HOLD_FIRST = 0.6
MIN_HOLD_REPEAT = 0.12
MAX_HOLD = 3.0
TITLE_MIN_HOLD = 1.5

MOTION_PLACE = 0.18
MOTION_BACKTRACK = 0.12
MOTION_SOLUTION = 0.35
CLOSING_HOLD = 1.5

#: Scale every hold (never the motion) for fast iteration. 0 drops them all.
HOLD_SCALE = float(os.environ.get("NQUEENS_HOLD_SCALE", "1"))

_DIGITS = re.compile(r"\d+")


def caption_template(caption: str) -> str:
    return _DIGITS.sub("#", caption)


def novel_characters(previous: str | None, current: str) -> int:
    """How much of `current` actually needs re-reading, given `previous`.

    Row and column digits sit in unrelated positions in a caption ("Place
    row 0 at column 3" -> "Place row 1 at column 2"), and backtracking
    crosses a row boundary on almost every step -- so both digits usually
    change together. A naive common-prefix/common-suffix trim breaks badly
    here: the first mismatch (the row digit) ends the prefix scan, and the
    *second* mismatch (the column digit) sits right at the end of the
    string, so the suffix scan finds nothing and the entire middle -- most
    of the caption -- is counted as "changed" even though only two digits
    actually differ. `SequenceMatcher` finds every matching run regardless
    of position, so the unchanged "at column "/" row " text in between two
    separately-changed digits is correctly recognized as already-read.
    """
    if previous is None:
        return len(current)
    matcher = SequenceMatcher(None, previous, current, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return len(current) - matched


def reading_time(text: str, minimum: float) -> float:
    return max(minimum, min(len(text) / CPS_FIRST, MAX_HOLD))


def motion_time(step) -> float:
    if step.action == "place":
        return MOTION_PLACE
    if step.action == "solution":
        return MOTION_SOLUTION
    return MOTION_BACKTRACK


class CaptionPacer:
    """Stateful hold budget: remembers what the viewer has already read."""

    def __init__(self) -> None:
        self.previous: str | None = None
        self.seen: set[str] = set()

    def hold_for(self, caption: str) -> float:
        if caption == self.previous:
            hold = 0.0
        elif caption_template(caption) in self.seen:
            changed = novel_characters(self.previous, caption)
            hold = max(MIN_HOLD_REPEAT, min(changed / CPS_REPEAT, MAX_HOLD))
        else:
            hold = max(MIN_HOLD_FIRST, min(len(caption) / CPS_FIRST, MAX_HOLD))
        self.seen.add(caption_template(caption))
        self.previous = caption
        return max(0.0, hold)


def scene_duration(steps, title: str = "", subtitle: str = "") -> dict[str, float]:
    pacer = CaptionPacer()
    motion = 0.0
    hold = reading_time(f"{title} {subtitle}", TITLE_MIN_HOLD) + CLOSING_HOLD
    count = 0
    for step in steps:
        motion += motion_time(step)
        hold += pacer.hold_for(step.caption)
        count += 1
    return {
        "steps": count,
        "motion": motion,
        "hold": hold * HOLD_SCALE,
        "total": motion + hold * HOLD_SCALE,
    }


def _report() -> None:
    from nqueens import solve_bitmask_steps

    for n, limit in [(6, 4), (8, 3)]:
        steps = list(solve_bitmask_steps(n, limit=limit))
        info = scene_duration(steps, f"N-Queens (n={n})")
        print(
            f"n={n} limit={limit}: {info['steps']:4d} steps, "
            f"motion={info['motion']:.1f}s hold={info['hold']:.1f}s "
            f"total={info['total']:.1f}s"
        )
    if HOLD_SCALE != 1.0:
        print(f"\n(NQUEENS_HOLD_SCALE={HOLD_SCALE} is in effect)")


if __name__ == "__main__":
    _report()
