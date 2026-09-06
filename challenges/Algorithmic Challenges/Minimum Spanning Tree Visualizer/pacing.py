"""How long each edge decision stays on screen, and why.

The first version of `visualize.py` gave every step a flat ~0.4s `run_time`
covering both the line's motion *and* its caption change, with no separate
pause to actually read the caption -- unwatchable without pausing the video.
This module is the same fix used in this repo's sorting-algorithm
visualizer: split a step into a short **motion** phase (the edge animates
in/out) and a still **hold** phase, sized by subtitle-reading-speed
practice (Netflix caps English at 17 characters/second with a 5/6s minimum
cue; these captions are denser than prose, so the first-read rate here sits
below that ceiling at 14 cps).

Consecutive captions are frequently the same sentence with different edge
endpoints ("Accept 0-1 (w=4)" -> "Accept 1-2 (w=7)"), so a repeat of an
already-seen caption *shape* is charged only for the characters that
changed, at roughly double the rate; an exact repeat costs nothing.

    uv run python pacing.py
"""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher

CPS_FIRST = 14.0
CPS_REPEAT = 30.0
MIN_HOLD_FIRST = 0.7
MIN_HOLD_REPEAT = 0.35
MAX_HOLD = 4.0
TITLE_MIN_HOLD = 1.5

MOTION_ACCEPT = 0.4
MOTION_REJECT = 0.35
CLOSING_HOLD = 1.5

#: Scale every hold (never the motion) for fast iteration. 0 drops them all.
HOLD_SCALE = float(os.environ.get("MST_HOLD_SCALE", "1"))

_DIGITS = re.compile(r"\d+")


def caption_template(caption: str) -> str:
    return _DIGITS.sub("#", caption)


def novel_characters(previous: str | None, current: str) -> int:
    """How much of `current` actually needs re-reading, given `previous`.

    An edge caption like "Accept 0-1 (w=4)" -> "Accept 1-2 (w=7)" changes
    three separate digits (u, v, w) in unrelated positions. A naive
    common-prefix/common-suffix trim would count almost the entire caption
    as "changed" the moment the first digit differs, since the last
    character (part of the weight) also differs and stops the suffix scan
    immediately. `SequenceMatcher` finds every matching run regardless of
    position, so the unchanged "-", " (w=", ")" text between the three
    changed digits is correctly recognized as already-read.
    """
    if previous is None:
        return len(current)
    matcher = SequenceMatcher(None, previous, current, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return len(current) - matched


def reading_time(text: str, minimum: float) -> float:
    return max(minimum, min(len(text) / CPS_FIRST, MAX_HOLD))


def motion_time(step) -> float:
    return MOTION_ACCEPT if step.accepted else MOTION_REJECT


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
    from mst import ALGORITHMS

    demo_edges = [
        (0, 1, 4), (0, 2, 4), (1, 2, 2), (1, 3, 5),
        (2, 3, 8), (2, 4, 10), (3, 4, 2), (3, 5, 6), (4, 5, 3),
    ]  # fmt: skip
    print(f"{'scene':<10} {'steps':>6} {'motion':>9} {'hold':>9} {'total':>9}")
    for name, algo in ALGORITHMS.items():
        steps = list(algo(6, demo_edges))
        info = scene_duration(steps, name)
        print(
            f"{name:<10} {info['steps']:>6d} {info['motion']:>8.1f}s "
            f"{info['hold']:>8.1f}s {info['total']:>8.1f}s"
        )
    if HOLD_SCALE != 1.0:
        print(f"\n(MST_HOLD_SCALE={HOLD_SCALE} is in effect)")


if __name__ == "__main__":
    _report()
