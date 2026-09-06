"""How long each step of a sorting animation stays on screen, and why.

The original driver gave every step a single 0.15-0.25 s ``run_time``, which
is not a duration a caption can be read in -- and worse, the caption is being
*transformed* for the whole of it, so there is no legible frame at all. This
module splits a step in two:

* a **motion** phase, short, where the bars and the caption change;
* a **hold** phase, still, sized so the caption can actually be read.

The hold budget follows subtitle practice rather than taste. Netflix's Timed
Text Style Guide caps English at 17 characters per second with a minimum cue
duration of 5/6 s; the BBC's subtitle guidelines target 160-180 words per
minute, which works out to about the same rate. These captions are denser than
prose -- indices and comparisons, read as symbols rather than skimmed as
words -- so the first-read rate here sits below the ceiling rather than at it.

The second rate is where the running time is won. Consecutive captions are
overwhelmingly the same sentence with different numbers:

    "Compare adjacent pair 3, 4"  ->  "Compare adjacent pair 4, 5"

A viewer who has read the template once only has to re-read what changed, so a
repeat is charged for the differing characters alone, at roughly double the
rate, and an exactly repeated caption is charged nothing. Charging every
caption a full first read would put Selection Sort past two minutes and Cycle
Sort past three, for no gain in comprehension.

Run this file to see the resulting per-scene durations without rendering
anything (it imports no Manim):

    uv run python pacing.py
"""

from __future__ import annotations

import os
import re

CPS_FIRST = 14.0  # characters/second, first time a caption shape appears
CPS_REPEAT = 30.0  # characters/second for the changed part of a repeat
MIN_HOLD_FIRST = 0.9  # even "Sorted" needs a beat to register
MIN_HOLD_REPEAT = 0.40  # a changed digit still needs a fixation
MAX_HOLD = 4.0  # nothing here is long enough to justify more
TITLE_MIN_HOLD = 2.0  # title + subtitle, before the first step

MOTION_SWAP = 0.30  # a swap moves two bars and needs to be followed
MOTION_WRITE = 0.30
MOTION_COMPARE = 0.20
MOTION_PLAIN = 0.25
CLOSING_HOLD = 1.5

#: Scale every hold (never the motion) for fast iteration. 0 drops them all.
HOLD_SCALE = float(os.environ.get("SORT_RACE_HOLD_SCALE", "1"))

_DIGITS = re.compile(r"\d+")


def caption_template(caption: str) -> str:
    """The caption with its numbers removed -- its *shape*.

    ``"Compare adjacent pair 3, 4"`` and ``"Compare adjacent pair 4, 5"`` are
    the same template, so the second is a repeat even though no earlier caption
    equals it character for character.
    """
    return _DIGITS.sub("#", caption)


def novel_characters(previous: str | None, current: str) -> int:
    """How much of ``current`` is new, given the viewer just read ``previous``.

    The common prefix and the common suffix are already on screen and already
    parsed; what is left between them is what the eye has to move to. For
    ``"Compare adjacent pair 3, 4"`` -> ``"...pair 4, 5"`` that is two
    characters, not twenty-six.
    """
    if previous is None:
        return len(current)
    limit = min(len(previous), len(current))
    head = 0
    while head < limit and previous[head] == current[head]:
        head += 1
    tail = 0
    while tail < limit - head and previous[-1 - tail] == current[-1 - tail]:
        tail += 1
    return len(current) - head - tail


def reading_time(text: str, minimum: float) -> float:
    """A first-read budget for a block of text, clamped at both ends."""
    return max(minimum, min(len(text) / CPS_FIRST, MAX_HOLD))


def motion_time(step) -> float:
    """How long the transform for one step should take.

    A swap or a write changes bar heights and deserves to be followed; a
    comparison only recolours two bars and can be quicker.
    """
    if step.swap:
        return MOTION_SWAP
    if step.write:
        return MOTION_WRITE
    if step.compare:
        return MOTION_COMPARE
    return MOTION_PLAIN


class CaptionPacer:
    """Stateful hold budget: remembers what the viewer has already read."""

    def __init__(self) -> None:
        self.previous: str | None = None
        self.seen: set[str] = set()

    def hold_for(self, caption: str) -> float:
        """Seconds this caption should sit still, before ``HOLD_SCALE``.

        Three cases, cheapest first:

        * identical to the last caption -- already read, no hold at all;
        * a template seen before -- charge only what changed, at the faster
          re-read rate;
        * a new template -- charge the whole line at the first-read rate.

        Cycle Sort is the case that makes this worth doing: 107 steps, but
        only 16 of them say anything the previous step did not.
        """
        if caption == self.previous:
            hold = 0.0
        elif caption_template(caption) in self.seen:
            changed = novel_characters(self.previous, caption)
            hold = max(MIN_HOLD_REPEAT, min(changed / CPS_REPEAT, MAX_HOLD))
        else:
            hold = max(MIN_HOLD_FIRST, min(len(caption) / CPS_FIRST, MAX_HOLD))
        self.seen.add(caption_template(caption))
        self.previous = caption
        return hold


def scene_duration(steps, title: str = "", subtitle: str = "") -> dict[str, float]:
    """Total seconds for one scene, split into motion and hold.

    Pure arithmetic over the step stream -- no rendering, no Manim -- so the
    pacing can be checked and tuned in milliseconds instead of minutes.
    """
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
    from sorting_algorithms import ALGORITHMS, BASE_ARRAY, SUBTITLES

    print(
        f"{'scene':<20} {'steps':>6} {'motion':>9} {'hold':>9} {'total':>9} "
        f"{'s/step':>8}"
    )
    grand = 0.0
    for key, fn in ALGORITHMS.items():
        steps = list(fn(list(BASE_ARRAY)))
        info = scene_duration(steps, key, SUBTITLES.get(key, ""))
        grand += info["total"]
        print(
            f"{key:<20} {info['steps']:>6d} {info['motion']:>8.1f}s "
            f"{info['hold']:>8.1f}s {info['total']:>8.1f}s "
            f"{info['total'] / info['steps']:>7.2f}s"
        )
    print(
        f"{'':<20} {'':>6} {'':>9} {'':>9} {grand:>8.1f}s "
        f"({grand / 60:.1f} min for all 14)"
    )
    if HOLD_SCALE != 1.0:
        print(f"\n(SORT_RACE_HOLD_SCALE={HOLD_SCALE} is in effect)")


if __name__ == "__main__":
    _report()
