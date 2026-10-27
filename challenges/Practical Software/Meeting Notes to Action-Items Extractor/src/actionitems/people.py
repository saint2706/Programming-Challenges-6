"""Owner resolution against the attendee list (and the speaker labels seen in the transcript).

No nicknames: "Bob" matches "Bob Chen" only through the first name, and a near-miss such as
"Alicia" for "Alice" is accepted only by a fuzzy ratio of 88 or more. Anything else keeps the
name as written with a lower confidence, so it can be reviewed rather than silently merged.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from rapidfuzz import fuzz, process

FUZZY_CUTOFF = 88

# Capitalised words that are not people's names in meeting text.
NON_NAMES = frozenset(
    {
        "i", "we", "you", "they", "he", "she", "it", "this", "that", "there", "here", "the",
        "let", "action", "okay", "ok", "sure", "yes", "yeah", "thanks", "thank", "hi", "hello",
        "please", "so", "and", "but", "if", "when", "next", "team", "everyone", "someone",
        "somebody", "anyone", "nobody", "both", "all", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday", "january", "february", "march", "april",
        "may", "june", "july", "august", "september", "october", "november", "december",
        "eod", "eow", "todo", "note", "notes", "decision", "question", "agenda", "summary",
        "tomorrow", "today", "tonight", "end", "week", "month", "customer", "client", "legal",
        "product", "engineering", "marketing", "sales", "finance", "design", "ops", "qa",
        "sprint", "release", "launch", "demo", "standup", "retro",
    }
)  # fmt: skip


@dataclass(frozen=True)
class Resolution:
    name: str | None
    confidence: float
    reason: str


def clean_name(raw: str) -> str:
    """Trim punctuation and possessives, collapse spaces, keep the spelling as written."""
    value = re.sub(r"(?:'s|’s)$", "", raw.strip())
    value = value.strip(" \t.,;:!?\"'()[]{}<>@")
    return " ".join(value.split())


def looks_like_name(raw: str) -> bool:
    words = clean_name(raw).split()
    return bool(words) and all(word.lower() not in NON_NAMES for word in words)


class Roster:
    """The people a task can be assigned to: the attendee list plus transcript speakers."""

    def __init__(
        self, attendees: Iterable[str] | None, speakers: Iterable[str] = ()
    ) -> None:
        self.attendees = [
            clean_name(name) for name in attendees or [] if clean_name(name)
        ]
        self.has_attendees = bool(self.attendees)
        seen: dict[str, None] = {}
        for name in [
            *self.attendees,
            *(clean_name(s) for s in speakers if clean_name(s)),
        ]:
            seen.setdefault(name, None)
        self.known = list(seen)

    def resolve(self, raw: str) -> Resolution:
        name = clean_name(raw)
        if not name:
            return Resolution(None, 0.0, "no name")
        lowered = name.lower()
        exact = [k for k in self.known if k.lower() == lowered]
        if exact:
            return Resolution(exact[0], 1.0, "exact match in the roster")
        if " " not in name:
            firsts = {k for k in self.known if k.split()[0].lower() == lowered}
            if len(firsts) == 1:
                return Resolution(firsts.pop(), 0.9, "first name matched the roster")
            if len(firsts) > 1:
                return Resolution(name, 0.5, "first name matches several people")
        if self.known:
            labels: list[str] = []
            owners: list[str] = []
            for full in self.known:
                for label in (full.lower(), full.split()[0].lower()):
                    labels.append(label)
                    owners.append(full)
            best = process.extractOne(
                lowered, labels, scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF
            )
            if best is not None:
                _, score, index = best
                return Resolution(
                    owners[index],
                    round(score / 100 * 0.85, 2),
                    f"fuzzy match to {owners[index]!r} (ratio {score:.0f})",
                )
            if self.has_attendees:
                return Resolution(name, 0.6, "name is not in the attendee list")
        return Resolution(name, 0.7, "name as written (no attendee list)")
