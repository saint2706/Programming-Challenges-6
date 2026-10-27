"""Plain data shapes shared by the parsers, the rules, the exporters and the evaluator."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Utterance:
    """One line (or cue) of the input after format parsing."""

    line: int  # 1-based line number in the source text
    text: str
    speaker: str | None = None
    timestamp: str | None = None
    # "text", "bullet", "checkbox_open", "checkbox_done", "heading", "action_section"
    kind: str = "text"
    # Section the line sits under, lower-cased: "action items", "decisions", ... or None.
    section: str | None = None


@dataclass(frozen=True)
class Source:
    """Where an action item was said or written."""

    line: int
    timestamp: str | None
    speaker: str | None
    excerpt: str


@dataclass
class ActionItem:
    task: str
    owner: str | None
    due: str | None  # ISO 8601 calendar date
    due_phrase: str | None
    due_confidence: float
    item_type: (
        str  # commitment, request, conditional, assigned, unassigned, todo, tentative
    )
    confidence: float
    reasons: list[str]
    sources: list[Source]
    mentions: int = 1
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["sources"] = [asdict(s) for s in self.sources]
        return data
