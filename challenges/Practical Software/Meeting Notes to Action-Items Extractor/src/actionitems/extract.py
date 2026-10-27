"""The rules pipeline: document -> utterances -> clauses -> candidates -> action items.

Confidence for each item is the rule's base confidence scaled by how sure we are of the owner
(roster match quality, or spaCy NER agreement for names the roster does not know). A conditional
commitment ("if the vendor agrees, I will...") is scaled down further. Dedupe merges items with
the same owner, the same due date and near-identical task text, keeping the strongest one.
"""

from dataclasses import dataclass, field
from datetime import date, datetime

from rapidfuzz import fuzz

from actionitems import nlp
from actionitems.dates import (
    date_from_header,
    date_from_mtime,
    parse_meeting_date,
    resolve_due,
    validate_timezone,
)
from actionitems.errors import InputError
from actionitems.formats import parse_document
from actionitems.ingest import Document
from actionitems.models import ActionItem, Source
from actionitems.people import Roster
from actionitems.rules import Context, classify, clean_task, split_clauses

DEDUPE_RATIO = 90
EXCERPT_CHARS = 200
# Fuzzy dedupe is quadratic within a bucket, so the candidate count per document is capped.
MAX_ITEMS_PER_DOCUMENT = 2000


@dataclass(frozen=True)
class Options:
    attendees: tuple[str, ...] = ()
    meeting_date: str | None = None
    timezone: str = "UTC"
    min_confidence: float = 0.4
    use_nlp: bool = True


@dataclass
class Extraction:
    name: str
    meeting_date: date
    timezone: str
    backend: str
    items: list[ActionItem] = field(default_factory=list)


def _meeting_date(doc: Document, options: Options, tz) -> date:
    if options.meeting_date:
        return parse_meeting_date(options.meeting_date, tz)
    from_header = date_from_header(doc.text, tz)
    if from_header is not None:
        return from_header
    if doc.mtime is not None:
        return date_from_mtime(doc.mtime, tz)
    return datetime.now(tz).date()


def _owner_factor(
    owner_raw: str | None, roster: Roster, clause: str, use_nlp: bool, notes: list[str]
) -> tuple[str | None, float]:
    if owner_raw is None:
        return None, 1.0
    resolution = roster.resolve(owner_raw)
    if resolution.name is None:
        return None, 1.0
    factor = resolution.confidence
    if use_nlp and resolution.name not in roster.known:
        names = nlp.person_names(clause)
        if names and any(resolution.name in n or n in resolution.name for n in names):
            factor = max(factor, 0.85)
            notes.append("owner confirmed as a person by spaCy NER")
    return resolution.name, factor


def extract_document(doc: Document, options: Options) -> Extraction:
    tz = validate_timezone(options.timezone)
    meeting = _meeting_date(doc, options, tz)
    utterances = parse_document(doc.text)
    speakers = [u.speaker for u in utterances if u.speaker]
    roster = Roster(options.attendees, speakers)
    backend = nlp.backend_name() if options.use_nlp else "rules-only"
    result = Extraction(doc.name, meeting, options.timezone, backend)
    for utterance in utterances:
        if utterance.kind == "checkbox_done":
            continue
        ctx = Context(utterance.speaker, utterance.kind, utterance.section)
        for clause in split_clauses(utterance):
            candidate = classify(clause, ctx)
            if candidate is None:
                continue
            notes: list[str] = []
            owner, factor = _owner_factor(
                candidate.owner_raw, roster, clause, options.use_nlp, notes
            )
            confidence = candidate.base * factor
            if candidate.conditional:
                confidence *= 0.85
            due = due_phrase = None
            due_confidence = 0.0
            if candidate.due_phrase:
                resolved = resolve_due(candidate.due_phrase, meeting)
                if resolved is not None:
                    due, due_phrase, due_confidence = (
                        resolved.due.isoformat(),
                        resolved.phrase,
                        resolved.confidence,
                    )
                    candidate.reasons.append(resolved.reason)
                else:
                    due_phrase = candidate.due_phrase
                    notes.append("due phrase found but not resolvable")
            task = clean_task(candidate.task)
            if not task:
                continue
            item = ActionItem(
                task=task,
                owner=owner,
                due=due,
                due_phrase=due_phrase,
                due_confidence=round(due_confidence, 2),
                item_type=candidate.item_type,
                confidence=round(min(max(confidence, 0.0), 1.0), 2),
                reasons=[*candidate.reasons, *notes],
                sources=[
                    Source(
                        utterance.line,
                        utterance.timestamp,
                        utterance.speaker,
                        " ".join(clause.split())[:EXCERPT_CHARS],
                    )
                ],
                notes=notes,
            )
            result.items.append(item)
            if len(result.items) > MAX_ITEMS_PER_DOCUMENT:
                raise InputError(
                    f"{doc.name} yields more than {MAX_ITEMS_PER_DOCUMENT} candidate items; "
                    "split the file"
                )
    result.items = merge_items(result.items)
    result.items = [i for i in result.items if i.confidence >= options.min_confidence]
    return result


def merge_items(items: list[ActionItem]) -> list[ActionItem]:
    """Merge duplicates (same owner and due date, near-identical task); keep the strongest.

    Items are bucketed by (owner, due) first, so fuzzy comparison only happens within a bucket.
    """
    merged: list[ActionItem] = []
    buckets: dict[tuple[str | None, str | None], list[int]] = {}
    for item in items:
        bucket = buckets.setdefault((item.owner, item.due), [])
        index = next((i for i in bucket if _same_task(merged[i], item)), None)
        if index is None:
            bucket.append(len(merged))
            merged.append(item)
            continue
        twin = merged[index]
        mentions = twin.mentions + item.mentions
        sources = [*twin.sources, *item.sources]
        winner = item if item.confidence > twin.confidence else twin
        winner.mentions, winner.sources = mentions, sources
        merged[index] = winner
    return merged


def _same_task(a: ActionItem, b: ActionItem) -> bool:
    return fuzz.token_set_ratio(a.task.lower(), b.task.lower()) >= DEDUPE_RATIO


def extract_documents(documents: list[Document], options: Options) -> list[Extraction]:
    return [extract_document(doc, options) for doc in documents]
