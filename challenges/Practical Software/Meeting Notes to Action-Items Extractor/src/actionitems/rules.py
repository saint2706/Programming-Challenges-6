"""Clause-level rules: which sentences are commitments, requests, or neither.

A clause is classified in this order: hedges and conditionals are peeled off, a due phrase
is removed, then decisions, questions, requests, negations and commitments are tried. Owners
come from the speaker ("I'll"), a named subject ("Bob will"), an addressee ("Bob, can you"),
an @mention, or stay unassigned ("we will", "someone should"). Checkbox and action-section
lines are tasks even without a modal, because the document says so.
"""

import re
from dataclasses import dataclass, field

from actionitems.dates import find_due_phrase
from actionitems.models import Utterance
from actionitems.people import looks_like_name

ACTION_VERBS = frozenset(
    [
        "send",
        "share",
        "book",
        "draft",
        "write",
        "update",
        "review",
        "prepare",
        "schedule",
        "follow",
        "check",
        "fix",
        "set",
        "create",
        "add",
        "remove",
        "file",
        "email",
        "call",
        "ping",
        "finish",
        "submit",
        "open",
        "close",
        "merge",
        "deploy",
        "test",
        "investigate",
        "look",
        "remind",
        "revisit",
        "decide",
        "shorten",
        "finalise",
        "finalize",
        "reach",
        "get",
        "make",
        "organize",
        "plan",
        "circulate",
        "document",
        "publish",
        "post",
        "upload",
        "migrate",
        "refactor",
        "ship",
        "deliver",
        "order",
        "buy",
        "pay",
        "invoice",
        "sign",
        "approve",
        "move",
        "ask",
        "take",
        "own",
        "cover",
        "start",
        "wrap",
        "draw",
        "connect",
        "link",
        "attach",
        "forward",
        "reply",
        "respond",
        "answer",
        "notify",
        "inform",
        "invite",
        "pick",
        "reserve",
        "renew",
        "cancel",
        "collect",
        "gather",
        "list",
        "map",
        "tag",
        "label",
        "sync",
        "stage",
        "release",
        "commit",
        "debug",
        "profile",
        "monitor",
        "restart",
        "rotate",
        "archive",
        "delete",
        "clean",
        "triage",
        "prototype",
        "research",
        "compare",
        "estimate",
        "measure",
        "verify",
        "validate",
        "audit",
        "analyze",
        "analyse",
        "summarize",
        "summarise",
        "outline",
        "sketch",
        "design",
        "implement",
        "port",
        "hire",
        "onboard",
        "train",
        "interview",
        "contact",
        "chase",
        "address",
        "resolve",
        "handle",
        "confirm",
        "pull",
        "push",
        "run",
        "build",
        "install",
        "configure",
        "rename",
        "reschedule",
        "loop",
        "kick",
        "wrap",
        "backup",
        "patch",
        "bump",
        "rebase",
        "pair",
        "page",
        "alert",
        "record",
        "transcribe",
        "translate",
        "chase",
        "coordinate",
        "fill",
        "sort",
        "prepare",
        "pitch",
        "quote",
        "estimate",
        "budget",
        "forecast",
        "log",
        "escalate",
    ]
)
MULTI_WORD_ACTIONS = frozenset(
    {"follow up", "reach out", "set up", "look into", "get back", "make sure", "check in",
     "sign off", "loop in", "kick off", "take over", "wrap up", "sort out", "fill in", "dig into"}
)  # fmt: skip
NON_ACTION_REQUESTS = frozenset(
    {"know", "tell", "explain", "remind", "say", "think", "see", "hear", "believe", "understand",
     "feel", "be", "let", "agree", "wait", "elaborate", "clarify", "mind"}
)  # fmt: skip

NAME = r"[A-Z][\w.'\u2019-]*(?: [A-Z][\w.'\u2019-]*){0,2}"
MODAL_FUTURE = (
    r"(?:'ll|'m going to|'m gonna|'re going to|'re gonna|'s going to|'s gonna"
    r"|\s+(?:will|shall|must)|\s+(?:am|is|are)\s+(?:going to|gonna)"
    r"|\s+(?:need|needs|has|have)\s+to|\s+(?:going to|gonna))"
)
MODAL_TENTATIVE = r"(?:'d|\s+(?:should|could|might|may|can|would|ought to))"
TAKING = r"(?:taking|handling|owning|covering|picking up)"

RE_FIRST = re.compile(rf"^(?P<subj>I|[Ww]e)(?P<modal>{MODAL_FUTURE})\s+(?P<rest>\S.*)$")
RE_FIRST_TENTATIVE = re.compile(
    rf"^(?P<subj>I|[Ww]e)(?P<modal>{MODAL_TENTATIVE})\s+(?P<rest>\S.*)$"
)
RE_I_TAKING = re.compile(rf"^(?P<subj>I)(?:'m|\s+am)\s+{TAKING}\s+(?P<rest>\S.*)$")
RE_NAMED = re.compile(rf"^(?P<subj>{NAME})(?P<modal>{MODAL_FUTURE})\s+(?P<rest>\S.*)$")
RE_NAMED_TO = re.compile(rf"^(?P<subj>{NAME})\s+to\s+(?P<rest>\S.*)$")
RE_NAMED_TAKING = re.compile(
    rf"^(?P<subj>{NAME})(?:\s+is|'s|\s+will be)\s+{TAKING}\s+(?P<rest>\S.*)$"
)
RE_SOMEONE = re.compile(
    rf"^(?P<subj>(?i:someone|somebody|anyone|one of us|everyone))(?P<modal>{MODAL_FUTURE}|\s+(?:should|ought to))\s+(?P<rest>\S.*)$"
)  # fmt: skip
RE_NEED_SOMEONE = re.compile(
    r"^(?i:(?:we|i|you|they)\s+(?:need|needs)\s+(?:someone|somebody|anyone)\s+to)\s+(?P<rest>\S.*)$"
)
RE_NAMED_SHOULD = re.compile(
    rf"^(?P<subj>{NAME})\s+(?:should|ought to)\s+(?P<rest>\S.*)$"
)
RE_ADDRESSED_SHOULD = re.compile(
    rf"^(?P<addr>{NAME})[,:]?\s+(?i:(?:you\s+)?(?:should|need to|have to|must))\s+(?P<rest>\S.*)$"
)
RE_NEED_YOU = re.compile(
    rf"^(?:(?P<addr>{NAME})[,:]\s+)?(?i:I need you to)\s+(?P<rest>\S.*)$"
)
RE_AGENDA_LETS = re.compile(
    r"(?i)^(?:let'?s|let us)\s+(?:start|begin|get started|kick off|go over|move on|wrap up)\b"
)
RE_TRAILING_HEDGE = re.compile(r"(?i)[,\s]+(?:maybe|perhaps|possibly)[.!?]?$")
RE_LEAD_OWNER = re.compile(rf"^(?P<name>{NAME})\s+[-–—]\s+(?P<rest>\S.*)$")
RE_LETS = re.compile(r"^(?i:let'?s|let us)\s+(?P<rest>\S.*)$")
RE_REQUEST_YOU = re.compile(
    rf"^(?:(?P<addr>{NAME})[,:]?\s+)?"
    r"(?i:(?:please,?\s+)?(?:can|could|would|will)\s+(?:you|u|ya))"
    r"(?:(?i:\s+(?:please|maybe|just|quickly|possibly)))*\s+(?P<rest>\S.*)$"
)
RE_REQUEST_ANYONE = re.compile(
    r"^(?:(?P<addr>" + NAME + r"),?\s+)?(?i:(?:can|could|would|will)\s+"
    r"(?:someone|somebody|anyone|one of you|you guys|you all|guys))\s+(?P<rest>\S.*)$"
)
RE_PLEASE = re.compile(
    rf"^(?:(?P<addr>{NAME})[,:]?\s+)?(?i:please)(?:\s+(?i:also|just))?\s+(?P<rest>\S.*)$"
)
RE_THINK_ABOUT = re.compile(r"(?i)^(?:i'?ll|i will|we'?ll|we will)?\s*think about\b")
RE_VOCATIVE = re.compile(rf"^(?P<addr>{NAME}),\s+(?P<rest>\S.*)$")
RE_MENTION = re.compile(r"(?<!\w)@([A-Za-z][\w.-]{0,40})")
RE_PAREN_OWNER = re.compile(
    r"\(\s*(?:owner|assignee|owned by)?:?\s*(" + NAME + r")\s*\)"
)
RE_TRAILING_OWNER = re.compile(r"\s[-\u2013\u2014]\s*(" + NAME + r")\s*$")
RE_DECISION = re.compile(
    r"(?i)\b(?:we|the team|everyone|they)\s+(?:decided|agreed|chose|voted|settled on|concluded)"
    r"\b(?:\s+(?:to|that))?\s*(?P<after>\S.*)?$"
)
RE_DECISION_PHRASE = re.compile(
    r"(?i)\b(?:we're going with|going forward|we are going with)\b"
)
RE_NEGATION = re.compile(
    r"(?i)\b(?:won't|wont|will not|not going to|not gonna|don't need to|do not need to"
    r"|doesn't need to|does not need to|no need to|shouldn't|should not|can't|cannot|never"
    r"|no longer|not planning|isn't going|aren't going|no one needs|nobody needs"
    r"|don't have to|do not have to|we're not|we are not|i'm not|i am not|not sure)\b"
)
RE_QUESTION_START = re.compile(
    r"(?i)^(?:should we|shall we|do we|does|did|is|are|was|were|who|what|when|where|why|how"
    r"|which|anyone|any idea|is anyone|can anyone|could anyone|are we|is there|would anyone)\b"
)
RE_HEDGE = re.compile(
    r"(?i)^(?:i think|i guess|i believe|i suppose|i expect|i'm pretty sure|probably|honestly"
    r"|basically|hopefully|i reckon|maybe)\s+(?=\S)"
)
RE_CONDITIONAL_PREFIX = re.compile(
    r"(?i)^(?:if|when|once|unless|assuming|as soon as|provided that)\b[^,]{0,160},\s*(?P<main>\S.*)$"
)
RE_CONDITIONAL_SUFFIX = re.compile(r"(?i)\s+(?:if|unless|assuming|once)\s+\S.*$")
RE_FILLER = re.compile(
    r"(?i)^(?:(?:so|um|uh|umm|and|also|then|yeah|yes|okay|ok|alright|right|well|hmm|like|just|sure|fine|done)[,]?\s+)+"
)
SPLIT_SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=\S)")
SPLIT_CONJUNCTION = re.compile(
    r"\s*,?\s+(?:and|also|plus|then)\s+(?=(?:I|we|We|[A-Z][a-z]+)(?:'ll|\s+will|\s+can|\s+should"
    r"|\s+needs?\s+to|\s+has\s+to|\s+is\s+going\s+to|\s+to\s+\w)|(?:I|we|We)'ll\b)"
)
SPLIT_ASR = re.compile(
    r"(?<=\w)\s+(?=(?:I'll|I will|we'll|we will|can you|could you|will you|would you|please)\b)"
)
SPLIT_ASR_NAME = re.compile(
    r"(?<=\w)\s+(?=[A-Z][a-z]+ (?:will|'ll|can|should|needs? to|has to|is going to)\b)"
)
STRIP_TASK_LEAD = re.compile(
    r"(?i)^(?:(?:can|could|would|will) you(?: please| just| maybe| quickly)?\s+|please\s+"
    r"|you (?:should|need to|have to)\s+|(?:let'?s|let us)\s+|to\s+"
    r"|(?:going to|gonna|need to|needs to|has to|have to|should|will|shall)\s+)"
)


@dataclass(frozen=True)
class Context:
    speaker: str | None
    kind: str  # Utterance.kind
    section: str | None


@dataclass
class Candidate:
    item_type: str  # commitment, unassigned, assigned, request, todo, tentative
    owner_raw: str | None
    task: str  # raw text after the subject and modal; cleaned later
    base: float
    reasons: list[str] = field(default_factory=list)
    due_phrase: str | None = None
    addressee: str | None = None
    needs_acceptance: bool = False
    conditional: bool = False
    hedged: bool = False


def split_clauses(utterance: Utterance) -> list[str]:
    """Split one utterance into clauses. Unpunctuated (ASR-style) text gets extra splits."""
    text = " ".join(utterance.text.split())
    if not text:
        return []
    pieces = [p for p in SPLIT_SENTENCE.split(text) if p]
    if not any(ch in text for ch in ".!?;"):
        pieces = [q for p in pieces for q in SPLIT_ASR.split(p)]
        pieces = [q for p in pieces for q in SPLIT_ASR_NAME.split(p)]
        # "Eli | can you ...": a lone capitalised word is the addressee, keep it with the rest
        joined: list[str] = []
        for piece in pieces:
            if joined and re.fullmatch(r"[A-Z][\w'-]*", joined[-1]):
                joined[-1] = f"{joined[-1]} {piece}"
            else:
                joined.append(piece)
        pieces = joined
    clauses: list[str] = []
    for piece in pieces:
        clauses.extend(c.strip() for c in SPLIT_CONJUNCTION.split(piece) if c.strip())
    return clauses


def _first_word(text: str) -> str:
    match = re.match(r"[A-Za-z']+", text)
    return match.group(0).lower() if match else ""


def is_action_verb_start(text: str) -> bool:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return False
    return words[0] in ACTION_VERBS or " ".join(words[:2]) in MULTI_WORD_ACTIONS


def clean_task(text: str) -> str:
    """Imperative task text: drop fillers, subjects, modals and trailing conditionals."""
    value = " ".join(text.split()).strip(" ,;:.!?")
    value = RE_CONDITIONAL_SUFFIX.sub("", value)
    value = RE_FILLER.sub("", value)
    value = STRIP_TASK_LEAD.sub("", value, count=1)
    value = STRIP_TASK_LEAD.sub("", value, count=1)
    taking = re.match(rf"(?i)^{TAKING}\s+(?P<rest>\S.*)$", value)
    if taking:
        value = "Take " + taking.group("rest")
    value = value.strip(" ,;:.!?")
    if not value:
        return " ".join(text.split()).strip(" ,;:.!?")
    return value[0].upper() + value[1:]


def owner_in_text(text: str) -> tuple[str | None, str]:
    """Find an owner marked as @name, (Name) or a trailing ' - Name'; return it and the text."""
    mention = RE_MENTION.search(text)
    if mention:
        return mention.group(1), (
            text[: mention.start()] + text[mention.end() :]
        ).strip()
    paren = RE_PAREN_OWNER.search(text)
    if paren and looks_like_name(paren.group(1)):
        return paren.group(1), (text[: paren.start()] + text[paren.end() :]).strip()
    trailing = RE_TRAILING_OWNER.search(text)
    if trailing and looks_like_name(trailing.group(1)):
        return trailing.group(1), text[: trailing.start()].strip()
    return None, text


def _named_owner(name: str) -> str | None:
    return name if looks_like_name(name) else None


def _commitment(clause: str, ctx: Context) -> Candidate | None:
    """Subject plus future modal, or a 'Name to <verb>' assignment."""
    match = RE_FIRST.match(clause)
    if match:
        rest = match.group("rest")
        needs_to = match.group("modal").rstrip().endswith("to")
        if match.group("subj") == "I":
            if ctx.speaker is None:
                return Candidate("unassigned", None, rest, 0.6,
                                 ["first person, but the speaker is unknown"])  # fmt: skip
            base = 0.85 if needs_to else 0.9
            return Candidate("commitment", ctx.speaker, rest, base,
                             [f"first-person commitment by {ctx.speaker}"])  # fmt: skip
        base = 0.55 if needs_to else 0.6
        return Candidate("unassigned", None, rest, base,
                         ["'we' commitment: no single owner"])  # fmt: skip
    match = RE_I_TAKING.match(clause)
    if match and ctx.speaker is not None:
        return Candidate("commitment", ctx.speaker, match.group("rest"), 0.85,
                         [f"first-person '{match.group(0).split()[1]}' by {ctx.speaker}"])  # fmt: skip
    match = RE_NAMED.match(clause)
    if match and _named_owner(match.group("subj")):
        owner = _named_owner(match.group("subj"))
        return Candidate("commitment", owner, match.group("rest"), 0.9,
                         [f"named subject '{owner}' with a future modal"])  # fmt: skip
    match = RE_NAMED_TAKING.match(clause)
    if match and _named_owner(match.group("subj")):
        owner = _named_owner(match.group("subj"))
        return Candidate("commitment", owner, "Take " + match.group("rest"), 0.85,
                         [f"named subject '{owner}' taking the work"])  # fmt: skip
    match = RE_NAMED_TO.match(clause)
    if (
        match
        and _named_owner(match.group("subj"))
        and is_action_verb_start(match.group("rest"))
    ):
        owner = _named_owner(match.group("subj"))
        return Candidate("assigned", owner, match.group("rest"), 0.85,
                         [f"'{owner} to <verb>' assignment"])  # fmt: skip
    match = RE_SOMEONE.match(clause)
    if match:
        return Candidate("unassigned", None, match.group("rest"), 0.55,
                         [f"'{match.group('subj')}': no single owner"])  # fmt: skip
    match = RE_NEED_SOMEONE.match(clause)
    if match:
        return Candidate("unassigned", None, match.group("rest"), 0.55,
                         ["'need someone to': no single owner"])  # fmt: skip
    match = RE_NAMED_SHOULD.match(clause)
    if match and _named_owner(match.group("subj")):
        owner = _named_owner(match.group("subj"))
        return Candidate("assigned", owner, match.group("rest"), 0.7,
                         [f"'{owner} should': assigned, softer than a commitment"])  # fmt: skip
    match = RE_LETS.match(clause)
    if match and is_action_verb_start(match.group("rest")):
        return Candidate("unassigned", None, match.group("rest"), 0.6,
                         ["'let's' suggestion to act: no single owner"])  # fmt: skip
    return None


def _tentative(clause: str) -> Candidate | None:
    match = RE_FIRST_TENTATIVE.match(clause)
    if match:
        return Candidate("tentative", None, match.group("rest"), 0.35,
                         ["hedged: 'should/could/might/can' is not a commitment"])  # fmt: skip
    return None


def _request(clause: str, ctx: Context) -> Candidate | None:
    match = RE_REQUEST_YOU.match(clause)
    if match:
        rest = match.group("rest")
        addr = _named_owner(match.group("addr") or "")
        if _first_word(rest) in NON_ACTION_REQUESTS or not rest.strip():
            return None
        if addr:
            return Candidate("request", addr, rest, 0.6,
                             [f"request addressed to {addr}"], addressee=addr,
                             needs_acceptance=True)  # fmt: skip
        return Candidate("request", None, rest, 0.4,
                         ["request without a named addressee"], needs_acceptance=True)  # fmt: skip
    match = RE_REQUEST_ANYONE.match(clause)
    if match:
        return Candidate("request", None, match.group("rest"), 0.4,
                         ["request to the room: needs an acceptance"], needs_acceptance=True)  # fmt: skip
    match = RE_NEED_YOU.match(clause)
    if match:
        addr = _named_owner(match.group("addr") or "")
        reason = (
            f"request addressed to {addr}"
            if addr
            else "request without a named addressee"
        )
        return Candidate(
            "request",
            addr,
            match.group("rest"),
            0.6 if addr else 0.4,
            [reason],
            addressee=addr,
            needs_acceptance=True,
        )
    match = RE_ADDRESSED_SHOULD.match(clause)
    if match and _named_owner(match.group("addr")):
        addr = match.group("addr")
        return Candidate("request", addr, match.group("rest"), 0.6,
                         [f"request addressed to {addr}"], addressee=addr,
                         needs_acceptance=True)  # fmt: skip
    match = RE_PLEASE.match(clause)
    if match and is_action_verb_start(match.group("rest")):
        addr = _named_owner(match.group("addr") or "")
        base = 0.6 if addr else 0.4
        return Candidate("request", addr, match.group("rest"), base,
                         ["imperative request"], addressee=addr, needs_acceptance=True)  # fmt: skip
    match = RE_VOCATIVE.match(clause)
    if (
        match
        and is_action_verb_start(match.group("rest"))
        and _named_owner(match.group("addr"))
    ):
        addr = match.group("addr")
        return Candidate("assigned", addr, match.group("rest"), 0.75,
                         [f"imperative addressed to {addr}"])  # fmt: skip
    return None


def classify(clause: str, ctx: Context) -> Candidate | None:
    """Classify one clause; ``None`` means it is not an action item (decision, question, report...)."""
    text = " ".join(clause.split())
    hedged = bool(RE_HEDGE.match(text))
    text = RE_HEDGE.sub("", text, count=1)
    trailing_hedge = RE_TRAILING_HEDGE.search(text)
    if trailing_hedge:
        text, hedged = text[: trailing_hedge.start()], True
    text = RE_FILLER.sub("", text, count=1).strip()
    conditional = False
    prefix = RE_CONDITIONAL_PREFIX.match(text)
    if prefix:
        text, conditional = prefix.group("main"), True
    if RE_CONDITIONAL_SUFFIX.search(text):
        text, conditional = RE_CONDITIONAL_SUFFIX.sub("", text), True
    due = find_due_phrase(text)
    due_text = due.text if due else None
    if due:
        text = " ".join((text[: due.start] + " " + text[due.end :]).split()).strip(" ,")
        text = re.sub(r"\s+([.?!])", r"\1", text)
    if not text:
        return None
    candidate = _classify_core(text, ctx)
    if candidate is None:
        return None
    candidate.due_phrase = due_text
    candidate.conditional = conditional
    candidate.hedged = hedged
    if conditional:
        candidate.reasons.append("conditional: depends on a later event")
    if hedged:
        candidate.reasons.append("hedged wording")
    return candidate


def _classify_core(text: str, ctx: Context) -> Candidate | None:
    decision = RE_DECISION.search(text)
    if decision:
        after = decision.group("after")
        if after:
            inner = _commitment(after, ctx) or _request(after, ctx)
            if inner is not None and inner.item_type in (
                "commitment",
                "assigned",
                "unassigned",
            ):
                inner.reasons.append("embedded in a decision sentence")
                return inner
        return None
    if RE_DECISION_PHRASE.search(text) or RE_AGENDA_LETS.match(text):
        return None
    if RE_THINK_ABOUT.match(text):
        return None
    if text.endswith("?") and _request(text, ctx) is None:
        return None
    if RE_QUESTION_START.match(text) and _request(text, ctx) is None:
        return None
    request = _request(text, ctx)
    if request is not None:
        return request
    if RE_NEGATION.search(text) and (
        _commitment(text, ctx)
        or _tentative(text)
        or RE_FIRST.match(text)
        or RE_NAMED.match(text)
    ):
        return None
    commitment = _commitment(text, ctx)
    if commitment is not None:
        return commitment
    tentative = _tentative(text)
    if tentative is not None:
        return tentative
    if ctx.kind == "checkbox_open" or ctx.kind == "action_section":
        return _task_line(text, ctx)
    return None


def _task_line(text: str, ctx: Context) -> Candidate | None:
    """A checkbox or action-section line: a task whether or not it has a modal."""
    if RE_NEGATION.search(text) or RE_QUESTION_START.match(text):
        return None
    lead = RE_LEAD_OWNER.match(text)
    if lead and looks_like_name(lead.group("name")):
        owner, rest = lead.group("name"), lead.group("rest")
    else:
        owner, rest = owner_in_text(text)
    speaker = ctx.speaker
    named_to = RE_NAMED_TO.match(rest)
    if owner is None and named_to and _named_owner(named_to.group("subj")):
        owner, rest = named_to.group("subj"), named_to.group("rest")
    # The speaker is only set for a turn with an explicit "Name:" prefix (formats.py clears
    # the carried speaker on checkbox lines), so it is a deliberate owner marker here.
    if owner is None and speaker is not None:
        owner = speaker
    if owner is None:
        verb_led = is_action_verb_start(rest)
        if ctx.kind == "checkbox_open":
            return Candidate("todo", None, rest, 0.7, ["checkbox without an owner"])
        return Candidate("unassigned", None, rest, 0.6 if verb_led else 0.5,
                         ["action-section line without an owner"])  # fmt: skip
    if ctx.kind == "checkbox_open":
        return Candidate("todo", owner, rest, 0.85, [f"open checkbox for {owner}"])
    return Candidate("assigned", owner, rest, 0.8, [f"action-section line for {owner}"])
