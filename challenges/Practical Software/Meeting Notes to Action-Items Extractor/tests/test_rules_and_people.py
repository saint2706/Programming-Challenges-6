import pytest
from actionitems.people import Roster, clean_name, looks_like_name
from helpers import run

# Each case is one speaker line on the Monday 2026-11-02 meeting with the default roster.
# expected: (owner, due ISO date or None, item_type)
COMMITMENTS = [
    (
        "Marco Rossi: I'll send the deck by Friday.",
        "Marco Rossi",
        "2026-11-06",
        "commitment",
    ),
    (
        "Dana Lee: I will book the room tomorrow.",
        "Dana Lee",
        "2026-11-03",
        "commitment",
    ),
    ("Dana Lee: I'll book it on Monday.", "Dana Lee", "2026-11-02", "commitment"),
    ("Dana Lee: I'll do it next Friday.", "Dana Lee", "2026-11-13", "commitment"),
    ("Dana Lee: I'll do it in 3 days.", "Dana Lee", "2026-11-05", "commitment"),
    ("Dana Lee: I'll fix it today.", "Dana Lee", "2026-11-02", "commitment"),
    ("Dana Lee: I'll do it by end of week.", "Dana Lee", "2026-11-06", "commitment"),
    (
        "Ana Costa: Yes, I will share it by Friday.",
        "Ana Costa",
        "2026-11-06",
        "commitment",
    ),
]

REQUESTS = [
    ("Priya Shah: Dana, can you book a room?", "Dana Lee", None, "request"),
    ("Priya Shah: Eli, please follow up with legal.", "Eli Park", None, "request"),
    (
        "Priya Shah: Dana to run the regression suite on Tuesday.",
        "Dana Lee",
        "2026-11-03",
        "assigned",
    ),
    (
        "Marco Rossi: Eli will confirm the seat count by 2026-11-20.",
        "Eli Park",
        "2026-11-20",
        "commitment",
    ),
]

UNASSIGNED = [
    (
        "Priya Shah: Someone should update the risk register before next week.",
        None,
        "2026-11-09",
        "unassigned",
    ),
    ("Priya Shah: We need someone to test the export flow.", None, None, "unassigned"),
]

CONDITIONAL = [
    (
        "Marco Rossi: If legal signs off, I will also publish the FAQ.",
        "Marco Rossi",
        None,
        "commitment",
    ),
]

CHECKBOXES = [
    (
        "- [ ] Send the pricing deck to the client (Marco Rossi) by Friday",
        "Marco Rossi",
        "2026-11-06",
    ),
    ("- [ ] Update the status page @Dana", "Dana Lee", None),
]


@pytest.mark.parametrize(
    ("line", "owner", "due", "item_type"),
    COMMITMENTS + REQUESTS + UNASSIGNED + CONDITIONAL,
)
def test_task_owner_due_and_type(line, owner, due, item_type):
    items = run(line).items
    assert len(items) == 1, [i.task for i in items]
    item = items[0]
    assert item.owner == owner
    assert item.due == due
    assert item.item_type == item_type


@pytest.mark.parametrize(
    "line",
    [
        "Priya Shah: Let's get started on the launch checklist.",
        "Marco Rossi: We decided to delay the beta until December.",
        "Dana Lee: Should we move the review to Monday?",
        "Ana Costa: Yes, I can do that.",
        "Priya Shah: Let's skip the renewal discussion this week.",
    ],
)
def test_non_actions_produce_nothing(line):
    assert run(line).items == []


def test_hedged_suggestion_is_filtered_by_default_threshold():
    text = "Dana Lee: I should probably check the logs."
    assert run(text).items == []
    kept = run(text, min_confidence=0.3).items
    assert len(kept) == 1
    assert kept[0].confidence < 0.4


def test_checkbox_open_items_take_owner_and_due_from_the_line():
    for line, owner, due in CHECKBOXES:
        (item,) = run(line, meeting="2026-11-02").items
        assert item.owner == owner
        assert item.due == due


def test_checkbox_done_items_are_not_actions():
    assert run("- [x] Book the review room (Dana Lee)").items == []


def test_speaker_is_owner_when_text_names_nobody():
    (item,) = run("Dana Lee: I'll book the room.").items
    assert item.owner == "Dana Lee"


def test_items_keep_their_source_line():
    text = "Priya Shah: Thanks.\nMarco Rossi: I'll send it by Friday."
    (item,) = run(text).items
    assert item.sources[0].line == 2
    assert item.sources[0].speaker == "Marco Rossi"


def test_duplicate_mentions_merge_into_one_item():
    text = "Marco Rossi: I'll send the deck by Friday.\nMarco Rossi: I'll send the deck by Friday."
    items = run(text).items
    assert len(items) == 1
    assert items[0].mentions == 2


# ---- owners --------------------------------------------------------------------------------


def test_roster_exact_match_is_full_confidence():
    res = Roster(["Priya Shah"]).resolve("Priya Shah")
    assert (res.name, res.confidence) == ("Priya Shah", 1.0)


def test_roster_match_is_case_insensitive():
    assert Roster(["Priya Shah"]).resolve("priya SHAH").name == "Priya Shah"


def test_roster_first_name_match():
    res = Roster(["Priya Shah"]).resolve("Priya")
    assert (res.name, res.confidence) == ("Priya Shah", 0.9)


def test_roster_fuzzy_match_is_accepted_with_lower_confidence():
    res = Roster(["Priya Shah"]).resolve("Priyaa")
    assert res.name == "Priya Shah"
    assert 0.6 < res.confidence < 0.9


def test_roster_unknown_name_is_kept_with_low_confidence():
    res = Roster(["Priya Shah"]).resolve("Zed")
    assert (res.name, res.confidence) == ("Zed", 0.6)


def test_roster_without_attendees_keeps_name_at_moderate_confidence():
    res = Roster(None).resolve("Zed")
    assert (res.name, res.confidence) == ("Zed", 0.7)


def test_ambiguous_first_name_is_flagged():
    res = Roster(["Dana Lee", "Dana Kim"]).resolve("Dana")
    assert res.confidence == 0.5


def test_speakers_join_the_roster():
    res = Roster(None, speakers=["Noah Kim"]).resolve("Noah")
    assert res.name == "Noah Kim"


def test_empty_name_resolves_to_nothing():
    res = Roster(["Priya Shah"]).resolve("   ")
    assert res.name is None and res.confidence == 0.0


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("Marco's", "Marco"),
        ("  Ana   Costa, ", "Ana Costa"),
        ("(Dana Lee)", "Dana Lee"),
    ],
)
def test_clean_name(raw, cleaned):
    assert clean_name(raw) == cleaned


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Priya Shah", True), ("Someone", False), ("Next Week", False)],
)
def test_looks_like_name(raw, expected):
    assert looks_like_name(raw) is expected
