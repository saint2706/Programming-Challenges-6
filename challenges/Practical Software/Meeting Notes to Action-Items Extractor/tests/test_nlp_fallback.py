from actionitems import nlp
from actionitems.extract import Options, extract_document
from actionitems.ingest import Document

TEXT = "Priya Shah: Let's start.\nMarco Rossi: I'll send the deck by Friday."


def _extract(use_nlp: bool):
    options = Options(
        attendees=("Priya Shah", "Marco Rossi"),
        meeting_date="2026-11-02",
        timezone="UTC",
        use_nlp=use_nlp,
    )
    return extract_document(Document("notes.txt", TEXT, None), options)


def test_missing_spacy_model_falls_back_to_rules_with_the_same_items(monkeypatch):
    monkeypatch.setattr(nlp, "_load", lambda: None)
    assert nlp.backend_name() == "rules-only"
    assert nlp.person_names(TEXT) is None
    with_nlp = _extract(use_nlp=True)
    rules_only = _extract(use_nlp=False)
    assert with_nlp.backend == "rules-only"
    assert [i.task for i in with_nlp.items] == [i.task for i in rules_only.items]
