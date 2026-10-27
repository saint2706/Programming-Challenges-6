"""Optional spaCy support: person entities (NER) used to confirm owners the roster does not know.

spaCy is a hard dependency of the package, but the model is loaded lazily and only once. If the
model cannot be loaded, every function here returns ``None`` and the pipeline runs rules-only;
the report records which backend was used so results are never silently mixed.
"""

from functools import lru_cache

MODEL_NAME = "en_core_web_sm"


@lru_cache(maxsize=1)
def _load():  # pragma: no cover - depends on the installed model
    try:
        import spacy

        return spacy.load(MODEL_NAME, disable=["lemmatizer"])
    except Exception:  # noqa: BLE001 - any failure means "no NLP backend"
        return None


def backend_name() -> str:
    return f"spacy:{MODEL_NAME}" if _load() is not None else "rules-only"


def person_names(text: str) -> set[str] | None:
    """Return PERSON entity strings in ``text``, or ``None`` when no model is available."""
    nlp = _load()
    if nlp is None:
        return None
    return {ent.text for ent in nlp(text).ents if ent.label_ == "PERSON"}
