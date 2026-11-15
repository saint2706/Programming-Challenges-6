"""Tag columns that likely hold personal data, from the column name and from sampled values.

A value rule fires when at least 80% of the sampled non-empty values match, so one stray email in
a free-text column does not tag it. Every hit also adds the umbrella tag `pii`.
"""

from __future__ import annotations

import re

NAME_RULES = {
    "email": r"e ?mail",
    "phone": r"phone|mobile|\bfax\b|\btel\b",
    "ip_address": r"\bip ?(addr|address)?\b",
    "person_name": r"(first|last|full|given|family|middle|sur) ?name|\bsurname\b",
    "address": r"address|street|postal ?code|\bzip( ?code)?\b",
    "birth_date": r"birth|\bdob\b",
    "national_id": r"\bssn\b|social ?security|passport|national ?id",
    "payment_card": r"card ?(no|num|number)|\bcc ?num|credit ?card|\bpan\b",
}
VALUE_RULES = {
    "email": re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$"),
    "phone": re.compile(r"^\+?\(?\d{1,4}\)?([ .-]?\d{2,4}){2,4}$"),
    "ip_address": re.compile(
        r"^(25[0-5]|2[0-4]\d|1?\d?\d)(\.(25[0-5]|2[0-4]\d|1?\d?\d)){3}$"
    ),
    "national_id": re.compile(r"^\d{3}-\d{2}-\d{4}$"),
}
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
PLAIN_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
NAME_RES = {tag: re.compile(rx) for tag, rx in NAME_RULES.items()}


def _names(name: str) -> tuple[str, str]:
    """customerEmail / Customer-Email -> ("customer email", "customeremail")."""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    spaced = " ".join(re.findall(r"[a-z0-9]+", name.lower()))
    return spaced, spaced.replace(" ", "")


def _name_hit(rx: re.Pattern, name: str) -> bool:
    return any(rx.search(n) for n in _names(name))


def luhn(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def _card(v: str) -> bool:
    d = re.sub(r"[ -]", "", v)
    return d.isdigit() and 13 <= len(d) <= 19 and luhn(d)


def tag_column(name: str, samples: list[str]) -> list[str]:
    tags = {tag for tag, rx in NAME_RES.items() if _name_hit(rx, name)}
    vals = [v.strip() for v in samples if v and v.strip()]
    if vals:
        need = 0.8 * len(vals)
        for tag, rx in VALUE_RULES.items():
            hits = [v for v in vals if rx.match(v)]
            if (
                tag == "phone"
            ):  # ISO dates and short codes have the shape but are not phones
                hits = [
                    v
                    for v in hits
                    if 7 <= sum(ch.isdigit() for ch in v) <= 15  # E.164 max
                    and not ISO_DATE.match(v)
                    and not PLAIN_NUMBER.match(v)
                ]
            if len(hits) >= need:
                tags.add(tag)
        if sum(_card(v) for v in vals) >= need:
            tags.add("payment_card")
    # A phone-shaped value is also any plain number; only trust it beside a phone-ish name or
    # with separators, so numeric ids are not tagged.
    if (
        "phone" in tags
        and not _name_hit(NAME_RES["phone"], name)
        and (
            tags & {"payment_card", "national_id"}
            or not all(re.search(r"[ .()+-]", v) for v in vals)
        )
    ):
        tags.discard("phone")
    if tags & {"ip_address", "email"}:
        tags.discard(
            "address"
        )  # "ip address" and "email address" are not postal addresses
    return sorted(tags | {"pii"}) if tags else []
