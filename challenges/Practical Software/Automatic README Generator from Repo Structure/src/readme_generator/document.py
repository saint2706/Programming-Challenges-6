"""Marker-delimited generated blocks, so regenerating never touches prose written around them."""

import re
from dataclasses import dataclass, field

from readme_generator.render import SECTION_ORDER, Section

BLOCK = re.compile(
    r"<!-- readme-gen:begin (?P<id>[\w-]+) -->\n(?P<body>.*?)\n<!-- readme-gen:end (?P=id) -->",
    re.DOTALL,
)
_REMOVED = "\x00removed\x00"  # placeholder so the blank lines around a dropped block go with it


def wrap(section: Section) -> str:
    return f"<!-- readme-gen:begin {section.id} -->\n{section.body}\n<!-- readme-gen:end {section.id} -->"


def render_document(sections: list[Section]) -> str:
    return "\n\n".join(wrap(s) for s in sections) + "\n"


def has_blocks(text: str) -> bool:
    return BLOCK.search(text) is not None


@dataclass
class MergeReport:
    updated: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def merge_document(existing: str, sections: list[Section]) -> tuple[str, MergeReport]:
    """Refresh the generated blocks inside ``existing``.

    Blocks still produced are replaced in place, blocks no longer applicable are deleted (a stale
    "Docker" section is worse than none), and new sections are inserted after the nearest earlier
    section in ``SECTION_ORDER``. Everything outside the markers is left byte-for-byte alone.
    """
    wanted = {s.id: s for s in sections}
    report = MergeReport()

    def replace(match: re.Match[str]) -> str:
        section = wanted.get(match.group("id"))
        if section is None:
            report.removed.append(match.group("id"))
            return _REMOVED
        if section.body != match.group("body"):
            report.updated.append(section.id)
        return wrap(section)

    text = BLOCK.sub(replace, existing)
    text = re.sub(f"{_REMOVED}\n*", "", text)
    present = {m.group("id") for m in BLOCK.finditer(text)}
    for section in sections:
        if section.id in present:
            continue
        text = _insert(text, section, present)
        present.add(section.id)
        report.added.append(section.id)
    return text, report


def _insert(text: str, section: Section, present: set[str]) -> str:
    rank = (
        SECTION_ORDER.index(section.id)
        if section.id in SECTION_ORDER
        else len(SECTION_ORDER)
    )
    earlier = [i for i in SECTION_ORDER[:rank] if i in present]
    block = wrap(section)
    if earlier:
        anchor = f"<!-- readme-gen:end {earlier[-1]} -->"
        position = text.index(anchor) + len(anchor)
        return text[:position] + "\n\n" + block + text[position:]
    later = [i for i in SECTION_ORDER[rank + 1 :] if i in present]
    if later:
        anchor = f"<!-- readme-gen:begin {later[0]} -->"
        position = text.index(anchor)
        return text[:position] + block + "\n\n" + text[position:]
    return text.rstrip("\n") + "\n\n" + block + "\n"
