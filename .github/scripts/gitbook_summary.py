"""Write challenges/SUMMARY.md, the table of contents GitBook builds the docs site from.

.gitbook.yaml makes challenges/ the GitBook root, so every page the site publishes is listed in
that file. Only challenges whose README says "Status: Implemented" are listed; the "Not started"
READMEs are a title and one sentence.

Run it after finishing or renaming a challenge (the scheduled auto-fix workflow also runs it):

    python .github/scripts/gitbook_summary.py
"""

import re
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
CHALLENGES = ROOT / "challenges"
SUMMARY = CHALLENGES / "SUMMARY.md"
HEADING = re.compile(r"^# (.+?)\s*$")
IMPLEMENTED = re.compile(r"^\*\*Status:\*\*\s+Implemented\b", re.MULTILINE)


def subdirs(parent: Path) -> list[Path]:
    return sorted(
        (p for p in parent.iterdir() if p.is_dir()), key=lambda p: p.name.casefold()
    )


def title_of(readme: str, fallback: str) -> str:
    match = HEADING.match(readme.splitlines()[0]) if readme else None
    title = match.group(1) if match else fallback
    return title.replace("[", r"\[").replace("]", r"\]")


def build() -> str:
    lines = ["# Summary", "", "- [Programming Challenges 6](README.md)"]
    for category in subdirs(CHALLENGES):
        entries = []
        for challenge in subdirs(category):
            readme = challenge / "README.md"
            if not readme.is_file():
                continue
            text = readme.read_text(encoding="utf-8")
            if IMPLEMENTED.search(text):
                path = quote(f"{category.name}/{challenge.name}/README.md")
                entries.append(f"- [{title_of(text, challenge.name)}]({path})")
        if entries:
            lines += ["", f"## {category.name}", "", *entries]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    SUMMARY.write_text(build(), encoding="utf-8", newline="\n")
    print(f"wrote {SUMMARY.relative_to(ROOT)}")
