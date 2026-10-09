"""Repository facts that are not dependency manifests: env vars, make targets, Docker, CI, license."""

import ast
import re
from dataclasses import dataclass, field
from posixpath import basename

from readme_generator.manifests import Manifest
from readme_generator.scan import RepoScan

ENV_TEMPLATES = (
    ".env.example",
    ".env.sample",
    ".env.template",
    ".env.dist",
    "example.env",
    "sample.env",
)
LICENSE_FILES = (
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "LICENCE",
    "COPYING",
    "UNLICENSE",
)
LICENSE_MARKERS = [
    ("MIT License", "MIT"), ("Apache License", "Apache-2.0"), ("Mozilla Public License", "MPL-2.0"),
    ("GNU AFFERO GENERAL PUBLIC LICENSE", "AGPL-3.0"), ("GNU LESSER GENERAL PUBLIC LICENSE", "LGPL"),
    ("GNU GENERAL PUBLIC LICENSE", "GPL"), ("This is free and unencumbered software", "Unlicense"),
    ("ISC License", "ISC"), ("BSD 3-Clause", "BSD-3-Clause"), ("BSD 2-Clause", "BSD-2-Clause"),
    ("Redistribution and use in source and binary forms", "BSD"),
]  # fmt: skip


@dataclass
class EnvVar:
    name: str
    default: str | None  # shown only when the template value is not secret-looking
    comment: str | None


@dataclass
class Signals:
    env_vars: list[EnvVar] = field(default_factory=list)
    env_template: str | None = None
    make_targets: list[tuple[str, str]] = field(
        default_factory=list
    )  # (target, doc or "")
    dockerfile: bool = False
    compose_file: str | None = None
    compose_services: list[str] = field(default_factory=list)
    ci: list[str] = field(default_factory=list)
    license: str | None = None
    contributing: bool = False
    description: str | None = None


_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_SECRET_NAME = re.compile(
    r"(SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|PRIVATE|CREDENTIAL|DSN)", re.IGNORECASE
)


def parse_env_template(text: str) -> list[EnvVar]:
    """Variables from an ``.env.example``; a ``#`` comment directly above a variable documents it."""
    variables: list[EnvVar] = []
    pending: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            pending.append(stripped.lstrip("# ").strip())
            continue
        match = _ENV_LINE.match(raw)
        if match:
            name, value = (
                match.group(1),
                match.group(2).split(" #", 1)[0].strip().strip("\"'"),
            )
            # A template should hold placeholders, but real keys get pasted into them; never echo one.
            hide = bool(_SECRET_NAME.search(name)) or len(value) > 40
            variables.append(
                EnvVar(
                    name,
                    None if hide or not value else value,
                    " ".join(pending) or None,
                )
            )
        pending = []
    return variables


_MAKE_TARGET = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)\s*:(?!=)(.*)$")


def parse_makefile(text: str) -> list[tuple[str, str]]:
    """Top-level targets with their ``## doc`` text, in file order (``.PHONY`` and pattern rules skipped)."""
    targets: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        match = _MAKE_TARGET.match(line)
        if not match or match.group(1) in seen:
            continue
        seen.add(match.group(1))
        doc = match.group(2).split("##", 1)[1].strip() if "##" in match.group(2) else ""
        targets.append((match.group(1), doc))
    return targets


def parse_compose_services(text: str) -> list[str]:
    """Service names from a compose file, read by indentation so no YAML dependency is needed."""
    services: list[str] = []
    in_services = False
    indent = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if re.match(r"^services\s*:", raw):
            in_services = True
            continue
        if in_services:
            if not raw.startswith((" ", "\t")):
                break
            width = len(raw) - len(raw.lstrip())
            indent = width if indent is None else indent
            match = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*:\s*$", raw)
            if match and width == indent:
                services.append(match.group(1))
    return services


def detect_license(scan: RepoScan, manifests: list[Manifest]) -> str | None:
    for manifest in manifests:
        if not manifest.directory and manifest.license:
            return manifest.license
    for name in LICENSE_FILES:
        if scan.has_file(name):
            head = scan.read_text(name, limit=4000)
            for marker, label in LICENSE_MARKERS:
                if marker.lower() in head.lower():
                    return label
            return "see LICENSE"
    return None


def package_docstring(scan: RepoScan, manifests: list[Manifest]) -> str | None:
    """First docstring line of the project's top-level package, as a last-resort description."""
    candidates = [
        f for f in scan.files if f.endswith("/__init__.py") and f.count("/") <= 2
    ]
    candidates.sort(key=lambda f: (not f.startswith("src/"), f.count("/"), f))
    for path in candidates:
        if "test" in path.lower():
            continue
        try:
            doc = ast.get_docstring(ast.parse(scan.read_text(path)))
        except SyntaxError:
            continue
        if doc:
            return doc.strip().splitlines()[0].strip()
    return None


def collect_signals(scan: RepoScan, manifests: list[Manifest]) -> Signals:
    signals = Signals()
    for template in ENV_TEMPLATES:
        if scan.has_file(template):
            signals.env_template = template
            signals.env_vars = parse_env_template(scan.read_text(template))
            break
    if scan.has_file("Makefile"):
        signals.make_targets = parse_makefile(scan.read_text("Makefile"))
    signals.dockerfile = scan.has_file("Dockerfile")
    for name in (
        "compose.yaml",
        "compose.yml",
        "docker-compose.yml",
        "docker-compose.yaml",
    ):
        if scan.has_file(name):
            signals.compose_file = name
            signals.compose_services = parse_compose_services(scan.read_text(name))
            break
    workflows = [
        f
        for f in scan.files
        if f.startswith(".github/workflows/") and f.endswith((".yml", ".yaml"))
    ]
    signals.ci = [basename(f) for f in workflows]
    if scan.has_file(".gitlab-ci.yml"):
        signals.ci.append(".gitlab-ci.yml")
    signals.license = detect_license(scan, manifests)
    signals.contributing = scan.has_file("CONTRIBUTING.md")
    signals.description = next(
        (m.description for m in manifests if not m.directory and m.description), None
    )
    if not signals.description:
        signals.description = package_docstring(scan, manifests)
    return signals
