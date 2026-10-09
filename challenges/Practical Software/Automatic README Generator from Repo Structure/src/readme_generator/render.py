"""Turn the inferred facts into README sections."""

from dataclasses import dataclass

from readme_generator.manifests import Manifest
from readme_generator.scan import RepoScan
from readme_generator.signals import Signals
from readme_generator.stack import ECOSYSTEM_LABEL, Stack
from readme_generator.tree import render_tree

TODO_DESCRIPTION = "_TODO: describe what this project does and who it is for._"
SECTION_ORDER = [
    "header", "stack", "getting-started", "make", "packages", "configuration",
    "docker", "structure", "ci", "contributing", "license",
]  # fmt: skip
MAX_LANGUAGES = 4


@dataclass(frozen=True)
class Section:
    id: str
    body: str  # Markdown including its heading


def project_name(scan: RepoScan, manifests: list[Manifest]) -> str:
    for manifest in manifests:
        if not manifest.directory and manifest.name:
            return manifest.name
    return scan.root.name


def _code(lines: list[str]) -> str:
    return "```bash\n" + "\n".join(lines) + "\n```"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def header_section(name: str, signals: Signals) -> Section:
    return Section("header", f"# {name}\n\n{signals.description or TODO_DESCRIPTION}")


def stack_section(scan: RepoScan, stack: Stack) -> Section | None:
    lines: list[str] = []
    shares = scan.language_shares()[:MAX_LANGUAGES]
    if shares:
        lines.append(
            "- **Languages:** "
            + ", ".join(f"{name} ({share:.0%})" for name, share in shares)
        )
    for category, names in stack.technologies.items():
        lines.append(f"- **{category}:** {', '.join(names)}")
    if stack.package_managers:
        label = (
            "Package manager"
            if len(stack.package_managers) == 1
            else "Package managers"
        )
        lines.append(f"- **{label}:** {', '.join(stack.package_managers)}")
    return Section("stack", "## Tech stack\n\n" + "\n".join(lines)) if lines else None


def getting_started_section(stack: Stack) -> Section | None:
    c = stack.commands
    parts: list[str] = []
    if c.prerequisites:
        parts.append(
            "### Prerequisites\n\n" + "\n".join(f"- {p}" for p in c.prerequisites)
        )
    if c.install:
        parts.append("### Installation\n\n" + _code(c.install))
    if c.build:
        parts.append("### Build\n\n" + _code(c.build))
    if c.run:
        rows = [[f"`{command}`", what] for command, what in c.run]
        parts.append("### Usage\n\n" + _table(["Command", "What it runs"], rows))
    if c.test:
        parts.append("### Testing\n\n" + _code(c.test))
    if c.lint:
        parts.append("### Linting\n\n" + _code(c.lint))
    return (
        Section("getting-started", "## Getting started\n\n" + "\n\n".join(parts))
        if parts
        else None
    )


def make_section(signals: Signals) -> Section | None:
    if not signals.make_targets:
        return None
    rows = [
        [f"`make {target}`", doc or "—"] for target, doc in signals.make_targets[:15]
    ]
    return Section(
        "make", "## Make targets\n\n" + _table(["Command", "Description"], rows)
    )


def packages_section(manifests: list[Manifest]) -> Section | None:
    packages = [
        m for m in manifests if m.directory and m.name and not m.path.endswith(".txt")
    ]
    if len(packages) < 2:
        return None
    rows = [
        [
            f"`{m.directory}`",
            m.name or "",
            ECOSYSTEM_LABEL[m.ecosystem],
            m.description or "—",
        ]
        for m in packages
    ]
    return Section(
        "packages",
        "## Packages\n\n" + _table(["Path", "Name", "Ecosystem", "Description"], rows),
    )


def configuration_section(signals: Signals) -> Section | None:
    if not signals.env_vars:
        return None
    rows = [
        [f"`{v.name}`", f"`{v.default}`" if v.default else "—", v.comment or "—"]
        for v in signals.env_vars
    ]
    intro = f"Copy `{signals.env_template}` to `.env` and fill in the values:\n\n{_code(['cp ' + signals.env_template + ' .env'])}"
    return Section(
        "configuration",
        "## Configuration\n\n"
        + intro
        + "\n\n"
        + _table(["Variable", "Default", "Description"], rows),
    )


def docker_section(name: str, signals: Signals) -> Section | None:
    lines: list[str] = []
    if signals.dockerfile:
        tag = name.split("/")[-1].lower().replace(" ", "-")
        lines += [f"docker build -t {tag} ."]
    if signals.compose_file:
        lines.append("docker compose up")
    if not lines:
        return None
    body = "## Docker\n\n" + _code(lines)
    if signals.compose_services:
        body += "\n\nCompose services: " + ", ".join(
            f"`{s}`" for s in signals.compose_services
        )
    return Section("docker", body)


def structure_section(
    scan: RepoScan, name: str, depth: int, exclude: frozenset[str]
) -> Section | None:
    if not scan.files:
        return None
    return Section(
        "structure",
        "## Project structure\n\n```text\n"
        + render_tree(scan, name, depth, exclude)
        + "\n```",
    )


def ci_section(signals: Signals) -> Section | None:
    if not signals.ci:
        return None
    return Section(
        "ci",
        "## Continuous integration\n\n" + "\n".join(f"- `{w}`" for w in signals.ci),
    )


def build_sections(
    scan: RepoScan, manifests: list[Manifest], stack: Stack, signals: Signals,
    tree_depth: int = 2, exclude: frozenset[str] = frozenset(),
) -> list[Section]:  # fmt: skip
    name = project_name(scan, manifests)
    candidates = [
        header_section(name, signals),
        stack_section(scan, stack),
        getting_started_section(stack),
        make_section(signals),
        packages_section(manifests),
        configuration_section(signals),
        docker_section(name, signals),
        structure_section(scan, scan.root.name, tree_depth, exclude),
        ci_section(signals),
        Section(
            "contributing", "## Contributing\n\nSee [CONTRIBUTING.md](CONTRIBUTING.md)."
        )
        if signals.contributing
        else None,
        Section("license", f"## License\n\n{signals.license}.")
        if signals.license
        else None,
    ]
    return [s for s in candidates if s is not None]
