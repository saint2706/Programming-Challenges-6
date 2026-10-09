"""Parse the dependency manifests a repository ships into one ecosystem-neutral shape."""

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from posixpath import basename, dirname

import tomllib

from readme_generator.scan import RepoScan

# Directories whose manifests describe fixtures or samples, not part of the project itself.
NON_PROJECT_DIRS = frozenset(
    {
        "tests",
        "test",
        "fixtures",
        "testdata",
        "examples",
        "example",
        "docs",
        "samples",
        "third_party",
    }
)
MAX_MANIFEST_DEPTH = 3


@dataclass
class Manifest:
    ecosystem: str  # python | node | rust | go | ruby | php | java
    path: str  # posix path of the manifest file relative to the repo root
    name: str | None = None
    version: str | None = None
    description: str | None = None
    license: str | None = None
    requires: str | None = None  # runtime constraint, e.g. ">=3.12" or "1.22"
    dependencies: list[str] = field(default_factory=list)
    dev_dependencies: list[str] = field(default_factory=list)
    scripts: dict[str, str] = field(
        default_factory=dict
    )  # entry points or package scripts
    extras: dict[str, str] = field(default_factory=dict)  # ecosystem-specific facts

    @property
    def directory(self) -> str:
        return dirname(self.path)

    @property
    def all_dependencies(self) -> list[str]:
        return [*self.dependencies, *self.dev_dependencies]


_PEP508_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def python_requirement_name(spec: str) -> str | None:
    """The normalised distribution name of a PEP 508 requirement string."""
    match = _PEP508_NAME.match(spec)
    return re.sub(r"[-_.]+", "-", match.group(1)).lower() if match else None


def _python_names(specs: list) -> list[str]:
    names = (python_requirement_name(s) for s in specs if isinstance(s, str))
    return [n for n in names if n and n != "python"]


def parse_pyproject(path: str, text: str) -> Manifest:
    data = tomllib.loads(text)
    project = data.get("project", {})
    poetry = data.get("tool", {}).get("poetry", {})
    manifest = Manifest("python", path, name=project.get("name") or poetry.get("name"))
    manifest.version = project.get("version") or poetry.get("version")
    manifest.description = project.get("description") or poetry.get("description")
    license_value = project.get("license") or poetry.get("license")
    if isinstance(license_value, dict):
        license_value = license_value.get("text") or license_value.get("file")
    manifest.license = license_value if isinstance(license_value, str) else None
    manifest.requires = project.get("requires-python")
    manifest.dependencies = _python_names(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        manifest.dev_dependencies += _python_names(group)
    for group in data.get("dependency-groups", {}).values():
        manifest.dev_dependencies += _python_names(group)
    poetry_deps = {
        k: v for k, v in poetry.get("dependencies", {}).items() if k.lower() != "python"
    }
    manifest.dependencies += [re.sub(r"[-_.]+", "-", k).lower() for k in poetry_deps]
    poetry_dev = list(poetry.get("dev-dependencies", {}))
    for group in poetry.get("group", {}).values():
        poetry_dev += list(group.get("dependencies", {}))
    manifest.dev_dependencies += [re.sub(r"[-_.]+", "-", k).lower() for k in poetry_dev]
    if not manifest.requires and isinstance(
        poetry.get("dependencies", {}).get("python"), str
    ):
        manifest.requires = poetry["dependencies"]["python"]
    scripts = {**poetry.get("scripts", {}), **project.get("scripts", {})}
    manifest.scripts = {k: str(v) for k, v in scripts.items()}
    manifest.dependencies = list(dict.fromkeys(manifest.dependencies))
    manifest.dev_dependencies = list(dict.fromkeys(manifest.dev_dependencies))
    manifest.extras["backend"] = str(
        data.get("build-system", {}).get("build-backend", "")
    )
    if "uv" in data.get("tool", {}):
        manifest.extras["tool.uv"] = "1"
    if poetry:
        manifest.extras["tool.poetry"] = "1"
    return manifest


def parse_requirements(path: str, text: str) -> Manifest:
    names: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "git+", "http://", "https://", ".", "/")):
            continue
        name = python_requirement_name(line)
        if name:
            names.append(name)
    is_dev = "dev" in basename(path).lower() or "test" in basename(path).lower()
    manifest = Manifest("python", path)
    if is_dev:
        manifest.dev_dependencies = list(dict.fromkeys(names))
    else:
        manifest.dependencies = list(dict.fromkeys(names))
    return manifest


def parse_package_json(path: str, text: str) -> Manifest:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError("package.json is not an object")
    license_value = data.get("license")
    manifest = Manifest(
        "node", path,
        name=data.get("name"), version=data.get("version"), description=data.get("description"),
        license=license_value if isinstance(license_value, str) else None,
        requires=(data.get("engines") or {}).get("node"),
    )  # fmt: skip
    manifest.dependencies = list(data.get("dependencies", {}))
    manifest.dev_dependencies = list(data.get("devDependencies", {})) + list(
        data.get("peerDependencies", {})
    )
    manifest.scripts = {k: str(v) for k, v in data.get("scripts", {}).items()}
    if isinstance(data.get("bin"), dict):
        manifest.extras["bin"] = ", ".join(data["bin"])
    elif data.get("bin"):
        manifest.extras["bin"] = str(data.get("name", ""))
    if data.get("packageManager"):
        manifest.extras["packageManager"] = str(data["packageManager"]).split("@")[0]
    if data.get("workspaces"):
        manifest.extras["workspaces"] = "1"
    if data.get("type"):
        manifest.extras["type"] = str(data["type"])
    return manifest


def parse_cargo(path: str, text: str) -> Manifest:
    data = tomllib.loads(text)
    package = data.get("package", {})
    manifest = Manifest(
        "rust", path, name=package.get("name"), version=package.get("version")
    )
    manifest.description = package.get("description")
    license_value = package.get("license")
    manifest.license = license_value if isinstance(license_value, str) else None
    manifest.requires = package.get("rust-version")
    manifest.dependencies = list(data.get("dependencies", {}))
    manifest.dev_dependencies = list(data.get("dev-dependencies", {}))
    manifest.scripts = {
        b["name"]: b.get("path", "") for b in data.get("bin", []) if "name" in b
    }
    if "workspace" in data:
        manifest.extras["workspace"] = "1"
    return manifest


_GO_REQUIRE_LINE = re.compile(r"^\s*([^\s/]+(?:/[^\s]+)+)\s+v\S+")


def parse_go_mod(path: str, text: str) -> Manifest:
    manifest = Manifest("go", path)
    in_block = False
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].rstrip()
        if line.startswith("module "):
            manifest.name = line.split(None, 1)[1].strip().strip('"')
        elif line.startswith("go ") and not manifest.requires:
            manifest.requires = line.split()[1]
        elif line.startswith("require ("):
            in_block = True
        elif in_block and line.strip() == ")":
            in_block = False
        elif in_block or line.startswith("require "):
            match = _GO_REQUIRE_LINE.match(line.removeprefix("require "))
            if match:
                manifest.dependencies.append(match.group(1))
    return manifest


_GEM_LINE = re.compile(r"""^\s*gem\s+['"]([^'"]+)['"]""")


def parse_gemfile(path: str, text: str) -> Manifest:
    manifest = Manifest("ruby", path)
    for line in text.splitlines():
        match = _GEM_LINE.match(line)
        if match:
            manifest.dependencies.append(match.group(1))
    ruby = re.search(r"""^\s*ruby\s+['"]([^'"]+)['"]""", text, re.MULTILINE)
    manifest.requires = ruby.group(1) if ruby else None
    return manifest


def parse_composer(path: str, text: str) -> Manifest:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError("composer.json is not an object")
    license_value = data.get("license")
    if isinstance(license_value, list):
        license_value = ", ".join(map(str, license_value))
    manifest = Manifest("php", path, name=data.get("name"), version=data.get("version"))
    manifest.description = data.get("description")
    manifest.license = license_value if isinstance(license_value, str) else None
    manifest.requires = (data.get("require") or {}).get("php")
    manifest.dependencies = [
        k for k in data.get("require", {}) if k != "php" and not k.startswith("ext-")
    ]
    manifest.dev_dependencies = list(data.get("require-dev", {}))
    manifest.scripts = {
        k: json.dumps(v) if not isinstance(v, str) else v
        for k, v in data.get("scripts", {}).items()
    }
    return manifest


def parse_pom(path: str, text: str) -> Manifest:
    root = ET.fromstring(text)
    ns = {"m": root.tag[1:].split("}")[0]} if root.tag.startswith("{") else {}

    def child(parent: ET.Element, tag: str) -> str | None:
        found = parent.find(f"m:{tag}", ns) if ns else parent.find(tag)
        return found.text.strip() if found is not None and found.text else None

    manifest = Manifest(
        "java", path, name=child(root, "artifactId"), version=child(root, "version")
    )
    manifest.description = child(root, "description")
    deps = (
        root.findall("m:dependencies/m:dependency", ns)
        if ns
        else root.findall("dependencies/dependency")
    )
    for dep in deps:
        artifact = child(dep, "artifactId")
        if artifact:
            (
                manifest.dev_dependencies
                if child(dep, "scope") == "test"
                else manifest.dependencies
            ).append(artifact)
    manifest.extras["build"] = "maven"
    return manifest


Parser = Callable[[str, str], Manifest]
PARSERS: dict[str, Parser] = {
    "pyproject.toml": parse_pyproject,
    "package.json": parse_package_json,
    "Cargo.toml": parse_cargo,
    "go.mod": parse_go_mod,
    "Gemfile": parse_gemfile,
    "composer.json": parse_composer,
    "pom.xml": parse_pom,
}
_REQUIREMENTS = re.compile(
    r"^requirements([-_.][\w.-]+)?\.txt$|^[\w.-]*requirements\.txt$"
)


def parse_manifests(scan: RepoScan) -> tuple[list[Manifest], list[str]]:
    """Every manifest within ``MAX_MANIFEST_DEPTH`` directories, shallowest first, plus warnings.

    A malformed manifest becomes a warning rather than an error: one broken fixture file must not
    stop a README from being drafted. Manifests under test/fixture/example directories are skipped
    because they describe sample projects, not this one.
    """
    manifests: list[Manifest] = []
    warnings: list[str] = []
    for path in scan.files:
        file_name = basename(path)
        parser = PARSERS.get(file_name)
        if parser is None and _REQUIREMENTS.match(file_name):
            parser = parse_requirements
        if parser is None:
            continue
        parts = path.split("/")[:-1]
        if len(parts) > MAX_MANIFEST_DEPTH or any(
            p.lower() in NON_PROJECT_DIRS for p in parts
        ):
            continue
        try:
            manifests.append(parser(path, scan.read_text(path)))
        except (ValueError, TypeError, ET.ParseError) as exc:
            warnings.append(f"Could not parse {path}: {exc}")
    manifests.sort(key=lambda m: (m.path.count("/"), m.path))
    return manifests, warnings
