"""Infer technologies, package managers and the install/run/test commands from manifests."""

import re
from dataclasses import dataclass, field

from readme_generator.manifests import Manifest
from readme_generator.scan import RepoScan

# dependency name -> (display name, category). Names are matched after per-ecosystem normalisation.
PYTHON_TECH = {
    "fastapi": ("FastAPI", "Web framework"), "flask": ("Flask", "Web framework"),
    "django": ("Django", "Web framework"), "starlette": ("Starlette", "Web framework"),
    "streamlit": ("Streamlit", "Web framework"), "gradio": ("Gradio", "Web framework"),
    "uvicorn": ("Uvicorn", "Web framework"), "typer": ("Typer", "CLI"), "click": ("Click", "CLI"),
    "textual": ("Textual", "CLI"), "rich": ("Rich", "CLI"), "sqlalchemy": ("SQLAlchemy", "Data"),
    "sqlmodel": ("SQLModel", "Data"), "pandas": ("pandas", "Data"), "polars": ("Polars", "Data"),
    "numpy": ("NumPy", "Data"), "duckdb": ("DuckDB", "Data"), "scikit-learn": ("scikit-learn", "ML"),
    "torch": ("PyTorch", "ML"), "tensorflow": ("TensorFlow", "ML"), "pydantic": ("Pydantic", "Data"),
    "celery": ("Celery", "Infrastructure"), "requests": ("Requests", "Networking"),
    "httpx": ("HTTPX", "Networking"), "pytest": ("pytest", "Testing"),
    "ruff": ("Ruff", "Tooling"), "black": ("Black", "Tooling"), "mypy": ("mypy", "Tooling"),
    "pyright": ("Pyright", "Tooling"), "flake8": ("Flake8", "Tooling"),
}  # fmt: skip
NODE_TECH = {
    "react": ("React", "UI"), "next": ("Next.js", "Web framework"), "vue": ("Vue", "UI"),
    "nuxt": ("Nuxt", "Web framework"), "svelte": ("Svelte", "UI"), "@sveltejs/kit": ("SvelteKit", "Web framework"),
    "@angular/core": ("Angular", "UI"), "express": ("Express", "Web framework"),
    "fastify": ("Fastify", "Web framework"), "koa": ("Koa", "Web framework"),
    "@nestjs/core": ("NestJS", "Web framework"), "astro": ("Astro", "Web framework"),
    "vite": ("Vite", "Tooling"), "webpack": ("webpack", "Tooling"), "typescript": ("TypeScript", "Tooling"),
    "tailwindcss": ("Tailwind CSS", "UI"), "eslint": ("ESLint", "Tooling"), "prettier": ("Prettier", "Tooling"),
    "jest": ("Jest", "Testing"), "vitest": ("Vitest", "Testing"), "mocha": ("Mocha", "Testing"),
    "playwright": ("Playwright", "Testing"), "@playwright/test": ("Playwright", "Testing"),
    "cypress": ("Cypress", "Testing"), "prisma": ("Prisma", "Data"), "@prisma/client": ("Prisma", "Data"),
    "mongoose": ("Mongoose", "Data"), "electron": ("Electron", "UI"),
}  # fmt: skip
RUST_TECH = {
    "tokio": ("Tokio", "Infrastructure"), "axum": ("Axum", "Web framework"),
    "actix-web": ("Actix Web", "Web framework"), "rocket": ("Rocket", "Web framework"),
    "clap": ("clap", "CLI"), "serde": ("Serde", "Data"), "sqlx": ("SQLx", "Data"),
    "diesel": ("Diesel", "Data"), "rayon": ("Rayon", "Infrastructure"), "criterion": ("Criterion", "Testing"),
}  # fmt: skip
GO_TECH = {
    "github.com/gin-gonic/gin": ("Gin", "Web framework"), "github.com/labstack/echo": ("Echo", "Web framework"),
    "github.com/gofiber/fiber": ("Fiber", "Web framework"), "github.com/go-chi/chi": ("chi", "Web framework"),
    "github.com/spf13/cobra": ("Cobra", "CLI"), "gorm.io/gorm": ("GORM", "Data"),
    "github.com/stretchr/testify": ("testify", "Testing"),
}  # fmt: skip
RUBY_TECH = {
    "rails": ("Rails", "Web framework"), "sinatra": ("Sinatra", "Web framework"),
    "rspec": ("RSpec", "Testing"), "rubocop": ("RuboCop", "Tooling"),
}  # fmt: skip
PHP_TECH = {
    "laravel/framework": ("Laravel", "Web framework"), "symfony/framework-bundle": ("Symfony", "Web framework"),
    "phpunit/phpunit": ("PHPUnit", "Testing"),
}  # fmt: skip
JAVA_TECH = {
    "spring-boot-starter-web": ("Spring Boot", "Web framework"), "junit-jupiter": ("JUnit", "Testing"),
    "junit": ("JUnit", "Testing"),
}  # fmt: skip
TECH_TABLES = {
    "python": PYTHON_TECH, "node": NODE_TECH, "rust": RUST_TECH, "go": GO_TECH,
    "ruby": RUBY_TECH, "php": PHP_TECH, "java": JAVA_TECH,
}  # fmt: skip

CATEGORY_ORDER = [
    "Web framework",
    "UI",
    "CLI",
    "Data",
    "ML",
    "Networking",
    "Infrastructure",
    "Testing",
    "Tooling",
]
ECOSYSTEM_LABEL = {
    "python": "Python", "node": "Node.js", "rust": "Rust", "go": "Go", "ruby": "Ruby", "php": "PHP", "java": "Java",
}  # fmt: skip


@dataclass
class Commands:
    prerequisites: list[str] = field(default_factory=list)
    install: list[str] = field(default_factory=list)
    run: list[tuple[str, str]] = field(default_factory=list)  # (command, what it does)
    test: list[str] = field(default_factory=list)
    lint: list[str] = field(default_factory=list)
    build: list[str] = field(default_factory=list)


@dataclass
class Stack:
    ecosystems: list[str]  # ecosystems with a root manifest, in manifest order
    technologies: dict[str, list[str]]  # category -> display names
    package_managers: list[str]
    commands: Commands


def _lookup(
    table: dict[str, tuple[str, str]], dependency: str
) -> tuple[str, str] | None:
    if dependency in table:
        return table[dependency]
    # Go modules and Rust crates are matched by module prefix (``.../gin/v2`` is still Gin).
    for key, value in table.items():
        if "/" in key and dependency.startswith(key + "/"):
            return value
    return None


def detect_technologies(manifests: list[Manifest]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for manifest in manifests:
        table = TECH_TABLES.get(manifest.ecosystem, {})
        for dependency in manifest.all_dependencies:
            hit = _lookup(table, dependency)
            if hit and hit[0] not in found.setdefault(hit[1], []):
                found[hit[1]].append(hit[0])
    ordered = {
        category: sorted(found[category], key=str.lower)
        for category in CATEGORY_ORDER
        if category in found
    }
    return ordered


_IMPORTS_STREAMLIT = re.compile(r"^\s*(import streamlit|from streamlit)", re.MULTILINE)
_ASGI_APP = re.compile(r"^(\w+)\s*(?::[^=\n]+)?=\s*(FastAPI|Flask)\(", re.MULTILINE)
MAX_APP_FILES = 300


def python_app_hints(
    scan: RepoScan, runner: str, dependencies: set[str]
) -> list[tuple[str, str]]:
    """Start commands for web apps, found by locating the ``app = FastAPI()`` object or Streamlit script."""
    if not dependencies & {"fastapi", "flask", "streamlit"}:
        return []
    hints: list[tuple[str, str]] = []
    python_files = [
        f
        for f in scan.files
        if f.endswith(".py") and not f.startswith(("tests/", "test/"))
    ]
    for path in python_files[:MAX_APP_FILES]:
        text = scan.read_text(path, limit=64_000)
        module = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
        for variable, framework in _ASGI_APP.findall(text):
            if framework == "FastAPI" and "uvicorn" in dependencies:
                hints.append(
                    (
                        f"{runner}uvicorn {module}:{variable} --reload",
                        "FastAPI development server",
                    )
                )
            elif framework == "Flask":
                hints.append(
                    (
                        f"{runner}flask --app {module}:{variable} run --debug",
                        "Flask development server",
                    )
                )
        is_streamlit_script = path.rsplit("/", 1)[-1] in {"app.py", "streamlit_app.py"}
        if (
            "streamlit" in dependencies
            and is_streamlit_script
            and _IMPORTS_STREAMLIT.search(text)
        ):
            hints.append((f"{runner}streamlit run {path}", "Streamlit app"))
    return hints[:4]


def _python_commands(
    scan: RepoScan, root: Manifest, commands: Commands, manifests: list[Manifest]
) -> list[str]:
    dependencies = {
        d for m in manifests if m.ecosystem == "python" for d in m.all_dependencies
    }
    has_pyproject = scan.has_file(root.path) and root.path.endswith("pyproject.toml")
    if (
        scan.has_file("uv.lock")
        or "tool.uv" in root.extras
        or root.extras.get("backend", "").startswith("uv_build")
    ):
        manager, runner = "uv", "uv run "
        commands.prerequisites.append("[uv](https://docs.astral.sh/uv/)")
        commands.install.append("uv sync")
    elif scan.has_file("poetry.lock") or "tool.poetry" in root.extras:
        manager, runner = "Poetry", "poetry run "
        commands.prerequisites.append("[Poetry](https://python-poetry.org/)")
        commands.install.append("poetry install")
    elif has_pyproject:
        manager, runner = "pip", ""
        commands.install += [
            "python -m venv .venv",
            "source .venv/bin/activate",
            "pip install -e .",
        ]
    else:
        manager, runner = "pip", ""
        commands.install += ["python -m venv .venv", "source .venv/bin/activate"]
        requirement_files = [
            m.path
            for m in manifests
            if m.ecosystem == "python" and m.path.endswith(".txt")
        ]
        requirement_files.sort(key=lambda p: ("dev" in p or "test" in p, p))
        commands.install += [f"pip install -r {p}" for p in requirement_files[:2]]
    if root.requires:
        commands.prerequisites.insert(0, f"Python `{root.requires}`")
    for script, target in root.scripts.items():
        commands.run.append((f"{runner}{script}", f"entry point `{target}`"))
    commands.run += python_app_hints(scan, runner, dependencies)
    if not root.scripts:
        entry = next(
            (f for f in scan.files if f.endswith("/__main__.py") and f.count("/") <= 2),
            None,
        )
        if entry:
            module = entry.rsplit("/", 1)[0].removeprefix("src/").replace("/", ".")
            commands.run.append((f"{runner}python -m {module}", "package entry point"))
    if "pytest" in dependencies or any(
        f.startswith("tests/") and f.endswith(".py") for f in scan.files
    ):
        commands.test.append(f"{runner}pytest")
    if "ruff" in dependencies or scan.has_file("ruff.toml"):
        commands.lint.append(f"{runner}ruff check .")
    return [manager]


def _node_commands(scan: RepoScan, root: Manifest, commands: Commands) -> list[str]:
    declared = root.extras.get("packageManager")
    if declared in {"pnpm", "yarn", "npm", "bun"}:
        manager = declared
    elif scan.has_file("pnpm-lock.yaml"):
        manager = "pnpm"
    elif scan.has_file("yarn.lock"):
        manager = "yarn"
    elif scan.has_file("bun.lockb") or scan.has_file("bun.lock"):
        manager = "bun"
    else:
        manager = "npm"
    commands.install.append(f"{manager} install")
    if root.requires:
        commands.prerequisites.insert(0, f"Node.js `{root.requires}`")
    else:
        commands.prerequisites.insert(0, "Node.js")

    def script_cmd(name: str) -> str:
        shorthand = name in {"start", "test"} and manager in {
            "npm",
            "pnpm",
            "yarn",
            "bun",
        }
        return (
            f"{manager} {name}"
            if shorthand or manager == "yarn"
            else f"{manager} run {name}"
        )

    for name in ("dev", "start"):
        if name in root.scripts:
            commands.run.append((script_cmd(name), f"`{root.scripts[name]}`"))
    if "build" in root.scripts:
        commands.build.append(script_cmd("build"))
    if "test" in root.scripts:
        commands.test.append(script_cmd("test"))
    for name in ("lint", "format"):
        if name in root.scripts:
            commands.lint.append(script_cmd(name))
    if "bin" in root.extras:
        commands.run.append(
            (root.extras["bin"].split(", ")[0], "installed CLI (`bin` entry)")
        )
    return [manager]


def _rust_commands(scan: RepoScan, root: Manifest, commands: Commands) -> list[str]:
    commands.prerequisites.append(
        "[Rust toolchain](https://rustup.rs/)"
        + (f" `{root.requires}`" if root.requires else "")
    )
    commands.build.append("cargo build --release")
    if root.scripts:
        commands.run += [
            (f"cargo run --bin {name}", "binary target") for name in root.scripts
        ]
    elif scan.has_file(f"{root.directory + '/' if root.directory else ''}src/main.rs"):
        commands.run.append(("cargo run", "default binary"))
    commands.test.append("cargo test")
    commands.lint.append("cargo clippy")
    return ["Cargo"]


def _go_commands(scan: RepoScan, root: Manifest, commands: Commands) -> list[str]:
    commands.prerequisites.append(
        "[Go](https://go.dev/dl/)" + (f" {root.requires}" if root.requires else "")
    )
    commands.install.append("go mod download")
    commands.build.append("go build ./...")
    if scan.has_file("main.go"):
        commands.run.append(("go run .", "root `main` package"))
    for directory in sorted(
        {
            f.rsplit("/", 1)[0]
            for f in scan.files
            if f.startswith("cmd/") and f.endswith(".go")
        }
    ):
        commands.run.append((f"go run ./{directory}", "command"))
    commands.test.append("go test ./...")
    return ["Go modules"]


def _ruby_commands(scan: RepoScan, root: Manifest, commands: Commands) -> list[str]:
    commands.prerequisites.append(
        "Ruby" + (f" `{root.requires}`" if root.requires else "")
    )
    commands.install.append("bundle install")
    if "rails" in root.dependencies:
        commands.run.append(("bin/rails server", "Rails development server"))
    if "rspec" in root.all_dependencies or scan.has_file(".rspec"):
        commands.test.append("bundle exec rspec")
    return ["Bundler"]


def _php_commands(scan: RepoScan, root: Manifest, commands: Commands) -> list[str]:
    commands.prerequisites.append(
        "PHP" + (f" `{root.requires}`" if root.requires else "")
    )
    commands.prerequisites.append("[Composer](https://getcomposer.org/)")
    commands.install.append("composer install")
    if "test" in root.scripts:
        commands.test.append("composer test")
    elif "phpunit/phpunit" in root.all_dependencies:
        commands.test.append("vendor/bin/phpunit")
    return ["Composer"]


def _java_commands(scan: RepoScan, root: Manifest, commands: Commands) -> list[str]:
    commands.prerequisites.append("JDK")
    commands.build.append("mvn package")
    commands.test.append("mvn test")
    return ["Maven"]


def analyze_stack(scan: RepoScan, manifests: list[Manifest]) -> Stack:
    """Technologies from every manifest; commands from the root manifest of each ecosystem."""
    commands = Commands()
    ecosystems: list[str] = []
    managers: list[str] = []
    handlers = {
        "python": lambda root: _python_commands(scan, root, commands, manifests),
        "node": lambda root: _node_commands(scan, root, commands),
        "rust": lambda root: _rust_commands(scan, root, commands),
        "go": lambda root: _go_commands(scan, root, commands),
        "ruby": lambda root: _ruby_commands(scan, root, commands),
        "php": lambda root: _php_commands(scan, root, commands),
        "java": lambda root: _java_commands(scan, root, commands),
    }
    for manifest in manifests:
        if manifest.directory or manifest.ecosystem in ecosystems:
            continue
        # A requirements file only stands in for a project when no pyproject.toml is present.
        if manifest.path.endswith(".txt") and any(
            m.path == "pyproject.toml" for m in manifests
        ):
            continue
        ecosystems.append(manifest.ecosystem)
        managers += handlers[manifest.ecosystem](manifest)
    has_gradle = scan.has_file("build.gradle") or scan.has_file("build.gradle.kts")
    if has_gradle and "java" not in ecosystems:
        ecosystems.append("java")
        commands.prerequisites.append("JDK")
        commands.build.append("./gradlew build")
        commands.test.append("./gradlew test")
        managers.append("Gradle")
    return Stack(
        ecosystems,
        detect_technologies(manifests),
        list(dict.fromkeys(managers)),
        commands,
    )
