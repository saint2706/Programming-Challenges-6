import json

from helpers import analyze, make_repo
from readme_generator.manifests import (
    parse_cargo,
    parse_composer,
    parse_gemfile,
    parse_go_mod,
    parse_manifests,
    parse_package_json,
    parse_pom,
    parse_pyproject,
    parse_requirements,
    python_requirement_name,
)
from readme_generator.scan import scan_repo


def test_python_requirement_name_strips_extras_markers_and_normalises():
    assert (
        python_requirement_name("Uvicorn[standard]>=0.5; python_version>'3.9'")
        == "uvicorn"
    )
    assert python_requirement_name("scikit_learn==1.4") == "scikit-learn"
    assert python_requirement_name("== nonsense") is None


def test_pyproject_pep621_with_groups_and_scripts():
    manifest = parse_pyproject(
        "pyproject.toml",
        """
[project]
name = "demo"
version = "1.2.3"
description = "A demo."
requires-python = ">=3.12"
license = {text = "MIT"}
dependencies = ["fastapi>=0.1", "Typer[all]"]
[project.optional-dependencies]
extra = ["rich"]
[project.scripts]
demo = "demo.cli:app"
[dependency-groups]
dev = ["pytest>=9", "ruff"]
[build-system]
build-backend = "uv_build"
""",
    )
    assert (manifest.name, manifest.version, manifest.license, manifest.requires) == (
        "demo",
        "1.2.3",
        "MIT",
        ">=3.12",
    )
    assert manifest.dependencies == ["fastapi", "typer"]
    assert manifest.dev_dependencies == ["rich", "pytest", "ruff"]
    assert manifest.scripts == {"demo": "demo.cli:app"}
    assert manifest.extras["backend"] == "uv_build"


def test_pyproject_poetry_layout():
    manifest = parse_pyproject(
        "pyproject.toml",
        """
[tool.poetry]
name = "poet"
description = "Poetry project."
[tool.poetry.dependencies]
python = "^3.11"
Django = "^5"
[tool.poetry.group.dev.dependencies]
pytest = "*"
[tool.poetry.scripts]
poet = "poet.main:run"
""",
    )
    assert manifest.dependencies == ["django"]
    assert manifest.dev_dependencies == ["pytest"]
    assert manifest.requires == "^3.11"
    assert manifest.extras["tool.poetry"] == "1"
    assert manifest.scripts == {"poet": "poet.main:run"}


def test_requirements_skips_flags_urls_and_comments_and_classifies_dev_files():
    text = "# comment\n-r base.txt\nflask==3.0  # web\ngit+https://x/y.git\nrequests\n"
    assert parse_requirements("requirements.txt", text).dependencies == [
        "flask",
        "requests",
    ]
    assert parse_requirements("requirements-dev.txt", "pytest\n").dev_dependencies == [
        "pytest"
    ]


def test_package_json_scripts_bin_and_package_manager():
    manifest = parse_package_json(
        "package.json",
        json.dumps(
            {
                "name": "web",
                "version": "2.0.0",
                "license": "ISC",
                "engines": {"node": ">=20"},
                "dependencies": {"react": "^18"},
                "devDependencies": {"vite": "^5"},
                "scripts": {"dev": "vite", "test": "vitest"},
                "bin": {"web-cli": "cli.js"},
                "packageManager": "pnpm@9.1.0",
            }
        ),
    )
    assert manifest.requires == ">=20"
    assert manifest.dependencies == ["react"] and manifest.dev_dependencies == ["vite"]
    assert manifest.extras["packageManager"] == "pnpm"
    assert manifest.extras["bin"] == "web-cli"
    assert manifest.scripts["dev"] == "vite"


def test_cargo_go_gemfile_composer_pom():
    cargo = parse_cargo(
        "Cargo.toml",
        '[package]\nname="c"\nversion="0.1.0"\nrust-version="1.75"\n[dependencies]\ntokio="1"\n[[bin]]\nname="c-cli"\npath="src/cli.rs"\n',
    )
    assert (cargo.name, cargo.requires, cargo.dependencies, cargo.scripts) == (
        "c",
        "1.75",
        ["tokio"],
        {"c-cli": "src/cli.rs"},
    )

    go = parse_go_mod(
        "go.mod",
        "module example.com/svc\n\ngo 1.22\n\nrequire (\n\tgithub.com/gin-gonic/gin v1.9.1\n\tgolang.org/x/net v0.1.0 // indirect\n)\nrequire github.com/spf13/cobra v1.8.0\n",
    )
    assert (go.name, go.requires) == ("example.com/svc", "1.22")
    assert go.dependencies == [
        "github.com/gin-gonic/gin",
        "golang.org/x/net",
        "github.com/spf13/cobra",
    ]

    gem = parse_gemfile(
        "Gemfile", "ruby '3.3.0'\ngem 'rails', '~> 7'\n  gem \"rspec\"\n"
    )
    assert gem.dependencies == ["rails", "rspec"] and gem.requires == "3.3.0"

    composer = parse_composer(
        "composer.json",
        json.dumps(
            {
                "name": "a/b",
                "license": ["MIT"],
                "require": {
                    "php": ">=8.2",
                    "ext-json": "*",
                    "laravel/framework": "^11",
                },
            }
        ),
    )
    assert (
        composer.requires == ">=8.2"
        and composer.dependencies == ["laravel/framework"]
        and composer.license == "MIT"
    )

    pom = parse_pom(
        "pom.xml",
        '<project xmlns="http://maven.apache.org/POM/4.0.0"><artifactId>svc</artifactId><version>1.0</version>'
        "<dependencies><dependency><artifactId>spring-boot-starter-web</artifactId></dependency>"
        "<dependency><artifactId>junit</artifactId><scope>test</scope></dependency></dependencies></project>",
    )
    assert (
        pom.name == "svc"
        and pom.dependencies == ["spring-boot-starter-web"]
        and pom.dev_dependencies == ["junit"]
    )


def test_malformed_manifest_becomes_a_warning_not_a_crash(tmp_path):
    make_repo(
        tmp_path,
        {"package.json": "{not json", "pyproject.toml": "[project]\nname = 'ok'\n"},
    )
    manifests, warnings = parse_manifests(scan_repo(tmp_path))
    assert [m.name for m in manifests] == ["ok"]
    assert len(warnings) == 1 and "package.json" in warnings[0]


def test_fixture_and_deep_manifests_are_ignored(tmp_path):
    _, manifests, _, _, _ = analyze(
        tmp_path,
        {
            "package.json": '{"name": "root"}',
            "tests/fixtures/package.json": '{"name": "fixture"}',
            "examples/demo/package.json": '{"name": "example"}',
            "a/b/c/d/package.json": '{"name": "too-deep"}',
            "packages/ui/package.json": '{"name": "ui"}',
        },
    )
    assert [m.name for m in manifests] == ["root", "ui"]


def test_manifests_are_sorted_shallowest_first(tmp_path):
    _, manifests, _, _, _ = analyze(
        tmp_path,
        {
            "z/package.json": '{"name": "z"}',
            "package.json": '{"name": "root"}',
            "a/b/package.json": '{"name": "ab"}',
        },
    )
    assert [m.path for m in manifests] == [
        "package.json",
        "z/package.json",
        "a/b/package.json",
    ]
