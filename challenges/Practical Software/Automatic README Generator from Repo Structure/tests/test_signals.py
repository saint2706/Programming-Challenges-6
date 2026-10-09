from helpers import analyze
from readme_generator.signals import (
    parse_compose_services,
    parse_env_template,
    parse_makefile,
)


def test_env_template_documents_variables_and_never_echoes_secret_looking_values():
    text = (
        "# Public URL of the site\nSITE_URL=https://example.com\n\n"
        "API_TOKEN=sk-live-realkey\nDATABASE_DSN=postgres://u:p@h/db\n"
        "export PORT=8080 # listen port\nLONG=" + "x" * 50 + "\nEMPTY=\n"
    )
    by_name = {v.name: v for v in parse_env_template(text)}
    assert by_name["SITE_URL"].default == "https://example.com"
    assert by_name["SITE_URL"].comment == "Public URL of the site"
    assert by_name["API_TOKEN"].default is None
    assert by_name["DATABASE_DSN"].default is None
    assert by_name["PORT"].default == "8080"
    assert by_name["LONG"].default is None
    assert by_name["EMPTY"].default is None
    assert by_name["PORT"].comment is None  # a blank line ends a comment's reach


def test_makefile_targets_with_docs_skipping_phony_variables_and_recipes():
    text = ".PHONY: test\nCC := gcc\ntest: deps ## Run the tests\n\tpytest\nbuild:\n\tgo build\n# note\nbuild:\n%.o: %.c\n"
    assert parse_makefile(text) == [("test", "Run the tests"), ("build", "")]


def test_compose_services_read_by_indentation():
    text = "version: '3'\nservices:\n  web:\n    image: x\n    ports:\n      - '80:80'\n  db:\n    image: y\nvolumes:\n  data:\n"
    assert parse_compose_services(text) == ["web", "db"]
    assert parse_compose_services("name: x\n") == []


def test_license_comes_from_manifest_then_license_file(tmp_path):
    _, _, _, _, signals = analyze(
        tmp_path / "a", {"package.json": '{"name": "a", "license": "ISC"}'}
    )
    assert signals.license == "ISC"
    _, _, _, _, signals = analyze(
        tmp_path / "b", {"LICENSE": "MIT License\n\nCopyright (c) 2026"}
    )
    assert signals.license == "MIT"
    _, _, _, _, signals = analyze(tmp_path / "c", {"LICENSE": "Some bespoke terms"})
    assert signals.license == "see LICENSE"
    _, _, _, _, signals = analyze(tmp_path / "d", {"a.py": ""})
    assert signals.license is None


def test_description_prefers_manifest_then_package_docstring(tmp_path):
    files = {
        "pyproject.toml": '[project]\nname="p"\ndescription="From manifest."\n',
        "src/p/__init__.py": '"""Docstring."""',
    }
    assert analyze(tmp_path / "a", files)[4].description == "From manifest."
    files = {
        "pyproject.toml": '[project]\nname="p"\n',
        "src/p/__init__.py": '"""Docstring line.\n\nMore."""',
    }
    assert analyze(tmp_path / "b", files)[4].description == "Docstring line."
    assert (
        analyze(tmp_path / "c", {"pyproject.toml": '[project]\nname="p"\n'})[
            4
        ].description
        is None
    )


def test_ci_docker_and_contributing_are_detected(tmp_path):
    files = {
        ".github/workflows/test.yml": "",
        ".github/workflows/readme.md": "",
        "Dockerfile": "FROM x",
        "compose.yaml": "services:\n  app:\n    build: .\n",
        "CONTRIBUTING.md": "# Contributing",
    }
    _, _, _, _, signals = analyze(tmp_path, files)
    assert signals.ci == ["test.yml"]
    assert (
        signals.dockerfile
        and signals.compose_file == "compose.yaml"
        and signals.compose_services == ["app"]
    )
    assert signals.contributing
